import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils
from torchvision.transforms import functional

from models.network.crm_transferCoord_transferFeat import CRMNet

class SingleClassRefiner:
    
    def __init__(self, batch_size: int = 1, memory_chunk: int = 50176*16,
                 checkpoint_path: str = "weights/model_45705",
                 device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")) -> None:
        
        self.batch_size = batch_size
        self.memory_chunk = memory_chunk
        
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.model = nn.DataParallel(CRMNet(backend='resnet50').cuda())
        # self.model = nnCRMNet(backend='resnet50').to(self.device)
        self.model.load_state_dict(torch.load(checkpoint_path))
        
        self.im_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        self.seg_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5],
                std=[0.5]
            ),
        ])

        self.inv_im_trans = transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                std=[1/0.229, 1/0.224, 1/0.225]
                )

        self.inv_seg_trans = transforms.Normalize(
            mean=[-0.5/0.5],
            std=[1/0.5]
            )
        
    def __to_pixel_samples(self, img):
        """ Convert the image to coord-RGB pairs.
            img: Tensor, (3, H, W)
        """
        coord = self.__make_coord(img.shape[-2:])
        rgb = img.view(1, -1).permute(1, 0)
        return coord, rgb
    
    def __make_coord(self, shape, ranges=None, flatten=True):
        """ Make coordinates at grid centers.
        """
        coord_seqs = []
        for i, n in enumerate(shape):
            if ranges is None:
                v0, v1 = -1, 1
            else:
                v0, v1 = ranges[i]
            r = (v1 - v0) / (2 * n)
            seq = v0 + r + (2 * r) * torch.arange(n).float()
            coord_seqs.append(seq)
        ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
        if flatten:
            ret = ret.view(-1, ret.shape[-1])
        return ret

    def inv_transform_mask(self, x):
        x = x.detach().cpu()
        x = self.inv_seg_trans(x)
        return x

    @torch.no_grad()
    def __call__(self, image: torch.Tensor, mask_init: torch.Tensor) -> torch.Tensor:
        """
        image - [B=1, C=3, H, W]
        mask_init - [B=1, H, W]
        Returns:
             mask: [B=1, H, W]
        """

        # TODO: not batch-friendly
        im, seg = self.im_transform(image), self.seg_transform(mask_init)
        im, seg = im[None, :], seg[None, :]
        print(seg.shape)

        hr_coord, hr_rgb = self.__to_pixel_samples(seg.contiguous())

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / seg.shape[-2] 
        cell[:, 1] *= 2 / seg.shape[-1]

        crm_data = {}
        crm_data['coord'] = hr_coord[None, :]
        crm_data['cell'] = cell[None, :]

        torch.cuda.synchronize()
        start_batch_time = torch.cuda.Event(enable_timing=True)
        end_batch_time = torch.cuda.Event(enable_timing=True)
        start_batch_time.record()
        
        s = 1.0
        images = {}
        im_ = F.interpolate(im, size=(round(im.shape[-2] * s), round(im.shape[-1] * s)), mode='bilinear', align_corners=True).cuda()
        seg_ = F.interpolate(seg, size=(round(im.shape[-2] * s), round(im.shape[-1] * s)), mode='bilinear', align_corners=True).cuda()
    
        transferFeat = None
        transferCoord = None
        for i in range(0, seg.shape[-2]*seg.shape[-1], self.memory_chunk):
            torch.cuda.synchronize()
            start_chunk_time = torch.cuda.Event(enable_timing=True)
            end_chunk_time = torch.cuda.Event(enable_timing=True)
            start_chunk_time.record()
            if transferFeat is None:
                chunk_images, transferCoord, transferFeat = self.model(
                    im_, seg_,
                    coord=crm_data['coord'][:, i:i+self.memory_chunk, :],
                    cell=crm_data['cell'][:, i:i+self.memory_chunk, :],
                    transferCoord=transferCoord,
                    transferFeat=transferFeat)
            else:
                chunk_images = self.model(im_, seg_,
                                     coord=crm_data['coord'][:, i:i+self.memory_chunk, :],
                                     cell=crm_data['cell'][:, i:i+self.memory_chunk, :],
                                     transferCoord=transferCoord,
                                     transferFeat=transferFeat)
            
            if 'pred_224' not in images.keys():
                images = chunk_images
            else:
                for key in images.keys():
                    images[key] = torch.cat((images[key], chunk_images[key]), axis=1)

            torch.cuda.empty_cache()
            end_chunk_time.record()
            torch.cuda.synchronize()
            print("chunk_time:", start_chunk_time.elapsed_time(end_chunk_time))

        for key in images.keys(): 
            images[key] = images[key].view(images[key].shape[0], images[key].shape[1]//(seg.shape[-2]*seg.shape[-1]), * seg.shape[-2:])

        images['im'] = im
        # images['seg_'+str(turn)] = seg
        images['seg'] = seg
                
        # Suppress close-to-zero segmentation input
        for b in range(seg.shape[0]):
             if (seg[b]+1).sum() < 2:
                    images['pred_224'][b] = 0

        seg = (((images['pred_224'][0]).float()-0.5)*2).unsqueeze(0)

        end_batch_time.record()
        torch.cuda.synchronize()
        print("batch_time:", start_batch_time.elapsed_time(end_batch_time))
        
        return self.inv_transform_mask(seg)


class MultiClassSegmRefiner:

    def __init__(self, single_class_refiner: SingleClassRefiner, class_scaler: float = 1.5, ignore: int = 0):
        self.ignore = 0
        self.class_scaler = class_scaler
        self.single_class_refiner = single_class_refiner

    def __call__(self, image: torch.Tensor, mask_init: torch.Tensor) -> torch.Tensor:
        cropped_refs = {}
        for cl in np.unique(mask_init):
            if cl == self.ignore: continue

            cur_cls = mask_init.copy()
            cur_cls[cur_cls != cl] = 0
            bbox = self.extract_bboxes(cur_cls[None, :], self.class_scaler)[0]

            crp = cur_cls[bbox[1]:bbox[3],bbox[0]:bbox[2]]            
            refined_crop = self.single_class_refiner(image[bbox[1]:bbox[3],bbox[0]:bbox[2]], crp)

            restored_reff = torch.zeros(*mask_init.shape)
            restored_reff[bbox[1]:bbox[3],bbox[0]:bbox[2]] = refined_crop
            cropped_refs[cl] = restored_reff

        from_cropped_ref = torch.zeros(list(cropped_refs.items())[0][1].shape)
        for cl, ref in cropped_refs.items():
            from_cropped_ref[ref > 0.95] = cl

        return from_cropped_ref

    # extracts bounding boxes aroung people and scales correspondingly
    def extract_bboxes(self, mask, scale: float = 1.5):
        boxes = np.zeros([mask.shape[0], 4], dtype=np.int32)
        
        for i in range(mask.shape[0]):
            msk = mask[i, :, :]
            
            horizontal_indicies = np.where(np.any(msk, axis=0))[0]
            vertical_indicies = np.where(np.any(msk, axis=1))[0]
        
            h, w = msk.shape

            if horizontal_indicies.shape[0]:
                x1, x2 = horizontal_indicies[[0, -1]]
                y1, y2 = vertical_indicies[[0, -1]]
                x2 += 1
                y2 += 1
                
                bbwd = x2 - x1
                bbht = y2 - y1
                
                inv_width_scale = bbwd * scale**-1
                inv_height_scale = bbwd * scale**-1
                
                x1 = max(0, x1 - inv_width_scale)
                y1 = max(0, y1 - inv_height_scale)
                x2 = min(w, x2 + inv_width_scale)
                y2 = min(h, y2 + inv_height_scale)
                
            else:
                x1, x2, y1, y2 = 0, mask.shape[1], 0, mask.shape[2]

            boxes[i] = np.array([x1, y1, x2, y2])

        return boxes.astype(np.int32)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import imageio

    img_path, mask_path = "examples/im.jpg", "examples/original.png"
    im, mask = Image.open(img_path).convert('RGB'), Image.open(mask_path).convert('L')

    sr = SingleClassRefiner()    
    ref = sr(im, mask)

    imageio.imwrite('ref.png', (ref[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8))
