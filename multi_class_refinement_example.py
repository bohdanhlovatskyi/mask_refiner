import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import imageio

hp_path, gt_path = "examples/parsing_original.png", "examples/im.jpg"
im, mask = np.array(Image.open(gt_path).convert('RGB')), np.array(Image.open(hp_path).convert('L'))

from mask_refiner import MultiClassSegmRefiner
from mask_refiner import SingleClassRefiner

if __name__ == "__main__":
    sr = SingleClassRefiner()
    mcls_ref = MultiClassSegmRefiner(sr)

    ref = mcls_ref(im, mask)

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(22, 18))
    ax[0].imshow(im)
    ax[1].imshow(mask)
    ax[2].imshow(ref)

    plt.savefig("example_res.png")
