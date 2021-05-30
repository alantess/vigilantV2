from PIL import Image
from scipy import misc, ndimage
import matplotlib.pyplot as plt
import torch
import numpy as np

from torchvision import models
from torchvision import transforms

from captum.attr import visualization as viz
from captum.attr import LayerGradCam, FeatureAblation, LayerActivation, LayerAttribution


def intepret_semantic_model(model, device, alpha=50):
    invTrans = transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.],
                             std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
    ])
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512, 512)),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    model = model.to(device).eval()
    model.load()
    image = Image.open("../etc/dash.jpg")
    preproc_img = preprocess(image)

    preproc_img = preproc_img.unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(preproc_img)
        mask = out[0].permute(1, 2, 0)
        mask = mask.mul(255).clamp(0, 255)
        mask = mask.detach().cpu().numpy().astype(np.float32)

    preproc_img = invTrans(preproc_img)
    plt.figure()
    plt.axis('off')
    plt.imshow(preproc_img[0].permute(1, 2, 0).cpu())
    plt.imshow(apply_sharpen_filter(mask, alpha),
               alpha=0.4,
               cmap='winter',
               interpolation='gaussian')
    plt.show()


def apply_sharpen_filter(img, alpha):
    blurred_filter = ndimage.gaussian_filter(img, 3)
    filter_blurred = ndimage.gaussian_filter(blurred_filter, 1)
    img = blurred_filter + alpha * (blurred_filter - filter_blurred)
    return img
