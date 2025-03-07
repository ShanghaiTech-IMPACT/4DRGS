import torch
import numpy as np
import math
import cv2
from skimage.metrics import structural_similarity

def clear(x):
    return x.detach().cpu().numpy()

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    if isinstance(img1, np.ndarray):
        img1 = torch.from_numpy(img1)
    if isinstance(img2, np.ndarray):
        img2 = torch.from_numpy(img2)
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def data_norm(data):
    return (data-data.min())/(data.max()-data.min())

def clip_norm(data, clamp_max):
    if isinstance(data, torch.Tensor):
        return torch.clamp(data, min=0, max=clamp_max)/clamp_max
    if isinstance(data, np.ndarray):
        return np.clip(data, a_min=0, a_max=clamp_max)/clamp_max

def array2video(images, write_path, fps=20):

    images = images * 255
    images = images.astype(np.uint8)

    height, width = images[0].shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
    out = cv2.VideoWriter(write_path, fourcc, fps, (width, height), isColor=False)

    for image in images:
        out.write(image)
    
    out.release()

def get_psnr(pred, target):
    """
    Compute PSNR of two tensors (2D/3D) in decibels.
    pred/target should be of same size or broadcastable
    The max intensity should be 1, thus, it's better
    to normalize into [0,1]
    """

    mse = ((pred - target) ** 2).mean()
    if mse!=0:
      psnr = -10 * math.log10(mse)
    else:
      psnr = 'INF'
    return psnr

def get_ssim_2d(pred, target, data_range=1):
    """
    Compute SSIM of two tensors (2D) in decibels.
    pred/target should be of same size or broadcastable
    The max intensity should be 1, thus, it's better
    to normalize into [0,1]
    """
    ssim = structural_similarity(pred, target, data_range=data_range)
    return ssim

def curve_contrast_adjustment(image, curve_points):
    # 将图像的像素值映射到曲线上的新像素值
    adjusted_image = np.interp(image, curve_points[:, 0], curve_points[:, 1])
    return adjusted_image


class Averager:
    def __init__(self):
        self.sum = 0
        self.count = 0
    
    def add(self, value):
        self.sum += value
        self.count += 1
    
    def avg(self):
        if self.count == 0:
            return 0
        return self.sum / self.count