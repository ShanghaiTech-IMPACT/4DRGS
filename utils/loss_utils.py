import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import numpy as np
from tqdm import tqdm

def tv_3d_loss(vol, reduction="sum"):

    dx = torch.abs(torch.diff(vol, dim=0))
    dy = torch.abs(torch.diff(vol, dim=1))
    dz = torch.abs(torch.diff(vol, dim=2))

    tv = torch.sum(dx) + torch.sum(dy) + torch.sum(dz)

    if reduction == "mean":
        total_elements = (
            (vol.shape[0] - 1) * vol.shape[1] * vol.shape[2]
            + vol.shape[0] * (vol.shape[1] - 1) * vol.shape[2]
            + vol.shape[0] * vol.shape[1] * (vol.shape[2] - 1)
        )
        tv = tv / total_elements
    return tv

def l1_loss(network_output, gt, weight=1):
    return torch.abs((network_output - gt)*weight).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    if isinstance(img1, np.ndarray):
        img1 = torch.from_numpy(img1)
    if isinstance(img2, np.ndarray):
        img2 = torch.from_numpy(img2)

    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def s_ssim(img1, img2, window_size=11, size_average=True):

    if isinstance(img1, np.ndarray):
        img1 = torch.from_numpy(img1)
    if isinstance(img2, np.ndarray):
        img2 = torch.from_numpy(img2)

    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C2 = 0.03 ** 2   

    s_ssim_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)

    if size_average:
        return s_ssim_map.mean()
    else:
        return s_ssim_map.mean(1).mean(1).mean(1)

def gradient_map(img):
    # Compute gradients in x and y directions using Sobel filters
    sobel_x = torch.tensor([[-1, 0, 1], 
                            [-2, 0, 2], 
                            [-1, 0, 1]], dtype=img.dtype, device=img.device).unsqueeze(0).unsqueeze(0)
    
    sobel_y = torch.tensor([[-1, -2, -1], 
                            [0,  0,  0], 
                            [1,  2,  1]], dtype=img.dtype, device=img.device).unsqueeze(0).unsqueeze(0)
    
    grad_x = F.conv2d(img, sobel_x, padding=1)
    grad_y = F.conv2d(img, sobel_y, padding=1)
    
    # Combine gradients to form a gradient magnitude map
    gradient = torch.sqrt(grad_x ** 2 + grad_y ** 2)
    return gradient

def G_ssim(img1, img2, window_size=11, size_average=True):
    
    grad1 = gradient_map(img1)
    grad2 = gradient_map(img2)

    return ssim(grad1, grad2, window_size, size_average)

def correlation_coefficient_loss(geomap, dsa_map):
    # Flatten the maps to vectors
    geomap_flat = geomap.view(geomap.shape[0], -1)  # Shape: [N, H*W]
    dsa_map_flat = dsa_map.view(dsa_map.shape[0], -1)  # Shape: [N, H*W]    

    # Mean of each map
    geomap_mean = geomap_flat.mean(dim=1, keepdim=True)  # Shape: [N, 1]
    dsa_map_mean = dsa_map_flat.mean(dim=1, keepdim=True)  # Shape: [N, 1]
    
    # Subtract the mean from each map (zero-centering)
    geomap_zero_mean = geomap_flat - geomap_mean
    dsa_map_zero_mean = dsa_map_flat - dsa_map_mean
    
    # Compute the numerator (covariance)
    numerator = (geomap_zero_mean * dsa_map_zero_mean).sum(dim=1)
    
    # Compute the denominator (product of standard deviations)
    geomap_std = torch.sqrt((geomap_zero_mean ** 2).sum(dim=1))
    dsa_map_std = torch.sqrt((dsa_map_zero_mean ** 2).sum(dim=1))
    denominator = geomap_std * dsa_map_std
    
    # Avoid division by zero by adding a small epsilon
    epsilon = 1e-8
    correlation = numerator / (denominator + epsilon)
    
    # Return mean correlation over the batch
    return 1-correlation[0]

def imgTVloss(img):
    # geomap [H, W]
    dx = torch.abs(img[:, :, :-1] - img[:, :, 1:]).mean()
    dy = torch.abs(img[:, :-1, :] - img[:, 1:, :]).mean()
    tv = (dx + dy) / 2
    return tv