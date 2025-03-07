import torch
import math
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):   # pcd for X-ray imaging
    points : np.array
    opacities : np.array

rot_primary = lambda PrimaryAngle : np.array([
        [-np.sin(PrimaryAngle), np.cos(PrimaryAngle), 0, 0],
        [0, 0, -1, 0],
        [-np.cos(PrimaryAngle), -np.sin(PrimaryAngle), 0, 0],
        [0, 0, 0, 1]], dtype=np.float32)

rot_secondary = lambda SecondaryAngle : np.array([
        [1, 0, 0, 0],
        [0, np.cos(SecondaryAngle), -np.sin(SecondaryAngle), 0],
        [0, np.sin(SecondaryAngle), np.cos(SecondaryAngle), 0],
        [0, 0, 0, 1]], dtype=np.float32)

def get_extrinsic(PrimaryAngle, SecondaryAngle, sad):

    extr = np.array([[-np.sin(PrimaryAngle), np.cos(PrimaryAngle), 0, 0],
                     [np.sin(SecondaryAngle) * np.cos(PrimaryAngle), np.sin(SecondaryAngle) * np.sin(PrimaryAngle), -np.cos(SecondaryAngle), 0],
                     [-np.cos(SecondaryAngle) * np.cos(PrimaryAngle), -np.cos(SecondaryAngle) * np.sin(PrimaryAngle), -np.sin(SecondaryAngle), sad],
                     [0, 0, 0, 1]])
    
    return extr

def focal2fov(focal, d):
    return 2*math.atan(d/(2*focal))

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R
    Rt[:3, 3] = t
    Rt[3, 3] = 1
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def make_coords(volume_resolution, volume_phy, volume_origin):
    s1, s2, s3 = volume_phy
    o1, o2, o3 = volume_origin
    n1, n2, n3 = volume_resolution
    xyz = np.meshgrid(np.linspace(o1, s1 + o1, n1, endpoint=False) + s1/(2*n1),
                      np.linspace(o2, s2 + o2, n2, endpoint=False) + s2/(2*n2),
                      np.linspace(o3, s3 + o3, n3, endpoint=False) + s3/(2*n3), indexing='ij')
    xyz = np.asarray(xyz).transpose([1, 2, 3, 0])
    return xyz