import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View, getProjectionMatrix

class Camera(nn.Module):
    def __init__(self, frame_id, timestamp, R, T, PrimaryAngle, SecondaryAngle, FoVx, FoVy, sid, sad, near, far, 
                 width, height, sx, sy, image, image_name, uid, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.frame_id = frame_id
        self.timestamp = timestamp
        self.R = R
        self.T = T
        self.PrimaryAngle = PrimaryAngle
        self.SecondaryAngle = SecondaryAngle
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.sid = sid
        self.sad = sad
        self.zfar = far     # zfar, znear
        self.znear = near
        self.width = width
        self.height = height
        self.sx = sx
        self.sy = sy
        self.image = image  # save for fdk recon
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = torch.from_numpy(image).to(self.data_device)
        self.image_width = width
        self.image_height = height

        self.world_view_transform = torch.tensor(getWorld2View(R, T)).transpose(0, 1).cuda()   # world2camera
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()   # camera2NDC
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)  # world2NDC # world coords * world2NDC = NDC (P, 4) * (4, 4) -> (P, 4)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

