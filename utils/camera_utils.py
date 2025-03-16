from scene.cameras import Camera
import numpy as np
import torch

def loadCam(args, id, cam_info):
    if args == None: data_device = 'cuda'
    else: data_device = args.data_device
    return Camera(frame_id=cam_info.uid, timestamp=cam_info.timestamp, R=cam_info.R, T=cam_info.T, PrimaryAngle=cam_info.PrimaryAngle,
                  SecondaryAngle=cam_info.SecondaryAngle, FoVx=cam_info.FovX, FoVy=cam_info.FovY, sid=cam_info.sid, sad=cam_info.sad, 
                  near=cam_info.near, far=cam_info.far, width=cam_info.width, height=cam_info.height, sx=cam_info.sx, sy=cam_info.sy,
                  image=cam_info.image, image_name=cam_info.image_name, uid=id, data_device=data_device)


def cameraList_from_camInfos(cam_infos, args=None):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c))

    return camera_list
