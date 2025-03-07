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

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    pos = C2W[:3, 3]  # camera center
    rot = C2W[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'sid': camera.sid
    }
    return camera_entry
