import os
import json
from typing import NamedTuple
import numpy as np
from tqdm import tqdm
from utils.graphics_utils import get_extrinsic, focal2fov, getWorld2View, make_coords, BasicPointCloud
import SimpleITK as sitk
from ct.tigre_ct import tigre_ct
from plyfile import PlyData, PlyElement
from utils.camera_utils import cameraList_from_camInfos
from utils.image_utils import data_norm, clear

class CameraInfo(NamedTuple):
    uid: int
    timestamp: float
    R: np.array
    T: np.array
    PrimaryAngle: np.array
    SecondaryAngle: np.array
    sad: np.array
    sid: np.array
    FovY: np.array
    FovX: np.array
    sx: np.array
    sy: np.array
    near: np.array
    far: np.array
    image: np.array
    image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    recon_args: dict
    train_indice: np.array
    eval_indice: np.array
    all_indice: np.array

def get_near_far(sad, volume_phy):
    dist = np.linalg.norm([volume_phy[0]/2, volume_phy[1]/2])
    near = sad - dist
    far = sad + dist
    return near, far

def readCameras(camera_file, datapath):
    # load DSA proj
    mask_run_file = os.path.join(datapath, 'mask_run.nii.gz')
    fill_run_file = os.path.join(datapath, 'fill_run.nii.gz')
    DSA_file = os.path.join(datapath, 'DSA.nii.gz')

    mask_run = sitk.GetArrayFromImage(sitk.ReadImage(mask_run_file))
    fill_run = sitk.GetArrayFromImage(sitk.ReadImage(fill_run_file))

    proj = np.log(mask_run) - np.log(fill_run)
    proj = np.clip(proj, 0, proj.max())
    if not os.path.exists(DSA_file):
        sitk.WriteImage(sitk.GetImageFromArray(proj), DSA_file)

    # load paras
    with open(camera_file) as f:
        camera_paras = json.load(f)

    volume_phy = np.asarray(camera_paras['volume_phy'])
    volume_origin = -volume_phy/2
    recon_args = {'volume_resolution': camera_paras['volume_resolution'],
                   'volume_phy': volume_phy.tolist(),
                   'volume_spacing': camera_paras['volume_spacing'],
                   'volume_origin': volume_origin.tolist()}

    sad = camera_paras['sad']

    sid = camera_paras['sid']

    width, height = camera_paras['proj_resolution']

    dx, dy = camera_paras['proj_phy']

    FovX = focal2fov(sid, dx)
    FovY = focal2fov(sid, dy)

    sx, sy = camera_paras['proj_spacing']

    near, far = get_near_far(sad, volume_phy)

    # load camera
    cam_infos = []
    N_views = camera_paras['N_views']

    for idx in tqdm(range(0, N_views), desc='camera loading'):

        frame = camera_paras['frames'][idx]

        PrimaryAngle = frame['PrimaryAngle']
        SecondaryAngle = 0
        # if 'SecondaryAngle' in frame:
        #     SecondaryAngle = -frame['SecondaryAngle']
        # else:
        #     SecondaryAngle = 0

        uid = idx
        timestamp = uid/N_views

        extr = get_extrinsic(PrimaryAngle, SecondaryAngle, sad)   # world 2 camera
        R = extr[:3, :3]
        T = extr[:3, 3]

        image = proj[idx]  # [H, W]
        image = image[np.newaxis, :, :]  # [1, H, W]
        image_name = frame['file']

        cam_info = CameraInfo(uid=uid, timestamp=timestamp, R=R, T=T, PrimaryAngle=PrimaryAngle, SecondaryAngle=SecondaryAngle, sad=sad, sid=sid, 
                              FovY=FovY, FovX=FovX, sx=sx, sy=sy, near=near, far=far, image=image, image_name=image_name, width=width, height=height)

        cam_infos.append(cam_info)

    return cam_infos, recon_args

def create_camera_list(params, angle_list):
    # directly used for rendering

    sad = params['sad']
    sid = params['sid']
    width, height = params['proj_resolution']
    FovX, FovY = params['FovX'], params['FovY']
    sx, sy = params['sx'], params['sy']
    near, far = params['near'], params['far']
    
    cam_infos = []
    blank_image = np.zeros((1, height, width))  ## no available image
    for idx, PrimaryAngle in enumerate(tqdm(angle_list, desc='camera creating')):
        uid = idx
        timestamp = -1   ## no available timestamp
        extr = get_extrinsic(PrimaryAngle, 0, sad)
        R = extr[:3, :3]
        T = extr[:3, 3]
        image_name = str(idx).zfill(4)
        cam_info = CameraInfo(uid=uid, timestamp=timestamp, R=R, T=T, PrimaryAngle=PrimaryAngle, SecondaryAngle=0, sad=sad, sid=sid, FovY=FovY, FovX=FovX, sx=sx, sy=sy,
                              near=near, far=far, image=blank_image, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    cam_list = cameraList_from_camInfos(cam_infos)

    return cam_list

def get_indice(N_views, train_views):
    all_indice = np.arange(N_views)
    train_indice = np.arange(0, N_views, N_views/train_views).astype(int)   # note that train views could not be zero
    if train_indice[-1] >= N_views:
        train_indice[-1] = N_views - 1
    # train_indice = np.linspace(0, Nviews-1, train_views).astype(int)
    eval_indice = np.delete(all_indice, train_indice).astype(int)
    return train_indice, eval_indice, all_indice

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])
    
    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def vol_initializor(recon_vol, recon_args, init_args, type='fdk'):

    recon_vol = np.clip(recon_vol, a_min=0, a_max=recon_vol.max())
 
    M1 = init_args['M1']
    M2 = init_args['M2']
    if type == 'fdk':
        thres_percent = init_args['thres_percent_fdk']
    elif type == 'query':
        thres_percent = init_args['thres_percent_query']
    
    # img_max = recon_vol.max()
    # thres = img_max * thres_percent
    # mask = recon_vol > thres
    mask = recon_vol > thres_percent
    mask = mask.flatten()

    coords_matrix = make_coords(recon_args['volume_resolution'], recon_args['volume_phy'], recon_args['volume_origin'])
    coords_matrix = coords_matrix.reshape(-1, 3)

    coords_matrix_mask = coords_matrix[mask]
    recon_img_mask = recon_vol.reshape(-1, 1)[mask]
    # 从 mask 为 True 的点中随机取 M1 个点
    selected_indices = np.random.choice(coords_matrix_mask.shape[0], M1, replace=False)
    selected_points = coords_matrix_mask[selected_indices]
    selected_opacities = recon_img_mask[selected_indices]

    if M2 > 0:
        # # 从 mask 为 False 的点中选择 M2 个点
        # remaining_mask = ~mask
        # remaining_coords_matrix = coords_matrix[remaining_mask]
        # remaining_opacities = recon_vol.reshape(-1, 1)[remaining_mask]
        # # 按照 remaining_opacities 的大小作为被选中的概率进行随机选择
        # probabilities = np.exp(remaining_opacities*170) / np.exp(remaining_opacities*170).sum()
        # # probabilities = remaining_opacities / remaining_opacities.sum()
        # random_indices = np.random.choice(remaining_coords_matrix.shape[0], M2, replace=False, p=probabilities.flatten())
        # random_points = remaining_coords_matrix[random_indices]
        # random_opacities = remaining_opacities[random_indices]

        # 从整个场景中随机选择 M2 个点
        # probabilities = recon_vol.reshape(-1, 1) / recon_vol.reshape(-1, 1).sum()
        # random_indices = np.random.choice(coords_matrix.shape[0], M2, replace=False, p=probabilities.flatten())
        random_indices = np.random.choice(coords_matrix.shape[0], M2, replace=False)
        random_points = coords_matrix[random_indices]
        random_opacities = recon_vol.reshape(-1, 1)[random_indices]

        concat_points = np.concatenate([selected_points, random_points], axis=0)
        concat_opacities = np.concatenate([selected_opacities, random_opacities], axis=0)
    else:
        concat_points = selected_points
        concat_opacities = selected_opacities

    pcd = BasicPointCloud(points=concat_points, opacities=concat_opacities)

    return pcd
    

def random_initializor(recon_args, init_args):

    M = init_args['M1'] + init_args['M2']

    coords_matrix = make_coords(recon_args['volume_resolution'], recon_args['volume_phy'], recon_args['volume_origin']).reshape(-1, 3)
    selected_indices  = np.random.choice(coords_matrix.shape[0], M, replace=False)
    selected_points = coords_matrix[selected_indices]
    selected_opacities = np.ones((M,1)) * 0.1
    
    pcd = BasicPointCloud(points=selected_points, opacities=selected_opacities)

    return pcd

def storeply(path, pcd):
    xyz = pcd.points
    opacity = pcd.opacities

    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('opacity', 'f4')]

    elements = np.empty(xyz.shape[0], dtype=dtype)

    attributes = np.concatenate((xyz, opacity), axis=1)
    elements[:] = list(map(tuple, attributes))

    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readSceneInfo(datapath, outpath, train_views, init_args, loaded_iter, resampled):

    camera_file = os.path.join(datapath, 'transforms.json')

    cam_infos, recon_args = readCameras(camera_file, datapath)
    N_views = len(cam_infos)

    train_indice, eval_indice, all_indice = get_indice(N_views, train_views)
    print('Trainig cameras:', train_indice)
    print('Testing cameras:', eval_indice)

    train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx in train_indice]
    test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx in eval_indice]

    if not loaded_iter:
        if resampled:
            print('Resampled initialization')   
            pcd = None
        else:    
            if init_args['fdk_initial']:
                print('FDK initialization')
                fdk_file = os.path.join(outpath, 'fdk_reocn.nii.gz')
                CT_reconstructor = tigre_ct(train_cam_infos, recon_args)
                FDK_recon = CT_reconstructor.fdk(CT_reconstructor.projs, CT_reconstructor.PrimaryAngles) 
                FDK_recon = np.clip(FDK_recon, a_min=0, a_max=FDK_recon.max())
                pcd = vol_initializor(FDK_recon, recon_args, init_args, type='fdk')
                sitk.WriteImage(sitk.GetImageFromArray(FDK_recon.transpose(2, 1, 0)), fdk_file)   # [H, W, D] ---> [D, H, W]
            else:
                print('Random initialization')
                pcd = random_initializor(recon_args, init_args)
    else:
        pcd = None
    
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           recon_args=recon_args,
                           train_indice=train_indice,
                           eval_indice=eval_indice,
                           all_indice=all_indice)
    return scene_info




