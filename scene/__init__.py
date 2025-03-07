import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import readSceneInfo, storeply
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos
import numpy as np
import SimpleITK as sitk

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, scale_bound=None, resampled_pcds=None, \
                 load_iteration=None, shuffle=True):

        self.use_resampling = resampled_pcds is not None
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.scale_bound = scale_bound

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, f"point_cloud"))  # 自动找最大的iteration
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        init_args = {'M1': args.M1,
                     'M2': args.M2,
                     'fdk_initial': args.fdk_initial,
                     'thres_percent_fdk': args.thres_percent_fdk}
        
        self.init_args = init_args
        scene_info = readSceneInfo(args.source_path, args.model_path, args.train_views, init_args, self.loaded_iter, self.use_resampling)
        self.recon_args = scene_info.recon_args
        self.train_indice = scene_info.train_indice
        self.eval_indice = scene_info.eval_indice
        self.all_indice = scene_info.all_indice

        ply_path = os.path.join(self.model_path, f"input.ply")
        if self.use_resampling:
            storeply(ply_path, resampled_pcds)
        elif scene_info.point_cloud is not None:    
            storeply(ply_path, scene_info.point_cloud)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = np.asarray(self.recon_args['volume_phy']).max()
        self.scene_spacing = np.asarray(self.recon_args['volume_spacing']).min()
        if self.scale_bound is not None:
            self.gaussians.setup_functions(self.scale_bound * self.scene_spacing)
        else:
            self.gaussians.setup_functions()

        print("Load Training Cameras: ", len(scene_info.train_cameras))
        self.train_cameras = cameraList_from_camInfos(scene_info.train_cameras, args)
        print("Load Testing Cameras: ", len(scene_info.test_cameras))
        self.test_cameras = cameraList_from_camInfos(scene_info.test_cameras, args)
        print("Merge Cameras")
        self.all_cameras = self.camera_merge()

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           f"point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
            self.gaussians.load_field(os.path.join(self.model_path,
                                                            f"point_cloud",
                                                            "iteration_" + str(self.loaded_iter),
                                                            "field.pth"))
        else:
            if self.use_resampling:
                self.gaussians.create_from_pcd(resampled_pcds, self.cameras_extent)
            else:
                self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)


    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_field(os.path.join(point_cloud_path, "field.pth"))

    def getTrainCameras(self):
        return self.train_cameras

    def getTestCameras(self):
        return self.test_cameras
    
    def getAllCameras(self):
        return self.all_cameras
    
    def camera_merge(self):
        all_cameras = self.train_cameras + self.test_cameras
        all_cameras = sorted(all_cameras, key=lambda cam: cam.frame_id)
        return all_cameras

    def get_indice(self):
        indice = {}
        indice['train_indice'] = self.train_indice
        indice['eval_indice'] = self.eval_indice
        indice['all_indice'] = self.all_indice
        return indice

