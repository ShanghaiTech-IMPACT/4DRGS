from leapctype import *
import numpy as np
import time
from leaptorch import Projector
import copy
from tqdm import tqdm
import math
import torch

class leap_ct():
    def __init__(self, cam_infos, recon_args, use_device='cuda:0'):
        
        self.use_device = use_device
        self.device = torch.device(use_device)
        self.projector = Projector(forward_project=True, use_static=True, use_gpu=True, gpu_device=self.device)
        # self.projector.leapct.set_projector('VD')

        # set up the projection geometry
        numAngles = len(cam_infos)
        numRows = cam_infos[0].height 
        numCols = cam_infos[0].width
        pixelHeight = cam_infos[0].sy
        pixelWidth = cam_infos[0].sx
        sod = cam_infos[0].sad
        sdd = cam_infos[0].sid
        centerRow = 0.5*(numRows-1)
        centerCol = 0.5*(numCols-1)

        projs = []
        PrimaryAngles = []
        for cam in cam_infos:
            projs.append(cam.image.squeeze())
            PrimaryAngles.append(cam.PrimaryAngle)
        projs = np.stack(projs, axis=0)
        PrimaryAngles = np.stack(PrimaryAngles, axis=0)
        self.projs = np.ascontiguousarray(projs, dtype=np.float32)
        PrimaryAngles = np.rad2deg(PrimaryAngles)
        PrimaryAngles += 90.0   # align DSA imaging system with LEAP-toolbox convention
        self.PrimaryAngles = np.ascontiguousarray(PrimaryAngles, dtype=np.float32)

        self.projector.leapct.set_conebeam(numAngles, numRows, numCols, pixelHeight, pixelWidth, 
                                           centerRow, centerCol, self.PrimaryAngles, sod, sdd)
            
        numX, numY, numZ = recon_args["volume_resolution"]
        voxelWidth, voxelHeight = recon_args["volume_spacing"][0], recon_args["volume_spacing"][-1]

        self.projector.leapct.set_volume(numX, numY, numZ, voxelWidth, voxelHeight)

        self.projector.allocate_batch_data()   # necessary for torch version

    def fdk(self, projs_, PrimaryAngles_, VERSE=False):
        if VERSE:
            print('Reconstruction with LEAP-toolbox fdk...')

        startTime = time.time()

        projs = copy.deepcopy(projs_)   # do not let inner operation influence input
        projs = np.flip(projs, axis=1)
        g = np.ascontiguousarray(projs, dtype=np.float32)   # shape is numAngles, numRows, numCols

        PrimaryAngles = copy.deepcopy(PrimaryAngles_)
        self.projector.leapct.set_phis(PrimaryAngles)
        self.projector.allocate_batch_data()

        f = self.projector.leapct.allocate_volume() # shape is numZ, numY, numX
        f[:] = 0.0 # initialize the volume to zero
        self.projector.leapct.FBP(g, f)

        if VERSE:
            print('fdk Reconstruction Elapsed Time: ' + str(time.time()-startTime))

        return f



