import tigre
import numpy as np
import copy

class tigre_ct:
    def __init__(self, cam_infos, recon_args):

        geo = tigre.geometry(mode="cone")

        geo.nVoxel = np.asarray(
            recon_args["volume_resolution"][::-1]
        )  # x-axis, y-axis, z-axis
        geo.sVoxel = np.asarray(recon_args["volume_phy"][::-1])
        geo.dVoxel = np.asarray(recon_args["volume_spacing"][::-1])

        projs_cols, projs_rows = cam_infos[0].width, cam_infos[0].height
        self.W = projs_cols
        self.H = projs_rows

        self.proj_spacing_x = cam_infos[0].sx
        self.proj_spacing_y = cam_infos[0].sy

        self.sid = cam_infos[0].sid
        self.sad = cam_infos[0].sad

        geo.DSD = self.sid
        geo.DSO = self.sad

        geo.nDetector = np.asarray([self.H, self.W])
        geo.sDetector = np.asarray(
            [self.H * self.proj_spacing_y, self.W * self.proj_spacing_x]
        )
        geo.dDetector = np.asarray([self.proj_spacing_y, self.proj_spacing_x])

        geo.offOrigin = np.asarray([0, 0, 0])
        geo.offDetector = np.asarray([0, 0, 0])

        geo.accuracy = 0.5
        self.geo = geo

        projs = []
        PrimaryAngles = []
        for cam in cam_infos:
            projs.append(cam.image.squeeze())
            PrimaryAngles.append(cam.PrimaryAngle)
        projs = np.stack(projs, axis=0)
        PrimaryAngles = np.stack(PrimaryAngles, axis=0)
        self.projs = projs
        self.PrimaryAngles = PrimaryAngles

    def fdk(self, projs, PrimaryAngles, filter='hann'):
        recon = tigre.algorithms.fdk(
            projs[:, ::-1, :], copy.deepcopy(self.geo), PrimaryAngles, filter=filter
        )

        return recon
