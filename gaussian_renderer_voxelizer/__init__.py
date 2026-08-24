#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_Xray_gaussian_rasterization_voxelization import (
    GaussianRasterizationSettings as SortFreeRasterizationSettings,
    GaussianRasterizer as SortFreeRasterizer,
    GaussianVoxelizationSettings as SortFreeVoxelizationSettings,
    GaussianVoxelizer as SortFreeVoxelizer,
)
from diff_Xray_gaussian_rasterization_voxelization_legacy import (
    GaussianRasterizationSettings as LegacyRasterizationSettings,
    GaussianRasterizer as LegacyRasterizer,
    GaussianVoxelizationSettings as LegacyVoxelizationSettings,
    GaussianVoxelizer as LegacyVoxelizer,
)
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
import numpy as np
import copy


def _backend_classes(pipe):
    backend = getattr(pipe, "rasterizer_backend", "sort_free")
    if backend == "sort_free":
        return (
            SortFreeRasterizationSettings,
            SortFreeRasterizer,
            SortFreeVoxelizationSettings,
            SortFreeVoxelizer,
        )
    if backend == "legacy":
        return (
            LegacyRasterizationSettings,
            LegacyRasterizer,
            LegacyVoxelizationSettings,
            LegacyVoxelizer,
        )
    raise ValueError(f"Unknown rasterizer backend: {backend}")

def query(pc: GaussianModel, 
          timestamp,
          recon_args, 
          nVoxel,
          sVoxel,
          center,
          pipe, 
          scaling_modifier=1.0,
          opacity_precomp=None,
          opacity_type='final_opacity'
          ):
    """
    Query a volume with voxelization.
    """

    
    _, _, VoxelizationSettings, Voxelizer = _backend_classes(pipe)
    voxel_settings = VoxelizationSettings(
        scale_modifier=scaling_modifier,
        nVoxel_x=int(nVoxel[0]),
        nVoxel_y=int(nVoxel[1]),
        nVoxel_z=int(nVoxel[2]),
        sVoxel_x=float(sVoxel[0]),
        sVoxel_y=float(sVoxel[1]),
        sVoxel_z=float(sVoxel[2]),
        center_x=float(center[0]),
        center_y=float(center[1]),
        center_z=float(center[2]),
        prefiltered=False,
        debug=pipe.debug,
    )
    voxelizer = Voxelizer(voxel_settings=voxel_settings)

    means3D = pc.get_xyz

    if timestamp != -1:
        ret = pc._field(pc, timestamp, recon_args)
        opacity = ret[opacity_type]

    if opacity_precomp is not None:
        opacity = opacity_precomp

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    vol_pred, radii, vol_geo = voxelizer(
        means3D=means3D,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )
                
    return {
        "vol": vol_pred,
        "radii": radii,
        "vol_geo": vol_geo,
    }


def render(
    viewpoint_camera: Camera,
    pc: GaussianModel,
    recon_args,
    TP_std,
    pipe,
    scaling_modifier=1.0,
    opacity_precomp=None,
    opacity_type='final_opacity',
):
    """
    Render the scene.
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = (torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0 )
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    RasterizationSettings, Rasterizer, _, _ = _backend_classes(pipe)
    raster_settings = RasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        mode=1,   # mode = 1, default cone beam mode;  mode = 0 represents parallel beam mode
        debug=pipe.debug,
    )

    rasterizer = Rasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points

    timestamp = viewpoint_camera.timestamp
    if TP_std > 0:
        timestamp = timestamp + np.random.normal(loc=0, scale=TP_std)
    timestamp = np.clip(timestamp, 0, 1)

    ret = pc._field(pc, timestamp, recon_args)
    opacity = ret[opacity_type]

    if opacity_precomp is not None:
        opacity = opacity_precomp

    # dummy_opacity = torch.zeros_like(opacity, dtype=opacity.dtype, requires_grad=True, device="cuda")
    dummy_opacity = opacity.clone().detach().requires_grad_(True)
    try:
        dummy_opacity.retain_grad()
    except:
        pass

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii, render_others = rasterizer(
        means3D=means3D,
        means2D=means2D,
        opacities=opacity,
        dummy_opacities=dummy_opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from valu×N¸êÚ$z{-®éÜj×ÄÔÀÀÀ4(€€€€€€€Í•±˜¹‘•¹Í¥™å}™É½µ}¥Ñ•È€ô€ÔÀÀ4(€€€€€€€Í•±˜¹‘•¹Í¥™å}É…‘}Ñ¡É•Í¡½±‘}¥¹¥Ğ€ô€À¸ÀÀÀÄ€€Œ€À¸ÀÀÀÄ4(€€€€€€€Í•±˜¹‘•¹Í¥™å}É…‘}Ñ¡É•Í¡½±‘}™¥¹…°€ô€À¸ÀÀÀÄ€Œ€À¸ÀÀÀÀØ4(€€€€€€€Í•±˜¹Á•É•¹Ñ}‘•¹Í•}¥¹¥Ğ€ô€È¸Ô€€Œ¹Õ´½˜Ù½á•±Ì4(€€€€€€€Í•±˜¹Á•É•¹Ñ}‘•¹Í•}™¥¹…°€ô€À¸Ô€€Œ¹Õ´½˜Ù½á•±Ì4(€€€€€€€Í•±˜¹É…¹‘½µ}ÁÉÕ¹”€ô…±Í”4(€€€€€€€Í•±˜¹Á•É•¹Ñ}É…¹‘½µ}ÁÉÕ¹•}¥¹¥Ğ€ô€À¸Àà€€Œ€À¸Ä4(€€€€€€€Í•±˜¹Á•É•¹Ñ}É…¹‘½µ}ÁÉÕ¹•}™¥¹…°€ô€À¸Àà€€Œ€À¸ÀØ4(€€€€€€€Í•±˜¹½Á…¥Ñå}ÁÉÕ¹”€ô…±Í”4(€€€€€€€Í•±˜¹µ¥¹}½Á…¥Ñå}¥¹¥Ğ€ô€Å”´Ø4(€€€€€€€Í•±˜¹µ¥¹}½Á…¥Ñå}™¥¹…°€ô€Å”´Ø4(€€€€€€€Í•±˜¹…Ù½Á…¥Ñå}ÁÉÕ¹”€ôQÉÕ”4(€€€€€€€Í•±˜¹µ¥¹}…Ù½Á…¥Ñå}¥¹¥Ğ€ô€Å”´Ø4(€€€€€€€Í•±˜¹µ¥¹}…Ù½Á…¥Ñå}™¥¹…°€ô€Å”´Ø4(€€€€€€€Í•±˜¹µ…á}ÍÉ••¹}Í¥é”€ô9½¹”4(€€€€€€€Í•±˜¹™±½İ}½¹Í¥ÍÑ•¹ä€ôQÉÕ”4(€€€€€€€Í•±˜¹QA}ÍÑ€ô€Ä¸À4(€€€€€€€ÍÕÁ•È ¤¹}}¥¹¥Ñ}|¡Á…ÉÍ•È°€‰=ÁÑ¥µ¥é…Ñ¥½¸A…É…µ•Ñ•ÉÌˆ¤4(4)‘•˜•Ñ}½µ‰¥¹•‘}…ÉÌ¡Á…ÉÍ•È€èÉÕµ•¹ÑA…ÉÍ•È¤è4(€€€µ‘±¹•}ÍÑÉ¥¹œ€ôÍåÌ¹…ÉÙlÄét4(€€€™™¥±•}ÍÑÉ¥¹œ€ô€‰9…µ•ÍÁ…” ¤ˆ4(€€€…ÉÍ}µ‘±¥¹”€ôÁ…ÉÍ•È¹Á…ÉÍ•}…ÉÌ¡µ‘±¹•}ÍÑÉ¥¹œ¤4(4(€€€ÑÉäè4(€€€€€€€™™¥±•Á…Ñ €ô½Ì¹Á…Ñ ¹©½¥¸¡…ÉÍ}µ‘±¥¹”¹µ½‘•±}Á…Ñ °€‰™}…ÉÌˆ¤4(€€€€€€€ÁÉ¥¹Ğ ‰1½½­¥¹œ™½È½¹™¥œ™¥±”¥¸ˆ°™™¥±•Á…Ñ ¤4(€€€€€€€İ¥Ñ ½Á•¸¡™™¥±•Á…Ñ ¤…Ì™}™¥±”è4(€€€€€€€€€€€ÁÉ¥¹Ğ ‰½¹™¥œ™¥±”™½Õ¹èíôˆ¹™½Éµ…Ğ¡™™¥±•Á…Ñ ¤¤4(€€€€€€€€€€€™™¥±•}ÍÑÉ¥¹œ€ô™}™¥±”¹É•… ¤4(€€€•á•ÁĞQåÁ•ÉÉ½Èè4(€€€€€€€ÁÉ¥¹Ğ ‰½¹™¥œ™¥±”¹½Ğ™½Õ¹…Ğˆ¤4(€€€€€€€Á…ÍÌ4(€€€…ÉÍ}™™¥±”€ô•Ù…°¡™™¥±•}ÍÑÉ¥¹œ¤4(4(€€€µ•É•‘}‘¥Ğ€ôÙ…ÉÌ¡…ÉÍ}™™¥±”¤¹½Áä ¤4(€€€™½È¬±Ø¥¸Ù…ÉÌ¡…ÉÍ}µ‘±¥¹”¤¹¥Ñ•µÌ ¤è4(€€€€€€€¥˜Ø€„ô9½¹”è4(€€€€€€€€€€€µ•É•‘}‘¥Ñm­t€ôØ4(€€€É•ÑÕÉ¸9…µ•ÍÁ…” ¨©µ•É•‘}‘¥Ğ¤4(4(