import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer_voxelizer import render
from gaussian_renderer_voxelizer import query
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene.gaussian_model import GaussianModel
from utils.image_utils import *
from utils.mesh_utils import *
import numpy as np
import SimpleITK as sitk
from pyhocon import ConfigFactory
import json

def render_set(scene, gaussians, pipe, dataset, phase='test'):
    
    print('Current phase:', phase)

    print(f"Full set rendering")

    all_views = scene.getAllCameras().copy()

    output_path = os.path.join(dataset.model_path, phase, f"render_output")
    os.makedirs(output_path, exist_ok=True)

    render_images = []
    gt_images = []
    for view in tqdm(all_views, desc="Rendering progress of full set"):
        render_pkg = render(view, gaussians, scene.recon_args, 0, pipe)
        render_images.append(render_pkg['render'][0].cpu().numpy())
        gt_images.append(view.original_image[0].cpu().numpy())
    render_images = np.stack(render_images)
    render_images = np.clip(render_images, 0, render_images.max())
    gt_images = np.stack(gt_images)
    gt_images = np.clip(gt_images, 0, gt_images.max())

    sitk.WriteImage(sitk.GetImageFromArray(render_images), os.path.join(output_path, 'proj.nii.gz'))

    train_indice = scene.get_indice()['train_indice']
    psnr_cnt_train = Averager()
    ssim_cnt_train = Averager()
    psnr_cnt_eval = Averager()
    ssim_cnt_eval = Averager()

    for idx in range(len(all_views)):
        rendering = render_images[idx]
        gt = gt_images[idx]
        rendering = data_norm(rendering)
        gt = data_norm(gt)
        psnr_cur = get_psnr(rendering, gt)
        ssim_cur = get_ssim_2d(rendering, gt)
        if idx in train_indice:
            psnr_cnt_train.add(psnr_cur)
            ssim_cnt_train.add(ssim_cur)
        else:
            psnr_cnt_eval.add(psnr_cur)
            ssim_cnt_eval.add(ssim_cur)
    
    PSNR_avg_train = np.round(psnr_cnt_train.avg(), 4)
    SSIM_avg_train = np.round(ssim_cnt_train.avg(), 4)
    PSNR_avg_eval = np.round(psnr_cnt_eval.avg(), 4)
    SSIM_avg_eval = np.round(ssim_cnt_eval.avg(), 4)
    print(f"PSNR train: {PSNR_avg_train} dB")
    print(f"SSIM train: {SSIM_avg_train}")    
    print(f"PSNR eval: {PSNR_avg_eval} dB")
    print(f"SSIM eval: {SSIM_avg_eval}") 

    log_message = f"GauN: {gaussians._xyz.shape[0]}\nPSNR train: {PSNR_avg_train} dB\nSSIM train: {SSIM_avg_train}\nPSNR eval: {PSNR_avg_eval} dB\nSSIM eval: {SSIM_avg_eval}"
    output_logs_path = os.path.join(output_path, "logs.txt")
    with open(output_logs_path, 'w') as file:
        file.write(log_message)

def render_set_fixview(scene, gaussians, pipe, dataset, phase='test'):

    print('Current phase:', phase)
    select_view_idx = 0
    print(f'Fix {select_view_idx} viewpoint rendering')

    all_views = scene.getAllCameras().copy()
    select_view = all_views[select_view_idx]
    time_list = np.arange(0, len(all_views), dtype=np.int32)

    output_path = os.path.join(dataset.model_path, phase, f"fixview_render", f"viewpoint_{select_view_idx}")
    makedirs(output_path, exist_ok=True)

    render_images = []
    for timestamp in tqdm(time_list, desc=f"Rendering progress of viewpoint:{select_view_idx}"):
        select_view.timestamp = timestamp/len(all_views)
        render_pkg = render(select_view, gaussians, scene.recon_args, 0, pipe)
        render_images.append(render_pkg['render'][0].cpu().numpy())
    
    render_images = np.stack(render_images)
    render_images = np.clip(render_images, 0, render_images.max())
    sitk.WriteImage(sitk.GetImageFromArray(render_images), os.path.join(output_path, 'proj.nii.gz'))


def voxel_query_reconstruction(scene, gaussians, pipe, dataset, phase='test'):

    print('Current phase:', phase)

    print(f"Voxel query reconstruction")

    all_views = scene.getAllCameras().copy()

    output_path = os.path.join(dataset.model_path, phase, f"VQR_output")
    os.makedirs(output_path, exist_ok=True)

    opacity_time_record = []
    for view in tqdm(all_views, desc="Opacity computing"):
        timestamp = view.timestamp
        opacity_time_record.append(gaussians._field(gaussians, timestamp, scene.recon_args)['final_opacity'].squeeze())
    opacity_time_record = torch.stack(opacity_time_record)
    opacity_query = torch.mean(opacity_time_record, dim=0).unsqueeze(-1)

    query_pkg = query(
        gaussians,
        -1,
        scene.recon_args,
        scene.recon_args['volume_resolution'],
        scene.recon_args['volume_phy'],
        [0, 0, 0],
        pipe,
        opacity_precomp = opacity_query
    )
    vol_pred = query_pkg["vol"]
    vol_pred = vol_pred.detach().cpu().numpy()
    sitk.WriteImage(sitk.GetImageFromArray(vol_pred.transpose(2, 1, 0)), os.path.join(output_path, f'vol_pred.nii.gz'))

    # extract mesh and evaluate CD and HD metric
    mesh_pred_path = os.path.join(output_path, 'mesh_pred.obj')
    mesh_extraction(vol_pred, 0.008, mesh_pred_path)
    mesh_pred_align_path = os.path.join(output_path, 'mesh_pred_align.obj')
    mesh_ref_path = os.path.join(dataset.source_path, 'mesh_ref.obj')
    
    ICP_align(mesh_ref_path, mesh_pred_path, mesh_pred_align_path)

    json_path = os.path.join(dataset.source_path, 'transforms.json')
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    volume_spacing = np.asarray(json_data['volume_spacing'])

    CD, HD95 = compute_metric(mesh_ref_path, mesh_pred_align_path, volume_spacing)
    CD = np.round(CD, 4)
    HD95 = np.round(HD95, 4)

    # note that we report HD95 instead of HD in our paper
    print(f"CD: {CD} mm")
    print(f"HD: {HD95} mm")  

    log_message = f"CD: {CD} mm\nHD: {HD95} mm"
    output_logs_path = os.path.join(output_path, "logs.txt")
    with open(output_logs_path, 'w') as file:
        file.write(log_message)

def render_sets(dataset : ModelParams, pipeline : PipelineParams, field_conf, args):
    with torch.no_grad():

        scale_bound = None
        if dataset.use_scale_bound:
            if dataset.scale_min is not None and dataset.scale_max is not None:
                scale_bound = np.asarray([dataset.scale_min, dataset.scale_max])
        gaussians = GaussianModel(field_conf)
        scene = Scene(dataset, gaussians, scale_bound, load_iteration=args.iteration, shuffle=False)

        if args.render_2d:
            render_set(scene, gaussians, pipeline, dataset, "test")
        if args.render_fixview:
            render_set_fixview(scene, gaussians, pipeline, dataset, "test")   
        if args.VQR:
            voxel_query_reconstruction(scene, gaussians, pipeline, dataset, "test")

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--Nviews", type=int, default=30)
    parser.add_argument("--field_conf", type=str, default="./arguments/nerf.conf", help='field config file')
    parser.add_argument("--render_2d", action="store_true")
    parser.add_argument("--render_fixview", action="store_true")
    parser.add_argument("--VQR", action="store_true")

    args = get_combined_args(parser)

    print("Rendering " + args.model_path)

    field_conf = ConfigFactory.parse_file(args.field_conf)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    
    model = model.extract(args)

    render_sets(model, pipeline.extract(args), field_conf, args)
