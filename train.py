import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer_voxelizer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import *
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from pyhocon import ConfigFactory, HOCONConverter
import numpy as np
import time as timeloger
from test import *

try:
    from tensorboardX import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(tb_writer, dataset, field_conf, opt, pipe, testing_iterations, saving_iterations, debug_from):
    
    start_time = timeloger.time()

    scale_bound = None
    if dataset.use_scale_bound:
        if dataset.scale_min is not None and dataset.scale_max is not None:
            scale_bound = np.asarray([dataset.scale_min, dataset.scale_max])
    gaussians = GaussianModel(field_conf)

    scene = Scene(dataset, gaussians, scale_bound)
    first_iter = 0
    
    testing_iterations = np.array(testing_iterations) + first_iter
    saving_iterations = np.array(saving_iterations) + first_iter
    gaussians.training_setup(opt, first_iter)
    gaussians.densify_setup(opt, first_iter)
    densify_from_iter = opt.densify_from_iter + first_iter
    densify_until_iter = opt.densify_until_iter + first_iter
    
    viewpoint_stack = None
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress", initial=first_iter, total=opt.iterations)
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        
        # with torch.autograd.set_detect_anomaly(True):

            gaussians.update_learning_rate(iteration)
            if opt.flow_consistency:
                TP_std = opt.TP_std / dataset.train_views
            else:
                TP_std = 0

            # Pick a random Camera
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()

            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True

            render_pkg = render(viewpoint_cam, gaussians, scene.recon_args, TP_std, pipe)
            image, viewspace_point_tensor, visibility_filter, radii, dummy_opacity = \
            render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["dummy_opacity"]

            # Loss
            loss = {}
            gt_image = viewpoint_cam.original_image.cuda()

            loss['Ll1'] = l1_loss(image, gt_image)
            loss['Ldssim'] = (1.0 - ssim(image, gt_image))
            loss['Ltotal'] = (1.0 - opt.lambda_dssim) * loss['Ll1'] + opt.lambda_dssim * loss['Ldssim']
            loss['Ltotal'].backward()

            with torch.no_grad():
                # Progress bar
                total_points = gaussians._xyz.shape[0]

                if iteration % 10 == 0:
                    info_dict = {"Loss": f"{loss['Ltotal'].item():.{5}f}",
                                "point": f"{total_points}"}

                    progress_bar.set_postfix(info_dict)

                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()
            
                if (iteration in saving_iterations):
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration) 

                prune_num_record = None
                split_prune_mask = None
                d_clone = None
                d_split = None
                # Densification
                if iteration < densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, dummy_opacity, visibility_filter)

                    if iteration > densify_from_iter and iteration % opt.densification_interval == 0:
                        d_clone, d_split, split_prune_mask = gaussians.densify(scene.scene_spacing, iteration)
                        prune_num_record = gaussians.prune(split_prune_mask, opt.random_prune, opt.avgopacity_prune, opt.opacity_prune, dummy_opacity,
                                                            opt.max_screen_size, scene.recon_args, iteration)
                        scene.gaussians.densification_postfix(d_clone)
                        scene.gaussians.densification_postfix(d_split)

                # Log and save
                training_report(tb_writer, iteration, loss, prune_num_record, testing_iterations, scene, render, pipe)

                # Optimizer step
                if iteration < opt.iterations:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

    end_time = timeloger.time()
    elapsed_time = end_time - start_time
    minutes = elapsed_time // 60
    seconds = elapsed_time % 60
    # All done
    print(f"\nTraining complete in {int(minutes)} minutes and {int(seconds)} seconds.")
    with open(os.path.join(args.model_path, "time.txt"), 'w') as file:
        file.write(f"Training complete in {int(minutes)} minutes and {int(seconds)} seconds.")

    with torch.no_grad():
        voxel_query_reconstruction(scene, gaussians, pipe, dataset, 'train')
        render_set(scene, gaussians, pipe, dataset, 'train')

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, loss, prune_num_record, testing_iterations, scene : Scene, renderFunc, pipe):
    
    if tb_writer:
        for key in list(loss.keys()):
            tb_writer.add_scalar(f"train_loss_{key}", loss[key].item(), iteration)

        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

        avgopacity = scene.gaussians.avgopacity_accum/scene.gaussians.denom
        avgopacity[avgopacity.isnan()] = 0.0
        tb_writer.add_scalar('avgopacity_min', avgopacity.min().item(), iteration)
        tb_writer.add_scalar('avgopacity_max', avgopacity.max().item(), iteration)
        tb_writer.add_scalar('avgopacity_mean', avgopacity.mean().item(), iteration)
        tb_writer.add_scalar('avgopacity_median', avgopacity.median().item(), iteration)            
        tb_writer.add_histogram('avgopacity', avgopacity, iteration)

        if prune_num_record is not None:
            for key in list(prune_num_record.keys()):
                tb_writer.add_scalar(f"{key}", prune_num_record[key], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},    # test set全部进行了测试计算PNSR，但是只取前五个进行展示
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]}) # train set只取五个计算PNSR和展示

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, scene.recon_args, 0, pipe)
                    image = render_pkg["render"]
                    gt_image = viewpoint.original_image.to("cuda")
                    image = data_norm(torch.clamp(image, 0, image.max()))                    
                    gt_image = data_norm(torch.clamp(gt_image, 0, gt_image.max()))
                    geomap = render_pkg["render_geo"]
                    geomap = data_norm(torch.clamp(geomap, 0, geomap.max()))
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + f"_view_{viewpoint.image_name}/render", image[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + f"_view_{viewpoint.image_name}/geomap", geomap[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + f"_view_{viewpoint.image_name}/ground_truth", gt_image[None], global_step=iteration)

                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])      
                print("\n [ITER {}] Evaluating {}: PSNR {}".format(iteration, config['name'], psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    op = OptimizationParams(parser)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--ADC_until_iter", default=15000, type=int)
    parser.add_argument("--Nviews", type=int, default=30)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--field_conf", type=str, default="./arguments/nerf.conf", help='field config file')
    parser.add_argument("--TP", type=int, default=1, help="whether to use temporal perturbation rendering loss")
    parser.add_argument("--randomprune", type=int, default=0, help="whether to use random pruning")
    parser.add_argument("--thresprune", type=int, default=0, help="whether to use threshold pruning")
    parser.add_argument("--hisprune", type=int, default=1, help="whether to use historical pruning")
    parser.add_argument("--boundscale", type=int, default=1, help="whether to use bounded scaling activation")

    args = parser.parse_args(sys.argv[1:])
    args.TP = bool(args.TP)
    args.randomprune = bool(args.randomprune)
    args.thresprune = bool(args.thresprune)
    args.hisprune = bool(args.hisprune)
    args.boundscale = bool(args.boundscale)

    op.flow_consistency = args.TP
    op.random_prune = args.randomprune
    op.opacity_prune = args.thresprune
    op.avgopacity_prune = args.hisprune
    op.iterations = args.iteration
    op.densify_until_iter = args.ADC_until_iter


    lp = lp.extract(args)
    lp.use_scale_bound = args.boundscale

    field_conf = ConfigFactory.parse_file(args.field_conf)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    resampled_ret = None
    tbw = prepare_output_and_logger(lp)

    print("Training")
    training(tbw, lp, field_conf, op, pp.extract(args), args.test_iterations, args.save_iterations, args.debug_from)
