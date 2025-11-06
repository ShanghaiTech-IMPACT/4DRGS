# gaussian model for X-ray imaging

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, inverse_softplus, inverse_sigmoid_clamp, exp_decay, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation, build_scaling_rotation_inv
import torch.nn.functional as F
from scene.field import field
import matplotlib.pyplot as plt
import time
from utils.image_utils import data_norm

class GaussianModel:

    def setup_functions(self, scale_bound=None):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        if scale_bound is not None:
            scale_min_bound, scale_max_bound = scale_bound
            assert(scale_min_bound < scale_max_bound), "scale_min_bound should be less than scale_max_bound"
            self.scaling_activation = lambda x: torch.sigmoid(x) * (scale_max_bound - scale_min_bound) + scale_min_bound
            self.scaling_inverse_activation = lambda x: inverse_sigmoid_clamp((x - scale_min_bound) / (scale_max_bound - scale_min_bound))
        else:
            self.scaling_activation = torch.exp   # 保证 scale 是正数
            self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = F.relu 
        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, field_conf):
        self._xyz = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._field = field(field_conf)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.avgopacity_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.spatial_lr_scale = 0

    def capture(self):
        return (
            self._xyz,
            self._scaling,
            self._rotation,
            self._opacity,
            self._field.state_dict(),
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.avgopacity_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self._xyz, 
        self._scaling, 
        self._rotation, 
        self._opacity,
        field_state,
        self.max_radii2D, 
        xyz_gradient_accum,
        avgopacity_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.avgopacity_accum = avgopacity_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        self._field.load_state_dict(field_state)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)    

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = self.scaling_inverse_activation(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = torch.tensor(np.asarray(pcd.opacities)).float().cuda()
        
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))

        self._field = self._field.to("cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args, start_iter=0):
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.avgopacity_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init*self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr_init, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr_init, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr_init, "name": "rotation"},
            {'params': list(self._field.parameters()), 'lr': training_args.field_lr_init, "name": "field", "weight_decay": training_args.field_decay},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        
        self.xyz_scheduler_args = exp_decay(training_args.position_lr_init*self.spatial_lr_scale, 
                                            training_args.position_lr_final*self.spatial_lr_scale, 
                                            start_iter, training_args.iterations + start_iter)

        self.opacity_scheduler_args = exp_decay(training_args.opacity_lr_init, 
                                                training_args.opacity_lr_final, 
                                                start_iter, training_args.iterations + start_iter)
    
        self.scaling_scheduler_args = exp_decay(training_args.scaling_lr_init, 
                                                training_args.scaling_lr_final, 
                                                start_iter, training_args.iterations + start_iter)

        self.rotation_scheduler_args = exp_decay(training_args.rotation_lr_init, 
                                                training_args.rotation_lr_final, 
                                                start_iter, training_args.iterations + start_iter)

        self.field_scheduler_args = exp_decay(training_args.field_lr_init,
                                              training_args.field_lr_final,
                                              start_iter, training_args.iterations + start_iter)
        
    def densify_setup(self, densify_args, start_iter=0):
        self.percent_dense_args = exp_decay(densify_args.percent_dense_init,
                                            densify_args.percent_dense_final,
                                            densify_args.densify_from_iter + start_iter,
                                            densify_args.densify_until_iter + start_iter)

        self.densify_grad_threshold_args = exp_decay(densify_args.densify_grad_threshold_init,
                                                     densify_args.densify_grad_threshold_final,
                                                     densify_args.densify_from_iter + start_iter,
                                                     densify_args.densify_until_iter + start_iter)

        self.percent_random_prune_args = exp_decay(densify_args.percent_random_prune_init,
                                                   densify_args.percent_random_prune_final,
                                                   densify_args.densify_from_iter + start_iter,
                                                   densify_args.densify_until_iter + start_iter)

        self.min_avgopacity_args = exp_decay(densify_args.min_avgopacity_init,
                                            densify_args.min_avgopacity_final,
                                            densify_args.densify_from_iter + start_iter,
                                            densify_args.densify_until_iter + start_iter)
        
        self.min_opacity_args = exp_decay(densify_args.min_opacity_init,
                                         densify_args.min_opacity_final,
                                         densify_args.densify_from_iter + start_iter,
                                         densify_args.densify_until_iter + start_iter)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "opacity":
                lr = self.opacity_scheduler_args(iteration)
                param_group['lr'] = lr     
            if param_group["name"] == "scaling":
                lr = self.scaling_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "rotation":
                lr = self.rotation_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "field":
                lr = self.field_scheduler_args(iteration)
                param_group['lr'] = lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z']
        # All channels except the 3 DC
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def save_field(self, path):
        torch.save(self._field.state_dict(), path)

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
    
    def load_field(self, path):
        print("loading field from exists {}".format(path))
        self._field.load_state_dict(torch.load(path, map_location="cuda"))
        self._field = self._field.to("cuda")

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == "field": continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.avgopacity_accum = self.avgopacity_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == "field": continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, d):

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.avgopacity_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")


    def densify_and_split(self, grads, grad_threshold, scene_extent):
        N=2
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)  # 梯度满足条件，并且scale足够大，则进行split densify

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)

        # derive new_xyz through sampling
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)  
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))  #scale为原本点的scale/1.6, 其他属性则完全继承
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        d = {"xyz": new_xyz,
        "opacity": new_opacity,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        return d, selected_pts_mask


    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)  # 梯度满足条件，并且scale足够小，则进行clone densify

        new_xyz = self._xyz[selected_pts_mask]
        new_opacity = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        d = {"xyz": new_xyz,
        "opacity": new_opacity,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        return d
    
    def densify(self, extent, iteration):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.percent_dense = self.percent_dense_args(iteration)
        max_grad = self.densify_grad_threshold_args(iteration)

        d_clone = self.densify_and_clone(grads, max_grad, extent)
        d_split, split_prune_mask = self.densify_and_split(grads, max_grad, extent)
        torch.cuda.empty_cache()

        return d_clone, d_split, split_prune_mask

    def prune(self, split_prune_mask, random_prune, avgopacity_prune, opacity_prune, dummy_opacity, 
              max_screen_size, recon_args, iteration):
       
        prune_mask = self.exclude_outbbx(self.get_xyz, recon_args)
        if split_prune_mask is not None:
            prune_mask = torch.logical_or(prune_mask, split_prune_mask)  # split densify 产生的点也必须被剔除

        prune_num_record = {}
        if random_prune:  #适用于动态场景 仅对血管外围点进行剔除 但是会造成噪声
            percent_random_prune = self.percent_random_prune_args(iteration)
            random_prune_mask = self.get_random_prune_mask(percent_random_prune)
            random_prune_num = random_prune_mask.sum().item()
            prune_mask = torch.logical_or(prune_mask, random_prune_mask)        
            prune_num_record["random_prune_num"] = random_prune_num

        if avgopacity_prune:
            avgopacity = self.avgopacity_accum / self.denom
            avgopacity[avgopacity.isnan()] = 0.0
            min_avgopacity_thres = self.min_avgopacity_args(iteration)
            min_avgopacity_mask = avgopacity.squeeze() < min_avgopacity_thres
            min_avgopacity_prune_num = min_avgopacity_mask.sum().item()
            prune_mask = torch.logical_or(prune_mask, min_avgopacity_mask)
            prune_num_record["min_avgopacity_prune_num"] = min_avgopacity_prune_num

        if opacity_prune:
            min_opacity_thres = self.min_opacity_args(iteration)
            min_opacity_mask = dummy_opacity.squeeze() < min_opacity_thres
            min_opacity_prune_num = min_opacity_mask.sum().item()
            prune_mask = torch.logical_or(prune_mask, min_opacity_mask)
            prune_num_record["min_opacity_prune_num"] = min_opacity_prune_num

        if max_screen_size:
            # 如果radii2d 大于 max screen size 进行剔除
            big_points_vs = self.max_radii2D > max_screen_size
            max_screen_prune_num = big_points_vs.sum().item()
            prune_mask = torch.logical_or(prune_mask, big_points_vs)
            prune_num_record["max_screen_prune_num"] = max_screen_prune_num

        self.prune_points(prune_mask)
        
        torch.cuda.empty_cache()

        return prune_num_record

    def get_random_prune_mask(self, percent_random_prune):
        prune_mask = torch.zeros_like(self.get_opacity, dtype=bool).squeeze()
        P = prune_mask.shape[0]
        indices = torch.arange(0, P).to('cuda')
        select_indices = torch.randperm(int(P))[:int((P)*percent_random_prune)]
        prune_mask[indices[select_indices]] = True
        return prune_mask
        
    def exclude_outbbx(self, pts, recon_args):
        bbx_min = torch.tensor(recon_args['volume_origin'], dtype=torch.float32, device=pts.device)
        bbx_max = bbx_min + torch.tensor(recon_args['volume_phy'], dtype=torch.float32, device=pts.device)
        mask_outside = torch.logical_or(pts<bbx_min, pts>bbx_max)
        mask_outside = mask_outside.any(dim=1)
        return mask_outside

    def add_densification_stats(self, viewspace_point_tensor, dummy_opacity, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)
        # self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, 2:], dim=-1, keepdim=True)
        self.avgopacity_accum[update_filter] += dummy_opacity[update_filter]    
        self.denom[update_filter] += 1



