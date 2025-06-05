import numpy as np
import open3d as o3d
from scipy import ndimage
from skimage import measure
import copy
import mcubes
import pymeshlab

def mesh_extraction(vol_pred, thres, mesh_pred_path):
    verts, faces = mcubes.marching_cubes(vol_pred, thres)
    mcubes.export_obj(verts, faces, mesh_pred_path)
    print('Mesh extraction done! save to:', mesh_pred_path)

def ICP_align(mesh_ref_path, mesh_pred_path, mesh_pred_align_path):

    ms = pymeshlab.MeshSet()

    ms.load_new_mesh(mesh_ref_path)  # set ref_mesh as reference
    ms.load_new_mesh(mesh_pred_path)  # set pred_mesh as source

    ms.compute_matrix_by_icp_between_meshes(
        referencemesh = 0,
        sourcemesh = 1,
        samplenum = 2000,
        mindistabs = 10,
        trgdistabs = 0.005,
        maxiternum = 75,
    )

    ms.set_current_mesh(1)
    ms.apply_matrix_freeze()
    ms.save_current_mesh(mesh_pred_align_path)
    print('ICP align done! save to:', mesh_pred_align_path)

def compute_metric(mesh_ref_path, mesh_pred_align_path, spacing):

    mesh_ref = o3d.io.read_triangle_mesh(mesh_ref_path)
    mesh_pred_align = o3d.io.read_triangle_mesh(mesh_pred_align_path)
    CD = mesh_distance(mesh_ref, mesh_pred_align, spacing, type='chamfer')
    HD95 = mesh_distance(mesh_ref, mesh_pred_align, spacing, type='hausdorff_95')
    return CD, HD95

def mesh_distance(mesh_1, mesh_2, spacing, type='chamfer'):
    vertices_1 = np.asarray(mesh_1.vertices)
    vertices_1 = vertices_1 * spacing
    vertices_2 = np.asarray(mesh_2.vertices)
    vertices_2 = vertices_2 * spacing
    pcd_1 = o3d.geometry.PointCloud()
    pcd_2 = o3d.geometry.PointCloud()
  
    pcd_1.points = o3d.utility.Vector3dVector(vertices_1)
    pcd_2.points = o3d.utility.Vector3dVector(vertices_2)

    dist_1 = np.array(pcd_1.compute_point_cloud_distance(pcd_2))
    dist_2 = np.array(pcd_2.compute_point_cloud_distance(pcd_1))

    if type=='chamfer':
        cd1 = np.mean(dist_1)
        cd2 = np.mean(dist_2)
        cd = cd1 + cd2
        return cd
    elif type=='hausdorff':
        hd1 = np.max(dist_1)
        hd2 = np.max(dist_2)
        hd = max(hd1, hd2)
        return hd
    elif type=='hausdorff_95':
        hd_95_1 = np.percentile(dist_1, 95)
        hd_95_2 = np.percentile(dist_2, 95)
        hd_95 = max(hd_95_1, hd_95_2)
        return hd_95
