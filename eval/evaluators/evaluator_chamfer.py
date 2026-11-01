import numpy as np

import torch
import trimesh
import open3d as o3d
from chamferdist import ChamferDistance

from .evaluator import Evaluator


def downsample_farthest_points(pnts, num_points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pnts)
    pcd = pcd.farthest_point_down_sample(num_points)
    return np.asarray(pcd.points)

class ChamferEvaluation(Evaluator):
    def __init__(self, 
                 ground_truth_stl_file, generated_stl_file, args=None):
        super().__init__(args)
        self.ground_truth_stl_file = ground_truth_stl_file
        self.generated_stl_file = generated_stl_file

    def get_metric_names():
        return ['chamfer_distance']

    def evaluate(self):
        # Sample the surface points of the ground truth and generated meshes
        generated_mesh = trimesh.load(self.generated_stl_file)
        if not generated_mesh.is_watertight:
            print(f"{self.generated_stl_file} is non-watertight")
            return None

        generated_face_pnts, _ = trimesh.sample.sample_surface(generated_mesh, 4096)
        generated_face_pnts = downsample_farthest_points(generated_face_pnts, 2000)
        if generated_face_pnts is None:
            return None

        ground_truth_mesh = trimesh.load(self.ground_truth_stl_file)
        if not ground_truth_mesh.is_watertight:
            print(f"{self.ground_truth_stl_file} is non-watertight")
            return {
                'chamfer_distance': None
            }

        ground_truth_face_pnts, _ = trimesh.sample.sample_surface(ground_truth_mesh, 4096)
        ground_truth_face_pnts = downsample_farthest_points(ground_truth_face_pnts, 2000)
        if ground_truth_face_pnts is None:
            return {
                'chamfer_distance': None
            }

        generated_pnts = generated_face_pnts.reshape(-1, 3)
        ground_truth_pnts = ground_truth_face_pnts.reshape(-1, 3)

        # Zero-centering
        max_generated_pnts = generated_pnts.max(axis=0)
        min_generated_pnts = generated_pnts.min(axis=0)
        max_ground_truth_pnts = ground_truth_pnts.max(axis=0)
        min_ground_truth_pnts = ground_truth_pnts.min(axis=0)
        generated_pnts = (generated_pnts - (max_generated_pnts + min_generated_pnts) / 2)
        ground_truth_pnts = (ground_truth_pnts - (max_ground_truth_pnts + min_ground_truth_pnts) / 2)

        # Scale two point clouds by the maximum dimension
        all_pnts = np.concatenate([generated_pnts, ground_truth_pnts], axis=0)
        max_all_pnts = all_pnts.max(axis=0)
        min_all_pnts = all_pnts.min(axis=0)
        dim_all_pnts = max_all_pnts - min_all_pnts
        global_scale = dim_all_pnts.max()
        generated_pnts = generated_pnts / global_scale
        ground_truth_pnts = ground_truth_pnts / global_scale

        generated_pnts_cuda = torch.from_numpy(generated_pnts).unsqueeze(0).cuda()
        ground_truth_pnts_cuda = torch.from_numpy(ground_truth_pnts).unsqueeze(0).cuda()

        chamferDist = ChamferDistance()
        chamfer_distance = chamferDist(generated_pnts_cuda, ground_truth_pnts_cuda, bidirectional=True)

        return {
            'chamfer_distance': chamfer_distance.detach().cpu().item(),
        }