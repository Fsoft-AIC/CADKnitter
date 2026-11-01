import os
import sys
import argparse
import numpy as np
import pickle
import json
import trimesh
import open3d as o3d
from tqdm import tqdm
from OCC.Extend.DataExchange import write_step_file, write_stl_file
from occwl.compound import Compound

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils import construct_brep, extract_primitive


def sample_pc(step_file, num_u=32, num_v=32):
    cad_solid = list(Compound.load_from_step(step_file).solids())
    if len(cad_solid) != 1:
        return None, None, None, None, None
    cad_solid = cad_solid[0].split_all_closed_faces(num_splits=0)
    face_pnts, edge_pnts, edge_corner_pnts, edgeFace_IncM, faceEdge_IncM = extract_primitive(cad_solid, num_u, num_v)
    return face_pnts, edge_pnts, edge_corner_pnts, edgeFace_IncM, faceEdge_IncM


def downsample_farthest_points(pnts, num_points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pnts)
    pcd = pcd.farthest_point_down_sample(num_points)
    return np.asarray(pcd.points)

def write_txt(text, output_file):
    with open(output_file, "w") as f:
        f.write(text)

def check_output_watertight(filenames):
    cond_step = filenames["cond_step"]
    gt_step = filenames["gt_step"]
    cond_stl = filenames["cond_stl"]
    gt_stl = filenames["gt_stl"]
    
    condition_shape = list(Compound.load_from_step(cond_step).solids())
    gt_shape = list(Compound.load_from_step(gt_step).solids())

    if len(condition_shape) != 1 or len(gt_shape) != 1:
        print(f"{cond_step} or {gt_step} is not single solid")
        return False
    
    try:
        sample_pc(cond_step)
        sample_pc(gt_step)
    except AssertionError as e:
        print(e)
        return False
    
    ground_truth_mesh = trimesh.load(gt_stl)
    cond_mesh = trimesh.load(cond_stl)

    if not ground_truth_mesh.is_watertight or not cond_mesh.is_watertight:
        print(f"{cond_stl} or {gt_stl} is not watertight")
        return False

    ground_truth_face_pnts, _ = trimesh.sample.sample_surface(ground_truth_mesh, 4096)
    ground_truth_face_pnts = downsample_farthest_points(ground_truth_face_pnts, 2000)
    if ground_truth_face_pnts is None:
        print(f"{gt_stl}: Cannot sample points")
        return False

    cond_face_pnts, _ = trimesh.sample.sample_surface(cond_mesh, 4096)
    cond_face_pnts = downsample_farthest_points(cond_face_pnts, 2000)
    if cond_face_pnts is None:
        print(f"{cond_stl}: Cannot sample points")
        return False
    
    return True


def create_eval_gt(input_object, save_folder):
    object_name = input_object.split('/')[-1].split('.')[0].replace('merged_', '').replace('output_', '')

    # Load data
    with open(input_object, "rb") as tf:
        data = pickle.load(tf)
        
    cond_surfPos = data['object_2']['surf_bbox_wcs']
    cond_surf_wcs = data['object_2']['surf_wcs']
    cond_edge_wcs = data['object_2']['edge_wcs']
    cond_FaceEdgeAdj = data['object_2']['faceEdge_adj']
    cond_EdgeVertexAdj = data['object_2']['edgeCorner_adj']

    gt_surf_wcs = data['object_1']['surf_wcs']
    gt_edge_wcs = data['object_1']['edge_wcs']
    gt_FaceEdgeAdj = data['object_1']['faceEdge_adj']
    gt_EdgeVertexAdj = data['object_1']['edgeCorner_adj']

    prompt = data['object_1']['prompt']

    # Preprocess the data
    # Compute the global offset and scale based on cond_surfPos
    bbox_scaled = 3.0 # This is a legacy from BrepGen
    cond_total_points = np.array(cond_surfPos).reshape(-1, 3)
    min_vals = np.min(cond_total_points, axis=0)
    max_vals = np.max(cond_total_points, axis=0)
    global_offset = min_vals + (max_vals - min_vals)/2 
    global_scale = max(max_vals - min_vals)

    # Scale and translate the condition object
    cond_surf_wcs = (cond_surf_wcs - global_offset[np.newaxis,np.newaxis,np.newaxis,:]) / (global_scale * 0.5)
    cond_edge_wcs = (cond_edge_wcs - global_offset[np.newaxis,np.newaxis,:]) / (global_scale * 0.5)

    # Scale and translate the ground truth object
    gt_surf_wcs = (gt_surf_wcs - global_offset[np.newaxis,np.newaxis,np.newaxis,:]) / (global_scale * 0.5)
    gt_edge_wcs = (gt_edge_wcs - global_offset[np.newaxis,np.newaxis,:]) / (global_scale * 0.5)

    try:
        cond_solid = construct_brep(cond_surf_wcs, cond_edge_wcs, cond_FaceEdgeAdj, cond_EdgeVertexAdj)
        gt_solid = construct_brep(gt_surf_wcs, gt_edge_wcs, gt_FaceEdgeAdj, gt_EdgeVertexAdj)
    except Exception as e:
        print(e)
        return None

    write_step_file(cond_solid, f'{save_folder}/{object_name}_cond.step')
    write_step_file(gt_solid, f'{save_folder}/{object_name}_gt.step')
    write_stl_file(cond_solid, f'{save_folder}/{object_name}_cond.stl', linear_deflection=0.001, angular_deflection=0.5)
    write_stl_file(gt_solid, f'{save_folder}/{object_name}_gt.stl', linear_deflection=0.001, angular_deflection=0.5)
    write_txt(prompt, f'{save_folder}/{object_name}_prompt_gt.txt')
    return {
        "cond_step": f'{save_folder}/{object_name}_cond.step',
        "gt_step": f'{save_folder}/{object_name}_gt.step',
        "cond_stl": f'{save_folder}/{object_name}_cond.stl',
        "gt_stl": f'{save_folder}/{object_name}_gt.stl',
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_list", type=str, default='data/knitcad_data_split.json',
                        help="Path to the test list")
    parser.add_argument("--split", type=str, default='test',
                        help="Split to use")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Start index")
    parser.add_argument("--end_idx", type=int, default=None,
                        help="End index")
    parser.add_argument("--data_dir", type=str, default='data/pkl',
                        help="Path to the data directory")
    parser.add_argument("--output_folder", type=str, default='data/eval_gt',
                        help="Path to the output folder")
    args = parser.parse_args()
    
    # Load the test list
    if args.test_list.endswith(".pkl"):
        with open(args.test_list, 'rb') as f:
            test_list = pickle.load(f)
    else:
        with open(args.test_list, 'r') as f:
            test_list = json.load(f)
    test_list = test_list[args.split]

    end_idx = len(test_list) if args.end_idx is None else args.end_idx
    test_list = test_list[args.start_idx:end_idx]

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    failed_file = []
    success_file = []
    for test_file in tqdm(test_list, desc='Processing test list'):
        data_file = f'{args.data_dir}/{test_file}'
        print(f'Processing {data_file}')
        output_filenames = create_eval_gt(data_file, args.output_folder)

        if output_filenames is None:
            failed_file.append(test_file)
        else:
            if check_output_watertight(output_filenames):
                success_file.append(test_file)
            else:
                failed_file.append(test_file)

    print(f"Success: {len(success_file)}. Failed: {len(failed_file)}.")
