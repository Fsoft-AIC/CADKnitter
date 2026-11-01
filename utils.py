import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import base64
import numpy as np 
import math
import torch
import torch.nn as nn
import argparse
import ot
from torch.nn import functional as F
from chamferdist import ChamferDistance
from typing import List, Optional, Tuple, Union

try:
    from OCC.Core.gp import gp_Pnt, gp_Pnt
    from OCC.Core.TColgp import TColgp_Array2OfPnt
    from OCC.Core.GeomAPI import GeomAPI_PointsToBSplineSurface, GeomAPI_PointsToBSpline
    from OCC.Core.GeomAbs import GeomAbs_C2
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeEdge
    from OCC.Extend.TopologyUtils import TopologyExplorer, WireExplorer
    from OCC.Core.TColgp import TColgp_Array1OfPnt
    from OCC.Core.gp import gp_Pnt
    from OCC.Core.ShapeFix import ShapeFix_Face, ShapeFix_Wire, ShapeFix_Edge
    from OCC.Core.ShapeAnalysis import ShapeAnalysis_Wire
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeSolid
except:
    print("No OCC available")
    pass

from occwl.uvgrid import ugrid, uvgrid
from occwl.compound import Compound
from occwl.solid import Solid
from occwl.shell import Shell
from occwl.entity_mapper import EntityMapper

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import io
import wandb
from PIL import Image
import torch.nn.functional as F
from pytorch3d.loss import point_mesh_face_distance
from pytorch3d.structures import Meshes, Pointclouds

def plot_edges(edges, save_path, colors=None, log_wandb=False, title=None):
    # edges: numpy array (num_edges, num_points, 3)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if colors is None:
        for edge_id in range(edges.shape[0]):
            color = [cm.tab20(edge_id % 20)] * len(edges[edge_id])
            ax.scatter(edges[edge_id, :, 0], edges[edge_id, :, 1], edges[edge_id, :, 2], c=color, alpha=0.8)
    else:
        for edge_id in range(edges.shape[0]):
            ax.scatter(edges[edge_id, :, 0], edges[edge_id, :, 1], edges[edge_id, :, 2], c=colors[edge_id], alpha=0.8)

    all_coords = edges.reshape(-1, 3)
    x_min, y_min, z_min = all_coords.min(axis=0)
    x_max, y_max, z_max = all_coords.max(axis=0)

    mid_x = (x_min + x_max) / 2
    mid_y = (y_min + y_max) / 2
    mid_z = (z_min + z_max) / 2

    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2

    # Set equal limits on all axes
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set title
    if title is not None:
        ax.set_title(title)

    # Save the plot
    if log_wandb:
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)
        wandb.log({save_path: wandb.Image(image)})
        buf.close()
    else:
        dir_path = os.path.dirname(save_path)
        if dir_path != '':
            os.makedirs(dir_path, exist_ok=True)
        plt.savefig(save_path)
    plt.close()

# Define modern UI color palette
UI_COLORS = [
    '#3B82F6',  # Blue
    '#EF4444',  # Red
    '#10B981',  # Green
    '#F59E0B',  # Amber
    '#8B5CF6',  # Purple
    '#EC4899',  # Pink
    '#14B8A6',  # Teal
    '#F97316',  # Orange
    '#6366F1',  # Indigo
    '#84CC16',  # Lime
    '#06B6D4',  # Cyan
    '#FBBF24',  # Yellow
    '#A855F7',  # Violet
    '#FB923C',  # Light Orange
    '#22D3EE',  # Sky
    '#4ADE80',  # Emerald
]

COLOR_DICT = {
    'r': '#EF4444',
    'g': '#10B981',
    'b': '#3B82F6',
    'y': '#F59E0B',
    'purple': '#8B5CF6',
    'pink': '#EC4899',
    'teal': '#14B8A6',
    'orange': '#F97316',
    'indigo': '#6366F1',
    'lime': '#84CC16',
    'cyan': '#06B6D4',
    'violet': '#A855F7',
    'light_orange': '#FB923C',
    'sky': '#22D3EE',
    'emerald': '#4ADE80',
    'gray': '#6B7280',
    'brown': '#854D3D',
    'black': '#111827',
    'white': '#F3F4F6',
}

def draw_3d_box(ax, x1, y1, z1, x2, y2, z2, color='#3B82F6', linewidths=1.5, alpha=1.0):
    """
    Draw a 3D bounding box with enhanced styling.
    
    Args:
        ax: matplotlib 3D axis
        x1, y1, z1: minimum coordinates
        x2, y2, z2: maximum coordinates
        color: box edge color
        linewidths: width of box edges
        alpha: transparency of edges
    """
    # Define the 8 corners of the box
    corners = [
        [x1, y1, z1], [x2, y1, z1], [x2, y2, z1], [x1, y2, z1],  # bottom face
        [x1, y1, z2], [x2, y1, z2], [x2, y2, z2], [x1, y2, z2]   # top face
    ]

    # Define the 12 edges as line segments
    edges = [
        [corners[0], corners[1]], [corners[1], corners[2]],
        [corners[2], corners[3]], [corners[3], corners[0]],  # bottom
        [corners[4], corners[5]], [corners[5], corners[6]],
        [corners[6], corners[7]], [corners[7], corners[4]],  # top
        [corners[0], corners[4]], [corners[1], corners[5]],
        [corners[2], corners[6]], [corners[3], corners[7]],  # verticals
    ]

    # Add edges to plot with enhanced styling
    edge_collection = Line3DCollection(edges, colors=color, linewidths=linewidths, alpha=alpha)
    ax.add_collection3d(edge_collection)

def plot_surf_pos(surfPos_cpu, save_path, title="3D Bounding Boxes", 
                  log_wandb=False, colors=None, linewidths=None, 
                  keep_origin=False, white_background=True, 
                  minimal_mode=False, show_grid=True, 
                  figsize=(10, 8), dpi=100, alpha=1.0):
    """
    Plot 3D bounding boxes with enhanced styling and flexible display options.
    
    Args:
        surfPos_cpu: numpy array of shape (n, 6) containing box coordinates
        save_path: path to save the plot
        title: title for the plot
        log_wandb: whether to log to wandb
        colors: custom colors for boxes (uses UI palette by default)
        linewidths: custom linewidths for boxes
        keep_origin: whether to keep origin in view
        white_background: whether to use white background (True) or transparent (False)
        minimal_mode: if True, hides all grids, ticks, labels, and axes
        show_grid: whether to show grid lines (only used if minimal_mode=False)
        figsize: figure size as tuple (width, height)
        dpi: dots per inch for saved figure
        alpha: transparency of box edges (0-1)
    """
    if surfPos_cpu.shape[0] == 0:
        return
    
    assert len(surfPos_cpu.shape) == 2, f'surfPos_cpu must be 2D, but got {surfPos_cpu.shape}'
    if surfPos_cpu.shape[1] == 3:
        surfPos_cpu = surfPos_cpu.reshape(-1, 6)
    
    # Set up 3D plot with enhanced styling
    fig = plt.figure(figsize=figsize)
    
    # Set background color
    if white_background:
        fig.patch.set_facecolor('white')
    else:
        fig.patch.set_facecolor('none')
        fig.patch.set_alpha(0)
    
    ax = fig.add_subplot(111, projection='3d')
    
    # Set axes background
    if white_background:
        ax.xaxis.pane.set_facecolor('white')
        ax.yaxis.pane.set_facecolor('white')
        ax.zaxis.pane.set_facecolor('white')
        ax.xaxis.pane.set_alpha(1)
        ax.yaxis.pane.set_alpha(1)
        ax.zaxis.pane.set_alpha(1)
    else:
        ax.xaxis.pane.set_alpha(0)
        ax.yaxis.pane.set_alpha(0)
        ax.zaxis.pane.set_alpha(0)
    
    # Apply minimal mode if requested
    if minimal_mode:
        # Hide everything except the boxes
        ax.set_axis_off()
        ax.grid(False)
        # Remove title in minimal mode
        # title = ""
    else:
        # Set title with improved styling
        if title:
            ax.set_title(title, fontsize=14, fontweight='medium', pad=20)
        
        # Configure grid
        if show_grid:
            ax.grid(True, linestyle='--', alpha=0.3, color='#E5E7EB')
        else:
            ax.grid(False)
        
        # Style axes
        ax.set_xlabel('X', fontsize=11, labelpad=10)
        ax.set_ylabel('Y', fontsize=11, labelpad=10)
        ax.set_zlabel('Z', fontsize=11, labelpad=10)
        
        # Style ticks
        ax.tick_params(axis='both', which='major', labelsize=9, 
                      colors='#6B7280', pad=5)
        
        # Make pane edges lighter
        ax.xaxis.pane.set_edgecolor('#E5E7EB')
        ax.yaxis.pane.set_edgecolor('#E5E7EB')
        ax.zaxis.pane.set_edgecolor('#E5E7EB')
        ax.xaxis.pane.set_linewidth(0.5)
        ax.yaxis.pane.set_linewidth(0.5)
        ax.zaxis.pane.set_linewidth(0.5)

    num_boxes = surfPos_cpu.shape[0]
    
    # Use UI color palette if colors not provided
    if colors is None:
        colors = [UI_COLORS[i % len(UI_COLORS)] for i in range(num_boxes)]
    else:
        colors = [COLOR_DICT[color] for color in colors]
    
    # Default linewidths with slightly thicker lines for better visibility
    if linewidths is None:
        linewidths = [2.0 for _ in range(num_boxes)]

    # Draw each box with enhanced styling
    for i, box in enumerate(surfPos_cpu):
        x1, y1, z1, x2, y2, z2 = box
        draw_3d_box(ax, x1, y1, z1, x2, y2, z2, 
                   color=colors[i], linewidths=linewidths[i], alpha=alpha)

    # Get all coordinates and compute common scale
    all_coords = surfPos_cpu.reshape(-1, 3)
    x_min, y_min, z_min = all_coords.min(axis=0)
    x_max, y_max, z_max = all_coords.max(axis=0)

    # Add small padding to bounds
    padding = 0.05
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    
    x_min -= x_range * padding
    x_max += x_range * padding
    y_min -= y_range * padding
    y_max += y_range * padding
    z_min -= z_range * padding
    z_max += z_range * padding

    mid_x = (x_min + x_max) / 2
    mid_y = (y_min + y_max) / 2
    mid_z = (z_min + z_max) / 2

    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2

    # Set equal limits on all axes
    if not keep_origin:
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Adjust layout to prevent clipping
    plt.tight_layout()

    # Save the plot with high quality
    if log_wandb:
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', 
                   facecolor=fig.get_facecolor(), edgecolor='none')
        buf.seek(0)
        image = Image.open(buf)
        wandb.log({save_path: wandb.Image(image)})  # Uncomment if using wandb
        buf.close()
    else:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight',
                   facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()

def get_bbox_norm(point_cloud):
    # Find the minimum and maximum coordinates along each axis
    min_x = np.min(point_cloud[:, 0])
    max_x = np.max(point_cloud[:, 0])

    min_y = np.min(point_cloud[:, 1])
    max_y = np.max(point_cloud[:, 1])

    min_z = np.min(point_cloud[:, 2])
    max_z = np.max(point_cloud[:, 2])

    # Create the 3D bounding box using the min and max values
    min_point = np.array([min_x, min_y, min_z])
    max_point = np.array([max_x, max_y, max_z])
    return np.linalg.norm(max_point - min_point)


def compute_bbox_center_and_size(min_corner, max_corner):
    # Calculate the center
    center_x = (min_corner[0] + max_corner[0]) / 2
    center_y = (min_corner[1] + max_corner[1]) / 2
    center_z = (min_corner[2] + max_corner[2]) / 2
    center = np.array([center_x, center_y, center_z])
    # Calculate the size
    size_x = max_corner[0] - min_corner[0]
    size_y = max_corner[1] - min_corner[1]
    size_z = max_corner[2] - min_corner[2]
    size = max(size_x, size_y, size_z)
    return center, size


def randn_tensor(
    shape: Union[Tuple, List],
    generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
    device: Optional["torch.device"] = None,
    dtype: Optional["torch.dtype"] = None,
    layout: Optional["torch.layout"] = None,
):
    """This is a helper function that allows to create random tensors on the desired `device` with the desired `dtype`. When
    passing a list of generators one can seed each batched size individually. If CPU generators are passed the tensor
    will always be created on CPU.
    """
    # device on which tensor is created defaults to device
    rand_device = device
    batch_size = shape[0]

    layout = layout or torch.strided
    device = device or torch.device("cpu")

    if generator is not None:
        gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

    return latents


def pad_repeat(x, max_len):
    repeat_times = math.floor(max_len/len(x))
    sep = max_len-repeat_times*len(x)
    sep1 = np.repeat(x[:sep], repeat_times+1, axis=0)
    sep2 = np.repeat(x[sep:], repeat_times, axis=0)
    x_repeat = np.concatenate([sep1, sep2], 0)
    return x_repeat


def pad_zero(x, max_len, return_mask=False):
    keys = np.ones(len(x))
    padding = np.zeros((max_len-len(x))).astype(int)  
    mask = 1-np.concatenate([keys, padding]) == 1  
    padding = np.zeros((max_len-len(x), *x.shape[1:]))
    x_padded = np.concatenate([x, padding], axis=0)
    if return_mask:
        return x_padded, mask
    else:
        return x_padded

def get_args_vae():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data_process/deepcad_parsed', 
                        help='Path to data folder')  
    parser.add_argument('--train_list', type=str, default='data_process/deepcad_data_split_6bit_surface.pkl', 
                        help='Path to training list')  
    parser.add_argument('--val_list', type=str, default='data_process/deepcad_data_split_6bit.pkl', 
                        help='Path to validation list')  
    # Training parameters
    parser.add_argument("--option", type=str, choices=['surface', 'edge'], default='surface', 
                        help="Choose between option surface or edge (default: surface)")
    parser.add_argument('--batch_size', type=int, default=512, help='input batch size')  
    parser.add_argument('--train_nepoch', type=int, default=200, help='number of epochs to train for')
    parser.add_argument('--save_nepoch', type=int, default=20, help='number of epochs to save model')
    parser.add_argument('--test_nepoch', type=int, default=10, help='number of epochs to test model')
    parser.add_argument("--data_aug",  action='store_true', help='Use data augmentation')
    parser.add_argument("--finetune",  action='store_true', help='Finetune from existing weights')
    parser.add_argument("--weight",  type=str, default=None, help='Weight path when finetuning')  
    parser.add_argument("--gpu", type=int, nargs='+', default=[0], help="GPU IDs to use for training (default: [0])")
    # Save dirs and reload
    parser.add_argument('--env', type=str, default="surface_vae", help='environment')
    parser.add_argument('--dir_name', type=str, default="proj_log", help='name of the log folder.')
    args = parser.parse_args()
    # saved folder
    args.save_dir = f'{args.dir_name}/{args.env}'
    return args


def get_args_ldm():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data_process/deepcad_parsed', 
                        help='Path to data folder')  
    parser.add_argument('--list', type=str, default='data_process/deepcad_data_split_6bit.pkl', 
                        help='Path to data list')  
    parser.add_argument('--surfvae', type=str, default='proj_log/deepcad_surfvae/epoch_400.pt', 
                        help='Path to pretrained surface vae weights')  
    parser.add_argument('--edgevae', type=str, default='proj_log/deepcad_edgevae/epoch_300.pt', 
                        help='Path to pretrained edge vae weights')  
    parser.add_argument("--option", type=str, choices=['surfpos', 'surfz', 'edgepos', 'edgez'], default='surfpos', 
                        help="Choose between option [surfpos,edgepos,surfz,edgez] (default: surfpos)")
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=512, help='input batch size')  
    parser.add_argument('--train_nepoch', type=int, default=3000, help='number of epochs to train for')
    parser.add_argument('--test_nepoch', type=int, default=25, help='number of epochs to test model')
    parser.add_argument('--save_nepoch', type=int, default=50, help='number of epochs to save model')
    parser.add_argument('--max_face', type=int, default=50, help='maximum number of faces')
    parser.add_argument('--max_contact_face', type=int, default=10, help='maximum number of contact faces')
    parser.add_argument('--max_edge', type=int, default=30, help='maximum number of edges per face')
    parser.add_argument('--threshold', type=float, default=0.05, help='minimum threshold between two faces')
    parser.add_argument('--bbox_scaled', type=float, default=3, help='scaled the bbox')
    parser.add_argument('--z_scaled', type=float, default=1, help='scaled the latent z')
    parser.add_argument("--gpu", type=int, nargs='+', default=[0, 1], help="GPU IDs to use for training (default: [0, 1])")
    parser.add_argument("--data_aug",  action='store_true', help='Use data augmentation')
    parser.add_argument("--cf",  action='store_true', help='Use data augmentation')
    # model
    parser.add_argument("--n_layers", type=int, default=12, help="Number of layers in the transformer")
    # Save dirs and reload
    parser.add_argument('--env', type=str, default="surface_pos", help='environment')
    parser.add_argument('--dir_name', type=str, default="proj_log", help='name of the log folder.')
    # finetune
    parser.add_argument("--finetune",  action='store_true', help='Finetune from existing weights')
    parser.add_argument("--weight",  type=str, default=None, help='Weight path when finetuning')  
    # text model
    parser.add_argument("--text_model_name", type=str, default="ViT-B-32", help="Text model name")
    parser.add_argument("--max_text_length", type=int, default=148, help="Maximum text length")
    # loss
    parser.add_argument("--loss_type", type=str, default="vanila", help="Loss type")
    parser.add_argument("--contrastive_weight", type=float, default=1e-3, help="Weight for contrastive loss term")
    args = parser.parse_args()
    # saved folder
    args.save_dir = f'{args.dir_name}/{args.env}'
    return args


def rotate_point_cloud(point_cloud, angle_degrees, axis):
    """
    Rotate a point cloud around its center by a specified angle in degrees along a specified axis.

    Args:
    - point_cloud: Numpy array of shape (N, 3) representing the point cloud.
    - angle_degrees: Angle of rotation in degrees.
    - axis: Axis of rotation. Can be 'x', 'y', or 'z'.

    Returns:
    - rotated_point_cloud: Numpy array of shape (N, 3) representing the rotated point cloud.
    """

    # Convert angle to radians
    angle_radians = np.radians(angle_degrees)

    # Compute rotation matrix based on the specified axis
    if axis == 'x':
        rotation_matrix = np.array([[1, 0, 0],
                                    [0, np.cos(angle_radians), -np.sin(angle_radians)],
                                    [0, np.sin(angle_radians), np.cos(angle_radians)]])
    elif axis == 'y':
        rotation_matrix = np.array([[np.cos(angle_radians), 0, np.sin(angle_radians)],
                                    [0, 1, 0],
                                    [-np.sin(angle_radians), 0, np.cos(angle_radians)]])
    elif axis == 'z':
        rotation_matrix = np.array([[np.cos(angle_radians), -np.sin(angle_radians), 0],
                                    [np.sin(angle_radians), np.cos(angle_radians), 0],
                                    [0, 0, 1]])
    else:
        raise ValueError("Invalid axis. Must be 'x', 'y', or 'z'.")

    # Center the point cloud
    center = np.mean(point_cloud, axis=0)
    centered_point_cloud = point_cloud - center

    # Apply rotation
    rotated_point_cloud = np.dot(centered_point_cloud, rotation_matrix.T)

    # Translate back to original position
    rotated_point_cloud += center

    # Find the maximum absolute coordinate value
    max_abs_coord = np.max(np.abs(rotated_point_cloud))

    # Scale the point cloud to fit within the -1 to 1 cube
    normalized_point_cloud = rotated_point_cloud / max_abs_coord

    return normalized_point_cloud


def get_bbox(pnts):
    """
    Get the tighest fitting 3D (axis-aligned) bounding box giving a set of points
    """
    bbox_corners = []
    for point_cloud in pnts:
        # Find the minimum and maximum coordinates along each axis
        min_x = np.min(point_cloud[:, 0])
        max_x = np.max(point_cloud[:, 0])

        min_y = np.min(point_cloud[:, 1])
        max_y = np.max(point_cloud[:, 1])

        min_z = np.min(point_cloud[:, 2])
        max_z = np.max(point_cloud[:, 2])

        # Create the 3D bounding box using the min and max values
        min_point = np.array([min_x, min_y, min_z])
        max_point = np.array([max_x, max_y, max_z])
        bbox_corners.append([min_point, max_point])
    return np.array(bbox_corners)


def bbox_corners(bboxes):
    """
    Given the bottom-left and top-right corners of the bbox
    Return all eight corners 
    """
    bboxes_all_corners = []
    for bbox in bboxes:
        bottom_left, top_right = bbox[:3], bbox[3:]
        # Bottom 4 corners
        bottom_front_left = bottom_left
        bottom_front_right = (top_right[0], bottom_left[1], bottom_left[2])
        bottom_back_left = (bottom_left[0], top_right[1], bottom_left[2])
        bottom_back_right = (top_right[0], top_right[1], bottom_left[2])

        # Top 4 corners
        top_front_left = (bottom_left[0], bottom_left[1], top_right[2])
        top_front_right = (top_right[0], bottom_left[1], top_right[2])
        top_back_left = (bottom_left[0], top_right[1], top_right[2])
        top_back_right = top_right

        # Combine all coordinates
        all_corners = [
            bottom_front_left,
            bottom_front_right,
            bottom_back_left,
            bottom_back_right,
            top_front_left,
            top_front_right,
            top_back_left,
            top_back_right,
        ]
        bboxes_all_corners.append(np.vstack(all_corners))
    bboxes_all_corners = np.array(bboxes_all_corners)
    return bboxes_all_corners


def rotate_axis(pnts, angle_degrees, axis, normalized=False):
    """
    Rotate a point cloud around its center by a specified angle in degrees along a specified axis.

    Args:
    - point_cloud: Numpy array of shape (N, ..., 3) representing the point cloud.
    - angle_degrees: Angle of rotation in degrees.
    - axis: Axis of rotation. Can be 'x', 'y', or 'z'.

    Returns:
    - rotated_point_cloud: Numpy array of shape (N, 3) representing the rotated point cloud.
    """

    # Convert angle to radians
    angle_radians = np.radians(angle_degrees)
    
    # Convert points to homogeneous coordinates
    shape = list(np.shape(pnts))
    shape[-1] = 1
    pnts_homogeneous = np.concatenate((pnts, np.ones(shape)), axis=-1)

    # Compute rotation matrix based on the specified axis
    if axis == 'x':
        rotation_matrix = np.array([
            [1, 0, 0, 0],
            [0, np.cos(angle_radians), -np.sin(angle_radians), 0],
            [0, np.sin(angle_radians), np.cos(angle_radians), 0],
            [0, 0, 0, 1]
        ])
    elif axis == 'y':
        rotation_matrix = np.array([
            [np.cos(angle_radians), 0, np.sin(angle_radians), 0],
            [0, 1, 0, 0],
            [-np.sin(angle_radians), 0, np.cos(angle_radians), 0],
            [0, 0, 0, 1]
        ])
    elif axis == 'z':
        rotation_matrix = np.array([
            [np.cos(angle_radians), -np.sin(angle_radians), 0, 0],
            [np.sin(angle_radians), np.cos(angle_radians), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    else:
        raise ValueError("Invalid axis. Must be 'x', 'y', or 'z'.")

    # Apply rotation
    rotated_pnts_homogeneous = np.dot(pnts_homogeneous, rotation_matrix.T)
    rotated_pnts = rotated_pnts_homogeneous[...,:3]

    # Scale the point cloud to fit within the -1 to 1 cube
    if normalized:
        max_abs_coord = np.max(np.abs(rotated_pnts))
        rotated_pnts = rotated_pnts / max_abs_coord

    return rotated_pnts

def edge2loop(face_edges):
    face_edges_flatten = face_edges.reshape(-1,3)     
    # connect end points by closest distance
    merged_vertex_id = []
    for edge_idx, startend in enumerate(face_edges):
        self_id = [2*edge_idx, 2*edge_idx+1]
        # left endpoint 
        distance = np.linalg.norm(face_edges_flatten - startend[0], axis=1)
        min_id = list(np.argsort(distance))
        min_id_noself = [x for x in min_id if x not in self_id]
        merged_vertex_id.append(sorted([2*edge_idx, min_id_noself[0]]))
        # right endpoint
        distance = np.linalg.norm(face_edges_flatten - startend[1], axis=1)
        min_id = list(np.argsort(distance))
        min_id_noself = [x for x in min_id if x not in self_id]
        merged_vertex_id.append(sorted([2*edge_idx+1, min_id_noself[0]]))   

    merged_vertex_id = np.unique(np.array(merged_vertex_id),axis=0)
    return merged_vertex_id

def keep_largelist(int_lists):
    # Initialize a list to store the largest integer lists
    largest_int_lists = []

    # Convert each list to a set for efficient comparison
    sets = [set(lst) for lst in int_lists]

    # Iterate through the sets and check if they are subsets of others
    for i, s1 in enumerate(sets):
        is_subset = False
        for j, s2 in enumerate(sets):
            if i!=j and s1.issubset(s2) and s1 != s2:
                is_subset = True
                break
        if not is_subset:
            largest_int_lists.append(list(s1))

    # Initialize a set to keep track of seen tuples
    seen_tuples = set()

    # Initialize a list to store unique integer lists
    unique_int_lists = []

    # Iterate through the input list
    for int_list in largest_int_lists:
        # Convert the list to a tuple for hashing
        int_tuple = tuple(sorted(int_list))
        
        # Check if the tuple is not in the set of seen tuples
        if int_tuple not in seen_tuples:
            # Add the tuple to the set of seen tuples
            seen_tuples.add(int_tuple)
            
            # Add the original list to the list of unique integer lists
            unique_int_lists.append(int_list)
    
    return unique_int_lists

def detect_shared_vertex(edgeV_cad, edge_mask_cad, edgeV_bbox):
    """
    Find the shared vertices 
    """
    edge_id_offset = 2 * np.concatenate([np.array([0]),np.cumsum((edge_mask_cad==False).sum(1))])[:-1]
    valid = True
    
    # Detect shared-vertex on seperate face loop
    used_vertex = []
    face_sep_merges = []
    for face_idx, (face_edges, face_edges_mask, bbox_edges) in enumerate(zip(edgeV_cad, edge_mask_cad, edgeV_bbox)):
        face_edges = face_edges[~face_edges_mask]
        face_edges = face_edges.reshape(len(face_edges),2,3)
        face_start_id = edge_id_offset[face_idx]  

        # connect end points by closest distance (edge bbox)
        merged_vertex_id = edge2loop(bbox_edges)            
        if len(merged_vertex_id) == len(face_edges):
            merged_vertex_id = face_start_id + merged_vertex_id
            face_sep_merges.append(merged_vertex_id)
            used_vertex.append(bbox_edges*3)
            print('[PASS]')
            continue
        
        # connect end points by closest distance (vertex pos)
        merged_vertex_id = edge2loop(face_edges)
        if len(merged_vertex_id) == len(face_edges):
            merged_vertex_id = face_start_id + merged_vertex_id
            face_sep_merges.append(merged_vertex_id)
            used_vertex.append(face_edges)
            print('[PASS]')
            continue

        print('[FAILED]')
        valid = False
        break

    # Invalid
    if not valid:
        assert False

    # Detect shared-vertex across faces 
    total_pnts = np.vstack(used_vertex)
    total_pnts = total_pnts.reshape(len(total_pnts),2,3)
    total_pnts_flatten = total_pnts.reshape(-1,3)

    total_ids = []
    for face_idx, face_merge in enumerate(face_sep_merges):
        # non-self merge centers 
        nonself_face_idx = list(set(np.arange(len(face_sep_merges))) - set([face_idx]))
        nonself_face_merges = [face_sep_merges[x] for x in nonself_face_idx]
        nonself_face_merges = np.vstack(nonself_face_merges)
        nonself_merged_centers = total_pnts_flatten[nonself_face_merges].mean(1)

        # connect end points by closest distance
        across_merge_id = []
        for merge_id in face_merge:
            merged_center = total_pnts_flatten[merge_id].mean(0)
            distance = np.linalg.norm(nonself_merged_centers - merged_center, axis=1)
            nonself_match_id = nonself_face_merges[np.argsort(distance)[0]]
            joint_merge_id = list(nonself_match_id) + list(merge_id)
            across_merge_id.append(joint_merge_id) 
        total_ids += across_merge_id

    # Merge T-junctions
    while (True):
        no_merge = True
        final_merge_id = []

        # iteratelly merge until no changes happen
        for i in range(len(total_ids)):
            perform_merge = False

            for j in range(i+1,len(total_ids)):
                # check if vertex can be further merged 
                max_num = max(len(total_ids[i]), len(total_ids[j]))
                union = set(total_ids[i]).union(set(total_ids[j]))
                common = set(total_ids[i]).intersection(set(total_ids[j]))
                if len(union) > max_num and len(common)>0:
                    final_merge_id.append(list(union))
                    perform_merge = True
                    no_merge = False
                    break

            if not perform_merge:
                final_merge_id.append(total_ids[i]) # no-merge

        total_ids = final_merge_id
        if no_merge: break

    # remove subsets 
    total_ids = keep_largelist(total_ids)

    # merge again base on absolute coordinate value, required for >3 T-junction
    tobe_merged_centers = [total_pnts_flatten[x].mean(0) for x in total_ids]
    tobe_centers = np.array(tobe_merged_centers)
    distances = np.linalg.norm(tobe_centers[:, np.newaxis, :] - tobe_centers, axis=2)
    close_points = distances < 0.1
    mask = np.tril(np.ones_like(close_points, dtype=bool), k=-1)
    non_diagonal_indices = np.where(close_points & mask)
    row_indices, column_indices = non_diagonal_indices

    # update the total_ids
    total_ids_updated = []
    for row, col in zip(row_indices, column_indices):
        total_ids_updated.append(total_ids[row] + total_ids[col])
    for index, ids in enumerate(total_ids):
        if index not in list(row_indices) and index not in list(column_indices):
            total_ids_updated.append(ids)
    total_ids = total_ids_updated

    # merged vertices 
    unique_vertices = []
    for center_id in total_ids:
        center_pnts = total_pnts_flatten[center_id].mean(0) / 3.0
        unique_vertices.append(center_pnts)
    unique_vertices = np.vstack(unique_vertices)

    new_vertex_dict = {}
    for new_id, old_ids in enumerate(total_ids):
        new_vertex_dict[new_id] = old_ids

    return [unique_vertices, new_vertex_dict]

def detect_shared_edge(unique_vertices, new_vertex_dict, edge_z_cad, surf_z_cad, z_threshold, edge_mask_cad):
    """
    Find the shared edges 
    """
    init_edges = edge_z_cad

    # re-assign edge start/end to unique vertices
    new_ids = []
    for old_id in np.arange(2*len(init_edges)):
        new_id = []
        for key, value in new_vertex_dict.items():
            # Check if the desired number is in the associated list
            if old_id in value:
                new_id.append(key)
        assert len(new_id) == 1 # should only return one unique value
        new_ids.append(new_id[0])

    EdgeVertexAdj = np.array(new_ids).reshape(-1,2)

    # find edges assigned to the same start/end 
    similar_edges = []
    for i, s1 in enumerate(EdgeVertexAdj):
        for j, s2 in enumerate(EdgeVertexAdj):
            if i!=j and set(s1) == set(s2): # same start/end
                z1 = init_edges[i]
                z2 = init_edges[j]
                z_diff = np.abs(z1-z2).mean()
                if z_diff < z_threshold: # check z difference
                    similar_edges.append(sorted([i,j]))
                # else:
                #     print('z latent beyond...')
    similar_edges = np.unique(np.array(similar_edges),axis=0)

    # should reduce total edges by two
    if not 2*len(similar_edges) == len(EdgeVertexAdj):
        assert False, 'edge not reduced by 2'

    # unique edges
    unique_edge_id = similar_edges[:,0]
    EdgeVertexAdj = EdgeVertexAdj[unique_edge_id]
    unique_edges = init_edges[unique_edge_id]

    # unique faces 
    unique_faces = surf_z_cad
    FaceEdgeAdj = []
    ranges = np.concatenate([np.array([0]),np.cumsum((edge_mask_cad==False).sum(1))])
    for index in range(len(ranges)-1):
        adj_ids = np.arange(ranges[index], ranges[index+1])
        new_ids = []
        for id in adj_ids:
            new_id = np.where(similar_edges == id)[0]
            assert len(new_id) == 1
            new_ids.append(new_id[0])
        FaceEdgeAdj.append(new_ids)

    print(f'Post-process: F-{len(unique_faces)} E-{len(unique_edges)} V-{len(unique_vertices)}')

    return [unique_faces, unique_edges, FaceEdgeAdj, EdgeVertexAdj]


class STModel(nn.Module):
    def __init__(self, num_edge, num_surf):
        super().__init__()
        self.edge_t = nn.Parameter(torch.zeros((num_edge, 3)))
        self.surf_st = nn.Parameter(torch.FloatTensor([1,0,0,0]).unsqueeze(0).repeat(num_surf,1)) 


def get_bbox_minmax(point_cloud):
    # Find the minimum and maximum coordinates along each axis
    min_x = np.min(point_cloud[:, 0])
    max_x = np.max(point_cloud[:, 0])

    min_y = np.min(point_cloud[:, 1])
    max_y = np.max(point_cloud[:, 1])

    min_z = np.min(point_cloud[:, 2])
    max_z = np.max(point_cloud[:, 2])

    # Create the 3D bounding box using the min and max values
    min_point = np.array([min_x, min_y, min_z])
    max_point = np.array([max_x, max_y, max_z])
    return (min_point, max_point)


def joint_optimize(surf_ncs, edge_ncs, surfPos, unique_vertices, EdgeVertexAdj, FaceEdgeAdj, num_edge, num_surf):
    """
    Jointly optimize the face/edge/vertex based on topology
    """
    loss_func = ChamferDistance()

    model = STModel(num_edge, num_surf)
    model = model.cuda().train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.95, 0.999),
        weight_decay=1e-6,
        eps=1e-08,
    )

    # Optimize edges (directly compute)
    edge_ncs_se = edge_ncs[:,[0,-1]]
    edge_vertex_se = unique_vertices[EdgeVertexAdj]

    edge_wcs = []
    print('Joint Optimization...')
    for wcs, ncs_se, vertex_se in zip(edge_ncs, edge_ncs_se, edge_vertex_se):
        # scale
        scale_target = np.linalg.norm(vertex_se[0] - vertex_se[1])
        scale_ncs = np.linalg.norm(ncs_se[0] - ncs_se[1])
        edge_scale = scale_target / scale_ncs

        edge_updated = wcs*edge_scale
        edge_se = ncs_se*edge_scale  

        # offset
        offset = (vertex_se - edge_se)
        offset_rev = (vertex_se - edge_se[::-1])

        # swap start / end if necessary 
        offset_error = np.abs(offset[0] - offset[1]).mean()
        offset_rev_error =np.abs(offset_rev[0] - offset_rev[1]).mean()
        if offset_rev_error < offset_error:
            edge_updated = edge_updated[::-1]
            offset = offset_rev
    
        edge_updated = edge_updated + offset.mean(0)[np.newaxis,np.newaxis,:]
        edge_wcs.append(edge_updated)

    edge_wcs = np.vstack(edge_wcs)

    # Replace start/end points with corner, and backprop change along curve
    for index in range(len(edge_wcs)):
        start_vec = edge_vertex_se[index,0] - edge_wcs[index, 0]
        end_vec = edge_vertex_se[index,1] - edge_wcs[index, -1]
        weight = np.tile((np.arange(32)/31)[:,np.newaxis], (1,3))
        weighted_vec = np.tile(start_vec[np.newaxis,:],(32,1))*(1-weight) + np.tile(end_vec,(32,1))*weight
        edge_wcs[index] += weighted_vec            

    # Optimize surfaces 
    face_edges = []
    for adj in FaceEdgeAdj:
        all_pnts = edge_wcs[adj]
        face_edges.append(torch.FloatTensor(all_pnts).cuda())

    # Initialize surface in wcs based on surface pos
    surf_wcs_init = [] 
    bbox_threshold_min = []
    bbox_threshold_max = []   
    for edges_perface, ncs, bbox in zip(face_edges, surf_ncs, surfPos):
        surf_center, surf_scale = compute_bbox_center_and_size(bbox[0:3], bbox[3:])
        edges_perface_flat = edges_perface.reshape(-1, 3).detach().cpu().numpy()
        min_point, max_point = get_bbox_minmax(edges_perface_flat)
        edge_center, edge_scale = compute_bbox_center_and_size(min_point, max_point)
        bbox_threshold_min.append(min_point)
        bbox_threshold_max.append(max_point)

        # increase surface size if does not fully cover the wire bbox 
        if surf_scale < edge_scale:
            surf_scale = 1.05*edge_scale
    
        wcs = ncs * (surf_scale/2) + surf_center
        surf_wcs_init.append(wcs)

    surf_wcs_init = np.stack(surf_wcs_init)

    # optimize the surface offset
    surf = torch.FloatTensor(surf_wcs_init).cuda()
    for iters in range(200):
        surf_scale = model.surf_st[:,0].reshape(-1,1,1,1)
        surf_offset = model.surf_st[:,1:].reshape(-1,1,1,3)
        surf_updated = surf + surf_offset 
        
        surf_loss = 0
        for surf_pnt, edge_pnts in zip(surf_updated, face_edges):
            surf_pnt = surf_pnt.reshape(-1,3)
            edge_pnts = edge_pnts.reshape(-1,3).detach()
            surf_loss += loss_func(surf_pnt.unsqueeze(0), edge_pnts.unsqueeze(0), bidirectional=False, reverse=True) 
        surf_loss /= len(surf_updated) 

        optimizer.zero_grad()
        (surf_loss).backward()
        optimizer.step()

        # print(f'Iter {iters} surf:{surf_loss:.5f}') 

    surf_wcs = surf_updated.detach().cpu().numpy()

    return (surf_wcs, edge_wcs)


def add_pcurves_to_edges(face):
    edge_fixer = ShapeFix_Edge()
    top_exp = TopologyExplorer(face)
    for wire in top_exp.wires():
        wire_exp = WireExplorer(wire)
        for edge in wire_exp.ordered_edges():
            edge_fixer.FixAddPCurve(edge, face, False, 0.001)


def fix_wires(face, debug=False):
    top_exp = TopologyExplorer(face)
    for wire in top_exp.wires():
        if debug:
            wire_checker = ShapeAnalysis_Wire(wire, face, 0.01)
            print(f"Check order 3d {wire_checker.CheckOrder()}")
            print(f"Check 3d gaps {wire_checker.CheckGaps3d()}")
            print(f"Check closed {wire_checker.CheckClosed()}")
            print(f"Check connected {wire_checker.CheckConnected()}")
        wire_fixer = ShapeFix_Wire(wire, face, 0.01)

        # wire_fixer.SetClosedWireMode(True)
        # wire_fixer.SetFixConnectedMode(True)
        # wire_fixer.SetFixSeamMode(True)

        assert wire_fixer.IsReady()
        ok = wire_fixer.Perform()
        # assert ok


def fix_face(face):
    fixer = ShapeFix_Face(face)
    fixer.SetPrecision(0.01)
    fixer.SetMaxTolerance(0.1)
    ok = fixer.Perform()
    # assert ok
    fixer.FixOrientation()
    face = fixer.Face()
    return face


def construct_brep(surf_wcs, edge_wcs, FaceEdgeAdj, EdgeVertexAdj):
    """
    Fit parametric surfaces / curves and trim into B-rep
    """
    print('Building the B-rep...')
    # Fit surface bspline
    recon_faces = []  
    for points in surf_wcs:
        num_u_points, num_v_points = 32, 32
        uv_points_array = TColgp_Array2OfPnt(1, num_u_points, 1, num_v_points)
        for u_index in range(1,num_u_points+1):
            for v_index in range(1,num_v_points+1):
                pt = points[u_index-1, v_index-1]
                point_3d = gp_Pnt(float(pt[0]), float(pt[1]), float(pt[2]))
                uv_points_array.SetValue(u_index, v_index, point_3d)
        approx_face =  GeomAPI_PointsToBSplineSurface(uv_points_array, 3, 8, GeomAbs_C2, 5e-2).Surface() 
        recon_faces.append(approx_face)

    recon_edges = []
    for points in edge_wcs:
        num_u_points = 32
        u_points_array = TColgp_Array1OfPnt(1, num_u_points)
        for u_index in range(1,num_u_points+1):
            pt = points[u_index-1]
            point_2d = gp_Pnt(float(pt[0]), float(pt[1]), float(pt[2]))
            u_points_array.SetValue(u_index, point_2d)
        try:
            approx_edge = GeomAPI_PointsToBSpline(u_points_array, 0, 8, GeomAbs_C2, 5e-3).Curve()  
        except Exception as e:
            print('high precision failed, trying mid precision...')
            try:
                approx_edge = GeomAPI_PointsToBSpline(u_points_array, 0, 8, GeomAbs_C2, 8e-3).Curve()  
            except Exception as e:
                print('mid precision failed, trying low precision...')
                approx_edge = GeomAPI_PointsToBSpline(u_points_array, 0, 8, GeomAbs_C2, 5e-2).Curve()
        recon_edges.append(approx_edge)

    # Create edges from the curve list
    edge_list = []
    for curve in recon_edges:
        edge = BRepBuilderAPI_MakeEdge(curve).Edge()
        edge_list.append(edge)

    # Cut surface by wire 
    post_faces = []
    post_edges = []
    for idx,(surface, edge_incides) in enumerate(zip(recon_faces, FaceEdgeAdj)):
        corner_indices = EdgeVertexAdj[edge_incides]
        
        # ordered loop
        loops = []
        ordered = [0]
        seen_corners = [corner_indices[0,0], corner_indices[0,1]]
        next_index = corner_indices[0,1]

        while len(ordered)<len(corner_indices):
            while True:
                next_row = [idx for idx, edge in enumerate(corner_indices) if next_index in edge and idx not in ordered]
                if len(next_row) == 0:
                    break
                ordered += next_row
                next_index = list(set(corner_indices[next_row][0]) - set(seen_corners))
                if len(next_index)==0:break
                else: next_index = next_index[0]
                seen_corners += [corner_indices[next_row][0][0], corner_indices[next_row][0][1]]
            
            cur_len = int(np.array([len(x) for x in loops]).sum()) # add to inner / outer loops
            loops.append(ordered[cur_len:])
            
            # Swith to next loop
            next_corner =  list(set(np.arange(len(corner_indices))) - set(ordered))
            if len(next_corner)==0:break
            else: next_corner = next_corner[0]
            next_index = corner_indices[next_corner][0]
            ordered += [next_corner]
            seen_corners += [corner_indices[next_corner][0], corner_indices[next_corner][1]]
            next_index = corner_indices[next_corner][1]

        # Determine the outer loop by bounding box length (?)
        bbox_spans = [get_bbox_norm(edge_wcs[x].reshape(-1,3)) for x in loops]
        
        # Create wire from ordered edges
        _edge_incides_ = [edge_incides[x] for x in ordered]
        edge_post = [edge_list[x] for x in _edge_incides_]
        post_edges += edge_post

        out_idx = np.argmax(np.array(bbox_spans))
        inner_idx = list(set(np.arange(len(loops))) - set([out_idx]))

        # Outer wire
        wire_builder = BRepBuilderAPI_MakeWire()
        for edge_idx in loops[out_idx]:
            wire_builder.Add(edge_list[edge_incides[edge_idx]])
        outer_wire = wire_builder.Wire()

        # Inner wires
        inner_wires = []
        for idx in inner_idx:
            wire_builder = BRepBuilderAPI_MakeWire()
            for edge_idx in loops[idx]:
                wire_builder.Add(edge_list[edge_incides[edge_idx]])
            inner_wires.append(wire_builder.Wire())
    
        # Cut by wires
        face_builder = BRepBuilderAPI_MakeFace(surface, outer_wire)
        for wire in inner_wires:
            face_builder.Add(wire)
        face_occ = face_builder.Shape()
        fix_wires(face_occ)
        add_pcurves_to_edges(face_occ)
        fix_wires(face_occ)
        face_occ = fix_face(face_occ)
        post_faces.append(face_occ)

    # Sew faces into solid 
    sewing = BRepBuilderAPI_Sewing()
    for face in post_faces:
        sewing.Add(face)
        
    # Perform the sewing operation
    sewing.Perform()
    sewn_shell = sewing.SewedShape()

    # Make a solid from the shell
    maker = BRepBuilderAPI_MakeSolid()
    # print(f"Type of sewn shell: {type(sewn_shell)}")
    # if not isinstance(sewn_shell, TopoDS_Shell):
    #     breakpoint()
    maker.Add(sewn_shell)
    maker.Build()
    solid = maker.Solid()

    return solid

def save_surface(surf_wcs, path, title=None, colors=None, log_wandb=False, return_base64=False):
    # surf_wcs is num_faces x 32 x 32 x 3
    # each face has different color

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if colors is None:
        colors = [cm.tab20(i / len(surf_wcs)) for i in range(len(surf_wcs))]

    # Initialize min/max bounds
    all_points = []

    for face_idx, face in enumerate(surf_wcs):
        ax.plot_surface(face[:, :, 0], face[:, :, 1], face[:, :, 2],
                        color=colors[face_idx], alpha=0.5)
        all_points.append(face.reshape(-1, 3))

    # Flatten and concatenate all points
    all_points = np.concatenate(all_points, axis=0)

    # Compute axis limits
    x_min, y_min, z_min = np.min(all_points, axis=0)
    x_max, y_max, z_max = np.max(all_points, axis=0)
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2

    mid_x = (x_max + x_min) / 2
    mid_y = (y_max + y_min) / 2
    mid_z = (z_max + z_min) / 2

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    if title is not None:
        ax.set_title(title)

    # Save the plot
    if return_base64:
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        buf.close()
        plt.close()
        return image_base64
    elif log_wandb:
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)
        wandb.log({path: wandb.Image(image)})
        buf.close()
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)
        print(f"Saved surface plot at {path}")
    plt.close()

def save_edge(edge_wcs, path):
    # edge_wcs is num_edges x 32 x 3
    # each edge has different color
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = [cm.tab20(i / len(edge_wcs)) for i in range(len(edge_wcs))]
    for edge_idx, edge in enumerate(edge_wcs):
        ax.plot(edge[:,0], edge[:,1], edge[:,2], color=colors[edge_idx], alpha=0.5)
    plt.savefig(path)
    plt.close()

def decode_surf_wcs(surf_vae, surfz, surfPos):
    # surfz: [num_faces, 48] or [num_faces, 32, 32, 3]
    # surfPos: [num_faces, 6]
    # return: [num_faces, 32, 32, 3]
    assert surfz.shape[0] == surfPos.shape[0], f'surfz and surfPos must have the same number of faces, but got {surfz.shape[0]} and {surfPos.shape[0]}'
    
    with torch.no_grad():
        if len(surfz.shape) == 2:
            # The surfz is in the latent space
            surf_ncs_cad = surf_vae(torch.FloatTensor(surfz).cuda().unflatten(-1,torch.Size([16,3])).permute(0,2,1).unflatten(-1,torch.Size([4,4])))
            surf_ncs_cad = surf_ncs_cad.permute(0,2,3,1).detach().cpu().numpy()
        else:
            # The surfz is already in the spatial space
            assert len(surfz.shape) == 4 and surfz.shape[1] == 32 and surfz.shape[2] == 32, f'surfz must be 2D or 3D, but got {surfz.shape}'
            surf_ncs_cad = surfz

        surf_wcs_cad = []
        for ncs, bbox in zip(surf_ncs_cad, surfPos):
            surf_center, surf_scale = compute_bbox_center_and_size(bbox[0:3], bbox[3:])
            wcs = ncs * (surf_scale/2) + surf_center
            surf_wcs_cad.append(wcs)
    return np.stack(surf_wcs_cad)

def edge_len_loss(a, b, mask=None):
    edge_diff_a = a[:, 1:, :] - a[:, :-1, :]  # (E, N-1, 3)
    edge_diff_b = b[:, 1:, :] - b[:, :-1, :]  # (E, N-1, 3)

    corner_a = (a[:, 0, :] - a[:, -1, :]).unsqueeze(1)
    corner_b = (b[:, 0, :] - b[:, -1, :]).unsqueeze(1)

    edge_diff_a = torch.cat([corner_a, edge_diff_a], dim=1)
    edge_diff_b = torch.cat([corner_b, edge_diff_b], dim=1)

    len_a = torch.norm(edge_diff_a, dim=-1)  # (E, N-1)
    len_b = torch.norm(edge_diff_b, dim=-1)  # (E, N-1)

    if mask is not None:
        len_a = len_a[mask]
        len_b = len_b[mask]
    
    if len_a.numel() == 0:  # all masked out
        # return a differentiable zero
        return (a.sum() * 0.0)
    else:
        return F.mse_loss(len_a, len_b)

def angle_loss(a, b, mask=None):
    vec_a = a[:, 1:, :] - a[:, :-1, :]  # (E, N-1, 3)
    vec_b = b[:, 1:, :] - b[:, :-1, :]  # (E, N-1, 3)

    # Normalize vectors
    vec_a_norm = F.normalize(vec_a, dim=-1)  # (E, N-1, 3)
    vec_b_norm = F.normalize(vec_b, dim=-1)  # (E, N-1, 3)

    if mask is not None:
        vec_a_norm = vec_a_norm[mask]
        vec_b_norm = vec_b_norm[mask]
    
    if vec_a_norm.numel() == 0:
        return (a.sum() * 0.0)
    else:
        cos_sim = torch.sum(vec_a_norm * vec_b_norm, dim=-1)  # (E, N-1)

        # Angle loss as 1 - cosine similarity
        return torch.mean(1 - cos_sim)

def plot_surf_pos_ordered(surfPos_cpu, save_path, title="3D Bounding Boxes (Ordered)", log_wandb=False, colors=None, linewidths=None, keep_origin=False):
    """
    Plot surface positions with consistent color coding based on surface index order.
    This allows easy reference to attention weights matrices.
    """
    if surfPos_cpu.shape[0] == 0:
        return
    assert len(surfPos_cpu.shape) == 2, f'surfPos_cpu must be 2D, but got {surfPos_cpu.shape}'
    if surfPos_cpu.shape[1] == 3:
        surfPos_cpu = surfPos_cpu.reshape(-1, 6)
    
    # Set up 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title, fontsize=14)

    num_boxes = surfPos_cpu.shape[0]
    
    # Use consistent colors based on surface index order
    if colors is None:
        # Use a color map that provides good distinction between adjacent indices
        colors = [cm.tab20(i % 20) for i in range(num_boxes)]
    
    if linewidths is None:
        linewidths = [2.0 for _ in range(num_boxes)]

    # Draw each box with a unique color and add index labels
    for i, box in enumerate(surfPos_cpu):
        x1, y1, z1, x2, y2, z2 = box
        draw_3d_box(ax, x1, y1, z1, x2, y2, z2, color=colors[i], linewidths=linewidths[i])
        
        # Add surface index label at the center of each box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        center_z = (z1 + z2) / 2
        ax.text(center_x, center_y, center_z, str(i), fontsize=12, fontweight='bold', 
                color='black', ha='center', va='center')

    # Set limits with equal scaling
    all_coords = surfPos_cpu.reshape(-1, 3)
    x_min, y_min, z_min = all_coords.min(axis=0)
    x_max, y_max, z_max = all_coords.max(axis=0)

    mid_x = (x_min + x_max) / 2
    mid_y = (y_min + y_max) / 2
    mid_z = (z_min + z_max) / 2

    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2

    # Set equal limits on all axes
    if keep_origin == False:
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    
    # Set tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=10)

    # Save the plot
    if log_wandb:
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)
        wandb.log({save_path: wandb.Image(image)})
        buf.close()
    else:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def compute_intra_distances(boxes: torch.Tensor, metric='iou'):
    """
    Compute pairwise distances between boxes within a set.
    
    Args:
        boxes: Tensor of shape (N, 6) with format (x_min, y_min, z_min, x_max, y_max, z_max)
        metric: Distance metric to use ('iou', 'center', 'hausdorff', 'combined')
    
    Returns:
        Distance matrix of shape (N, N)
    """
    N = boxes.shape[0]
    
    if metric == 'iou':
        # IoU-based distance (1 - IoU)
        dist_matrix = 1.0 - compute_iou_3d(boxes, boxes)
        
    elif metric == 'center':
        # Normalized center distance
        centers = (boxes[:, :3] + boxes[:, 3:]) / 2.0  # (N, 3)
        # Normalize by the diagonal of the bounding box of all boxes
        all_points = boxes.reshape(-1, 3)
        all_min = all_points.min(dim=0)[0]
        all_max = all_points.max(dim=0)[0]
        diagonal = torch.norm(all_max - all_min)
        if diagonal > 0:
            centers = centers / diagonal
        dist_matrix = torch.cdist(centers, centers, p=2)
        
    elif metric == 'hausdorff':
        # Hausdorff distance between box corners
        dist_matrix = compute_hausdorff_distance(boxes, boxes)
        
    elif metric == 'combined':
        # Combine multiple metrics
        iou_dist = 1.0 - compute_iou_3d(boxes, boxes)
        centers = (boxes[:, :3] + boxes[:, 3:]) / 2.0
        all_min = boxes.reshape(-1, 3).min(dim=0)[0]
        all_max = boxes.reshape(-1, 3).max(dim=0)[0]
        diagonal = torch.norm(all_max - all_min)
        if diagonal > 0:
            centers = centers / diagonal
        center_dist = torch.cdist(centers, centers, p=2)
        # Weighted combination
        dist_matrix = 0.5 * iou_dist + 0.5 * center_dist
        
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return dist_matrix


def compute_iou_3d(boxes1: torch.Tensor, boxes2: torch.Tensor):
    """
    Compute 3D IoU between all pairs of boxes.
    
    Args:
        boxes1: Tensor of shape (N, 6)
        boxes2: Tensor of shape (M, 6)
    
    Returns:
        IoU matrix of shape (N, M)
    """
    N = boxes1.shape[0]
    M = boxes2.shape[0]
    
    # Expand dimensions for broadcasting
    boxes1_exp = boxes1.unsqueeze(1)  # (N, 1, 6)
    boxes2_exp = boxes2.unsqueeze(0)  # (1, M, 6)
    
    # Compute intersection
    inter_min = torch.maximum(boxes1_exp[..., :3], boxes2_exp[..., :3])  # (N, M, 3)
    inter_max = torch.minimum(boxes1_exp[..., 3:], boxes2_exp[..., 3:])  # (N, M, 3)
    inter_dims = torch.clamp(inter_max - inter_min, min=0)  # (N, M, 3)
    inter_volume = inter_dims.prod(dim=-1)  # (N, M)
    
    # Compute volumes
    vol1 = (boxes1[:, 3:] - boxes1[:, :3]).prod(dim=-1)  # (N,)
    vol2 = (boxes2[:, 3:] - boxes2[:, :3]).prod(dim=-1)  # (M,)
    
    # Compute union
    union_volume = vol1.unsqueeze(1) + vol2.unsqueeze(0) - inter_volume  # (N, M)
    
    # Compute IoU
    iou = inter_volume / (union_volume + 1e-10)
    
    return iou


def compute_hausdorff_distance(boxes1: torch.Tensor, boxes2: torch.Tensor):
    """
    Compute normalized Hausdorff distance between box corners.
    """
    def get_corners(boxes):
        """Get 8 corners of 3D boxes."""
        N = boxes.shape[0]
        corners = torch.zeros(N, 8, 3, device=boxes.device, dtype=boxes.dtype)
        for i, (dx, dy, dz) in enumerate([(0,0,0), (1,0,0), (0,1,0), (0,0,1), 
                                          (1,1,0), (1,0,1), (0,1,1), (1,1,1)]):
            corners[:, i, 0] = boxes[:, 0] * (1-dx) + boxes[:, 3] * dx
            corners[:, i, 1] = boxes[:, 1] * (1-dy) + boxes[:, 4] * dy
            corners[:, i, 2] = boxes[:, 2] * (1-dz) + boxes[:, 5] * dz
        return corners
    
    corners1 = get_corners(boxes1)  # (N, 8, 3)
    corners2 = get_corners(boxes2)  # (M, 8, 3)
    
    # Normalize by diagonal
    all_corners = torch.cat([corners1.view(-1, 3), corners2.view(-1, 3)], dim=0)
    bbox_min = all_corners.min(dim=0)[0]
    bbox_max = all_corners.max(dim=0)[0]
    diagonal = torch.norm(bbox_max - bbox_min)
    
    N, M = boxes1.shape[0], boxes2.shape[0]
    dist_matrix = torch.zeros(N, M, device=boxes1.device, dtype=boxes1.dtype)
    
    for i in range(N):
        for j in range(M):
            # Compute directed Hausdorff distances
            d1 = torch.cdist(corners1[i], corners2[j], p=2).min(dim=1)[0].max()
            d2 = torch.cdist(corners2[j], corners1[i], p=2).min(dim=1)[0].max()
            dist_matrix[i, j] = max(d1, d2) / (diagonal + 1e-10)
    
    return dist_matrix


def gromov_wasserstein_distance(boxes1: torch.Tensor, boxes2: torch.Tensor, 
                                metric='combined', epsilon=1e-3, max_iter=100):
    """
    Compute Gromov-Wasserstein distance between two sets of 3D bounding boxes.
    
    Args:
        boxes1: Tensor of shape (N, 6) 
        boxes2: Tensor of shape (M, 6)
        metric: Distance metric for intra-distances ('iou', 'center', 'hausdorff', 'combined')
        epsilon: Entropic regularization parameter (0 for exact GW)
        max_iter: Maximum number of iterations
    
    Returns:
        GW distance value
    """
    # Compute intra-distance matrices
    C1 = compute_intra_distances(boxes1, metric=metric)
    C2 = compute_intra_distances(boxes2, metric=metric)
    
    # Uniform distributions
    n1, n2 = boxes1.shape[0], boxes2.shape[0]
    p = torch.ones(n1, dtype=torch.float64) / n1
    q = torch.ones(n2, dtype=torch.float64) / n2
    
    # Convert to numpy for POT
    C1_np = C1.detach().cpu().numpy().astype(np.float64)
    C2_np = C2.detach().cpu().numpy().astype(np.float64)
    p_np = p.detach().cpu().numpy()
    q_np = q.detach().cpu().numpy()
    
    # Compute GW distance
    if epsilon > 0:
        # Entropic GW
        gw_dist = ot.gromov.entropic_gromov_wasserstein2(
            C1_np, C2_np, p_np, q_np, 
            loss_fun='square_loss', 
            epsilon=epsilon,
            max_iter=max_iter,
            tol=1e-9
        )
    else:
        # Exact GW
        gw_dist = ot.gromov.gromov_wasserstein2(
            C1_np, C2_np, p_np, q_np, 
            loss_fun='square_loss',
            max_iter=max_iter,
            tol=1e-9
        )
    
    return gw_dist


def fused_gromov_wasserstein_distance(boxes1: torch.Tensor, boxes2: torch.Tensor,
                                      alpha=0.5, metric='combined', 
                                      feature_type='ratios', epsilon=0.1, max_iter=100):
    """
    Compute Fused Gromov-Wasserstein distance between two sets of 3D bounding boxes.
    
    Args:
        boxes1: Tensor of shape (N, 6)
        boxes2: Tensor of shape (M, 6) 
        alpha: Trade-off parameter between feature and structure (0=pure Wasserstein, 1=pure Gromov)
        metric: Distance metric for intra-distances
        feature_type: Type of features to use ('ratios', 'volumes', 'centers', 'combined')
        epsilon: Entropic regularization parameter
        max_iter: Maximum number of iterations
    
    Returns:
        FGW distance value
    """
    # Compute structural distances
    C1 = compute_intra_distances(boxes1, metric=metric)
    C2 = compute_intra_distances(boxes2, metric=metric)
    
    # Compute features
    features1 = extract_box_features(boxes1, feature_type)
    features2 = extract_box_features(boxes2, feature_type)
    
    # Feature distance matrix
    M = torch.cdist(features1, features2, p=2)
    
    # Uniform distributions
    n1, n2 = boxes1.shape[0], boxes2.shape[0]
    p = torch.ones(n1, dtype=torch.float64) / n1
    q = torch.ones(n2, dtype=torch.float64) / n2
    
    # Convert to numpy
    C1_np = C1.detach().cpu().numpy().astype(np.float64)
    C2_np = C2.detach().cpu().numpy().astype(np.float64)
    M_np = M.detach().cpu().numpy().astype(np.float64)
    p_np = p.detach().cpu().numpy()
    q_np = q.detach().cpu().numpy()
    
    # Compute FGW distance
    if epsilon > 0:
        # Entropic FGW
        fgw_dist = ot.gromov.entropic_fused_gromov_wasserstein2(
            M_np, C1_np, C2_np, p_np, q_np,
            loss_fun='square_loss',
            epsilon=epsilon,
            alpha=alpha,
            max_iter=max_iter,
            tol=1e-9
        )
    else:
        # Exact FGW
        fgw_dist = ot.gromov.fused_gromov_wasserstein2(
            M_np, C1_np, C2_np, p_np, q_np,
            loss_fun='square_loss',
            alpha=alpha,
            max_iter=max_iter,
            tol=1e-9
        )
    
    return fgw_dist


def extract_box_features(boxes: torch.Tensor, feature_type='ratios'):
    """
    Extract features from bounding boxes.
    
    Args:
        boxes: Tensor of shape (N, 6)
        feature_type: Type of features ('ratios', 'volumes', 'centers', 'combined')
    
    Returns:
        Feature tensor of shape (N, D) where D depends on feature_type
    """
    dims = torch.abs(boxes[:, 3:] - boxes[:, :3])  # (N, 3)
    
    if feature_type == 'ratios':
        # Normalized aspect ratios
        max_dim = torch.max(dims, dim=1, keepdim=True)[0]
        features = dims / (max_dim + 1e-10)
        
    elif feature_type == 'volumes':
        # Log-normalized volumes
        volumes = dims.prod(dim=1, keepdim=True)  # (N, 1)
        log_volumes = torch.log(volumes + 1e-10)
        # Normalize
        features = (log_volumes - log_volumes.mean()) / (log_volumes.std() + 1e-10)
        
    elif feature_type == 'centers':
        # Normalized centers
        centers = (boxes[:, :3] + boxes[:, 3:]) / 2.0
        all_min = boxes[:, :3].min(dim=0)[0]
        all_max = boxes[:, 3:].max(dim=0)[0]
        range_val = all_max - all_min
        features = (centers - all_min) / (range_val + 1e-10)
        
    elif feature_type == 'combined':
        # Combine multiple features
        max_dim = torch.max(dims, dim=1, keepdim=True)[0]
        ratios = dims / (max_dim + 1e-10)
        
        volumes = dims.prod(dim=1, keepdim=True)
        log_volumes = torch.log(volumes + 1e-10)
        norm_volumes = (log_volumes - log_volumes.mean()) / (log_volumes.std() + 1e-10)
        
        centers = (boxes[:, :3] + boxes[:, 3:]) / 2.0
        all_min = boxes[:, :3].min(dim=0)[0]
        all_max = boxes[:, 3:].max(dim=0)[0]
        range_val = all_max - all_min
        norm_centers = (centers - all_min) / (range_val + 1e-10)
        
        features = torch.cat([ratios, norm_volumes, norm_centers], dim=1)
        
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")
    
    return features

def compute_bb_features(boxes: torch.Tensor, **norm_params):
    """
    Input:
        boxes: (N_faces, 6) --> (min_x, min_y, min_z, max_x, max_y, max_z)
    Output:
        Z: (N_faces, 5) --> (x/sx, y/sy, zc/sz, r1, r2)
        norm_params: (sx, sy, sz, beta)
    """
    sx = norm_params.get('sx', None)
    sy = norm_params.get('sy', None)
    sz = norm_params.get('sz', None)
    beta = norm_params.get('beta', "auto")
    eps = norm_params.get('eps', 1e-9)

    mins = boxes[:, :3]
    maxs = boxes[:, 3:]
    sizes = np.maximum(maxs - mins, eps)  # clamp to avoid zeros
    centers = (mins + maxs) / 2.0

    x, y, zc = centers[:, 0], centers[:, 1], centers[:, 2]
    w, h, d = sizes[:, 0], sizes[:, 1], sizes[:, 2]

    # scale-invariant shape coordinates (log-aspects)
    r1 = np.log(w) - np.log(h)         # log(w/h)
    r2 = np.log(d) - np.log(h)         # log(d/h)

    # center scaling
    if sx is None: sx = np.std(x) + eps
    if sy is None: sy = np.std(y) + eps
    if sz is None: sz = np.std(zc) + eps

    Xc = np.stack([x / sx, y / sy, zc / sz], axis=1)

    # choose beta to roughly match variances (so coupling isn't dominated by one side)
    if beta == "auto":
        std_center = np.mean(np.std(Xc, axis=0)) + eps
        std_shape = np.mean(np.std(np.stack([r1, r2], axis=1), axis=0)) + eps
        beta_val = std_center / std_shape
    else:
        beta_val = float(beta)

    R = beta_val * np.stack([r1, r2], axis=1)
    Z = np.concatenate([Xc, R], axis=1)  # (N, 5)

    norm_params = dict(sx=sx, sy=sy, sz=sz, beta=beta_val, eps=eps)
    return Z, norm_params

def vector_to_np(vector):
    """
    Convert a vector3d dict into a numpy vector
    """
    x = vector["x"]
    y = vector["y"]
    z = vector["z"]
    h = 0.0
    return np.array([x, y, z, h])

def transform_to_np(transform):
    """
    Convert a transform dict into a
    numpy 4x4 affine transformation matrix
    """
    x_axis = vector_to_np(transform["x_axis"])
    y_axis = vector_to_np(transform["y_axis"])
    z_axis = vector_to_np(transform["z_axis"])
    translation = vector_to_np(transform["origin"])
    translation[3] = 1.0
    return np.transpose(np.stack([x_axis, y_axis, z_axis, translation]))

def pad_pts(pts):
    """
    Pad n,3 points with a 1 to become n,4
    suitable for use with a 4x4 matrix
    """
    return np.pad(
        pts,
        ((0, 0), (0, 1)),
        mode="constant",
        constant_values=1
    )

def transform_pts_by_matrix(pts, matrix, copy=False):
    """
    Transform an array of xyz pts (n, 3) by a 4x4 matrix using numpy
    """
    # Transpose first
    if pts.shape[1] == 3:
        # If necessary pad the points with a zero to be (n, 4)
        v = pad_pts(pts).T
    elif pts.shape[1] == 4:
        v = pts.T
    v = matrix @ v
    # Transpose and crop back to (n, 3)
    if copy:
        # Reorder to C-contiguous order
        return v.T[:, 0:3].copy()
    else:
        return v.T[:, 0:3]



def update_mapping(data_dict):
    """
    Remove unused key index from data dictionary.
    """
    dict_new = {}
    mapping = {}
    max_idx = max(data_dict.keys())
    skipped_indices = np.array(sorted(list(set(np.arange(max_idx)) - set(data_dict.keys()))))
    for idx, value in data_dict.items():
        skips = (skipped_indices < idx).sum()
        idx_new = idx - skips
        dict_new[idx_new] = value
        mapping[idx] = idx_new
    return dict_new, mapping


def face_edge_adj(shape):
    """
    *** COPY AND MODIFIED FROM THE ORIGINAL OCCWL SOURCE CODE ***
    Extract face/edge geometry and create a face-edge adjacency 
    graph from the given shape (Solid or Compound)

    Args:
    - shape (Shell, Solid, or Compound): Shape
        
    Returns:
    - face_dict: Dictionary of occwl faces, with face ID as the key
    - edge_dict: Dictionary of occwl edges, with edge ID as the key
    - edgeFace_IncM: Edge ID as the key, Adjacent faces ID as the value 
    """
    assert isinstance(shape, (Shell, Solid, Compound))
    mapper = EntityMapper(shape)
   
    ### Faces ###
    face_dict = {}
    for face in shape.faces():
        face_idx = mapper.face_index(face)
        face_dict[face_idx] = (face.surface_type(), face)

    ### Edges and IncidenceMat ###
    edgeFace_IncM = {}
    edge_dict = {}
    for edge in shape.edges():
        if not edge.has_curve():
            continue    

        connected_faces = list(shape.faces_from_edge(edge))
        if len(connected_faces) == 2 and not edge.seam(connected_faces[0]) and not edge.seam(connected_faces[1]):
            left_face, right_face = edge.find_left_and_right_faces(connected_faces)
            if left_face is None or right_face is None:
                continue
            edge_idx = mapper.edge_index(edge) 
            edge_dict[edge_idx] = edge 
            left_index = mapper.face_index(left_face)
            right_index = mapper.face_index(right_face)

            if edge_idx in edgeFace_IncM:
                edgeFace_IncM[edge_idx] += [left_index, right_index]
            else:
                edgeFace_IncM[edge_idx] = [left_index, right_index]
        else:
            pass # ignore seam

    return face_dict, edge_dict, edgeFace_IncM


def extract_primitive(solid, num_u=32, num_v=32):
    """
    Extract all primitive information from splitted solid

    Args:
    - solid (occwl.Solid): A single b-rep solid in occwl format

    Returns:
    - face_pnts (N x num_u x num_v x 3): Sampled uv-grid points on the bounded surface region (face)
    - edge_pnts (M x num_u x 3): Sampled u-grid points on the boundged curve region (edge)
    - edge_corner_pnts (M x 2 x 3): Start & end vertices per edge
    - edgeFace_IncM (M x 2): Edge-Face incident matrix, every edge is connect to two face IDs
    - faceEdge_IncM: A list of N sublist, where each sublist represents the adjacent edge IDs to a face
    """
    assert isinstance(solid, Solid)

    # Retrieve face, edge geometry and face-edge adjacency
    face_dict, edge_dict, edgeFace_IncM = face_edge_adj(solid)

    # Skip unused index key, and update the adj 
    face_dict, face_map = update_mapping(face_dict)
    edge_dict, edge_map = update_mapping(edge_dict)
    edgeFace_IncM_update = {}
    for key, value in edgeFace_IncM.items():
        new_face_indices = [face_map[x] for x in value]
        edgeFace_IncM_update[edge_map[key]] = new_face_indices 
    edgeFace_IncM = edgeFace_IncM_update
    
    # Face-edge adj
    num_faces = len(face_dict)
    edgeFace_IncM = np.stack([x for x in edgeFace_IncM.values()])
    faceEdge_IncM = []
    for surf_idx in range(num_faces):
        surf_edges, _ = np.where(edgeFace_IncM == surf_idx)
        faceEdge_IncM.append(surf_edges)
    
    # Sample uv-grid from surface (num_u x num_v)
    graph_face_feat = {}
    for face_idx, face_feature in face_dict.items():
        _, face = face_feature
        points = uvgrid(
            face, method="point", num_u=num_u, num_v=num_v
        )
        visibility_status = uvgrid(
            face, method="visibility_status", num_u=num_u, num_v=num_v
        )
        mask = np.logical_or(visibility_status == 0, visibility_status == 2)  # 0: Inside, 1: Outside, 2: On boundary
        # Concatenate channel-wise to form face feature tensor
        face_feat = np.concatenate((points, mask), axis=-1)
        graph_face_feat[face_idx] = face_feat
    face_pnts = np.stack([x for x in graph_face_feat.values()])[:,:,:,:3]
    
    # sample u-grid from curve (1xnum_u)
    graph_edge_feat = {}
    graph_corner_feat = {}
    for edge_idx, edge in edge_dict.items():
        points = ugrid(edge, method="point", num_u=num_u)
        graph_edge_feat[edge_idx] = points
        #### edge corners as start/end vertex ###
        v_start = points[0]  
        v_end = points[-1]
        graph_corner_feat[edge_idx] = (v_start, v_end)
    edge_pnts = np.stack([x for x in graph_edge_feat.values()])
    edge_corner_pnts = np.stack([x for x in graph_corner_feat.values()])

    return [face_pnts, edge_pnts, edge_corner_pnts, edgeFace_IncM, faceEdge_IncM]


def grid_to_triangles(H, W, device=None, dtype=torch.long):
    idx = torch.arange(H*W, device=device, dtype=dtype).view(H, W)
    faces = []
    for i in range(H-1):
        for j in range(W-1):
            v00 = idx[i, j];   v01 = idx[i, j+1]
            v10 = idx[i+1, j]; v11 = idx[i+1, j+1]
            faces.append(torch.tensor([v00, v10, v11], device=device, dtype=dtype))
            faces.append(torch.tensor([v00, v11, v01], device=device, dtype=dtype))
    return torch.stack(faces, dim=0)  # (F,3)


def proj_loss_p3d_min(A_bhw3, B_bhw3, faces=None, reduction="mean", return_distance=False):
    """
    A_bhw3, B_bhw3: (B,H,W,3), faces: (F,3)
    Returns scalar mean squared distance from A points to surface of B.
    Relies on the patched pytorch3d point_mesh_face_distance (reduction / return_distance).
    """
    if len(A_bhw3.shape) == 3:
        A_bhw3 = A_bhw3.unsqueeze(0)
    if len(B_bhw3.shape) == 3:
        B_bhw3 = B_bhw3.unsqueeze(0)

    Bsz, H, W, _ = A_bhw3.shape
    device = A_bhw3.device

    if faces is None:
        faces = grid_to_triangles(H, W, device=device)

    # Build batched Meshes for B
    verts = [B_bhw3[b].reshape(-1,3) for b in range(Bsz)]
    faces_b = [faces.detach().clone() for _ in range(Bsz)]   # same topology per batch
    meshB = Meshes(verts=verts, faces=faces_b)

    # Build batched Pointclouds for A
    points_list = [A_bhw3[b].reshape(-1, 3) for b in range(Bsz)]
    pcls = Pointclouds(points=points_list)

    ret = point_mesh_face_distance(meshB, pcls, reduction=reduction, return_distance=return_distance)  # (B,N)
    return ret


def pairwise_distance_wcs(src_wcs, tgt_wcs):
    # src_wcs: (N, H, W, 3), tgt_wcs: (M, H, W, 3)
    tgt_wcs_list = []
    src_wcs_list = []
    for i in range(tgt_wcs.shape[0]):
        for j in range(src_wcs.shape[0]):
            tgt_wcs_list.append(tgt_wcs[i])
            src_wcs_list.append(src_wcs[j])
    tgt_wcs_list = torch.stack(tgt_wcs_list) # (M, H, W, 3)
    src_wcs_list = torch.stack(src_wcs_list) # (N, H, W, 3)
    _, pairwise_distance = proj_loss_p3d_min(src_wcs_list, tgt_wcs_list, reduction="mean", return_distance=True) # _ , (M * N)
    pairwise_distance = pairwise_distance.view(tgt_wcs.shape[0], src_wcs.shape[0])
    return pairwise_distance.transpose(0,1)


def pairwise_chamfer_dist(src_shapes, target_shapes):
    if isinstance(src_shapes, np.ndarray): src_shapes = torch.from_numpy(src_shapes) # (N_src, N_points, 3)
    if isinstance(target_shapes, np.ndarray): target_shapes = torch.from_numpy(target_shapes) # (N_target, N_points, 3)
    assert len(src_shapes.shape) == 3 and len(target_shapes.shape) == 3
    # Create pairwise batch
    src_shapes_batch = []
    target_shapes_batch = []
    for i in range(src_shapes.shape[0]):
        for j in range(target_shapes.shape[0]):
            src_shapes_batch.append(src_shapes[i])
            target_shapes_batch.append(target_shapes[j])
    src_shapes_batch = torch.stack(src_shapes_batch).cuda()
    target_shapes_batch = torch.stack(target_shapes_batch).cuda()
    chamfer = ChamferDistance()
    pairwise_distance = chamfer(src_shapes_batch, target_shapes_batch, bidirectional=True, batch_reduction=None)
    return pairwise_distance.detach().cpu().view(src_shapes.shape[0], target_shapes.shape[0])

def edge_structure_assignment(edge_wcs_1, edge_wcs_2):
    cost_matrix = pairwise_chamfer_dist(edge_wcs_1, edge_wcs_2)
    # List 8 cases of the matching
    matching_cases = [
        [[2, 1, 3, 0], [2, 1, 3, 0]],
        [[2, 1, 3, 0], [1, 3, 0, 2]],
        [[2, 1, 3, 0], [3, 0, 2, 1]],
        [[2, 1, 3, 0], [0, 2, 1, 3]],
        [[2, 1, 3, 0], [2, 0, 3, 1]],
        [[2, 1, 3, 0], [0, 3, 1, 2]],
        [[2, 1, 3, 0], [3, 1, 2, 0]],
        [[2, 1, 3, 0], [1, 2, 0, 3]],
    ]
    # Check 8 cases of the matching
    cost_matching = []
    for matching_case in matching_cases:
        cost_matching.append(
            cost_matrix[matching_case[0][0], matching_case[1][0]] + 
            cost_matrix[matching_case[0][1], matching_case[1][1]] + 
            cost_matrix[matching_case[0][2], matching_case[1][2]] + 
            cost_matrix[matching_case[0][3], matching_case[1][3]])

    argmin_cost_matching = np.argmin(cost_matching)
    edge_idx = matching_cases[argmin_cost_matching][0]
    ref_edge_idx = matching_cases[argmin_cost_matching][1]
    return edge_idx, ref_edge_idx


def pairwise_chamfer_dist_wcs(src_wcs, tgt_wcs, bidirectional=False):
    # src_wcs: (N, H, W, 3), tgt_wcs: (M, H, W, 3)
    tgt_wcs_list = []
    src_wcs_list = []
    for i in range(src_wcs.shape[0]):
        for j in range(tgt_wcs.shape[0]):
            src_wcs_list.append(src_wcs[i])
            tgt_wcs_list.append(tgt_wcs[j])
    src_wcs_list = torch.stack(src_wcs_list).reshape(len(src_wcs_list), -1, 3) # (N, H * W, 3)
    tgt_wcs_list = torch.stack(tgt_wcs_list).reshape(len(tgt_wcs_list), -1, 3) # (M, H * W, 3)
    chamfer = ChamferDistance()
    pairwise_distance = chamfer(src_wcs_list, tgt_wcs_list, bidirectional=bidirectional, batch_reduction=None)
    pairwise_distance = pairwise_distance.view(src_wcs.shape[0], tgt_wcs.shape[0])
    return pairwise_distance