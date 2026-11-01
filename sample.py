import os
import yaml
import torch
import argparse
import numpy as np
import pickle
import time
from tqdm import tqdm
from model.network import SurfPosNet, SurfZNet, EdgePosNet, EdgeZNet, AutoencoderKLFastDecode, AutoencoderKLFastEncode, AutoencoderKL1DFastDecode
from diffusers import DDPMScheduler, PNDMScheduler
from OCC.Extend.DataExchange import write_step_file, write_stl_file
from utils import (
    randn_tensor,
    compute_bbox_center_and_size,
    construct_brep,
    detect_shared_vertex,
    detect_shared_edge,
    joint_optimize,
    pad_zero,
    decode_surf_wcs,
    angle_loss,
    edge_len_loss,
    fused_gromov_wasserstein_distance,
    pairwise_distance_wcs,
    edge_structure_assignment,
    pairwise_chamfer_dist_wcs,
    proj_loss_p3d_min,
    grid_to_triangles,
)

from chamferdist import ChamferDistance
from scipy.optimize import linear_sum_assignment

def compute_edge_length_loss(face_wcs, cond_face_wcs, cond_reference_edges_wcs, cond_face_reference_edges_mask):
    """
    Input:
        face_wcs: (n_faces, 32, 32, 3)
        cond_face_wcs: (n_faces, 32, 32, 3)
        cond_reference_edges_wcs: {'left': (n_faces, 32, 3), 'right': (n_faces, 32, 3), 'top': (n_faces, 32, 3), 'bottom': (n_faces, 32, 3)}
        cond_reference_edges_codes: {'left': (n_faces, ), 'right': (n_faces, ), 'top': (n_faces, ), 'bottom': (n_faces, )}
    Output:
        loss: (1, )
    """
    left_edges = face_wcs[:, :, 0, :] # (n_faces, 32, 3)
    right_edges = face_wcs[:, :, -1, :] # (n_faces, 32, 3)
    top_edges = face_wcs[:, 0, :, :] # (n_faces, 32, 3)
    bottom_edges = face_wcs[:, -1, :, :] # (n_faces, 32, 3)

    left_loss = edge_len_loss(left_edges, cond_reference_edges_wcs['left'], mask=cond_face_reference_edges_mask['left'])
    right_loss = edge_len_loss(right_edges, cond_reference_edges_wcs['right'], mask=cond_face_reference_edges_mask['right'])
    top_loss = edge_len_loss(top_edges, cond_reference_edges_wcs['top'], mask=cond_face_reference_edges_mask['top'])
    bottom_loss = edge_len_loss(bottom_edges, cond_reference_edges_wcs['bottom'], mask=cond_face_reference_edges_mask['bottom'])

    loss = left_loss + right_loss + top_loss + bottom_loss
    return loss

def compute_edge_angle_loss(face_wcs, cond_face_wcs, cond_reference_edges_wcs, cond_face_reference_edges_mask):
    """
    Input:
        face_wcs: (n_faces, 32, 32, 3)
        cond_face_wcs: (n_faces, 32, 32, 3)
        cond_reference_edges_wcs: {'left': (n_faces, 32, 3), 'right': (n_faces, 32, 3), 'top': (n_faces, 32, 3), 'bottom': (n_faces, 32, 3)}
        cond_reference_edges_codes: {'left': (n_faces, ), 'right': (n_faces, ), 'top': (n_faces, ), 'bottom': (n_faces, )}
    Output:
        loss: (1, )
    """
    # Extract edges from the face_wcs
    left_edges = face_wcs[:, :, 0, :] # (n_faces, 32, 3)
    right_edges = face_wcs[:, :, -1, :] # (n_faces, 32, 3)
    top_edges = face_wcs[:, 0, :, :] # (n_faces, 32, 3)
    bottom_edges = face_wcs[:, -1, :, :] # (n_faces, 32, 3)

    left_angle_loss = angle_loss(left_edges, cond_reference_edges_wcs['left'], mask=cond_face_reference_edges_mask['left'])
    right_angle_loss = angle_loss(right_edges, cond_reference_edges_wcs['right'], mask=cond_face_reference_edges_mask['right'])
    top_angle_loss = angle_loss(top_edges, cond_reference_edges_wcs['top'], mask=cond_face_reference_edges_mask['top'])
    bottom_angle_loss = angle_loss(bottom_edges, cond_reference_edges_wcs['bottom'], mask=cond_face_reference_edges_mask['bottom'])

    loss = left_angle_loss + right_angle_loss + top_angle_loss + bottom_angle_loss
    return loss

def optimize_contact_surf(face_wcs, cond_face_wcs, cond_face_reference_edges_wcs, cond_face_reference_edges_mask, eval_args=None):
    """Geometrically align generated contact surfaces to their matched condition contact surfaces.

    Learns three additive residuals per face via gradient descent:
      - residual_u: (B, H, 3) — per-U-row offset, broadcast over V to preserve the internal structure of the surface.
      - residual_v: (B, W, 3) — per-V-column offset, broadcast over U to preserve the internal structure of the surface.
      - translation: (B, 3) — global translation per face.

    The updated surface is:
        face_wcs + residual_u[:, :, None, :] + residual_v[:, None, :, :] + translation[:, None, None, :]

    Optimization alternates between two phases every 50 steps:
      - Epochs [0–4] mod 50  (C_pos):   Optimize `translation` only (residuals detached).
            Loss: Lproj — our modified point-to-mesh projection loss pushing generated points onto the
            condition surface mesh (proj_loss_p3d_min with reduction="min").
      - Epochs [5–49] mod 50 (C_shape): Optimize `residual_u` / `residual_v` only (translation detached).
            Loss: Ledge (ELO) — penalizes edge length mismatch between generated boundary edges
            and the condition reference edges; plus W_ANGLE * Leangle (currently disabled, W_ANGLE=0).

    Args:
        face_wcs: (B, 32, 32, 3) generated contact surface point clouds, requires no grad on entry.
        cond_face_wcs: (B, 32, 32, 3) condition contact surface point clouds used as the projection target.
        cond_face_reference_edges_wcs: dict with keys 'left'/'right'/'top'/'bottom', each (B, 32, 3),
            holding the condition open-boundary edge geometry to enforce via ELO.
        cond_face_reference_edges_mask: dict with same keys, each (B,) bool tensor — True when that
            edge slot is an active reference edge for the corresponding face.
        eval_args: dict, must contain 'num_optim_steps' (total gradient steps).

    Returns:
        updated_face_wcs: (B, 32, 32, 3) detached tensor of optimized surface point clouds.
    """
    with torch.enable_grad():
        B,H,W,_ = face_wcs.shape
        device = face_wcs.device
        faces = grid_to_triangles(H,W,device)

        residual_u = torch.zeros(B, face_wcs.shape[1], face_wcs.shape[-1], device=face_wcs.device).requires_grad_(True) # (B, H, 3): per-row offset, broadcast over columns
        residual_v = torch.zeros(B, face_wcs.shape[1], face_wcs.shape[-1], device=face_wcs.device).requires_grad_(True) # (B, W, 3): per-column offset, broadcast over rows
        # Interior points are updated through residual_u[i] (driven by left/right boundary edges at row i) 
        # and residual_v[j] (driven by top/bottom boundary edges at column j), 
        # giving each point 4 corresponding boundary edge points.
        translation = torch.zeros(B, 3, device=face_wcs.device).requires_grad_(True) # (B, 3): per-face global translation

        optimizer = torch.optim.Adam([residual_u, residual_v, translation], lr=0.01)
        W_ANGLE = 0

        for epoch in range(0, eval_args["num_optim_steps"]):
            loss = None
            optimizer.zero_grad()
            if epoch % 50 < 5:
                # C_pos
                res_u_detached = residual_u.detach()
                res_v_detached = residual_v.detach()
                updated_face_wcs = face_wcs + res_u_detached.unsqueeze(2) + res_v_detached.unsqueeze(1) 
                updated_face_wcs = updated_face_wcs.flatten(1, 2)
                updated_face_wcs = updated_face_wcs + translation.unsqueeze(1)
                updated_face_wcs = updated_face_wcs.unflatten(1, (face_wcs.shape[1], -1))

                Lproj = proj_loss_p3d_min(updated_face_wcs, cond_face_wcs, faces, reduction="min")
                loss = Lproj
            else:
                # C_shape
                translation_detached = translation.detach()
                updated_face_wcs = face_wcs + residual_u.unsqueeze(2) + residual_v.unsqueeze(1) 
                updated_face_wcs = updated_face_wcs.flatten(1, 2)
                updated_face_wcs = updated_face_wcs + translation_detached.unsqueeze(1)
                updated_face_wcs = updated_face_wcs.unflatten(1, (face_wcs.shape[1], -1))
                
                Ledge = compute_edge_length_loss(updated_face_wcs, cond_face_wcs, cond_face_reference_edges_wcs, cond_face_reference_edges_mask)
                Leangle = compute_edge_angle_loss(updated_face_wcs, cond_face_wcs, cond_face_reference_edges_wcs, cond_face_reference_edges_mask)
                loss = Ledge + W_ANGLE*Leangle

            if loss is not None:
                loss.backward()
                optimizer.step()

    updated_face_wcs = face_wcs + residual_u.unsqueeze(2) + residual_v.unsqueeze(1) 
    updated_face_wcs = updated_face_wcs.flatten(1, 2)
    updated_face_wcs = updated_face_wcs + translation.unsqueeze(1)
    updated_face_wcs = updated_face_wcs.unflatten(1, (face_wcs.shape[1], -1))
    return updated_face_wcs.detach()

def preprocess_condition_object(data, cond_contact_surface_ids, num_surfaces, num_contact_faces):
    """Normalize the condition object geometry and extract contact surface metadata for geometric guidance.

    The condition object's surfaces, edges, and bounding boxes are globally normalized to [-3, 3].
    Surface positions are then split into contact vs. rest slots and zero-padded to fixed sizes
    compatible with SurfPosNet's batched input layout (contact faces first, then rest).

    For geometric guidance during sampling, each contact surface's UV-boundary edges (left/right/top/bottom
    slices of the 32x32 grid) are matched against the surface's boundary edges — those edges
    belonging to exactly one contact face.
    Matched pairs are stored as "reference edges" and used by optimize_contact_surf to enforce alignment.

    Args:
        data: dict with keys 'surf_bbox_wcs' (N,6), 'surf_wcs' (N,32,32,3), 'edge_wcs' (E,32,3),
              'faceEdge_adj' (N, variable) for the condition object.
        cond_contact_surface_ids: list of surface indices in the condition object that are contact surfaces.
        num_surfaces: total padded surface count expected by SurfPosNet.
        num_contact_faces: number of contact face slots (placed first in the padded layout).

    Returns:
        padded_cond_surfPos: (num_surfaces, 6) normalized bbox array, contact faces first.
        cond_surf_mask: (num_surfaces,) bool mask, True where the slot is padding.
        cond_contact_surf: dict with contact surface geometry and reference edge metadata.
    """
    cond_surfPos = data['surf_bbox_wcs']
    cond_surf_wcs = data['surf_wcs']
    cond_edge_wcs = data['edge_wcs']
    cond_FaceEdgeAdj = data['faceEdge_adj']

    # Normalize all geometry to [-bbox_scaled, bbox_scaled] using a single global scale
    # derived from the axis-aligned bounding box of all surface bbox corners.
    bbox_scaled = 3.0
    bbox_corners = np.array(cond_surfPos).reshape(-1, 3)
    min_vals = np.min(bbox_corners, axis=0)
    max_vals = np.max(bbox_corners, axis=0)
    bbox_center = min_vals + (max_vals - min_vals) / 2
    global_scale = max(max_vals - min_vals)

    bbox_corners = ((bbox_corners - bbox_center[np.newaxis, :]) / (global_scale * 0.5)) * bbox_scaled
    cond_surf_wcs = (cond_surf_wcs - bbox_center[np.newaxis, np.newaxis, np.newaxis, :]) / (global_scale * 0.5) * bbox_scaled
    cond_edge_wcs = (cond_edge_wcs - bbox_center[np.newaxis, np.newaxis, :]) / (global_scale * 0.5) * bbox_scaled
    cond_surfPos = bbox_corners.reshape(-1, 6)

    # Split surfaces into contact and rest, then zero-pad each group to its fixed slot count.
    # SurfPosNet expects contact faces to occupy the first num_contact_faces positions.
    assert num_contact_faces > 0 and num_surfaces - num_contact_faces > 0, \
        f"num_contact_faces: {num_contact_faces}, num_surfaces: {num_surfaces} is not valid"
    cond_contact_surf_pos = cond_surfPos[cond_contact_surface_ids]
    cond_rest_surf_pos = cond_surfPos[list(set(range(cond_surfPos.shape[0])) - set(cond_contact_surface_ids))]
    padded_cond_contact_surf_pos, cond_contact_surf_mask = pad_zero(cond_contact_surf_pos, num_contact_faces, return_mask=True)
    padded_cond_rest_surf_pos, cond_rest_surf_mask = pad_zero(cond_rest_surf_pos, num_surfaces - num_contact_faces, return_mask=True)
    padded_cond_surfPos = np.concatenate([padded_cond_contact_surf_pos, padded_cond_rest_surf_pos], axis=0)
    cond_surf_mask = np.concatenate([cond_contact_surf_mask, cond_rest_surf_mask], axis=0)

    cond_contact_surf_wcs = torch.FloatTensor(cond_surf_wcs[cond_contact_surface_ids])  # (n_contact_faces, 32, 32, 3)
    cond_contact_surf_pos = torch.FloatTensor(cond_surfPos[cond_contact_surface_ids])   # (n_contact_faces, 6)

    # Find boundary edges of the contact region: edges that belong to exactly one
    # contact surface.
    all_contact_edge_ids = set()
    shared_edge_ids = set()
    for contact_surface_id in cond_contact_surface_ids:
        face_edges = set(cond_FaceEdgeAdj[contact_surface_id])
        shared_edge_ids.update(face_edges.intersection(all_contact_edge_ids))
        all_contact_edge_ids.update(face_edges)
    contact_boundary_edge_ids = list(all_contact_edge_ids - shared_edge_ids)

    # For each contact surface, record which of its edges are boundary edges.
    # Order of items in cond_contact_surf_boundary_edges_ids matches cond_contact_surf_wcs order 
    # because of using the same cond_contact_surface_ids
    cond_contact_surf_boundary_edges_ids = []
    for contact_surface_id in cond_contact_surface_ids:
        edge_ids = [eid for eid in cond_FaceEdgeAdj[contact_surface_id] if eid in contact_boundary_edge_ids]
        cond_contact_surf_boundary_edges_ids.append(np.array(edge_ids, dtype=np.int32))
    cond_contact_surf_boundary_edges_ids = np.array(cond_contact_surf_boundary_edges_ids, dtype=object)

    # For each contact surface, identify which UV-boundary slices (left/right/top/bottom edges of
    # the 32x32 grid) correspond to boundary edges. These matched pairs are "reference
    # edges": geometric anchors used by optimize_contact_surf to align the generated surface's edges
    # with the condition object's edges. Matching is done via bidirectional Chamfer distance.
    UV_EDGE_SLICES = ['left', 'right', 'top', 'bottom']
    CHAMFER_MATCH_THRESHOLD = 0.01
    chamfer_dist = ChamferDistance()

    cond_contact_surf_reference_edges_wcs = []
    cond_contact_surf_reference_edges_codes = []
    for surf_idx in range(len(cond_contact_surf_wcs)):
        face_wcs = cond_contact_surf_wcs[surf_idx]  # (32, 32, 3)
        boundary_edges_ids = cond_contact_surf_boundary_edges_ids[surf_idx].astype(np.int32)
        boundary_edges_wcs = torch.FloatTensor(cond_edge_wcs[boundary_edges_ids])  # (n_boundary, 32, 3)

        # Extract the four UV-boundary edge curves from the surface grid.
        # left=u-min column, right=u-max column, top=v-min row, bottom=v-max row.
        uv_edges = [
            face_wcs[:, 0, :].reshape(-1, 3),   # left
            face_wcs[:, -1, :].reshape(-1, 3),  # right
            face_wcs[0, :, :].reshape(-1, 3),   # top
            face_wcs[-1, :, :].reshape(-1, 3),  # bottom
        ]

        matched_ref_edge_ids = []
        matched_uv_codes = []
        for uv_idx, uv_edge in enumerate(uv_edges):
            for ref_edge_idx in range(boundary_edges_wcs.shape[0]):
                cd = chamfer_dist(uv_edge.unsqueeze(0), boundary_edges_wcs[ref_edge_idx].unsqueeze(0), bidirectional=True)
                if cd < CHAMFER_MATCH_THRESHOLD:
                    matched_ref_edge_ids.append(ref_edge_idx)
                    matched_uv_codes.append(UV_EDGE_SLICES[uv_idx])
                    break

        cond_contact_surf_reference_edges_wcs.append(
            np.array(boundary_edges_wcs[matched_ref_edge_ids], dtype=np.float32))
        cond_contact_surf_reference_edges_codes.append(
            np.array(matched_uv_codes, dtype=object))

    cond_contact_surf_reference_edges_wcs = np.array(cond_contact_surf_reference_edges_wcs, dtype=object)
    cond_contact_surf_reference_edges_codes = np.array(cond_contact_surf_reference_edges_codes, dtype=object)

    cond_contact_surf = {
        'cond_contact_surf_pos': cond_contact_surf_pos,
        'cond_contact_surf_wcs': cond_contact_surf_wcs,
        'cond_contact_surf_boundary_edges_ids': cond_contact_surf_boundary_edges_ids,
        'cond_contact_surf_reference_edges_wcs': cond_contact_surf_reference_edges_wcs,
        'cond_contact_surf_reference_edges_codes': cond_contact_surf_reference_edges_codes,
    }

    return padded_cond_surfPos, cond_surf_mask, cond_contact_surf

def pairwise_distance_boxes_dim(src_boxes, tgt_boxes):
    """Compute pairwise bbox distance between two sets of axis-aligned bounding boxes.

    Distance between two boxes = L2(center_src, center_tgt) + L2(dim_src, dim_tgt).
    This joint metric penalizes both positional and size mismatch.

    Args:
        src_boxes: (N, 6) tensor of bboxes encoded as [min_xyz, max_xyz].
        tgt_boxes: (M, 6) tensor of bboxes encoded as [min_xyz, max_xyz].

    Returns:
        (N, M) tensor of pairwise distances.
    """
    pairwise_distance = []
    for i in range(tgt_boxes.shape[0]):
        for j in range(src_boxes.shape[0]):
            src_center = (src_boxes[j, :3] + src_boxes[j, 3:]) / 2
            tgt_center = (tgt_boxes[i, :3] + tgt_boxes[i, 3:]) / 2
            coord_distance = torch.norm(src_center - tgt_center, p=2)

            src_dim = torch.abs(src_boxes[j, 3:] - src_boxes[j, :3])
            tgt_dim = torch.abs(tgt_boxes[i, 3:] - tgt_boxes[i, :3])
            dim_distance = torch.norm(src_dim - tgt_dim, p=2)

            pairwise_distance.append(coord_distance + dim_distance)
    pairwise_distance = torch.stack(pairwise_distance).view(tgt_boxes.shape[0], src_boxes.shape[0])
    return pairwise_distance.transpose(0, 1)

def match_contact_faces(cond_contact_surf_wcs, surfPos, surfZ, surf_vae, contact_surf_mask, surfMask, device='cpu'):
    """Match each generated contact surface to its corresponding condition contact surface.

    After SurfZNet sampling, the generated contact faces need to be associated with specific
    condition contact faces so that optimize_contact_surf can apply the correct reference edges.
    Matching is done per-sample via the Hungarian algorithm on a pairwise WCS distance matrix.

    Args:
        cond_contact_surf_wcs: (n_cond_faces, 32, 32, 3) condition contact surface geometry.
        surfPos: (B, max_num_surfaces, 6) generated surface bounding boxes.
        surfZ: (B, max_num_surfaces, 48) generated surface latent z.
        surf_vae: VAE decoder to convert latent z to UV-space point clouds.
        contact_surf_mask: (B, max_num_surfaces) bool mask, True for contact surface slots.
        surfMask: (B, max_num_surfaces) bool mask, True for padding slots.
        device: torch device string for distance computation.

    Returns:
        List of B dicts mapping generated_contact_surf_idx -> cond_contact_surf_idx,
        one dict per sample in the batch [{contact_surf_ind: cond_contact_surf_ind}]
    """
    cond_contact_surf_wcs = cond_contact_surf_wcs.to(device)
    aligned_contact_surf_records = []
    for sample_idx in range(surfPos.shape[0]):
        valid_mask = ~surfMask[sample_idx]
        # Decode latent z to WCS point clouds for all valid (non-padding) surfaces.
        surf_wcs_cad = decode_surf_wcs(
            surf_vae,
            surfZ[sample_idx][valid_mask].numpy(),
            surfPos[sample_idx][valid_mask].numpy(),
        )
        # Select only the contact surface slots among the valid surfaces.
        contact_surf_wcs = torch.FloatTensor(surf_wcs_cad[contact_surf_mask[sample_idx][valid_mask]]).to(device)  # (n_gen_contact, 32, 32, 3)

        # Hungarian matching on the (n_gen_contact × n_cond_contact) WCS distance matrix.
        cost_matrix = pairwise_distance_wcs(contact_surf_wcs, cond_contact_surf_wcs).cpu().numpy()
        gen_idx, cond_idx = linear_sum_assignment(cost_matrix)
        aligned_contact_surf_records.append(dict(zip(gen_idx, cond_idx)))

    return aligned_contact_surf_records

def remove_duplicate_surfaces(surfPos, bbox_threshold, num_surfaces, contact_mask):
    """Remove near-duplicate surface bounding boxes from SurfPosNet output and re-pad to fixed size.

    SurfPosNet can generate redundant surfaces with nearly identical bounding boxes. This function
    removes duplicates by scanning each sample's bbox list and dropping any box that is within
    `bbox_threshold` of an already-accepted box (checking both the box and its min/max-swapped
    reverse, since the same surface can be encoded in either orientation).

    Args:
        surfPos: (B, max_num_surfaces, 6) tensor of generated bounding boxes (flattened [min,max]).
        bbox_threshold: max-axis L-inf distance below which two bboxes are considered duplicates.
        num_surfaces: target padded surface count after deduplication.
        contact_mask: (B, max_num_surfaces) bool tensor, True for contact surface slots.

    Returns:
        surfPos_deduplicate: list of B (num_surfaces, 6) tensors.
        surfMask_deduplicate: list of B (1, num_surfaces) bool tensors, True for padding slots.
        contact_surf_mask_deduplicate: list of B (num_surfaces,) bool tensors, True for contact slots.
    """
    surfPos_deduplicate = []
    surfMask_deduplicate = []
    contact_surf_mask_deduplicate = []
    for ii in range(surfPos.shape[0]):
        # Reshape from (max_surfaces, 6) → (max_surfaces, 2, 3) for per-axis comparison.
        bboxes = np.round(surfPos[ii].unflatten(-1, torch.Size([2, 3])).detach().cpu().numpy(), 4)
        non_repeat = bboxes[:1]
        contact_surf_mask = [contact_mask[ii][0]]

        for bbox_idx, bbox in enumerate(bboxes):
            diff = np.max(np.max(np.abs(non_repeat - bbox), -1), -1)
            same = diff < bbox_threshold
            # A bbox stored as [min,max] and [max,min] represents the same surface.
            bbox_rev = bbox[::-1]
            diff_rev = np.max(np.max(np.abs(non_repeat - bbox_rev), -1), -1)
            same_rev = diff_rev < bbox_threshold
            if same.sum() >= 1 or same_rev.sum() >= 1:
                continue  # duplicate — skip
            non_repeat = np.concatenate([non_repeat, bbox[np.newaxis, :, :]], 0)
            contact_surf_mask.append(contact_mask[ii][bbox_idx])

        bboxes = non_repeat.reshape(len(non_repeat), -1)
        n_valid = len(bboxes)
        n_pad = num_surfaces - n_valid

        surf_mask = torch.zeros((1, n_valid)) == 1                                          # False for valid slots
        contact_surf_mask = torch.BoolTensor(contact_surf_mask)

        bbox_padded = torch.concat([torch.FloatTensor(bboxes), torch.zeros(n_pad, 6)])
        mask_padded = torch.concat([surf_mask, torch.zeros(1, n_pad) == 0], -1)            # True for padding slots
        contact_surf_mask_padded = torch.concat([contact_surf_mask, torch.zeros(n_pad) == 1], -1)

        surfPos_deduplicate.append(bbox_padded)
        surfMask_deduplicate.append(mask_padded)
        contact_surf_mask_deduplicate.append(contact_surf_mask_padded)

    return surfPos_deduplicate, surfMask_deduplicate, contact_surf_mask_deduplicate

def generate_surfZ(model, initial_surfZ, surfPos, surfMask, class_label):
    surfZ = initial_surfZ

    surfz_scheduler = PNDMScheduler(
        num_train_timesteps=1000,
        beta_schedule='linear',
        prediction_type='epsilon',
        beta_start = 0.0001,
        beta_end = 0.02,
    )
            
    surfz_scheduler.set_timesteps(200)    
    for t in surfz_scheduler.timesteps: 
        timesteps = t.reshape(-1).cuda()
        pred = model(surfZ, timesteps, surfPos, surfMask, class_label)
        surfZ = surfz_scheduler.step(pred, t, surfZ).prev_sample

    return surfZ

def prepare_optimization_batch_data(surfPos, surfMask, surf_vae, surfZ, aligned_contact_surf_records, cond_contact_surf, visualization=False, step_idx=0):
    """Assemble per-contact-pair data needed to run optimize_contact_surf across the batch.

    For each matched (generated contact surface, condition contact surface) pair from
    aligned_contact_surf_records, this function:
      1. Decodes the generated surface latent to a UV-space point cloud (32, 32, 3).
      2. Extracts the four boundary edges of both the generated and condition surfaces:
           index 0 = left  : face_wcs[:, 0,  :]   (first UV column)
           index 1 = right : face_wcs[:, -1, :]   (last UV column)
           index 2 = top   : face_wcs[0,  :, :]   (first UV row)
           index 3 = bottom: face_wcs[-1, :, :]   (last UV row)
      3. Calls edge_structure_assignment to find the best orientation-consistent matching
         between generated and condition boundary edges.
      4. Filters to only those generated edges whose matched condition edge is a
         boundary (reference) edge — i.e. listed in cond_contact_surf_reference_edges_codes.
         These are the edges that should snap to the condition geometry during optimization.
      5. Packs everything into flat batched lists where batched_sample_mask tracks which
         batch sample each entry belongs to.

    Args:
        surfPos: (B, max_num_surfaces, 6) generated surface bounding boxes.
        surfMask: (B, max_num_surfaces) bool mask, True for padding slots.
        surf_vae: VAE decoder to convert latent codes to UV-space point clouds.
        surfZ: (B, max_num_surfaces, 48) generated surface latent codes.
        aligned_contact_surf_records: list of B dicts mapping
            generated_contact_surf_idx -> cond_contact_surf_idx (from match_contact_faces).
        cond_contact_surf: dict with keys:
            'cond_contact_surf_wcs': (n_cond, 32, 32, 3) condition surface geometry.
            'cond_contact_surf_reference_edges_wcs': list of (n_ref, 32, 3) per condition surface.
            'cond_contact_surf_reference_edges_codes': list of string lists ('left'/'right'/'top'/'bottom').
        visualization: unused, reserved for future debug rendering.
        step_idx: unused, reserved for future debug rendering.

    Returns:
        batched_contact_surf_wcs: list of (32, 32, 3) tensors, one per contact pair.
        batched_cond_contact_surf_wcs: list of (32, 32, 3) tensors, one per contact pair.
        batched_reference_edges: dict with keys 'left'/'right'/'top'/'bottom', each a list of
            (1, 32, 3) tensors (real edge or zeros when that edge is not a reference).
        batched_reference_edges_mask: dict with same keys, each a list of scalar BoolTensors
            indicating whether the corresponding entry in batched_reference_edges is valid.
        list_surf_wcs_cad: list of B decoded surface arrays, one per batch sample.
        batched_sample_mask: list of int sample indices, one per contact pair.
    """
    EDGE_CODE_TO_IDX = {'left': 0, 'right': 1, 'top': 2, 'bottom': 3}
    EDGE_DIRECTIONS = ['left', 'right', 'top', 'bottom']

    batched_contact_surf_wcs = []
    batched_sample_mask = []
    batched_cond_contact_surf_wcs = []
    batched_reference_edges = {d: [] for d in EDGE_DIRECTIONS}
    batched_reference_edges_mask = {d: [] for d in EDGE_DIRECTIONS}

    list_surf_wcs_cad = []
    for sample_idx in range(surfPos.shape[0]):
        valid_mask = ~surfMask[sample_idx]
        surf_wcs_cad = decode_surf_wcs(
            surf_vae,
            surfZ[sample_idx][valid_mask].numpy(),
            surfPos[sample_idx][valid_mask].numpy(),
        )
        list_surf_wcs_cad.append(surf_wcs_cad)

        for contact_surf_idx, cond_contact_surf_idx in aligned_contact_surf_records[sample_idx].items():
            face_wcs = torch.FloatTensor(surf_wcs_cad[contact_surf_idx]) # (32, 32, 3)
            batched_contact_surf_wcs.append(face_wcs)
            batched_sample_mask.append(sample_idx)

            cond_face_wcs = torch.FloatTensor(cond_contact_surf['cond_contact_surf_wcs'][cond_contact_surf_idx]) # (32, 32, 3)
            batched_cond_contact_surf_wcs.append(cond_face_wcs)

            cond_reference_edges_codes = cond_contact_surf['cond_contact_surf_reference_edges_codes'][cond_contact_surf_idx] # list of 'left'/'right'/'top'/'bottom'
            cond_reference_edges_ids = set(EDGE_CODE_TO_IDX[c] for c in cond_reference_edges_codes)

            # Extract the 4 boundary edges of each surface from their UV-grid boundaries.
            edge_wcs = torch.stack([
                face_wcs[:, 0, :],   # left
                face_wcs[:, -1, :],  # right
                face_wcs[0, :, :],   # top
                face_wcs[-1, :, :]   # bottom
            ], dim=0) # (4, 32, 3)
            cond_edge_wcs = torch.stack([
                cond_face_wcs[:, 0, :],
                cond_face_wcs[:, -1, :],
                cond_face_wcs[0, :, :],
                cond_face_wcs[-1, :, :]
            ], dim=0) # (4, 32, 3)

            # Find orientation-consistent edge assignment, then filter to reference edges only.
            edge_idx = []
            ref_edge_idx = []
            if cond_reference_edges_ids:
                edge_idx_, ref_edge_idx_ = edge_structure_assignment(edge_wcs, cond_edge_wcs)
                for gen_e, cond_e in zip(edge_idx_, ref_edge_idx_):
                    if cond_e in cond_reference_edges_ids:
                        edge_idx.append(gen_e)
                        ref_edge_idx.append(cond_e)
                edge_idx = np.array(edge_idx)
                ref_edge_idx = np.array(ref_edge_idx)

            for dir_name, dir_id in EDGE_CODE_TO_IDX.items():
                if dir_id in edge_idx:
                    batched_reference_edges[dir_name].append(cond_edge_wcs[ref_edge_idx[edge_idx == dir_id]]) # (1, 32, 3)
                    batched_reference_edges_mask[dir_name].append(torch.BoolTensor([True]))
                else:
                    batched_reference_edges[dir_name].append(torch.zeros((1, 32, 3)))
                    batched_reference_edges_mask[dir_name].append(torch.BoolTensor([False]))

    return batched_contact_surf_wcs, batched_cond_contact_surf_wcs, batched_reference_edges, batched_reference_edges_mask, list_surf_wcs_cad, batched_sample_mask

def generate_surf_pos(surfPos_model, surfPos, step_t, prompt, cond_surfPos, cond_surf_mask, contact_mask, scheduler):
    """
    Args:
        surfPos_model: SurfPosNet
        surfPos: (N_faces, 6)
        timestep: (1,)
        prompt: str
        cond_surfPos: (N_faces, 6)
    """
    timestep = step_t.reshape(-1).cuda()
    pred = surfPos_model(surfPos, timestep, prompt, cond_surfPos, cond_surf_mask, contact_mask)
    surfPos = scheduler.step(pred, step_t, surfPos).prev_sample
    return surfPos

def perform_guidance(surfPos_input=None, surfPos_model=None, step_t=1000, prompt=None, cond_surfPos=None, cond_surf_mask=None, contact_mask=None,
                cond_contact_surf_wcs=None, batch_size=16, num_candidates=8,
                bbox_threshold=0.02, num_surfaces=70, class_label=None, surfZ_model=None, surf_vae=None, device='cpu',
                w_sem=1.0):
    """Zero-order search over surface layout candidates to find the best geometrically consistent sample.

    Runs the full SurfPos + SurfZ generation pipeline num_candidates times in parallel (by repeating
    surfPos_input num_candidates times), then scores each candidate against the condition contact
    surfaces and returns the best one.

    Scoring uses a combined geometric + semantic distance:
        score = d_geo + w_sem * d_sem
      - d_geo: mean min bidirectional Chamfer distance from generated contact surface point clouds
               to condition contact surface point clouds — measures how well the generated contact
               faces match the condition geometry.
      - d_sem: Fused Gromov-Wasserstein distance between the candidate surface positions and the
               original (pre-guidance) surface positions — penalizes layout drift from the input.
               Disabled when w_sem=0.

    Args:
        surfPos_input: (1, max_num_surfaces, 6) or (max_num_surfaces, 6) initial surface bounding
            boxes used as the starting point for all candidates.
        surfPos_model: SurfPosNet, denoises surface bounding box layouts.
        step_t: int, diffusion timestep at which to start denoising (supports partial denoising).
        prompt: list of B text embeddings; only the first num_candidates entries are used.
        cond_surfPos: (B, max_num_surfaces, 6) condition object surface positions; trimmed to num_candidates.
        cond_surf_mask: (B, max_num_surfaces) bool padding mask for condition surfaces; trimmed to num_candidates.
        contact_mask: (B, max_num_surfaces) bool mask marking contact surface slots; trimmed to num_candidates.
        cond_contact_surf_wcs: (n_cond, 32, 32, 3) condition contact surface point clouds.
        batch_size: int, must be >= num_candidates (reserved for batched forward passes).
        num_candidates: int, number of independent candidates to generate and score.
        bbox_threshold: float, IoU/distance threshold for deduplicating generated surfaces.
        num_surfaces: int, max surface slots after deduplication.
        class_label: unused, passed through to generate_surfZ for optional class conditioning.
        surfZ_model: SurfZNet, denoises surface latent codes.
        surf_vae: VAE decoder used to decode surface latents to point clouds for scoring.
        device: torch device string.
        w_sem: float, weight on the semantic layout distance term. Set to 0 to disable.

    Returns:
        best_surfPos: (max_num_surfaces, 6) surface bounding boxes of the highest-scoring candidate,
            on `device`.
    """
    surfPos = surfPos_input.detach().clone()
    if len(surfPos.shape) == 2:
        surfPos = surfPos.unsqueeze(0)

    assert len(surfPos.shape) == 3 and surfPos.shape[0] == 1
    assert batch_size >= num_candidates

    regularized_surf_pos = surfPos[0].detach().clone().to(device)

    ddpm_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule='linear',
        prediction_type='epsilon',
        beta_start=0.0001,
        beta_end=0.02,
        clip_sample=True,
        clip_sample_range=3,
    )
    ddpm_scheduler.set_timesteps(1000)

    surfPos = surfPos.repeat(num_candidates, 1, 1).to(device)
    prompt = prompt[:num_candidates]
    cond_surfPos = cond_surfPos[:num_candidates]
    cond_surf_mask = cond_surf_mask[:num_candidates]
    contact_mask = contact_mask[:num_candidates]

    surfPos = generate_surf_pos(surfPos_model, surfPos, step_t, prompt, cond_surfPos, cond_surf_mask, contact_mask, ddpm_scheduler)

    surfPos_deduplicate, surfMask_deduplicate, contact_surf_mask_deduplicate = remove_duplicate_surfaces(surfPos, bbox_threshold, num_surfaces, contact_mask)

    surfPos_deduplicate = torch.stack(surfPos_deduplicate).to(device)
    surfMask = torch.vstack(surfMask_deduplicate).to(device)
    contact_surf_mask = torch.stack(contact_surf_mask_deduplicate).cpu()

    surfZ = randn_tensor((num_candidates, num_surfaces, 48)).to(device)
    surfZ = generate_surfZ(surfZ_model, surfZ, surfPos_deduplicate, surfMask, class_label)

    surfZ = surfZ.detach().cpu()
    surfPos_deduplicate = surfPos_deduplicate.detach().cpu()
    surfMask = surfMask.detach().cpu()
    cond_contact_surf_wcs = cond_contact_surf_wcs.to(device)

    best_score = float('inf')
    best_idx = -1
    for sample_idx in range(num_candidates):
        valid_mask = ~surfMask[sample_idx]
        surf_wcs_cad = decode_surf_wcs(
            surf_vae,
            surfZ[sample_idx][valid_mask].numpy(),
            surfPos_deduplicate[sample_idx][valid_mask].numpy(),
        )
        contact_surf_wcs = torch.FloatTensor(surf_wcs_cad[contact_surf_mask[sample_idx][valid_mask]]).to(device)  # (n_contact, 32, 32, 3)
        # D_geo: per-face min-assignment rather than flat merged Chamfer (paper Eq. 4).
        # Equivalent for the common single-contact-face case; more discriminative for multi-face.
        dist_matrix = pairwise_chamfer_dist_wcs(contact_surf_wcs, cond_contact_surf_wcs, bidirectional=True)  # (n_contact, n_cond)
        d_geo = dist_matrix.min(dim=1).values.mean().item()

        d_sem = fused_gromov_wasserstein_distance(surfPos[sample_idx], regularized_surf_pos, alpha=0.5, metric='center', feature_type='ratios') if w_sem > 0.0 else 0.0

        # Implementation of Zero-Order Search (ZO) - Pick the best scored sample
        score = d_geo + w_sem * d_sem
        if score < best_score:
            best_score = score
            best_idx = sample_idx

    return surfPos[best_idx].to(device)


def sample(eval_args):
    # Inference configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = eval_args['batch_size']
    z_threshold = eval_args['z_threshold']
    bbox_threshold =eval_args['bbox_threshold']
    save_folder = eval_args['save_folder']
    num_surfaces = eval_args['num_surfaces'] 
    num_edges = eval_args['num_edges']
    num_contact_faces = eval_args['num_contact_faces']
    input_data_file = eval_args['input_data_file']
    guidance_timesteps = eval_args['guidance_timesteps']

    object_name = input_data_file.split('/')[-1].split('.')[0].replace('merged_', '')

    target_object = eval_args['target_object']
    cond_object = 'object_2' if target_object == 'object_1' else 'object_1'
    surface_id_cond = 'surface2_id' if target_object == 'object_1' else 'surface1_id'

    # Load data
    with open(input_data_file, "rb") as tf:
        data = pickle.load(tf)

    prompt = data[target_object]['prompt']
    print(f'Prompt: {prompt}')

    contact_surfaces = data['contact_surfaces']
    cond_contact_surface_ids = set()
    for contact_surface in contact_surfaces:
        cond_contact_surface_ids.update(contact_surface[surface_id_cond])
    cond_contact_surface_ids = list(cond_contact_surface_ids)

    padded_cond_surfPos, cond_surf_mask, cond_contact_surf = preprocess_condition_object(data[cond_object], cond_contact_surface_ids, num_surfaces, num_contact_faces)

    # Convert to tensor
    padded_cond_surfPos = torch.FloatTensor(padded_cond_surfPos).cuda()
    cond_surf_mask = torch.BoolTensor(cond_surf_mask).cuda()

    class_label = None

    surfPos_model = SurfPosNet(eval_args['use_cf'])
    surfPos_model.load_state_dict(torch.load(eval_args['surfpos_weight'], weights_only=True), strict=False)  
    surfPos_model = surfPos_model.to(device).eval()

    surfZ_model = SurfZNet(eval_args['use_cf'])
    surfZ_model.load_state_dict(torch.load(eval_args['surfz_weight'], weights_only=True), strict=False)
    surfZ_model = surfZ_model.to(device).eval()

    edgePos_model = EdgePosNet(eval_args['use_cf'])
    edgePos_model.load_state_dict(torch.load(eval_args['edgepos_weight'], weights_only=True))
    edgePos_model = edgePos_model.to(device).eval()

    edgeZ_model = EdgeZNet(eval_args['use_cf']) 
    edgeZ_model.load_state_dict(torch.load(eval_args['edgez_weight'], weights_only=True))
    edgeZ_model = edgeZ_model.to(device).eval()

    surf_vae = AutoencoderKLFastDecode(in_channels=3,
        out_channels=3,
        down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
        up_block_types= ['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
        block_out_channels=[128, 256, 512, 512],
        layers_per_block=2,
        act_fn='silu',
        latent_channels=3,
        norm_num_groups=32,
        sample_size=512,
    )
    surf_vae.load_state_dict(torch.load(eval_args['surfvae_weight'], weights_only=True), strict=False)
    surf_vae = surf_vae.to(device).eval()

    surf_vae_encoder = AutoencoderKLFastEncode(in_channels=3,
        out_channels=3,
        down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
        up_block_types= ['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
        block_out_channels=[128, 256, 512, 512],
        layers_per_block=2,
        act_fn='silu',
        latent_channels=3,
        norm_num_groups=32,
        sample_size=512,
    )
    surf_vae_encoder.load_state_dict(torch.load(eval_args['surfvae_weight'], weights_only=True), strict=False)
    surf_vae_encoder = surf_vae_encoder.to(device).eval()

    edge_vae = AutoencoderKL1DFastDecode(
        in_channels=3,
        out_channels=3,
        down_block_types=['DownBlock1D', 'DownBlock1D', 'DownBlock1D'],
        up_block_types=['UpBlock1D', 'UpBlock1D', 'UpBlock1D'],
        block_out_channels=[128, 256, 512],  
        layers_per_block=2,
        act_fn='silu',
        latent_channels=3,
        norm_num_groups=32,
        sample_size=512
    )
    edge_vae.load_state_dict(torch.load(eval_args['edgevae_weight'], weights_only=True), strict=False)
    edge_vae = edge_vae.to(device).eval()

    pndm_scheduler = PNDMScheduler(
        num_train_timesteps=1000,
        beta_schedule='linear',
        prediction_type='epsilon',
        beta_start = 0.0001,
        beta_end = 0.02,
        skip_prk_steps=True,
    )

    ddpm_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule='linear',
        prediction_type='epsilon',
        beta_start = 0.0001,
        beta_end = 0.02,
        clip_sample = True,
        clip_sample_range=3
    ) 
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=False):
            ##########################################
            # STEP 1: Generate surface positions     #
            ##########################################
            surfPos = randn_tensor((batch_size, num_surfaces, 6)).to(device) # (B, N, 6)
            cond_surfPos = padded_cond_surfPos.unsqueeze(0).repeat(batch_size, 1, 1)
            cond_surf_mask = cond_surf_mask.unsqueeze(0).repeat(batch_size, 1)
            batched_prompt = [prompt] * batch_size

            # contact_mask marks the first num_contact_faces slots as contact surface candidates.
            # These slots are reserved for surfaces that will be aligned to the condition object.
            contact_mask = np.zeros((num_surfaces,), dtype=bool)
            contact_mask[:num_contact_faces] = True
            contact_mask = torch.BoolTensor(contact_mask).unsqueeze(0).repeat(batch_size, 1).to(device)

            contact_surf_mask = None
            surfMask = None

            pndm_scheduler.set_timesteps(200)
            ddpm_scheduler.set_timesteps(1000)

            sampling_configs = [
                {'scheduler': pndm_scheduler, 'timesteps': pndm_scheduler.timesteps[:158]},
                {'scheduler': ddpm_scheduler, 'timesteps': ddpm_scheduler.timesteps[-250:]},
            ]

            for config_idx, sampling_config in enumerate(sampling_configs):
                scheduler = sampling_config['scheduler']
                timesteps = sampling_config['timesteps']
                for step_t in tqdm(timesteps):
                    step_idx = step_t.item()

                    surfPos = surfPos.to(device)
                    timestep = step_t.reshape(-1).cuda()
                    pred = surfPos_model(surfPos, timestep, batched_prompt, cond_surfPos, cond_surf_mask, contact_mask)
                    surfPos = scheduler.step(pred, step_t, surfPos).prev_sample

                    # Snapshot before any in-place modification; used as input to perform_guidance.
                    current_surfPos = surfPos.detach().clone()

                    is_last_phase = config_idx == len(sampling_configs) - 1
                    is_last_step = step_idx <= scheduler.timesteps[-1].item()

                    # At the final denoising step, deduplicate surfaces and generate their latent codes
                    # (surfZ), then exit the loop. surfPos / surfMask / surfZ are the outputs of Step 1.
                    if is_last_phase and is_last_step:
                        surfPos_deduplicate, surfMask_deduplicate, contact_surf_mask_deduplicate = remove_duplicate_surfaces(surfPos, bbox_threshold, num_surfaces, contact_mask)
                        surfPos = torch.stack(surfPos_deduplicate).cuda()
                        surfMask = torch.vstack(surfMask_deduplicate).cuda()
                        contact_surf_mask = torch.stack(contact_surf_mask_deduplicate)
                        surfZ = randn_tensor((batch_size, num_surfaces, 48)).to(device)
                        surfZ = generate_surfZ(surfZ_model, surfZ, surfPos, surfMask, class_label)
                        break

                    # Skip geometric enforcement if this timestep is not in the guidance schedule,
                    # or if we are already in the final denoising phase (enforcement only applies to Phase 1).
                    if step_idx not in guidance_timesteps or (is_last_phase and step_idx == timesteps[-1].item()):
                        continue

                    ########### Geometric Enforcement ###########
                    # At selected guidance timesteps during Phase 1, we:
                    #   1. Deduplicate the current surface layout candidates.
                    #   2. Decode surface latents to point clouds and match generated contact faces
                    #      to the condition object's contact faces.
                    #   3. Optimize matched contact face geometry to align with the condition.
                    #   4. Re-run SurfPosNet from current_surfPos using the optimized contact surfaces
                    #      as geometric guidance (zero-order search), updating surfPos in-place.
                    print(f"Geometric Enforcement at timestep {step_idx}")

                    surfPos_deduplicate, surfMask_deduplicate, contact_surf_mask_deduplicate = remove_duplicate_surfaces(surfPos, bbox_threshold, num_surfaces, contact_mask)
                    surfPos = torch.stack(surfPos_deduplicate).cuda()
                    surfMask = torch.vstack(surfMask_deduplicate).cuda()
                    contact_surf_mask = torch.stack(contact_surf_mask_deduplicate)

                    surfZ = randn_tensor((batch_size, num_surfaces, 48)).to(device)
                    surfZ = generate_surfZ(surfZ_model, surfZ, surfPos, surfMask, class_label)

                    surfZ = surfZ.detach().cpu()
                    surfPos = surfPos.detach().cpu()
                    surfMask = surfMask.detach().cpu()

                    # Match each generated contact face to the closest condition contact face.
                    matched_contact_face_records = match_contact_faces(
                        cond_contact_surf['cond_contact_surf_wcs'], surfPos, surfZ, surf_vae,
                        contact_surf_mask, surfMask, device=device,
                    )

                    # Collect matched contact face point clouds and their condition counterparts into
                    # a flat batch for efficient optimization.
                    batched_contact_surf_wcs, batched_cond_contact_surf_wcs, batched_reference_edges, \
                        batched_reference_edges_mask, list_surf_wcs_cad, _ = prepare_optimization_batch_data(
                            surfPos, surfMask, surf_vae, surfZ, matched_contact_face_records,
                            cond_contact_surf, visualization=False, step_idx=step_idx,
                        )

                    batched_contact_surf_wcs = torch.stack(batched_contact_surf_wcs).to(device)
                    batched_cond_contact_surf_wcs = torch.stack(batched_cond_contact_surf_wcs).to(device)
                    batched_reference_edges = {k: torch.concat(v).to(device) for k, v in batched_reference_edges.items()}
                    batched_reference_edges_mask = {k: torch.concat(v).to(device) for k, v in batched_reference_edges_mask.items()}

                    # Gradient-based optimization: move generated contact faces toward condition faces
                    batched_contact_surf_wcs = optimize_contact_surf(
                        batched_contact_surf_wcs,
                        batched_cond_contact_surf_wcs,
                        batched_reference_edges,
                        batched_reference_edges_mask,
                        eval_args=eval_args,
                    )

                    # Scatter the optimized contact faces back into per-sample surface lists,
                    # then update surfPos for each sample based on optimized contact faces.
                    updated_surf_wcs_idx = 0
                    for sample_idx in range(surfPos.shape[0]):
                        guiding_surf_wcs = []
                        for contact_surf_idx in matched_contact_face_records[sample_idx]:
                            list_surf_wcs_cad[sample_idx][contact_surf_idx] = batched_contact_surf_wcs[updated_surf_wcs_idx].detach().cpu().numpy()
                            guiding_surf_wcs.append(torch.FloatTensor(list_surf_wcs_cad[sample_idx][contact_surf_idx]))
                            updated_surf_wcs_idx += 1

                        guiding_surf_wcs = torch.stack(guiding_surf_wcs).to(device)
                        surfPos[sample_idx] = perform_guidance(
                            surfPos_input=current_surfPos[sample_idx],
                            surfPos_model=surfPos_model, step_t=step_t,
                            prompt=batched_prompt, cond_surfPos=cond_surfPos,
                            cond_surf_mask=cond_surf_mask, contact_mask=contact_mask,
                            cond_contact_surf_wcs=guiding_surf_wcs,
                            batch_size=batch_size, num_candidates=eval_args['num_candidates'],
                            bbox_threshold=bbox_threshold, num_surfaces=num_surfaces,
                            class_label=class_label, surfZ_model=surfZ_model, surf_vae=surf_vae,
                            device=device, w_sem=eval_args['weight_semantic'],
                        )


            ########################################
            # STEP 2-1: generate the edge position #
            ########################################
            edgePos = randn_tensor((surfPos.shape[0], num_surfaces, num_edges, 6)).cuda()
        
            pndm_scheduler.set_timesteps(200)  
            for t in tqdm(pndm_scheduler.timesteps[:158]):  
                timesteps = t.reshape(-1).cuda()   
                noise_pred = edgePos_model(edgePos, timesteps, surfPos, surfZ, surfMask, class_label)
                edgePos = pndm_scheduler.step(noise_pred, t, edgePos).prev_sample

            ddpm_scheduler.set_timesteps(1000)  
            for t in tqdm(ddpm_scheduler.timesteps[-250:]):
                timesteps = t.reshape(-1).cuda()   
                noise_pred = edgePos_model(edgePos, timesteps, surfPos, surfZ, surfMask, class_label)
                edgePos = ddpm_scheduler.step(noise_pred, t, edgePos).prev_sample

            ####################################################
            # STEP 2-2: remove duplicate edges per face (bbox) #
            ####################################################
            edgeM = surfMask.unsqueeze(-1).repeat(1, 1, num_edges)

            for ii in range(batch_size):
                edge_bboxs = edgePos[ii][~surfMask[ii]].detach().cpu().numpy()

                for surf_idx, bboxes in enumerate(edge_bboxs):
                    bboxes = bboxes.reshape(len(bboxes),2,3)
                    valid_bbox = bboxes[0:1]
                    for bbox_idx, bbox in enumerate(bboxes):
                        diff = np.max(np.max(np.abs(valid_bbox - bbox),-1),-1)
                        bbox_rev = bbox[::-1]  # also test reverse bbox for matching
                        diff_rev = np.max(np.max(np.abs(valid_bbox - bbox_rev),-1),-1)
                        same = diff < bbox_threshold
                        same_rev = diff_rev < bbox_threshold
                        if same.sum()>=1 or same_rev.sum()>=1:
                            edgeM[ii, surf_idx, bbox_idx] = True
                            continue # repeat value
                        else:
                            valid_bbox = np.concatenate([valid_bbox, bbox[np.newaxis,:,:]],0)
                    edgeM[ii, surf_idx, 0] = False  # set first one to False  


            ##############################
            # STEP 2-3: generate edge zv #
            ##############################   
            edgeZV = randn_tensor((surfPos.shape[0], num_surfaces, num_edges, 18)).cuda()

            pndm_scheduler.set_timesteps(200)   
            for t in tqdm(pndm_scheduler.timesteps):
                timesteps = t.reshape(-1).cuda()   
                noise_pred = edgeZ_model(edgeZV, timesteps, edgePos, surfPos, surfZ, edgeM, class_label)
                edgeZV = pndm_scheduler.step(noise_pred, t, edgeZV).prev_sample

            edgeZV[edgeM] = 0 # set removed data to 0
            edge_z = edgeZV[:,:,:,:12]
            edgeV = edgeZV[:,:,:,12:].detach().cpu().numpy()

            # Decode the surfaces
            surf_ncs = surf_vae(surfZ.unflatten(-1,torch.Size([16,3])).flatten(0,1).permute(0,2,1).unflatten(-1,torch.Size([4,4])))
            surf_ncs = surf_ncs.permute(0,2,3,1).unflatten(0, torch.Size([surfPos.shape[0], num_surfaces])).detach().cpu().numpy()
            
            # Decode the edges
            edge_ncs = edge_vae(edge_z.unflatten(-1,torch.Size([4,3])).reshape(-1,4,3).permute(0,2,1))
            edge_ncs = edge_ncs.permute(0,2,1).reshape(surfPos.shape[0], num_surfaces, num_edges, 32, 3).detach().cpu().numpy()

            edge_mask = edgeM.detach().cpu().numpy()     
            edge_pos = edgePos.detach().cpu().numpy() / 3.0
            surfPos = surfPos.detach().cpu().numpy()  / 3.0


    #############################################
    ### STEP 3: Post-process (per-single CAD) ###
    #############################################
    for batch_idx in range(surfPos.shape[0]):
        # Per cad (not including invalid faces)
        surfMask_cad = surfMask[batch_idx].detach().cpu().numpy()
        edge_mask_cad = edge_mask[batch_idx][~surfMask_cad]
        edge_pos_cad = edge_pos[batch_idx][~surfMask_cad]
        edge_ncs_cad = edge_ncs[batch_idx][~surfMask_cad]
        edgeV_cad = edgeV[batch_idx][~surfMask_cad]
        edge_z_cad = edge_z[batch_idx][~surfMask[batch_idx]].detach().cpu().numpy()[~edge_mask_cad]
        surf_z_cad = surfZ[batch_idx][~surfMask[batch_idx]].detach().cpu().numpy()
        surf_pos_cad = surfPos[batch_idx][~surfMask_cad]

        # Retrieve vertices based on edge start/end
        edgeV_bbox = []
        for bbox, ncs, mask in zip(edge_pos_cad, edge_ncs_cad, edge_mask_cad):
            epos = bbox[~mask]
            edge = ncs[~mask]
            bbox_startends = []
            for bb, ee in zip(epos, edge): 
                bcenter, bsize = compute_bbox_center_and_size(bb[0:3], bb[3:])
                wcs = ee*(bsize/2) + bcenter
                bbox_start_end = wcs[[0,-1]]
                bbox_start_end = bbox_start_end.reshape(2,3)
                bbox_startends.append(bbox_start_end.reshape(1,2,3))
            bbox_startends = np.vstack(bbox_startends)
            edgeV_bbox.append(bbox_startends)
        
        ### 3-1: Detect shared vertices ###
        try:
            unique_vertices, new_vertex_dict = detect_shared_vertex(edgeV_cad, edge_mask_cad, edgeV_bbox)
        except Exception as e:
            print('Vertex detection failed...')
            continue
        
        ### 3-2: Detect shared edges ###
        try:
            unique_faces, unique_edges, FaceEdgeAdj, EdgeVertexAdj = detect_shared_edge(unique_vertices, new_vertex_dict, edge_z_cad, surf_z_cad, z_threshold, edge_mask_cad)
        except Exception as e:
            print('Edge detection failed...')
            continue
        
        # Decode unique faces / edges
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                surf_ncs_cad = surf_vae(torch.FloatTensor(unique_faces).cuda().unflatten(-1,torch.Size([16,3])).permute(0,2,1).unflatten(-1,torch.Size([4,4])))
                surf_ncs_cad = surf_ncs_cad.permute(0,2,3,1).detach().cpu().numpy()
                edge_ncs_cad = edge_vae(torch.FloatTensor(unique_edges).cuda().unflatten(-1,torch.Size([4,3])).permute(0,2,1))
                edge_ncs_cad = edge_ncs_cad.permute(0,2,1).detach().cpu().numpy()

        #### 3-3: Joint Optimize ###
        num_edge = len(edge_ncs_cad)
        num_surf = len(surf_ncs_cad)
        surf_wcs, edge_wcs = joint_optimize(surf_ncs_cad, edge_ncs_cad, surf_pos_cad, unique_vertices, EdgeVertexAdj, FaceEdgeAdj, num_edge, num_surf)
        
        #### 3-4: Build the B-rep ###
        try:
            solid = construct_brep(surf_wcs, edge_wcs, FaceEdgeAdj, EdgeVertexAdj)
        except Exception as e:
            print('B-rep rebuild failed...')
            continue

        # Save CAD model
        write_step_file(solid, f'{save_folder}/output_{object_name}_{batch_idx}.step')
        write_stl_file(solid, f'{save_folder}/output_{object_name}_{batch_idx}.stl', linear_deflection=0.001, angular_deflection=0.5)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default='knitcad', choices=['knitcad'],
                        help="Choose between evaluation mode (default: knitcad)")
    parser.add_argument("--input_data_file", type=str, default='data/joint_set_18083.pkl',
                        help="Path to the data file")
    parser.add_argument("--env", type=str, default='test',
                        help="Environment name")
    parser.add_argument("--target_object", type=str, default='object_1', choices=['object_1', 'object_2'],
                        help="Target object to generate")
    args = parser.parse_args()    

    # Load evaluation config 
    with open('eval_config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    eval_args = config[args.mode]
    eval_args['input_data_file'] = args.input_data_file
    eval_args['env'] = args.env
    eval_args['target_object'] = args.target_object
    
    while(True):
        sample(eval_args)