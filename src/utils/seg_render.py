"""Render subtree leaf projections and volumes for binary tree nodes.

Model settings come from the checkpoint's saved Hydra config_used.yaml.
Only current active Gaussians contribute density; history retains ancestry.
Exported volumes use the GT axis order (W, D, H) and its actual grid size.

Usage:
    python -m src.utils.seg_render \
        --ckpt  output/synthetic_dataset/cone_ntrain_50_angle_360/0_chest_cone/ckpts/ckpt_29999.pt \
        --data  data/synthetic_dataset/cone_ntrain_50_angle_360/0_chest_cone \
        --out   output/tree_seg \
        --max_depth 5 \
        --view_indices 0 25 50 75
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data.ctproj import CTImageDataset, CTImageParser
from src.models.gaussian_model import GaussianModel
from src.models.binary_tree_gaussian_model import BinaryTreeGaussianModel
from src.models.student_gaussian_model import StudentGaussianModel
from src.utils.checkpoint_config import load_model_kwargs
from src.utils.volume import volume_shape_from_parser


def merge_tree(m: BinaryTreeGaussianModel) -> dict:
    """Keep original tree indices and record where current active Gaussians begin."""
    return {
        "active_start": int(m.history_means.shape[0]),
        "means": torch.cat([m.history_means, m.means.data], dim=0),
        "scales": torch.cat([m.history_scales, m.scales.data], dim=0),
        "quats": torch.cat([m.history_quats, m.quats.data], dim=0),
        "opacities": torch.cat([m.history_opacities, m.opacities.data], dim=0),
        "parents": torch.cat([m.history_parent, m.parent], dim=0).long(),
    }


def get_subtree_leaves(
    node: int, left: torch.Tensor, right: torch.Tensor, *, active_start: int
) -> list:
    """Return active descendants, excluding childless archived nodes after pruning."""
    leaves = []
    queue = deque([node])
    while queue:
        n = queue.popleft()
        l, r = int(left[n].item()), int(right[n].item())
        if l < 0 and r < 0:
            if n >= active_start:
                leaves.append(n)
        else:
            if l >= 0:
                queue.append(l)
            if r >= 0:
                queue.append(r)
    return leaves


def collect_nodes_up_to_depth(
    root: int, left: torch.Tensor, right: torch.Tensor, level: torch.Tensor, max_depth: int
) -> list:
    """BFS to collect all nodes with depth <= max_depth."""
    nodes = []
    queue = deque([root])
    while queue:
        n = queue.popleft()
        d = int(level[n].item())
        if d > max_depth:
            continue
        nodes.append(n)
        l, r = int(left[n].item()), int(right[n].item())
        if l >= 0:
            queue.append(l)
        if r >= 0:
            queue.append(r)
    return nodes


def build_subset_model(indices, merged, device, **scale_kwargs) -> GaussianModel:
    if isinstance(indices, torch.Tensor):
        idx = indices.to(device)
    else:
        idx = torch.tensor(indices, device=device, dtype=torch.long)
    if ((idx < merged["active_start"]) | (idx >= merged["means"].shape[0])).any():
        raise ValueError("Segmentation subsets must contain only current active Gaussian indices.")
    m = GaussianModel(device=str(device), **scale_kwargs)
    m.means = torch.nn.Parameter(merged["means"][idx].clone(), requires_grad=False)
    m.scales = torch.nn.Parameter(merged["scales"][idx].clone(), requires_grad=False)
    m.quats = torch.nn.Parameter(merged["quats"][idx].clone(), requires_grad=False)
    m.opacities = torch.nn.Parameter(merged["opacities"][idx].clone(), requires_grad=False)
    m.eval()
    return m


@torch.no_grad()
def render_projection(gm: GaussianModel, sample) -> np.ndarray:
    c2w = sample["camtoworld"].float().unsqueeze(0).to(gm.device)
    K = sample["K"].float().unsqueeze(0).to(gm.device)
    image = sample["image"].float()
    width, height = int(image.shape[1]), int(image.shape[0])
    renders_hw, _ = gm.rasterize(
        camtoworlds=c2w, Ks=K,
        width=width, height=height,
        packed=False, sparse_grad=False, absgrad=False,
    )
    render = renders_hw.permute(0, 2, 1, 3).contiguous()
    return render[0, :, :, 0].cpu().numpy()


@torch.no_grad()
def compute_volume(gm: GaussianModel, volume_shape, bbox) -> np.ndarray | None:
    if bbox is None:
        return None
    vol = gm.voxelize(volume_shape, bbox.to(gm.device))
    if vol is None:
        return None
    return vol.squeeze(-1).permute(2, 1, 0).contiguous().cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Render binary-tree subtree projections")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to BinaryTreeGaussianModel checkpoint")
    parser.add_argument("--data", type=str, required=True, help="Path to scene data directory")
    parser.add_argument("--out", type=str, required=True, help="Output directory for rendered images")
    parser.add_argument("--max_depth", type=int, default=5, help="Max tree depth to explore (0-indexed, default 5 => 6 levels)")
    parser.add_argument("--view_indices", type=int, nargs="+", default=[0], help="Test view indices to render")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Segmentation rendering requires a CUDA device; CPU voxelization is unsupported.")
    print(f"Device: {device}")

    print("Loading model...")
    model_kwargs = load_model_kwargs(args.ckpt)
    model = BinaryTreeGaussianModel(device=str(device), **model_kwargs)
    model.load_ckpt(args.ckpt)
    model.eval()
    print(f"  Active Gaussians: {model.num_gaussians}")
    print(f"  History nodes:    {model.history_means.shape[0]}")

    print("Loading scene data...")
    scene_parser = CTImageParser(args.data)
    dataset = CTImageDataset(scene_parser, split="test")
    bbox = torch.tensor(scene_parser.bbox, dtype=torch.float32) if scene_parser.bbox is not None else None
    volume_shape = volume_shape_from_parser(scene_parser)
    print(f"  Test views:   {len(dataset)}")
    print(f"  Volume shape: {volume_shape}")
    print(f"  BBox:         {bbox}")

    print("Building tree...")
    merged = merge_tree(model)
    parents = merged["parents"]
    left, right, level = StudentGaussianModel._build_tree(parents)
    active_start = merged["active_start"]
    active_leaves = list(range(active_start, parents.numel()))
    if not active_leaves:
        raise ValueError("Checkpoint has no current active Gaussians to render.")
    root_indices = torch.where(parents < 0)[0].tolist()
    max_tree_depth = int(level.max().item())
    print(f"  Total nodes: {parents.numel()}")
    print(f"  Active leaves: {len(active_leaves)}")
    print(f"  Root nodes:  {root_indices}")
    print(f"  Max depth:   {max_tree_depth}")

    root = root_indices[0]
    target_nodes = collect_nodes_up_to_depth(root, left, right, level, args.max_depth)
    subtree_leaves = {}
    for node in target_nodes:
        leaves = get_subtree_leaves(node, left, right, active_start=active_start)
        if leaves:
            subtree_leaves[node] = leaves
    print(f"\nNodes to render (depth 0..{args.max_depth}): {len(subtree_leaves)}")

    full_model = build_subset_model(active_leaves, merged, device, **model_kwargs)

    os.makedirs(args.out, exist_ok=True)

    # --- Voxelize all nodes ---
    print("\nVoxelizing full tree...")
    full_dir = os.path.join(args.out, "full_tree")
    os.makedirs(full_dir, exist_ok=True)
    full_vol = compute_volume(full_model, volume_shape, bbox)
    if full_vol is not None:
        np.save(os.path.join(full_dir, "volume.npy"), full_vol)
        print(f"  Full tree volume saved: shape {full_vol.shape}")

    print("Voxelizing subtree nodes...")
    for node, leaves in subtree_leaves.items():
        depth = int(level[node].item())
        n_leaves = len(leaves)

        node_dir = os.path.join(args.out, f"depth_{depth}", f"node_{node}")
        os.makedirs(node_dir, exist_ok=True)

        sub_model = build_subset_model(leaves, merged, device, **model_kwargs)
        sub_vol = compute_volume(sub_model, volume_shape, bbox)
        if sub_vol is not None:
            np.save(os.path.join(node_dir, "volume.npy"), sub_vol)

        del sub_model
        torch.cuda.empty_cache()
        print(f"  Node {node:6d} (depth {depth}, {n_leaves:6d} leaves) -> {node_dir}")

    # --- Render projections ---
    print("\nRendering projections...")
    for vi in args.view_indices:
        if vi < 0 or vi >= len(dataset):
            print(f"  Skipping invalid view index {vi}")
            continue
        sample = dataset[vi]
        full_proj = render_projection(full_model, sample)
        np.save(os.path.join(full_dir, f"view_{vi:03d}.npy"), full_proj)

        for node, leaves in subtree_leaves.items():
            depth = int(level[node].item())
            n_leaves = len(leaves)

            node_dir = os.path.join(args.out, f"depth_{depth}", f"node_{node}")

            sub_model = build_subset_model(leaves, merged, device, **model_kwargs)
            sub_proj = render_projection(sub_model, sample)
            np.save(os.path.join(node_dir, f"view_{vi:03d}.npy"), sub_proj)

            del sub_model
            torch.cuda.empty_cache()

            print(f"  [view {vi:3d}] Node {node:6d} (depth {depth}, {n_leaves:6d} leaves)")

    print(f"\nDone. Results saved to {args.out}")


if __name__ == "__main__":
    main()
