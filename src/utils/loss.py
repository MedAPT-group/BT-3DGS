from typing import Dict, Optional

import torch
import torch.nn.functional as F

from src.utils.math import normalized_quat_to_rotmat
from src.utils.metric import ssim2d


def compute_tv_loss(vol: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    if vol.dim() == 4 and vol.size(0) == 1:
        vol = vol[0]
    assert vol.dim() == 3, "tv_3d_loss expects 3D volume"
    dx = torch.abs(vol[1:, :, :] - vol[:-1, :, :])
    dy = torch.abs(vol[:, 1:, :] - vol[:, :-1, :])
    dz = torch.abs(vol[:, :, 1:] - vol[:, :, :-1])
    tv = dx.sum() + dy.sum() + dz.sum()
    if reduction == "mean":
        denom = dx.numel() + dy.numel() + dz.numel()
        tv = tv / max(1, denom)
    return tv

def compute_edge_loss(pred: torch.Tensor, target: torch.Tensor, edge_mask: torch.Tensor) -> torch.Tensor:
    return (edge_mask * torch.abs(pred - target)).mean()


def compute_hierarchy_loss(
    child_means: torch.Tensor,
    parent_indices: torch.Tensor,
    history_means: torch.Tensor,
    history_scales: torch.Tensor,
    history_quats: torch.Tensor,
    scale_activation,
    max_parent_distance_factor: float = 1.0,
) -> torch.Tensor:
    """Penalize children that drift outside their archived parent's ellipsoid."""
    if (
        child_means.numel() == 0
        or parent_indices.numel() == 0
        or history_means.numel() == 0
        or history_scales.numel() == 0
        or history_quats.numel() == 0
    ):
        return child_means.new_zeros(())

    history_count = min(
        history_means.shape[0],
        history_scales.shape[0],
        history_quats.shape[0],
    )
    valid = (parent_indices >= 0) & (parent_indices < history_count)
    if not torch.any(valid):
        return child_means.new_zeros(())

    parent_ids = parent_indices[valid].to(dtype=torch.long)
    child_centers = child_means[valid]
    parent_centers = history_means.index_select(0, parent_ids)
    parent_scales = scale_activation(history_scales.index_select(0, parent_ids))
    parent_quats = F.normalize(history_quats.index_select(0, parent_ids), dim=-1)
    parent_rotmats = normalized_quat_to_rotmat(parent_quats)

    delta_world = child_centers - parent_centers
    delta_local = torch.einsum("nji,nj->ni", parent_rotmats, delta_world)
    normalized_radius = torch.linalg.norm(
        delta_local / parent_scales.clamp_min(1e-6),
        dim=-1,
    )
    allowed_radius = max(float(max_parent_distance_factor), 1e-6)
    normalized_excess = F.relu(normalized_radius - allowed_radius) / allowed_radius
    return normalized_excess.square().mean()

class LossComputer:
    """Compute weighted sum of projection L1/SSIM and volume TV."""

    def __init__(self, w_l1: float, w_ssim: float, w_tv: float, ssim_window: int = 11):
        self.w_l1 = w_l1
        self.w_ssim = w_ssim
        self.w_tv = w_tv
        self.ssim_window = ssim_window

    def __call__(
        self,
        renders: torch.Tensor,
        targets: torch.Tensor,
        volume_pred: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        l1 = torch.abs(renders - targets).mean()
        out["proj_l1"] = l1
        renders_cf = renders.permute(0, 3, 1, 2).contiguous()
        targets_cf = targets.permute(0, 3, 1, 2).contiguous()
        ssim_val = ssim2d(renders_cf, targets_cf, window_size=self.ssim_window)
        ssim_loss = 1.0 - ssim_val
        out["proj_ssim"] = ssim_loss
        if self.w_tv > 0 and volume_pred is not None:
            try:
                tv = compute_tv_loss(volume_pred.squeeze(), reduction="mean")
            except Exception:
                tv = torch.zeros((), device=renders.device)
        else:
            tv = torch.zeros((), device=renders.device)
        out["vol_tv"] = tv
        out["total"] = self.w_l1 * l1 + self.w_ssim * ssim_loss + self.w_tv * tv
        return out

class BinaryTreeLossComputer:
    """Compute weighted sum of projection L1/SSIM and volume TV."""

    def __init__(
        self,
        w_l1: float,
        w_ssim: float,
        w_tv: float,
        w_edge: float,
        w_hierarchy: float = 0.0,
        ssim_window: int = 11,
    ):
        self.w_l1 = w_l1
        self.w_ssim = w_ssim
        self.w_tv = w_tv
        self.w_edge = w_edge
        self.w_hierarchy = w_hierarchy
        self.ssim_window = ssim_window

    def __call__(
        self,
        renders: torch.Tensor,
        targets: torch.Tensor,
        edge_mask: Optional[torch.Tensor],
        volume_pred: Optional[torch.Tensor] = None,
        hierarchy_loss: Optional[torch.Tensor] = None,
        edge_targets: Optional[torch.Tensor] = None,
        edge_weight_override: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        l1 = torch.abs(renders - targets).mean()
        out["proj_l1"] = l1
        renders_cf = renders.permute(0, 3, 1, 2).contiguous()
        targets_cf = targets.permute(0, 3, 1, 2).contiguous()
        ssim_val = ssim2d(renders_cf, targets_cf, window_size=self.ssim_window)
        ssim_loss = 1.0 - ssim_val

        if edge_targets is None:
            edge_targets = targets
        if edge_mask is None:
            edge_loss = torch.zeros((), device=renders.device)
        else:
            edge_loss = compute_edge_loss(renders, edge_targets, edge_mask)

        out["proj_ssim"] = ssim_loss
        if self.w_tv > 0 and volume_pred is not None:
            try:
                tv = compute_tv_loss(volume_pred.squeeze(), reduction="mean")
            except Exception:
                tv = torch.zeros((), device=renders.device)
        else:
            tv = torch.zeros((), device=renders.device)
        out["vol_tv"] = tv
        out["edge"] = edge_loss
        if hierarchy_loss is None:
            hierarchy_loss = torch.zeros((), device=renders.device)
        out["hierarchy"] = hierarchy_loss
        edge_weight = self.w_edge if edge_weight_override is None else float(edge_weight_override)
        out["edge_weight"] = torch.tensor(edge_weight, device=renders.device)
        out["total"] = (
            self.w_l1 * l1
            + self.w_ssim * ssim_loss
            + self.w_tv * tv
            + edge_weight * edge_loss
            + self.w_hierarchy * hierarchy_loss
        )
        return out
