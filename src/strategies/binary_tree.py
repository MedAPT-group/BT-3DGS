import os
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from .base import Strategy
from src.utils.math import normalized_quat_to_rotmat
from src.models.gaussian_model import GaussianModel
from src.utils.ops import (
    _update_param_with_optimizer,
    reset_opa,
)


def _append_history(
    model: torch.nn.Module,
    indices: Tensor,
) -> Tensor:
    """Archive the selected active nodes and return their new history indices."""
    if indices.numel() == 0:
        return torch.empty(0, dtype=torch.int32, device=model.means.device)

    start = model.history_parent.shape[0]
    hist_indices = torch.arange(
        start,
        start + indices.numel(),
        device=model.means.device,
        dtype=torch.int32,
    )

    model.history_means = torch.cat(
        [model.history_means, model.means.index_select(0, indices).detach().clone()],
        dim=0,
    )
    model.history_scales = torch.cat(
        [model.history_scales, model.scales.index_select(0, indices).detach().clone()],
        dim=0,
    )
    model.history_quats = torch.cat(
        [model.history_quats, model.quats.index_select(0, indices).detach().clone()],
        dim=0,
    )
    model.history_opacities = torch.cat(
        [model.history_opacities, model.opacities.index_select(0, indices).detach().clone()],
        dim=0,
    )
    model.history_parent = torch.cat(
        [model.history_parent, model.parent.index_select(0, indices).detach().clone()],
        dim=0,
    )
    model.history_level = torch.cat(
        [model.history_level, model.level.index_select(0, indices).detach().clone()],
        dim=0,
    )
    return hist_indices


def duplicate(
    model: torch.nn.Module,
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, Any],
    mask: Tensor,
    trainable_param_keys: List[str],
):
    device = mask.device
    sel = torch.where(mask)[0]

    def param_fn(name: str, p: Tensor) -> Tensor:
        return torch.nn.Parameter(torch.cat([p, p[sel]]), requires_grad=p.requires_grad)

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        return torch.cat([v, torch.zeros((len(sel), *v.shape[1:]), device=device)])

    history_indices = _append_history(model, sel)

    updated_parent = model.parent.detach().clone()
    updated_parent[sel] = history_indices
    model.parent = torch.cat([updated_parent, history_indices], dim=0)

    updated_level = model.level.detach().clone()
    updated_level[sel] = updated_level[sel] + 1
    model.level = torch.cat([updated_level, updated_level.index_select(0, sel)], dim=0)

    _update_param_with_optimizer(param_fn, optimizer_fn, model, optimizers, trainable_param_keys)
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = torch.cat((v, v[sel]))


def split(
    model: torch.nn.Module,
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, Any],
    mask: Tensor,
    trainable_param_keys: List[str],
    revised_opacity: bool = False,
):
    device = mask.device
    sel = torch.where(mask)[0]
    rest = torch.where(~mask)[0]

    scales = model.activated_scales[sel]  # [N, 3]
    quats = F.normalize(model.quats[sel], dim=-1)
    rotmats = normalized_quat_to_rotmat(quats)  # [N, 3, 3]

    longest_axis = scales.argmax(dim=-1)  # [N]
    axis_mask = torch.zeros_like(scales)
    axis_mask.scatter_(1, longest_axis.unsqueeze(1), 1.0)
    offset_local = scales * axis_mask  # [N, 3]
    offset_world = torch.einsum("nij,nj->ni", rotmats, offset_local)  # [N, 3]
    samples = torch.stack([offset_world, -offset_world], dim=0)  # [2, N, 3]

    history_indices = _append_history(model, sel)

    split_level = model.level.index_select(0, sel) + 1
    model.parent = model.parent.index_select(0, rest)
    model.parent = torch.cat([model.parent, history_indices.repeat(2)], dim=0)
    model.level = model.level.index_select(0, rest)
    model.level = torch.cat([model.level, split_level.repeat(2)], dim=0)

    def param_fn(name: str, p: Tensor) -> Tensor:
        repeats = [2] + [1] * (p.dim() - 1)
        if name == "means":
            p_split = (p[sel] + samples).reshape(-1, 3)
        elif name == "scales":
            target_scales = scales.clone()
            longest_scale = scales.gather(1, longest_axis.unsqueeze(1))
            target_scales.scatter_(1, longest_axis.unsqueeze(1), (longest_scale / 1.6).clamp_min(1e-12))
            p_split = model.inverse_activate_scale(target_scales).repeat(2, 1)
        elif name == "opacities" and revised_opacity:
            base = GaussianModel.activate_opacity(p[sel]).clamp(max=0.9999)
            new_opacities = 1.0 - torch.sqrt(1.0 - base)
            p_split = torch.log(torch.expm1(new_opacities)).repeat(repeats)
        else:
            p_split = p[sel].repeat(repeats)
        p_new = torch.cat([p[rest], p_split])
        return torch.nn.Parameter(p_new, requires_grad=p.requires_grad)

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        return torch.cat([v[rest], torch.zeros((2 * len(sel), *v.shape[1:]), device=device)])

    _update_param_with_optimizer(param_fn, optimizer_fn, model, optimizers, trainable_param_keys)
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            repeats = [2] + [1] * (v.dim() - 1)
            state[k] = torch.cat((v[rest], v[sel].repeat(repeats)))


def remove(
    model: torch.nn.Module,
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, Tensor],
    mask: Tensor,
    trainable_param_keys: List[str],
):
    keep = torch.where(~mask)[0]
    model.parent = model.parent.index_select(0, keep)
    model.level = model.level.index_select(0, keep)

    def param_fn(name: str, p: Tensor) -> Tensor:
        return torch.nn.Parameter(p[keep], requires_grad=p.requires_grad)

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        return v[keep]

    _update_param_with_optimizer(param_fn, optimizer_fn, model, optimizers, trainable_param_keys)
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v[keep]


class BinaryTreeStrategy(Strategy):

    def __init__(self, cfg: Any):
        self.prune_opa = cfg.strategy_prune_opa
        self.grow_grad2d = cfg.strategy_grow_grad2d
        self.grow_scale3d = cfg.strategy_grow_scale3d
        self.grow_scale2d = cfg.strategy_grow_scale2d
        self.prune_scale3d = cfg.strategy_prune_scale3d
        self.prune_scale2d = cfg.strategy_prune_scale2d
        self.refine_scale2d_stop_iter = cfg.strategy_refine_scale2d_stop_iter
        self.refine_start_iter = cfg.strategy_refine_start_iter
        self.refine_stop_iter = cfg.strategy_refine_stop_iter
        self.reset_every = cfg.strategy_reset_every
        self.refine_every = cfg.strategy_refine_every
        self.pause_refine_after_reset = cfg.strategy_pause_refine_after_reset
        self.absgrad = cfg.strategy_absgrad
        self.revised_opacity = cfg.strategy_revised_opacity
        self.key_for_gradient = cfg.strategy_key_for_gradient
        self.min_remaining = cfg.strategy_min_remaining
        self.max_gaussians = cfg.strategy_max_gaussians
        self.prune_start_iter = cfg.strategy_prune_start_iter
        self.max_prune_ratio = cfg.strategy_max_prune_ratio
        self.lod_dir = ""
        self.lod_thresholds = tuple()
        self._saved_lod_thresholds = set()

    def configure_lod_checkpoints(self, lod_dir: str, lod_thresholds: Tuple[int, ...]) -> None:
        self.lod_dir = lod_dir
        self.lod_thresholds = tuple(int(threshold) for threshold in lod_thresholds)
        os.makedirs(self.lod_dir, exist_ok=True)

    def initialize_state(self, scene_scale: float = 1.0) -> Dict[str, Any]:
        state = {"grad2d": None, "count": None, "scene_scale": scene_scale}
        if self.refine_scale2d_stop_iter > 0:
            state["radii"] = None
        return state

    def check_sanity(
        self,
        model: torch.nn.Module,
        optimizers: Dict[str, torch.optim.Optimizer],
    ):
        super().check_sanity(model, optimizers)
        params = model.param_dict()
        for key in ["means", "scales", "quats", "opacities"]:
            assert key in params, f"{key} is required in params but missing."

    def step_pre_backward(
        self,
        model: torch.nn.Module,
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
    ):
        assert self.key_for_gradient in info, "means2d gradient is required but missing."
        info[self.key_for_gradient].retain_grad()

    def step_post_backward(
        self,
        model: torch.nn.Module,
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
        packed: bool = False,
    ):
        if step >= self.refine_stop_iter:
            return

        self._update_state(model, state, info, packed=packed)
        if (
            step > self.refine_start_iter
            and step % self.refine_every == 0
            and step % self.reset_every >= self.pause_refine_after_reset
        ):
            self._save_lod_checkpoint_if_needed(model, step)
            n_dupli, n_split = self._grow_gs(
                model, optimizers, state, step, model.trainable_param_keys
            )
            n_prune, _ = self._prune_gs(
                model, optimizers, state, step, model.trainable_param_keys
            )

            state["grad2d"].zero_()
            state["count"].zero_()
            if self.refine_scale2d_stop_iter > 0:
                state["radii"].zero_()
            torch.cuda.empty_cache()

        if step > 0 and step % self.reset_every == 0:
            reset_opa(
                model=model,
                optimizers=optimizers,
                state=state,
                value=self.prune_opa * 2.0,
            )

    def _save_lod_checkpoint_if_needed(self, model: torch.nn.Module, step: int) -> None:
        if not self.lod_thresholds:
            return

        num_leaf_nodes = int(model.num_gaussians)
        crossed_thresholds = [
            threshold
            for threshold in self.lod_thresholds
            if threshold not in self._saved_lod_thresholds
            and num_leaf_nodes >= threshold * 0.95
        ]
        if not crossed_thresholds:
            return

        for threshold in crossed_thresholds:
            ckpt_path = os.path.join(
                self.lod_dir,
                f"ckpt_lod_{threshold}_step_{step}.pt",
            )
            model.save_ckpt(
                ckpt_path,
                step,
                extra={
                    "lod_threshold": threshold,
                    "num_leaf_nodes": num_leaf_nodes,
                },
            )
            self._saved_lod_thresholds.add(threshold)
            print(
                f"Saved LOD checkpoint at step {step}: "
                f"threshold={threshold}, leaf_nodes={num_leaf_nodes}, path={ckpt_path}"
            )

    def _update_state(
        self,
        model: torch.nn.Module,
        state: Dict[str, Any],
        info: Dict[str, Any],
        packed: bool = False,
    ):
        for key in ["width", "height", "n_cameras", "radii", "gaussian_ids", self.key_for_gradient]:
            assert key in info, f"{key} is required but missing."

        grads = (
            info[self.key_for_gradient].absgrad.clone()
            if self.absgrad
            else info[self.key_for_gradient].grad.clone()
        )
        grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
        grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]

        n_gaussian = model.num_gaussians
        if state["grad2d"] is None:
            state["grad2d"] = torch.zeros(n_gaussian, device=grads.device)
        if state["count"] is None:
            state["count"] = torch.zeros(n_gaussian, device=grads.device)
        if self.refine_scale2d_stop_iter > 0 and state["radii"] is None:
            state["radii"] = torch.zeros(n_gaussian, device=grads.device)

        if packed:
            gs_ids = info["gaussian_ids"]
            radii = info["radii"]
        else:
            sel = info["radii"] > 0.0
            gs_ids = torch.where(sel)[1]
            grads = grads[sel]
            radii = info["radii"][sel]

        state["grad2d"].index_add_(0, gs_ids, grads.norm(dim=-1))
        state["count"].index_add_(0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32))
        if self.refine_scale2d_stop_iter > 0:
            state["radii"][gs_ids] = torch.maximum(
                state["radii"][gs_ids],
                radii / float(max(info["width"], info["height"])),
            )

    @torch.no_grad()
    def _grow_gs(
        self,
        model: torch.nn.Module,
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        trainable_param_keys: List[str],
    ) -> Tuple[int, int]:
        current_n_gaussians = model.num_gaussians
        if self.max_gaussians > 0 and current_n_gaussians >= self.max_gaussians:
            return 0, 0

        grads = state["grad2d"] / state["count"].clamp_min(1)
        device = grads.device
        is_grad_high = grads > self.grow_grad2d
        is_small = (
            model.activated_scales.max(dim=-1).values
            <= self.grow_scale3d * state["scene_scale"]
        )
        is_dupli = is_grad_high & is_small
        n_dupli = is_dupli.sum().item()
        is_split = is_grad_high & (~is_small)
        if step < self.refine_scale2d_stop_iter:
            is_split |= state["radii"] > self.grow_scale2d
        n_split = is_split.sum().item()

        if self.max_gaussians > 0:
            available_slots = self.max_gaussians - current_n_gaussians
            if available_slots <= 0:
                return 0, 0
            if n_dupli > available_slots:
                dupli_grads = grads[is_dupli]
                topk_indices = torch.topk(dupli_grads, k=available_slots, largest=True).indices
                dupli_indices = torch.where(is_dupli)[0]
                new_dupli_mask = torch.zeros_like(is_dupli)
                new_dupli_mask[dupli_indices[topk_indices]] = True
                is_dupli = new_dupli_mask
                n_dupli = available_slots

        if n_dupli > 0:
            duplicate(model, optimizers, state, is_dupli, trainable_param_keys)
            if self.max_gaussians > 0:
                current_n_gaussians += n_dupli
                available_slots = self.max_gaussians - current_n_gaussians

        is_split = torch.cat([is_split, torch.zeros(n_dupli, dtype=torch.bool, device=device)])
        if self.max_gaussians > 0 and n_split > available_slots:
            if available_slots > 0:
                split_indices = torch.where(is_split)[0]
                split_grads = grads[split_indices[: model.num_gaussians]]
                topk_indices = torch.topk(split_grads, k=available_slots, largest=True).indices
                new_split_mask = torch.zeros_like(is_split)
                new_split_mask[split_indices[topk_indices]] = True
                is_split = new_split_mask
                n_split = available_slots
            else:
                n_split = 0
                is_split = torch.zeros_like(is_split)

        if n_split > 0:
            split(
                model=model,
                optimizers=optimizers,
                state=state,
                mask=is_split,
                trainable_param_keys=trainable_param_keys,
                revised_opacity=self.revised_opacity,
            )
        return n_dupli, n_split

    @torch.no_grad()
    def _prune_gs(
        self,
        model: torch.nn.Module,
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        trainable_param_keys: List[str],
    ) -> Tuple[int, Dict[str, int]]:
        prune_stats = {"low_opacity": 0, "too_big_3d": 0, "too_big_2d": 0, "limited_by_ratio": 0}
        if step < self.prune_start_iter:
            return 0, prune_stats
        if model.opacities.numel() <= self.min_remaining:
            return 0, prune_stats

        is_prune = model.activate_opacity(model.opacities.flatten()) < self.prune_opa
        prune_stats["low_opacity"] = is_prune.sum().item()
        if step > self.reset_every:
            is_too_big_3d = (
                model.activated_scales.max(dim=-1).values
                > self.prune_scale3d * state["scene_scale"]
            )
            prune_stats["too_big_3d"] = is_too_big_3d.sum().item()
            is_prune = is_prune | is_too_big_3d
            if step < self.refine_scale2d_stop_iter:
                is_too_big_2d = state["radii"] > self.prune_scale2d
                prune_stats["too_big_2d"] = is_too_big_2d.sum().item()
                is_prune = is_prune | is_too_big_2d

        n_total = is_prune.numel()
        n_prune_candidate = is_prune.sum().item()
        if n_total - n_prune_candidate < self.min_remaining and n_total > self.min_remaining:
            opacities_sig = model.activate_opacity(model.opacities).flatten()
            topk = torch.topk(opacities_sig, k=self.min_remaining, largest=True).indices
            is_prune[topk] = False
            n_prune_candidate = is_prune.sum().item()

        if n_prune_candidate > 0:
            max_remove = max(int(n_total * self.max_prune_ratio), 1)
            max_remove = min(max_remove, n_total - self.min_remaining)
            if n_prune_candidate > max_remove and max_remove > 0:
                candidate_idx = torch.where(is_prune)[0]
                cand_scores = model.activate_opacity(model.opacities).flatten()[candidate_idx]
                sorted_idx = torch.argsort(cand_scores, descending=False)
                new_mask = torch.zeros_like(is_prune)
                new_mask[candidate_idx[sorted_idx[:max_remove]]] = True
                is_prune = new_mask

        n_prune = is_prune.sum().item()
        if n_prune > 0:
            remove(model, optimizers, state, is_prune, trainable_param_keys)
        return n_prune, prune_stats
