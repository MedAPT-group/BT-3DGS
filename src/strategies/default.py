from typing import Any, Dict, Tuple

import torch

from .base import Strategy
from src.models.gaussian_model import GaussianModel
from src.utils.ops import duplicate, remove, reset_opa, split


class DefaultStrategy(Strategy):

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
            n_dupli, n_split = self._grow_gs(model, optimizers, state, step)

            n_prune, prune_stats = self._prune_gs(model, optimizers, state, step)

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

    def _update_state(
        self,
        model: torch.nn.Module,
        state: Dict[str, Any],
        info: Dict[str, Any],
        packed: bool = False,
    ):
        for key in [
            "width",
            "height",
            "n_cameras",
            "radii",
            "gaussian_ids",
            self.key_for_gradient,
        ]:
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
        if self.refine_scale2d_stop_iter > 0 and state.get("radii") is None:
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

        is_large = ~is_small
        is_split = is_grad_high & is_large
        if step < self.refine_scale2d_stop_iter:
            is_split |= state["radii"] > self.grow_scale2d
        n_split = is_split.sum().item()

        if self.max_gaussians > 0:
            available_slots = self.max_gaussians - current_n_gaussians
            if available_slots <= 0:
                return 0, 0
            if n_dupli > available_slots:
                if available_slots > 0:
                    dupli_grads = grads[is_dupli]
                    topk_indices = torch.topk(dupli_grads, k=available_slots, largest=True).indices
                    dupli_indices = torch.where(is_dupli)[0]
                    new_dupli_mask = torch.zeros_like(is_dupli)
                    new_dupli_mask[dupli_indices[topk_indices]] = True
                    is_dupli = new_dupli_mask
                    n_dupli = available_slots
                else:
                    n_dupli = 0
                    is_dupli = torch.zeros_like(is_dupli)

        if n_dupli > 0:
            duplicate(model=model, optimizers=optimizers, state=state, mask=is_dupli)
            if self.max_gaussians > 0:
                current_n_gaussians += n_dupli
                available_slots = self.max_gaussians - current_n_gaussians

        is_split = torch.cat(
            [is_split, torch.zeros(n_dupli, dtype=torch.bool, device=device)]
        )

        if self.max_gaussians > 0 and n_split > 0:
            max_splits = available_slots
            if n_split > max_splits:
                if max_splits > 0:
                    split_indices = torch.where(is_split)[0]
                    split_grads = grads[split_indices[: model.num_gaussians]]
                    topk_indices = torch.topk(split_grads, k=max_splits, largest=True).indices
                    new_split_mask = torch.zeros_like(is_split)
                    new_split_mask[split_indices[topk_indices]] = True
                    is_split = new_split_mask
                    n_split = max_splits
                else:
                    n_split = 0
                    is_split = torch.zeros_like(is_split)

        if n_split > 0:
            split(
                model=model,
                optimizers=optimizers,
                state=state,
                mask=is_split,
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
    ) -> Tuple[int, Dict[str, int]]:
        prune_stats = {
            "low_opacity": 0,
            "too_big_3d": 0,
            "too_big_2d": 0,
            "limited_by_ratio": 0,
        }
        if step < self.prune_start_iter:
            return 0, prune_stats

        current_total = model.opacities.numel()
        if current_total <= self.min_remaining:
            return 0, prune_stats

        is_prune_low_opa = GaussianModel.activate_opacity(model.opacities.flatten()) < self.prune_opa
        prune_stats["low_opacity"] = is_prune_low_opa.sum().item()
        is_prune = is_prune_low_opa

        if step > self.reset_every:
            is_too_big_3d = (
                model.activated_scales.max(dim=-1).values
                > self.prune_scale3d * state["scene_scale"]
            )
            prune_stats["too_big_3d"] = is_too_big_3d.sum().item()
            is_too_big = is_too_big_3d
            if step < self.refine_scale2d_stop_iter:
                is_too_big_2d = state["radii"] > self.prune_scale2d
                prune_stats["too_big_2d"] = is_too_big_2d.sum().item()
                is_too_big = is_too_big | is_too_big_2d
            is_prune = is_prune | is_too_big

        n_total = is_prune.numel()
        n_prune_candidate = is_prune.sum().item()
        if n_total - n_prune_candidate < self.min_remaining and n_total > self.min_remaining:
            opacities_sig = GaussianModel.activate_opacity(model.opacities).flatten()
            topk = torch.topk(opacities_sig, k=self.min_remaining, largest=True).indices
            is_prune[topk] = False
            n_prune_candidate = is_prune.sum().item()
        elif n_prune_candidate == n_total:
            opacities_sig = GaussianModel.activate_opacity(model.opacities).flatten()
            is_prune[torch.argmax(opacities_sig)] = False
            n_prune_candidate = is_prune.sum().item()

        if n_prune_candidate > 0:
            max_remove = max(int(n_total * self.max_prune_ratio), 1)
            max_remove = min(max_remove, n_total - self.min_remaining)
            max_remove = max(max_remove, 0)
            if n_prune_candidate > max_remove and max_remove > 0:
                prune_stats["limited_by_ratio"] = n_prune_candidate - max_remove
                opacities_sig = GaussianModel.activate_opacity(model.opacities).flatten()
                candidate_idx = torch.where(is_prune)[0]
                cand_scores = opacities_sig[candidate_idx]
                sorted_idx = torch.argsort(cand_scores, descending=False)
                new_mask = torch.zeros_like(is_prune)
                prune_indices = candidate_idx[sorted_idx[:max_remove]]
                new_mask[prune_indices] = True
                is_prune = new_mask

        n_prune = is_prune.sum().item()
        if n_prune > 0:
            remove(model=model, optimizers=optimizers, state=state, mask=is_prune)
        return n_prune, prune_stats
