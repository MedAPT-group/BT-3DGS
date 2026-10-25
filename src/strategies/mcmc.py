import math
from typing import Any, Dict

import torch
from torch import Tensor

from .base import Strategy
from src.utils.ops import inject_noise_to_position, relocate, sample_add


class MCMCStrategy(Strategy):
    def __init__(self, cfg: Any):
        self.cap_max = 1_000_000
        self.noise_lr = 5e5
        self.refine_start_iter = 500
        self.refine_stop_iter = 25_000
        self.refine_every = 100
        self.min_opacity = 0.005

    def initialize_state(self) -> Dict[str, Any]:
        n_max = 51
        binoms = torch.zeros((n_max, n_max))
        for n in range(n_max):
            for k in range(n + 1):
                binoms[n, k] = math.comb(n, k)
        return {"binoms": binoms}

    def check_sanity(
        self,
        model: torch.nn.Module,
        optimizers: Dict[str, torch.optim.Optimizer],
    ):
        super().check_sanity(model, optimizers)
        params = model.param_dict()
        for key in ["means", "scales", "quats", "opacities"]:
            assert key in params, f"{key} is required in params but missing."

    def step_post_backward(
        self,
        model: torch.nn.Module,
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
        packed: bool = False,
    ):
        lr = next(iter(optimizers.values())).param_groups[0]["lr"]
        state["binoms"] = state["binoms"].to(model.means.device)
        binoms = state["binoms"]

        if (
            step < self.refine_stop_iter
            and step > self.refine_start_iter
            and step % self.refine_every == 0
        ):
            n_relocated_gs = self._relocate_gs(model, optimizers, binoms)
            n_new_gs = self._add_new_gs(model, optimizers, binoms)
            torch.cuda.empty_cache()

        inject_noise_to_position(
            model=model, optimizers=optimizers, state={}, scaler=lr * self.noise_lr
        )

    @torch.no_grad()
    def _relocate_gs(
        self,
        model: torch.nn.Module,
        optimizers: Dict[str, torch.optim.Optimizer],
        binoms: Tensor,
    ) -> int:
        opacities = torch.sigmoid(model.opacities.flatten())
        dead_mask = opacities <= self.min_opacity
        n_gs = dead_mask.sum().item()
        if n_gs > 0:
            relocate(
                model=model,
                optimizers=optimizers,
                state={},
                mask=dead_mask,
                binoms=binoms,
                min_opacity=self.min_opacity,
            )
        return n_gs

    @torch.no_grad()
    def _add_new_gs(
        self,
        model: torch.nn.Module,
        optimizers: Dict[str, torch.optim.Optimizer],
        binoms: Tensor,
    ) -> int:
        current_n_points = model.num_gaussians
        n_target = min(self.cap_max, int(1.05 * current_n_points))
        n_gs = max(0, n_target - current_n_points)
        if n_gs > 0:
            sample_add(
                model=model,
                optimizers=optimizers,
                state={},
                n=n_gs,
                binoms=binoms,
                min_opacity=self.min_opacity,
            )
        return n_gs
