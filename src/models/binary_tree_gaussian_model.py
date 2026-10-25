import os
from typing import Any, Dict, Optional

import numpy as np
import torch

from .gaussian_model import GaussianModel


class BinaryTreeGaussianModel(GaussianModel):
    """Gaussian model that maintains a binary-tree history of split operations.

    Extends :class:`GaussianModel` with per-point parent tracking and
    historical snapshots of all Gaussian parameters, enabling tree-structured
    densification strategies.

    Extra attributes (initialised by ``load_from_init_npy``):

    * ``parent``              – (N,) int32, parent index for each active point (``-1`` = root).
    * ``level``               – (N,) int32, depth for each active point.
    * ``history_means``       – archived means from previous generations.
    * ``history_scales``      – archived scales.
    * ``history_quats``       – archived quaternions.
    * ``history_opacities``   – archived opacities.
    * ``history_parent``      – archived parent indices.
    * ``history_level``       – archived depth values.
    """

    # ------------------------------------------------------------------
    # Initialization (override)
    # ------------------------------------------------------------------

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Register buffers for binary-tree history.
        self.register_buffer("parent", torch.empty((0,), dtype=torch.int32))
        self.register_buffer("level", torch.empty((0,), dtype=torch.int32))
        self.register_buffer("history_means", torch.empty((0, 3)))
        self.register_buffer("history_scales", torch.empty((0, 3)))
        self.register_buffer("history_quats", torch.empty((0, 4)))
        self.register_buffer("history_opacities", torch.empty((0,)))
        self.register_buffer("history_parent", torch.empty((0,), dtype=torch.int32))
        self.register_buffer("history_level", torch.empty((0,), dtype=torch.int32))

    @staticmethod
    def build_level_from_parent(parent: torch.Tensor) -> torch.Tensor:
        """Reconstruct per-node depth from parent indices."""
        level = torch.zeros_like(parent)
        for idx in range(int(parent.numel())):
            parent_idx = int(parent[idx].item())
            if parent_idx < 0:
                continue
            level[idx] = level[parent_idx] + 1
        return level

    @torch.no_grad()
    def load_from_init_npy(
        self,
        npy_path: str,
        init_scale: float = 1.0,
        init_opa: Optional[float] = None,
        global_scale: float = 1.0,
        init_opa_scale: float = 1.0,
        init_scale_multiplier: float = 1.0,
    ):
        """Initialise Gaussians from an ``.npy`` file with uniform initial scale.

        Unlike the base class, this method does **not** compute nearest-neighbour
        distances; every point receives the same scale ``init_scale * init_scale_multiplier``.
        It also initialises the binary-tree history buffers.

        Returns:
            ``self`` for chaining.
        """
        assert os.path.exists(npy_path), f"Init file not found: {npy_path}"
        data = np.load(npy_path)
        assert data.shape[1] >= 4, "Init npy requires at least 4 columns (x,y,z,opacity)."

        xyz = torch.from_numpy(data[:, :3]).float() * global_scale
        opa_vals = torch.from_numpy(data[:, 3]).float()
        opa_vals = (opa_vals * init_opa_scale).clamp(min=1e-6, max=0.99)

        n_points = xyz.shape[0]
        if n_points == 0:
            raise ValueError("No points provided for initialization")
        xyz_dev = xyz.to(self.device).contiguous()

        per_point_scale = torch.ones(n_points, device=self.device) * float(init_scale) * float(init_scale_multiplier)
        raw_scales = self.inverse_activate_scale(
            per_point_scale[:, None].repeat(1, 3).to(self.device)
        )

        quats = torch.zeros((n_points, 4), device=self.device)
        quats[:, 0] = 1.0
        raw_opacities = self.inverse_activate_opacity(opa_vals.to(self.device))

        self.means = torch.nn.Parameter(xyz_dev)
        self.scales = torch.nn.Parameter(raw_scales)
        self.quats = torch.nn.Parameter(quats)
        self.opacities = torch.nn.Parameter(raw_opacities.to(self.device))

        self.trainable_param_keys = ["means", "scales", "quats", "opacities"]
        self.parent = torch.full((n_points,), -1, device=self.device, dtype=torch.int32)
        self.level = torch.zeros(n_points, device=self.device, dtype=torch.int32)
        self.history_means = torch.empty((0, 3), device=self.device)
        self.history_scales = torch.empty((0, 3), device=self.device)
        self.history_quats = torch.empty((0, 4), device=self.device)
        self.history_opacities = torch.empty((0,), device=self.device)
        self.history_parent = torch.empty((0,), device=self.device, dtype=torch.int32)
        self.history_level = torch.empty((0,), device=self.device, dtype=torch.int32)
        return self

    @torch.no_grad()
    def initialize_centered_sphere(
        self,
        center: torch.Tensor,
        init_scale: float,
        init_opa: float = 0.1,
        init_opa_scale: float = 1.0,
    ):
        """Initialise a single root Gaussian centred in the scene.

        The Gaussian is placed at the provided centre with a large isotropic
        scale so the tree can begin splitting from one coarse scene-covering
        node without an initialization point-cloud file.
        """
        center = torch.as_tensor(center, dtype=torch.float32, device=self.device)
        if center.numel() != 3:
            raise ValueError(f"center must have 3 elements, got shape {tuple(center.shape)}")
        center = center.reshape(1, 3).contiguous()

        per_point_scale = torch.full((1,), float(init_scale), device=self.device)
        raw_scales = self.inverse_activate_scale(
            per_point_scale[:, None].repeat(1, 3)
        )

        quats = torch.zeros((1, 4), device=self.device)
        quats[:, 0] = 1.0
        opacity = torch.tensor(
            [float(init_opa) * float(init_opa_scale)],
            dtype=torch.float32,
            device=self.device,
        ).clamp(min=1e-6, max=0.99)
        raw_opacities = self.inverse_activate_opacity(opacity)

        self.means = torch.nn.Parameter(center)
        self.scales = torch.nn.Parameter(raw_scales)
        self.quats = torch.nn.Parameter(quats)
        self.opacities = torch.nn.Parameter(raw_opacities)

        self.trainable_param_keys = ["means", "scales", "quats", "opacities"]
        self.parent = torch.full((1,), -1, device=self.device, dtype=torch.int32)
        self.level = torch.zeros(1, device=self.device, dtype=torch.int32)
        self.history_means = torch.empty((0, 3), device=self.device)
        self.history_scales = torch.empty((0, 3), device=self.device)
        self.history_quats = torch.empty((0, 4), device=self.device)
        self.history_opacities = torch.empty((0,), device=self.device)
        self.history_parent = torch.empty((0,), device=self.device, dtype=torch.int32)
        self.history_level = torch.empty((0,), device=self.device, dtype=torch.int32)
        return self

    # ------------------------------------------------------------------
    # Parameter access (override)
    # ------------------------------------------------------------------

    def param_dict(self) -> Dict[str, torch.nn.Parameter]:
        """Return all parameters including history tensors and parent indices."""
        params = super().param_dict()
        params.update(
            {
                "history_means": self.history_means,
                "history_scales": self.history_scales,
                "history_quats": self.history_quats,
                "history_opacities": self.history_opacities,
                "history_parent": self.history_parent,
                "history_level": self.history_level,
                "parent": self.parent,
                "level": self.level,
            }
        )
        return params

    # ------------------------------------------------------------------
    # Training utilities (override)
    # ------------------------------------------------------------------

    def build_optimizers(self, lrs: Dict[str, float]) -> Dict[str, torch.optim.Optimizer]:
        """Build Adam optimisers for trainable parameters only.

        History tensors and parent indices are excluded from optimisation.
        """
        optimizers = {}
        params = self.param_dict()
        for name in self.trainable_param_keys:
            lr = lrs.get(name, 1e-3)
            optimizers[name] = torch.optim.Adam(
                [{"params": params[name], "name": name}], lr=lr, eps=1e-15
            )
        return optimizers

    # ------------------------------------------------------------------
    # Serialization (override)
    # ------------------------------------------------------------------

    def save_ckpt(self, path: str, step: int, extra: Optional[Dict[str, Any]] = None):
        """Save checkpoint including binary-tree history under the ``"history"`` key."""
        history_params = {
            "parent": self.parent,
            "level": self.level,
            "history_means": self.history_means,
            "history_scales": self.history_scales,
            "history_quats": self.history_quats,
            "history_opacities": self.history_opacities,
            "history_parent": self.history_parent,
            "history_level": self.history_level,
        }
        payload = {"step": step, "model": self.state_dict(), "history": history_params}
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    def load_ckpt(self, path: str, strict: bool = True) -> int:
        """Restore model and binary-tree history from a checkpoint file.

        Returns:
            The training step stored in the checkpoint (``-1`` if absent).
        """
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:
            ckpt = torch.load(path, map_location=self.device)

        state = ckpt["model"]
        self.means = torch.nn.Parameter(state["means"].to(self.device))
        self.scales = torch.nn.Parameter(state["scales"].to(self.device))
        self.quats = torch.nn.Parameter(state["quats"].to(self.device))
        self.opacities = torch.nn.Parameter(state["opacities"].to(self.device))
        self.trainable_param_keys = ["means", "scales", "quats", "opacities"]

        h = ckpt["history"]
        self.parent = h["parent"].to(self.device, dtype=torch.int32)
        self.level = h["level"].to(self.device, dtype=torch.int32)
        self.history_means = h["history_means"].to(self.device)
        self.history_scales = h["history_scales"].to(self.device)
        self.history_quats = h["history_quats"].to(self.device)
        self.history_opacities = h["history_opacities"].to(self.device)
        self.history_parent = h["history_parent"].to(self.device, dtype=torch.int32)
        self.history_level = h["history_level"].to(self.device, dtype=torch.int32)

        return int(ckpt.get("step", -1))
