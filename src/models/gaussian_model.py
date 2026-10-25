import os
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from r2splat.ops import (
    ExplicitGaussianState,
    InferenceGaussianState,
)
from r2splat.ops.direct import (
    inference_rasterize,
    inference_voxelize,
    prepare_inference_gaussians as _prepare_inference_gaussians,
)
from r2splat.ops.explicit import (
    explicit_backward,
    explicit_rasterize,
    explicit_voxelize,
    prepare_explicit_gaussians as _prepare_explicit_gaussians,
)
from src.utils.math import invert_rigid_transform

try:
    from simple_knn._C import distCUDA2  # type: ignore
except Exception:
    distCUDA2 = None


class GaussianModel(torch.nn.Module):
    """Gaussian splat model for CT reconstruction.

    Each Gaussian is parameterised by four learnable tensors stored in raw
    (pre-activation) space:

    * ``means``      – (N, 3) centre positions
    * ``scales``     – (N, 3) raw scales  (activated via exp or bounded sigmoid)
    * ``quats``      – (N, 4) unit quaternion rotations
    * ``opacities``  – (N,)   raw opacities (activated via softplus)

    Scale bounds can optionally clamp the activated scales into
    ``[min_scale, max_scale]`` using a sigmoid parameterisation.

    Args:
        device: Torch device string.
        min_scale: Lower bound for activated scales (``None`` = unbounded exp).
        max_scale: Upper bound for activated scales (must pair with *min_scale*).
        scale_eps: Margin inside the sigmoid range to avoid saturation.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        device: str = "cpu",
        min_scale: Optional[Union[float, Sequence[float], torch.Tensor]] = None,
        max_scale: Optional[Union[float, Sequence[float], torch.Tensor]] = None,
        scale_eps: float = 1e-4,
        eps2d: float = 0.01,
        eps3d: float = 1e-6,
        img_tile_size: int = 16,
        vol_tile_size: int = 8,
        img_radius_clip: float = 3.0,
        vol_radius_clip: float = 3.0,
        backward_mode: str = "primitive",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.eps2d = float(eps2d)
        self.eps3d = float(eps3d)
        self.img_tile_size = int(img_tile_size)
        self.vol_tile_size = int(vol_tile_size)
        self.img_radius_clip = float(img_radius_clip)
        self.vol_radius_clip = float(vol_radius_clip)
        self.backward_mode = str(backward_mode).lower()
        if self.backward_mode not in {"pixel", "primitive"}:
            raise ValueError("backward_mode must be 'pixel' or 'primitive'.")
        self._gaussian_ids_cache: Optional[torch.Tensor] = None
        self._configure_scale_bounds(min_scale, max_scale, scale_eps)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_gaussians(self) -> int:
        """Number of Gaussian primitives currently held by the model."""
        return 0 if not hasattr(self, "means") else self.means.shape[0]

    @property
    def activated_scales(self) -> torch.Tensor:
        """Scales after activation (exp or bounded sigmoid)."""
        return self.activate_scale(self.scales)

    @property
    def activated_opacities(self) -> torch.Tensor:
        """Opacities after activation (softplus)."""
        return self.activate_opacity(self.opacities)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

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
        """Initialise Gaussians from an ``.npy`` file (columns: x, y, z, opacity, ...).

        Per-point initial scale is derived from nearest-neighbour distances
        (via ``distCUDA2`` when available, otherwise chunked ``cdist``).

        Args:
            npy_path: Path to the ``.npy`` point cloud.
            init_scale: Base multiplier applied to NN distances for initial scale.
            init_opa: *Unused* – kept for API compatibility.
            global_scale: Uniform spatial scaling applied to xyz coordinates.
            init_opa_scale: Multiplier applied to raw opacity values before clamping.
            init_scale_multiplier: Additional multiplier on top of *init_scale*.

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
        assert n_points > 0, "No points provided for initialization"
        xyz_dev = xyz.to(self.device).contiguous()

        nn_dist = None
        if distCUDA2 is not None and xyz_dev.is_cuda:
            try:
                nn_sq = distCUDA2(xyz_dev)
                nn_dist = torch.sqrt(torch.clamp_min(nn_sq, 1e-6))
            except Exception:
                nn_dist = None

        if nn_dist is None:
            bsz = 10000
            chunks = []
            for start in range(0, n_points, bsz):
                end = min(start + bsz, n_points)
                chunk = xyz_dev[start:end]
                min_dists = []
                for t_start in range(0, n_points, bsz):
                    t_end = min(t_start + bsz, n_points)
                    targets = xyz_dev[t_start:t_end]
                    d = torch.cdist(chunk, targets)
                    chunk_idx = torch.arange(start, end, device=xyz_dev.device).unsqueeze(1)
                    target_idx = torch.arange(t_start, t_end, device=xyz_dev.device).unsqueeze(0)
                    d = d.masked_fill(chunk_idx == target_idx, float("inf"))
                    min_dists.append(d.min(dim=1).values)
                min_dists = torch.stack(min_dists, dim=1).min(dim=1).values
                min_dists[~torch.isfinite(min_dists)] = 1e-2
                chunks.append(min_dists)
            nn_dist = torch.cat(chunks, dim=0)

        nn_dist = nn_dist.clamp_min(2e-4)
        per_point_scale = nn_dist * float(init_scale) * float(init_scale_multiplier)
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
        return self

    def initialize_from_data_dir(
        self,
        data_dir: str,
        global_scale: float = 1.0,
        init_scale: float = 1.0,
        init_opa: float = 0.1,
        init_opa_scale: float = 1.0,
        init_scale_multiplier: float = 1.0,
        bbox: Optional[torch.Tensor] = None,
    ):
        """Load ``init_<scene>.npy`` from *data_dir* and optionally crop to *bbox*.

        Args:
            data_dir: Directory containing the scene data and init file.
            global_scale: Uniform spatial scaling applied to xyz.
            init_scale: Base scale multiplier for Gaussian radii.
            init_opa: Default opacity value (forwarded to ``load_from_init_npy``).
            init_opa_scale: Opacity scaling factor.
            init_scale_multiplier: Additional scale multiplier.
            bbox: ``(2, 3)`` tensor ``[min_corner, max_corner]`` – points outside
                are removed after initialisation.

        Returns:
            ``self`` for chaining.
        """
        scene = os.path.basename(os.path.normpath(data_dir))
        init_path = os.path.join(data_dir, f"init_{scene}.npy")
        if not os.path.exists(init_path):
            raise FileNotFoundError(f"Initialization file required but missing: {init_path}")
        self.load_from_init_npy(
            init_path,
            init_scale=init_scale,
            init_opa=init_opa,
            global_scale=global_scale,
            init_opa_scale=init_opa_scale,
            init_scale_multiplier=init_scale_multiplier,
        )

        if bbox is not None:
            mins, maxs = bbox[0].to(self.means.device), bbox[1].to(self.means.device)
            mask = ((self.means >= mins) & (self.means <= maxs)).all(dim=-1)
            if mask.sum() < mask.numel():
                for name in ["means", "scales", "quats", "opacities"]:
                    param = getattr(self, name)
                    setattr(self, name, torch.nn.Parameter(param[mask].detach().clone()))
        return self

    # ------------------------------------------------------------------
    # Parameter access & activation functions
    # ------------------------------------------------------------------

    def param_dict(self) -> Dict[str, torch.nn.Parameter]:
        """Return a name → Parameter mapping for all Gaussian parameters."""
        return {
            "means": self.means,
            "scales": self.scales,
            "quats": self.quats,
            "opacities": self.opacities,
        }

    @staticmethod
    def activate_opacity(raw: torch.Tensor) -> torch.Tensor:
        """Map raw opacity values to positive opacities (softplus)."""
        return F.softplus(raw)

    @staticmethod
    def inverse_activate_opacity(opacity: torch.Tensor) -> torch.Tensor:
        """Map positive opacities back to raw (pre-softplus) space."""
        return GaussianModel._inverse_softplus(opacity)

    def activate_scale(self, raw_scales: torch.Tensor) -> torch.Tensor:
        """Map raw scales to positive scales (exp or bounded sigmoid)."""
        if not self._use_sigmoid_scale:
            return torch.exp(raw_scales)
        lower = self._scale_min.to(raw_scales.device)
        upper = self._scale_max.to(raw_scales.device)
        return lower + (upper - lower) * torch.sigmoid(raw_scales)

    def inverse_activate_scale(self, scales: torch.Tensor) -> torch.Tensor:
        """Map positive scales back to raw (pre-activation) space."""
        if not self._use_sigmoid_scale:
            return torch.log(scales.clamp_min(1e-12))
        lower = self._scale_min.to(scales.device)
        upper = self._scale_max.to(scales.device)
        clamped = scales.clamp(min=lower + self._scale_eps, max=upper - self._scale_eps)
        sigma = ((clamped - lower) / (upper - lower)).clamp(min=1e-6, max=1.0 - 1e-6)
        return torch.logit(sigma)

    # ------------------------------------------------------------------
    # Core computation (rendering & voxelization)
    # ------------------------------------------------------------------

    def prepare_gaussians(
        self, *, explicit: bool = False
    ) -> Union[InferenceGaussianState, ExplicitGaussianState]:
        """Prepare transformed parameters with the direct CUDA operator.

        ``explicit=True`` selects the training fast path: CUDA performs the
        parameter transform and its first-order VJP is invoked later through
        :meth:`backward_explicit`.  The default path returns a reusable,
        forward-only state for validation and standalone inference.
        """
        if not self._use_sigmoid_scale:
            raise RuntimeError(
                "Direct r2splat operators require bounded sigmoid scales."
            )
        if self._scale_min.numel() != 1 or self._scale_max.numel() != 1:
            raise RuntimeError(
                "Direct r2splat operators require scalar scale bounds."
            )
        scale_min = float(self._scale_min.item())
        scale_max = float(self._scale_max.item())

        if explicit:
            return _prepare_explicit_gaussians(
                self.quats,
                self.scales,
                self.opacities,
                scale_min=scale_min,
                scale_max=scale_max,
                backward_mode=self.backward_mode,
            )

        return _prepare_inference_gaussians(
            self.quats,
            self.scales,
            self.opacities,
            scale_min=scale_min,
            scale_max=scale_max,
        )

    def rasterize(
        self,
        camtoworlds: torch.Tensor,
        Ks: torch.Tensor,
        width: int,
        height: int,
        packed: bool = False,
        sparse_grad: bool = False,
        absgrad: bool = False,
        rasterize_mode: str = "classic",
        camera_model: str = "pinhole",
        viewmats: Optional[torch.Tensor] = None,
        gaussian_state: Optional[
            Union[InferenceGaussianState, ExplicitGaussianState]
        ] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Rasterise the Gaussian field into a 2-D image.

        Args:
            camtoworlds: ``(B, 4, 4)`` camera-to-world matrices.
            Ks: ``(B, 3, 3)`` intrinsic matrices.
            width: Output image width in pixels.
            height: Output image height in pixels.
            packed: Must be ``False``.
            sparse_grad: Must be ``False``.
            absgrad: Record absolute gradients for explicit training; ignored
                by forward-only inference.
            rasterize_mode: Must be ``"classic"``.
            camera_model: Must be ``"pinhole"``.
            viewmats: Optional precomputed world-to-camera matrices.
            gaussian_state: Optional tensors returned by ``prepare_gaussians``.
            **kwargs: No backend-specific options are accepted.

        Returns:
            ``(renders, info)`` – rendered images and auxiliary rasterisation info.
        """
        if packed:
            raise ValueError("r2splat does not support packed=True.")
        if sparse_grad:
            raise ValueError("r2splat does not support sparse_grad=True.")
        if rasterize_mode != "classic":
            raise ValueError("r2splat only supports rasterize_mode='classic'.")
        if camera_model != "pinhole":
            raise ValueError("r2splat only supports camera_model='pinhole'.")
        if kwargs:
            raise TypeError(f"Unsupported rasterization options: {sorted(kwargs)}")

        if viewmats is None:
            viewmats = invert_rigid_transform(camtoworlds)
        if viewmats.shape != camtoworlds.shape:
            raise ValueError(
                "viewmats must have the same shape as camtoworlds, got "
                f"{tuple(viewmats.shape)} and {tuple(camtoworlds.shape)}."
            )
        state = gaussian_state if gaussian_state is not None else self.prepare_gaussians()
        if isinstance(state, ExplicitGaussianState):
            result = explicit_rasterize(
                state,
                self.means,
                viewmats,
                Ks,
                img_w=width,
                img_h=height,
                tile_size=self.img_tile_size,
                eps2d=self.eps2d,
                radius_clip=self.img_radius_clip,
                absgrad=absgrad,
            )
        elif isinstance(state, InferenceGaussianState):
            result = inference_rasterize(
                state,
                self.means,
                viewmats,
                Ks,
                img_w=width,
                img_h=height,
                tile_size=self.img_tile_size,
                eps2d=self.eps2d,
                radius_clip=self.img_radius_clip,
            )
        else:
            raise TypeError(
                "gaussian_state must be an InferenceGaussianState or "
                "ExplicitGaussianState."
            )
        n_cameras = int(camtoworlds.shape[0])
        gaussian_ids = self._get_gaussian_ids(n_cameras)
        info = {
            "means2d": result.means2d,
            "radii": result.radii.amax(dim=-1),
            "gaussian_ids": gaussian_ids,
            "width": int(width),
            "height": int(height),
            "n_cameras": n_cameras,
        }
        renders = result.image.transpose(1, 2).unsqueeze(-1).contiguous()
        return renders, info

    def voxelize(
        self,
        volume_shape: Tuple[int, int, int],
        bbox: Optional[torch.Tensor],
        gaussian_state: Optional[
            Union[InferenceGaussianState, ExplicitGaussianState]
        ] = None,
        **kwargs,
    ) -> Optional[torch.Tensor]:
        """Voxelise the Gaussian field into a 3-D volume.

        Args:
            volume_shape: ``(width, depth, height)`` of the target voxel grid.
            bbox: ``(2, 3)`` bounding box ``[min_corner, max_corner]``.
                Returns ``None`` if not provided.
            gaussian_state: Optional tensors returned by ``prepare_gaussians``.
            **kwargs: No backend-specific options are accepted.

        Returns:
            The voxelised volume tensor, or ``None`` when *bbox* is missing.
        """
        if bbox is None:
            return None
        if kwargs:
            raise TypeError(f"Unsupported voxelization options: {sorted(kwargs)}")
        width, depth, height = volume_shape
        bbox_dev = bbox.to(self.device)
        bbox_min, bbox_max = bbox_dev[0], bbox_dev[1]
        ranges = bbox_max - bbox_min
        centers = (bbox_min + bbox_max) / 2.0
        state = gaussian_state if gaussian_state is not None else self.prepare_gaussians()
        if isinstance(state, ExplicitGaussianState):
            result = explicit_voxelize(
                state,
                self.means,
                ranges,
                centers,
                dim2=width,
                dim1=depth,
                dim0=height,
                tile_size=self.vol_tile_size,
                eps3d=self.eps3d,
                radius_clip=self.vol_radius_clip,
            )
        elif isinstance(state, InferenceGaussianState):
            result = inference_voxelize(
                state,
                self.means,
                ranges,
                centers,
                dim2=width,
                dim1=depth,
                dim0=height,
                tile_size=self.vol_tile_size,
                eps3d=self.eps3d,
                radius_clip=self.vol_radius_clip,
            )
        else:
            raise TypeError(
                "gaussian_state must be an InferenceGaussianState or "
                "ExplicitGaussianState."
            )
        return result.volume.unsqueeze(-1).contiguous()

    def backward_explicit(self, gaussian_state: ExplicitGaussianState) -> None:
        """Accumulate the explicit CUDA VJP into the four raw parameters."""
        if not isinstance(gaussian_state, ExplicitGaussianState):
            raise TypeError("backward_explicit requires an ExplicitGaussianState.")
        explicit_backward(
            gaussian_state,
            self.means,
            self.quats,
            self.scales,
            self.opacities,
        )

    def _get_gaussian_ids(self, n_cameras: int) -> torch.Tensor:
        """Return unpacked Gaussian ids without reallocating them every render."""
        cache = self._gaussian_ids_cache
        if (
            cache is None
            or cache.device != self.means.device
            or cache.numel() != self.num_gaussians
        ):
            cache = torch.arange(
                self.num_gaussians,
                device=self.means.device,
                dtype=torch.long,
            )
            self._gaussian_ids_cache = cache
        return cache.expand(n_cameras, -1)

    # ------------------------------------------------------------------
    # Training utilities
    # ------------------------------------------------------------------

    def build_optimizers(self, lrs: Dict[str, float]) -> Dict[str, torch.optim.Optimizer]:
        """Create one Adam optimiser per Gaussian parameter group.

        Args:
            lrs: Mapping from parameter name to learning rate.
                Missing names default to ``1e-3``.

        Returns:
            ``{name: optimizer}`` dict matching ``param_dict`` keys.
        """
        optimizers = {}
        for name, param in self.param_dict().items():
            lr = lrs.get(name, 1e-3)
            optimizers[name] = torch.optim.Adam(
                [{"params": param, "name": name}], lr=lr, eps=1e-15
            )
        return optimizers

    def clamp_opacity(self, max_opacity: Optional[float] = None):
        """Clamp activated opacities to ``[0, max_opacity]`` in-place."""
        if max_opacity is None:
            return
        with torch.no_grad():
            post = self.activated_opacities.clamp(min=0.0, max=max_opacity)
            self.opacities.copy_(self.inverse_activate_opacity(post))

    def scale_opacity(self, factor: float):
        """Multiply all activated opacities by *factor* in-place.

        Raises:
            ValueError: If *factor* is not positive.
        """
        if factor <= 0:
            raise ValueError("factor must be > 0")
        with torch.no_grad():
            post = self.activated_opacities * factor
            self.opacities.copy_(self.inverse_activate_opacity(post))

    # ------------------------------------------------------------------
    # Serialization (checkpoint, ply)
    # ------------------------------------------------------------------

    def save_ckpt(self, path: str, step: int, extra: Optional[Dict[str, Any]] = None):
        """Save model state dict and training step to a checkpoint file.

        Args:
            path: Destination file path.
            step: Current training step number.
            extra: Additional entries to include in the checkpoint dict.
        """
        payload = {"step": step, "model": self.state_dict()}
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    def load_ckpt(self, path: str, strict: bool = True) -> int:
        """Restore model from a checkpoint file.

        Args:
            path: Checkpoint file path.
            strict: Whether to enforce exact key matching in ``load_state_dict``.

        Returns:
            The training step stored in the checkpoint (``-1`` if absent).
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.load_state_dict(ckpt["model"], strict=strict)
        return int(ckpt.get("step", -1))

    def save_ply(self, path: str):
        """Serialise Gaussian parameters to a pickle-based ``.ply`` file.

        The file stores raw (pre-activation) scales and opacities together
        with scale-bound metadata so the model can be faithfully reloaded.
        """
        import pickle

        os.makedirs(os.path.dirname(path), exist_ok=True)
        xyz = self.means.detach().cpu().numpy()
        scale = self.scales.detach().cpu().numpy()
        rotation = self.quats.detach().cpu().numpy()
        density = self.opacities.detach().cpu().numpy()
        if density.ndim == 1:
            density = density[:, None]
        out = {
            "xyz": xyz,
            "density": density,
            "scale": scale,
            "rotation": rotation,
            "scale_bound": self._get_scale_bound(),
        }
        if self._use_sigmoid_scale:
            out["scale_eps"] = self._scale_eps
        with open(path, "wb") as f:
            pickle.dump(out, f, pickle.HIGHEST_PROTOCOL)

    def load_ply(self, path: str):
        """Load Gaussian parameters from a pickle-based ``.ply`` file.

        Scale bounds are restored from the file metadata via
        ``_load_and_configure_scale_bound``.
        """
        import pickle

        with open(path, "rb") as f:
            data = pickle.load(f)
        xyz = data.get("xyz", data.get("means"))
        scale = data.get("scale", data.get("scales"))
        rotation = data.get("rotation", data.get("quats"))
        density = data.get("density", data.get("opacities"))
        if xyz is None or scale is None or rotation is None or density is None:
            raise ValueError(f"Missing required fields. Found keys: {list(data.keys())}")
        if density.ndim == 2 and density.shape[1] == 1:
            density = density.squeeze(1)

        self.means = torch.nn.Parameter(
            torch.tensor(xyz, dtype=torch.float32, device=self.device).requires_grad_(True)
        )
        self.scales = torch.nn.Parameter(
            torch.tensor(scale, dtype=torch.float32, device=self.device).requires_grad_(True)
        )
        self.quats = torch.nn.Parameter(
            torch.tensor(rotation, dtype=torch.float32, device=self.device).requires_grad_(True)
        )
        self.opacities = torch.nn.Parameter(
            torch.tensor(density, dtype=torch.float32, device=self.device).requires_grad_(True)
        )
        self._load_and_configure_scale_bound(data)

    def export_ply(self, path: str, color_mode: str = "opacity"):
        """Export Gaussians as an ASCII PLY file with per-vertex greyscale colour.

        Opacity values are normalised to ``[0, 255]`` and written as RGB.

        Args:
            path: Destination file path.
            color_mode: ``"opacity"`` or ``"density"`` (both use activated
                opacities; kept for forward compatibility).

        Returns:
            The written file path.
        """
        means = self.means.detach().cpu().numpy()
        post_opa = self.activated_opacities.detach().cpu().numpy()
        if color_mode not in ("opacity", "density"):
            raise ValueError(f"Unsupported color_mode: {color_mode}")
        vals = post_opa.reshape(-1)
        vmin, vmax = float(vals.min()), float(vals.max())
        norm = (vals - vmin) if vmax - vmin < 1e-8 else (vals - vmin) / (vmax - vmin)
        colors = (norm * 255.0).clip(0, 255).astype("uint8")
        if colors.ndim == 2:
            colors = colors[:, 0]
        with open(path, "w") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {means.shape[0]}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            for (x, y, z), c in zip(means, colors):
                f.write(f"{x} {y} {z} {c} {c} {c}\n")
        return path

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _inverse_softplus(y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Numerically stable inverse of ``F.softplus``: ``log(expm1(y))``."""
        y_clamped = y.clamp_min(0.0) + eps
        return torch.log(torch.expm1(y_clamped))

    def _configure_scale_bounds(
        self,
        min_scale: Optional[Union[float, Sequence[float], torch.Tensor]],
        max_scale: Optional[Union[float, Sequence[float], torch.Tensor]],
        scale_eps: float,
    ):
        """Set internal scale-bound state used by activation functions.

        When both *min_scale* and *max_scale* are provided, a sigmoid-based
        parameterisation is used; otherwise plain ``exp`` is used.
        """
        if (min_scale is None) ^ (max_scale is None):
            raise ValueError("min_scale and max_scale must be both set or both None")
        if min_scale is None and max_scale is None:
            self._use_sigmoid_scale = False
            self._scale_min = None
            self._scale_max = None
            self._scale_eps = 0.0
            return

        min_tensor = torch.as_tensor(min_scale, dtype=torch.float32)
        max_tensor = torch.as_tensor(max_scale, dtype=torch.float32)
        assert min_tensor.shape == max_tensor.shape, "min_scale and max_scale must share the same shape"
        assert torch.all(max_tensor > min_tensor), "max_scale must be greater than min_scale"
        assert 0.0 <= scale_eps < 0.5, "scale_eps must lie in [0, 0.5)"

        self._use_sigmoid_scale = True
        self._scale_min = min_tensor
        self._scale_max = max_tensor
        self._scale_eps = float(scale_eps)

    def _get_scale_bound(self) -> Optional[Any]:
        """Return scale bounds as a numpy array for serialisation, or ``None``."""
        if not self._use_sigmoid_scale:
            return None
        scale_min = self._scale_min.cpu().numpy()
        scale_max = self._scale_max.cpu().numpy()
        if scale_min.ndim == 0:
            return np.array([float(scale_min), float(scale_max)])
        return np.stack([scale_min, scale_max], axis=0)

    def _load_and_configure_scale_bound(self, data: dict):
        """Restore scale-bound configuration from a deserialised data dict.

        Called by ``load_ply`` to re-establish ``_use_sigmoid_scale`` and
        related attributes from the saved ``scale_bound`` / ``scale_eps`` keys.
        """
        scale_bound = data.get("scale_bound", None)
        scale_eps = data.get("scale_eps", 1e-4)

        if scale_bound is None:
            self._configure_scale_bounds(min_scale=None, max_scale=None, scale_eps=0.0)
        else:
            scale_bound = np.asarray(scale_bound)
            if scale_bound.shape == (2,):
                min_scale, max_scale = float(scale_bound[0]), float(scale_bound[1])
            elif scale_bound.shape == (2, 3):
                min_scale, max_scale = scale_bound[0], scale_bound[1]
            else:
                raise ValueError(f"Invalid scale_bound shape: {scale_bound.shape}")
            self._configure_scale_bounds(
                min_scale=min_scale, max_scale=max_scale, scale_eps=scale_eps
            )
