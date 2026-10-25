"""Binary-tree trainer with custom __init__ and custom train loop."""

import math
import os
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.trainers.default import DefaultTrainer, TrainStepOutput
from src.strategies import BinaryTreeStrategy
from src.utils.loss import LossComputer, BinaryTreeLossComputer
from src.utils.math import normalized_quat_to_rotmat
from src.models.binary_tree_gaussian_model import BinaryTreeGaussianModel


class BinaryTreeTrainer(DefaultTrainer):
    """Trainer for CT 3D Gaussian Splatting with binary-tree densification."""

    def __init__(self, cfg, device: str):
        """Initialize the binary-tree trainer with binary-tree-specific components."""
        self.cfg = cfg
        self.device = device

        self.model = None
        self.optimizers = None
        self.strategy = None
        self.strategy_state = None
        self.loss_computer = None
        self._lrs_init = None
        self._lrs_final = None
        self._wandb_logger = None
        self._wandb_run = None
        self._wandb_enabled = False
        self._hierarchy_geometry_cache = None

        self._create_output_dirs()
        self._load_datasets()
        self._infer_scene_scale()
        self._load_ground_truth_volume()

        self.build_training(
            model_name="BinaryTreeGaussianModel",
            strategy_name="BinaryTreeStrategy",
            loss_name="BinaryTreeLossComputer",
        )

        self._validate_components()
        self._init_tv_geometry()
        self._init_wandb()

    def build_training(
        self,
        *,
        model_name: str = "GaussianModel",
        strategy_name: Optional[str] = "DefaultStrategy",
        loss_name: str = "LossComputer",
    ) -> None:
        """Create the model, optimizers, strategy, strategy state, and loss from names."""
        model_scale_meta = self._resolve_model_scale_meta()

        if model_name == "BinaryTreeGaussianModel":
            self.model = BinaryTreeGaussianModel(device=self.device, **model_scale_meta)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        bbox_tensor = None
        if getattr(self.parser, "bbox", None) is not None:
            bbox_tensor = torch.as_tensor(self.parser.bbox, dtype=torch.float32)

        if bbox_tensor is not None:
            center = bbox_tensor.mean(dim=0)
            scene_extent = float((bbox_tensor[1] - bbox_tensor[0]).abs().max())
        else:
            center = torch.zeros(3, dtype=torch.float32)
            scene_extent = float(self.cfg.model.init_extent)

        init_scale = max(scene_extent * 0.5, float(self.cfg.model.scale_min))
        init_scale = min(init_scale, float(self.cfg.model.scale_max))
        print(
            "Initializing binary-tree root Gaussian at scene center "
            f"{center.tolist()} with scale={init_scale:.4f}."
        )
        self.model.initialize_centered_sphere(
            center=center,
            init_scale=init_scale,
            init_opa=0.1,
            init_opa_scale=self.cfg.model.init_opa_scale,
        )

        global_scale = self.cfg.model.global_scale
        self._lrs_init = {
            "means": self.cfg.optim.lr_means_init * global_scale,
            "scales": self.cfg.optim.lr_scales_init,
            "quats": self.cfg.optim.lr_quats_init,
            "opacities": self.cfg.optim.lr_opacities_init,
        }
        self._lrs_final = {
            "means": self.cfg.optim.lr_means_final * global_scale,
            "scales": self.cfg.optim.lr_scales_final,
            "quats": self.cfg.optim.lr_quats_final,
            "opacities": self.cfg.optim.lr_opacities_final,
        }
        self.optimizers = self.model.build_optimizers(self._lrs_init)

        if strategy_name == "BinaryTreeStrategy":
            self.strategy = BinaryTreeStrategy(self.cfg.strategy)
            self.strategy.configure_lod_checkpoints(
                self.lod_dir,
                tuple(int(threshold) for threshold in self.cfg.trainer.lod_thresholds),
            )
            self.strategy_state = self.strategy.initialize_state(scene_scale=self.scene_scale)
        else:
            raise ValueError(f"Unknown strategy type: {strategy_name}")

        if loss_name == "LossComputer":
            self.loss_computer = LossComputer(
                w_l1=self.cfg.trainer.proj_l1_weight,
                w_ssim=self.cfg.trainer.proj_ssim_weight,
                w_tv=self.cfg.trainer.vol_tv_weight,
                ssim_window=self.cfg.trainer.ssim_window,
            )
        elif loss_name == "BinaryTreeLossComputer":
            self.loss_computer = BinaryTreeLossComputer(
                w_l1=self.cfg.trainer.proj_l1_weight,
                w_ssim=self.cfg.trainer.proj_ssim_weight,
                w_tv=self.cfg.trainer.vol_tv_weight,
                w_edge=self.cfg.trainer.edge_weight_final,
                w_hierarchy=self.cfg.trainer.hierarchy_weight,
                ssim_window=self.cfg.trainer.ssim_window,
            )
        else:
            raise ValueError(f"Unknown loss name: {loss_name}")

        self.strategy.check_sanity(self.model, self.optimizers)

    def train(self) -> None:
        """Run the binary-tree training loop with edge-aware supervision."""
        self._limit_threads_if_needed()

        train_loader = self._create_train_loader()
        train_iter = iter(train_loader)
        bbox_summary = self._format_bbox_summary()
        max_steps = self.cfg.trainer.max_steps
        progress_bar = tqdm(
            range(max_steps),
            desc="Training",
            dynamic_ncols=True,
            mininterval=1.0,
            miniters=50,
        )

        for step in progress_bar:
            self._update_learning_rates(step, max_steps)
            raw_batch, train_iter = self._fetch_next_batch(train_iter, train_loader)

            batch = self._prepare_train_batch(raw_batch)
            gaussian_state = self.model.prepare_gaussians(explicit=True)
            renders_hw, render_info = self.rasterize_splats(
                batch.camtoworlds,
                batch.intrinsics,
                batch.image_width,
                batch.image_height,
                viewmats=batch.viewmats,
                gaussian_state=gaussian_state,
            )
            renders = self._format_renders(renders_hw)
            volume_pred = self._sample_tv_volume(step, gaussian_state=gaussian_state)
            edge_mask = batch.raw["edge_mask"].to(self.device)
            gt_blur_sigma = self._compute_gt_blur_sigma(step)
            loss_targets = self._blur_targets(batch.target_images, gt_blur_sigma)
            edge_weight = self._compute_edge_weight(step)
            hierarchy_loss = self._compute_hierarchy_loss()
            loss_dict = self.loss_computer(
                renders,
                loss_targets,
                edge_mask,
                volume_pred,
                hierarchy_loss=hierarchy_loss,
                edge_targets=batch.target_images,
                edge_weight_override=edge_weight,
            )
            loss_dict["gt_blur_sigma"] = torch.tensor(gt_blur_sigma, device=self.device)

            self.strategy.step_pre_backward(
                model=self.model,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=render_info,
            )
            loss_dict["total"].backward()
            self.model.backward_explicit(gaussian_state)
            self.strategy.step_post_backward(
                model=self.model,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=render_info,
                packed=self.cfg.trainer.packed,
            )

            for optimizer in self.optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            step_output = TrainStepOutput(
                batch=batch,
                renders=renders,
                render_info=render_info,
                volume_pred=volume_pred,
                loss_dict=loss_dict,
            )

            self._save_outputs(step)
            if step % 250 == 0 or step + 1 == max_steps:
                progress_bar.set_description(
                    self._format_progress(step_output.loss_dict, bbox_summary),
                    refresh=False,
                )
            self.log(step, step_output.loss_dict)

        self._close_wandb()

    def log(self, step: int, loss_dict: Dict[str, torch.Tensor]) -> None:
        """Reuse the base trainer logging behavior."""
        super().log(step, loss_dict)

    def _create_output_dirs(self) -> None:
        """Create result directories including LOD checkpoints."""
        super()._create_output_dirs()
        self.lod_dir = os.path.join(self.cfg.exp.result_dir, "lod")
        os.makedirs(self.lod_dir, exist_ok=True)

    def _compute_hierarchy_loss(self) -> torch.Tensor:
        """Keep child Gaussians close to their archived parent support."""
        if self.cfg.trainer.hierarchy_weight <= 0:
            return self.model.means.new_zeros(())

        history_count = min(
            self.model.history_means.shape[0],
            self.model.history_scales.shape[0],
            self.model.history_quats.shape[0],
        )
        if self.model.num_gaussians == 0 or history_count == 0:
            return self.model.means.new_zeros(())

        cache_key = (
            self.model.parent.data_ptr(),
            self.model.parent.numel(),
            self.model.parent._version,
            self.model.history_means.data_ptr(),
            self.model.history_means._version,
            self.model.history_scales.data_ptr(),
            self.model.history_scales._version,
            self.model.history_quats.data_ptr(),
            self.model.history_quats._version,
            history_count,
        )
        cache = self._hierarchy_geometry_cache
        if cache is None or cache["key"] != cache_key:
            with torch.no_grad():
                valid = (self.model.parent >= 0) & (
                    self.model.parent < history_count
                )
                child_ids = torch.where(valid)[0]
                if child_ids.numel() == 0:
                    self._hierarchy_geometry_cache = {
                        "key": cache_key,
                        "child_ids": child_ids,
                        "parent_centers": self.model.history_means.new_empty((0, 3)),
                        "world_to_normalized": self.model.history_means.new_empty(
                            (0, 3, 3)
                        ),
                    }
                else:
                    parent_ids = self.model.parent.index_select(0, child_ids).long()
                    parent_centers = self.model.history_means.index_select(
                        0, parent_ids
                    )
                    parent_scales = self.model.activate_scale(
                        self.model.history_scales.index_select(0, parent_ids)
                    ).clamp_min(1e-6)
                    parent_quats = F.normalize(
                        self.model.history_quats.index_select(0, parent_ids),
                        dim=-1,
                    )
                    parent_rotmats = normalized_quat_to_rotmat(parent_quats)
                    world_to_normalized = (
                        parent_rotmats.transpose(-1, -2)
                        / parent_scales.unsqueeze(-1)
                    ).contiguous()
                    self._hierarchy_geometry_cache = {
                        "key": cache_key,
                        "child_ids": child_ids,
                        "parent_centers": parent_centers,
                        "world_to_normalized": world_to_normalized,
                    }
            cache = self._hierarchy_geometry_cache

        child_ids = cache["child_ids"]
        if child_ids.numel() == 0:
            return self.model.means.new_zeros(())
        delta_world = (
            self.model.means.index_select(0, child_ids) - cache["parent_centers"]
        )
        delta_normalized = torch.bmm(
            cache["world_to_normalized"], delta_world.unsqueeze(-1)
        ).squeeze(-1)
        normalized_radius = torch.linalg.vector_norm(delta_normalized, dim=-1)
        allowed_radius = max(
            float(self.cfg.trainer.hierarchy_parent_distance_factor), 1e-6
        )
        normalized_excess = F.relu(normalized_radius - allowed_radius) / allowed_radius
        return normalized_excess.square().mean()

    def _compute_gt_blur_sigma(self, step: int) -> float:
        """Cosine decay GT blur so training shifts from low to high frequency."""
        return self._cosine_schedule(
            step=step,
            start_value=float(self.cfg.trainer.gt_blur_sigma_init),
            end_value=float(self.cfg.trainer.gt_blur_sigma_final),
            start_step=int(self.cfg.trainer.gt_blur_start_step),
            end_step=int(self.cfg.trainer.gt_blur_end_step),
        )

    def _compute_edge_weight(self, step: int) -> float:
        """Cosine grow edge supervision weight as high-frequency fitting is introduced."""
        return self._cosine_schedule(
            step=step,
            start_value=float(self.cfg.trainer.edge_weight_init),
            end_value=float(self.cfg.trainer.edge_weight_final),
            start_step=int(self.cfg.trainer.edge_weight_start_step),
            end_step=int(self.cfg.trainer.edge_weight_end_step),
        )

    @staticmethod
    def _cosine_schedule(
        step: int,
        start_value: float,
        end_value: float,
        start_step: int,
        end_step: int,
    ) -> float:
        """Cosine interpolate between two values inside a step interval."""
        if end_step <= start_step:
            return end_value
        if step <= start_step:
            return start_value
        if step >= end_step:
            return end_value
        progress = (step - start_step) / float(end_step - start_step)
        cosine_progress = 0.5 * (1.0 - math.cos(math.pi * progress))
        return float(start_value + (end_value - start_value) * cosine_progress)

    def _blur_targets(self, targets: torch.Tensor, sigma: float) -> torch.Tensor:
        """Apply Gaussian blur on GT to emphasize low-frequency structure in early training."""
        if sigma <= 0:
            return targets

        radius = max(int(round(3.0 * sigma)), 1)
        kernel_size = radius * 2 + 1
        coords = torch.arange(
            kernel_size, device=targets.device, dtype=targets.dtype
        ) - radius
        kernel_1d = torch.exp(-(coords.square()) / (2.0 * sigma * sigma))
        kernel_1d = kernel_1d / kernel_1d.sum().clamp_min(1e-12)
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)

        targets_cf = targets.permute(0, 3, 1, 2).contiguous()
        channels = targets_cf.shape[1]
        kernel = kernel_2d.expand(channels, 1, kernel_size, kernel_size)
        padded = F.pad(targets_cf, (radius, radius, radius, radius), mode="reflect")
        blurred = F.conv2d(padded, kernel, groups=channels)
        return blurred.permute(0, 2, 3, 1).contiguous()
