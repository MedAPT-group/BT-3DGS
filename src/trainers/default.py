"""Default trainer implementation built on top of BaseTrainer."""

import os
from typing import Any, Dict, Iterator, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
try:
    import wandb
except ImportError:
    wandb = None
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

from src.data.ctproj import CTImageDataset, CTImageParser
from src.loggers.wandb import WandbLogger
from src.models.gaussian_model import GaussianModel
from src.strategies import DefaultStrategy, MCMCStrategy
from src.utils.loss import LossComputer
from src.utils.metric import metric_proj, metric_vol
from src.utils.math import invert_rigid_transform
from src.trainers.base import BaseTrainer, TrainBatch, TrainStepOutput


class DefaultTrainer(BaseTrainer):
    """Base trainer with three main customization points: ``__init__``, ``train``, and ``log``."""

    def __init__(self, cfg: Any, device: str):
        """Initialize the default trainer state, components, and utilities."""
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

        self._create_output_dirs()
        self._load_datasets()
        self._infer_scene_scale()
        self._load_ground_truth_volume()

        self.build_training(
            model_name="GaussianModel",
            strategy_name=self.cfg.strategy.strategy_type,
            loss_name="LossComputer",
        )

        self._validate_components()
        self._init_tv_geometry()
        self._init_wandb()

    def train(self) -> None:
        """Default training loop."""
        self._limit_threads_if_needed()

        train_loader = self._create_train_loader()
        train_iterator = iter(train_loader)
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
            raw_batch, train_iterator = self._fetch_next_batch(
                train_iterator, train_loader
            )

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
            loss_dict = self._compute_loss(batch, renders, volume_pred)

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
        """Default logging behavior."""
        if not self._wandb_enabled or self._wandb_run is None:
            return

        log_data: Dict[str, Any] = {
            "train/num_gaussians": float(self.model.num_gaussians)
        }
        for key, value in loss_dict.items():
            log_data[f"train/{key}"] = float(value.item())

        should_metrics = (
            self.cfg.trainer.metrics_every > 0
            and (step + 1) % self.cfg.trainer.metrics_every == 0
        )
        should_visualize = (
            self.cfg.trainer.visualize_every > 0
            and (step + 1) % self.cfg.trainer.visualize_every == 0
        )

        image_pairs = None
        if should_metrics:
            image_pairs = self._append_validation_metrics(log_data)
        if should_visualize:
            self._append_validation_visualization(log_data, step, image_pairs=image_pairs)

        self._wandb_run.log(log_data, step=step)

    def build_training(
        self,
        *,
        model_name: str = "GaussianModel",
        strategy_name: Optional[str] = "DefaultStrategy",
        loss_name: str = "LossComputer",
    ) -> None:
        """Create the model, optimizers, strategy, strategy state, and loss from names."""
        model_scale_meta = self._resolve_model_scale_meta()

        if model_name == "GaussianModel":
            self.model = GaussianModel(device=self.device, **model_scale_meta)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        bbox_tensor = None
        if getattr(self.parser, "bbox", None) is not None:
            bbox_tensor = torch.as_tensor(self.parser.bbox, dtype=torch.float32)

        self.model.initialize_from_data_dir(
            data_dir=self.cfg.exp.data_dir,
            global_scale=self.cfg.model.global_scale,
            bbox=bbox_tensor,
            init_opa_scale=self.cfg.model.init_opa_scale,
            init_scale_multiplier=self.cfg.model.init_scale_multiplier,
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

        if strategy_name == "DefaultStrategy":
            self.strategy = DefaultStrategy(self.cfg.strategy)
            self.strategy_state = self.strategy.initialize_state(
                scene_scale=self.scene_scale
            )
        elif strategy_name == "MCMCStrategy":
            self.strategy = MCMCStrategy(self.cfg.strategy)
            self.strategy_state = self.strategy.initialize_state()
        else:
            raise ValueError(f"Unknown strategy type: {strategy_name}")

        if loss_name == "LossComputer":
            self.loss_computer = LossComputer(
                w_l1=self.cfg.trainer.proj_l1_weight,
                w_ssim=self.cfg.trainer.proj_ssim_weight,
                w_tv=self.cfg.trainer.vol_tv_weight,
                ssim_window=self.cfg.trainer.ssim_window,
            )
        else:
            raise ValueError(f"Unknown loss name: {loss_name}")

        self.strategy.check_sanity(self.model, self.optimizers)

    def _create_output_dirs(self) -> None:
        """Create result directories used for checkpoints and exports."""
        os.makedirs(self.cfg.exp.result_dir, exist_ok=True)
        self.ckpt_dir = os.path.join(self.cfg.exp.result_dir, "ckpts")
        self.ply_dir = os.path.join(self.cfg.exp.result_dir, "ply")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.ply_dir, exist_ok=True)

    def _load_datasets(self) -> None:
        """Load the train/test datasets and create the validation dataloader."""
        self.parser = CTImageParser(self.cfg.exp.data_dir)
        self.trainset = CTImageDataset(self.parser, split="train")

        try:
            self.valset = CTImageDataset(self.parser, split="test")
        except Exception:
            self.valset = None

        self.val_loader = None
        if self.valset is not None:
            self.val_loader = torch.utils.data.DataLoader(
                self.valset,
                batch_size=1,
                shuffle=False,
                num_workers=self.cfg.trainer.num_workers,
                pin_memory=self.cfg.trainer.pin_memory,
                persistent_workers=(
                    self.cfg.trainer.num_workers > 0
                    and self.cfg.trainer.persistent_workers
                ),
            )

    def _infer_scene_scale(self) -> None:
        """Estimate a scene scale used by strategy initialization."""
        bbox = getattr(self.parser, "bbox", None)
        if bbox is not None:
            bbox_tensor = torch.as_tensor(bbox, dtype=torch.float32)
            self.scene_scale = (
                float((bbox_tensor[1] - bbox_tensor[0]).abs().max()) * 0.5
            )
            return
        self.scene_scale = self.cfg.model.init_extent * self.cfg.model.global_scale

    def _load_ground_truth_volume(self) -> None:
        """Load the ground-truth volume when available and cache its shape."""
        self.vol_gt = None
        try:
            self.vol_gt = self.trainset.get_vol_gt().to(self.device)
        except Exception:
            pass

        if self.vol_gt is not None:
            self.volume_shape_wdh = tuple(self.vol_gt.shape)
            return

        self.volume_shape_wdh = tuple(
            self.parser.meta.get(
                "nVoxel",
                getattr(self.cfg.trainer, "volume_shape", None),
            )
        )

    def _init_tv_geometry(self) -> None:
        """Precompute the random sub-volume geometry used by TV regularization."""
        self._bbox_tensor = (
            torch.as_tensor(self.parser.bbox, dtype=torch.float32, device=self.device)
            if getattr(self.parser, "bbox", None) is not None
            else None
        )

        if self.cfg.trainer.vol_tv_weight <= 0:
            self.tv_volume_shape = None
            self.tv_vol_sVoxel = None
            return

        if self._bbox_tensor is None:
            raise ValueError("bbox is required to calculate TV sub-volume size")

        tv_size = int(self.cfg.trainer.tv_vol_size)
        self.tv_volume_shape = (tv_size, tv_size, tv_size)
        bbox_size = self._bbox_tensor[1] - self._bbox_tensor[0]
        voxel_size = bbox_size / torch.tensor(
            self.volume_shape_wdh,
            device=self.device,
            dtype=torch.float32,
        )
        self.tv_vol_sVoxel = torch.tensor(
            self.tv_volume_shape,
            device=self.device,
            dtype=torch.float32,
        ) * voxel_size

    def _init_wandb(self) -> None:
        """Initialize the W&B logger and keep the active run handle."""
        self._wandb_logger = WandbLogger(
            wandb_cfg=getattr(self.cfg, "wandb", None),
            result_dir=self.cfg.exp.result_dir,
        )
        self._wandb_run, self._wandb_enabled = self._wandb_logger.initialize()

    def _validate_components(self) -> None:
        """Ensure all core training components were initialized before training."""
        required = {
            "model": self.model,
            "optimizers": self.optimizers,
            "strategy": self.strategy,
            "strategy_state": self.strategy_state,
            "loss_computer": self.loss_computer,
            "_lrs_init": self._lrs_init,
            "_lrs_final": self._lrs_final,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError("Trainer initialization is incomplete: " + ", ".join(missing))

    def _resolve_model_scale_meta(self) -> Dict[str, Any]:
        """Resolve optional scale bounds from config first, then dataset metadata."""
        meta = getattr(self.parser, "meta", {})
        min_scale = (
            self.cfg.model.scale_min
            if self.cfg.model.scale_min is not None
            else meta.get("scale_min")
        )
        max_scale = (
            self.cfg.model.scale_max
            if self.cfg.model.scale_max is not None
            else meta.get("scale_max")
        )
        result = {
            "eps2d": float(self.cfg.model.eps2d),
            "eps3d": float(self.cfg.model.eps3d),
            "img_tile_size": int(self.cfg.model.img_tile_size),
            "vol_tile_size": int(self.cfg.model.vol_tile_size),
            "img_radius_clip": float(self.cfg.model.img_radius_clip),
            "vol_radius_clip": float(self.cfg.model.vol_radius_clip),
            "backward_mode": str(self.cfg.model.backward_mode),
        }
        if min_scale is not None and max_scale is not None:
            result["min_scale"] = min_scale
            result["max_scale"] = max_scale
            result["scale_eps"] = self.cfg.model.scale_eps
        return result

    def _prepare_train_batch(self, raw_batch: Dict[str, torch.Tensor]) -> TrainBatch:
        """Move a raw dataloader batch to device and normalize its shape metadata."""
        target_images = raw_batch["image"].to(device=self.device, dtype=torch.float32)
        camtoworlds = raw_batch["camtoworld"].to(dtype=torch.float32)
        viewmats = raw_batch.get("viewmat")
        if viewmats is None:
            viewmats = invert_rigid_transform(camtoworlds)
        return TrainBatch(
            raw=raw_batch,
            camtoworlds=camtoworlds.to(device=self.device),
            viewmats=viewmats.to(device=self.device, dtype=torch.float32),
            intrinsics=raw_batch["K"].to(device=self.device, dtype=torch.float32),
            target_images=target_images,
            image_height=target_images.shape[1],
            image_width=target_images.shape[2],
        )

    def rasterize_splats(
        self,
        camtoworlds: torch.Tensor,
        intrinsics: torch.Tensor,
        width: int,
        height: int,
        *,
        viewmats: Optional[torch.Tensor] = None,
        gaussian_state=None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Render the current Gaussian set into 2D projections."""
        return self.model.rasterize(
            camtoworlds=camtoworlds,
            Ks=intrinsics,
            width=width,
            height=height,
            packed=self.cfg.trainer.packed,
            sparse_grad=self.cfg.trainer.sparse_grad,
            absgrad=bool(getattr(self.strategy, "absgrad", False)),
            viewmats=viewmats,
            gaussian_state=gaussian_state,
        )

    def voxelize_splats(
        self,
        volume_shape: Tuple[int, ...],
        bbox: torch.Tensor,
        *,
        gaussian_state=None,
    ) -> torch.Tensor:
        """Voxelize the current Gaussian set into a dense 3D volume."""
        return self.model.voxelize(
            volume_shape,
            bbox,
            gaussian_state=gaussian_state,
        )

    def _format_renders(self, renders_hw: torch.Tensor) -> torch.Tensor:
        """Convert renderer output into the layout expected by losses and metrics."""
        return renders_hw.permute(0, 2, 1, 3).contiguous()

    def _compute_loss(
        self,
        batch: TrainBatch,
        renders: torch.Tensor,
        volume_pred: Optional[torch.Tensor],
        *,
        loss_fn=None,
    ) -> Dict[str, torch.Tensor]:
        """Compute the step loss, optionally delegating to a custom loss callable."""
        if loss_fn is not None:
            return loss_fn(batch, renders, volume_pred)
        return self.loss_computer(renders, batch.target_images, volume_pred)

    def _sample_tv_volume(
        self,
        step: int,
        *,
        gaussian_state=None,
    ) -> Optional[torch.Tensor]:
        """Sample a random local volume used by TV regularization for this step."""
        if self.cfg.trainer.vol_tv_weight <= 0:
            if step == 0:
                print("TV loss weight is 0, voxelization is disabled for training.")
            return None

        try:
            bbox_min = self._bbox_tensor[0]
            bbox_max = self._bbox_tensor[1]
            half_size = self.tv_vol_sVoxel / 2
            valid_min = bbox_min + half_size
            valid_max = bbox_max - half_size
            center = valid_min + (valid_max - valid_min) * torch.rand(
                3, device=self.device
            )
            sampled_bbox = torch.stack([center - half_size, center + half_size])
            volume = self.voxelize_splats(
                self.tv_volume_shape,
                sampled_bbox,
                gaussian_state=gaussian_state,
            )
        except Exception as exc:
            if step % 100 == 0:
                print(
                    f"Step {step}: TV sub-volume voxelization failed: "
                    f"{type(exc).__name__}: {exc}"
                )
            return None

        if volume is not None and volume.ndim >= 3:
            volume = volume.squeeze()
        return volume

    def _limit_threads_if_needed(self) -> None:
        """Reduce PyTorch CPU threading when the trainer config asks for it."""
        if not self.cfg.trainer.limit_threads:
            return
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass

    def _create_train_loader(self) -> torch.utils.data.DataLoader:
        """Create the shuffled dataloader used by the train loop."""
        cfg = self.cfg.trainer
        preload_to = getattr(self.trainset, "preload_to", None)
        use_device_preload = self.device.startswith("cuda") and callable(preload_to)
        if use_device_preload:
            preload_to(self.device)
        return torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=0 if use_device_preload else cfg.num_workers,
            pin_memory=False if use_device_preload else cfg.pin_memory,
            persistent_workers=(
                False
                if use_device_preload
                else (cfg.num_workers > 0 and cfg.persistent_workers)
            ),
        )

    @staticmethod
    def _fetch_next_batch(
        loader_iter: Iterator,
        loader: torch.utils.data.DataLoader,
    ) -> Tuple[Dict[str, torch.Tensor], Iterator]:
        """Fetch the next batch and recycle the iterator when an epoch ends."""
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)
        return batch, loader_iter

    @staticmethod
    def _compute_lr(step: int, max_steps: int, lr_init: float, lr_final: float) -> float:
        """Exponentially interpolate between the initial and final learning rates."""
        t = min(step / max_steps, 1.0)
        return lr_init * ((lr_final / lr_init) ** t)

    def _update_learning_rates(self, step: int, max_steps: int) -> None:
        """Apply the current step learning rate to every optimizer."""
        for name, optimizer in self.optimizers.items():
            lr = self._compute_lr(
                step,
                max_steps,
                self._lrs_init[name],
                self._lrs_final[name],
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

    def _close_wandb(self) -> None:
        """Close the active W&B run when logging is enabled."""
        if self._wandb_enabled and self._wandb_run is not None:
            self._wandb_run.finish()

    def _save_outputs(self, step: int) -> None:
        """Save checkpoints and optional PLY exports for configured steps."""
        cfg = self.cfg.trainer
        if step in cfg.save_steps:
            ckpt_path = os.path.join(self.ckpt_dir, f"ckpt_{step}.pt")
            self.model.save_ckpt(
                ckpt_path,
                step,
                extra={
                    "optimizers": {
                        name: optimizer.state_dict()
                        for name, optimizer in self.optimizers.items()
                    },
                },
            )
            self.model.save_ply(
                os.path.join(self.ckpt_dir, f"point_cloud_{step}.pickle")
            )

        if step in cfg.ply_steps and cfg.save_ply:
            self.model.export_ply(
                os.path.join(self.ply_dir, f"point_cloud_{step}.ply")
            )

        self._save_gif_npy_frame(step)

    @torch.no_grad()
    def _save_gif_npy_frame(self, step: int) -> None:
        """Save projection and volume npy snapshots for GIF generation."""
        cfg = self.cfg.trainer
        save_every = int(getattr(cfg, "gif_npy_every", 0) or 0)
        frame_step = step + 1
        if save_every <= 0 or frame_step % save_every != 0:
            return

        output_dir = os.path.join(
            self.cfg.exp.result_dir,
            str(getattr(cfg, "gif_npy_dir", "gif_npy")),
        )
        os.makedirs(output_dir, exist_ok=True)

        was_training = self.model.training
        self.model.eval()
        try:
            if bool(getattr(cfg, "gif_npy_save_projection", True)):
                projection = self._render_gif_front_projection(
                    float(getattr(cfg, "gif_npy_projection_angle", 0.0))
                )
                if projection is not None:
                    np.save(
                        os.path.join(
                            output_dir,
                            f"step_{step:06d}_front_projection.npy",
                        ),
                        projection,
                    )

            if bool(getattr(cfg, "gif_npy_save_volume", True)):
                volume = self._voxelize_gif_volume()
                if volume is not None:
                    np.save(
                        os.path.join(output_dir, f"step_{step:06d}_volume.npy"),
                        volume,
                    )
        finally:
            if was_training:
                self.model.train()

    def _render_gif_front_projection(self, angle: float) -> Optional[np.ndarray]:
        """Render a single configured front-view projection as a 2D numpy array."""
        if len(self.trainset) == 0:
            return None

        reference_image = self.trainset.images[0]
        image_height = int(reference_image.shape[0])
        image_width = int(reference_image.shape[1])
        camtoworld = self.trainset._get_camtoworld(angle).to(self.device)[None]
        intrinsic = self.trainset.K.to(self.device)[None]
        renders_hw, _ = self.rasterize_splats(
            camtoworld,
            intrinsic,
            image_width,
            image_height,
        )
        projection = self._format_renders(renders_hw)[0]
        return projection.squeeze(-1).detach().float().cpu().numpy()

    def _voxelize_gif_volume(self) -> Optional[np.ndarray]:
        """Voxelize the current model and return a volume aligned to GT order."""
        if self._bbox_tensor is None or self.volume_shape_wdh is None:
            return None

        volume = self.voxelize_splats(self.volume_shape_wdh, self._bbox_tensor)
        if volume is None:
            return None
        volume = volume.squeeze()
        if volume.ndim != 3:
            return None
        volume = volume.permute(2, 1, 0).contiguous()
        return volume.detach().float().cpu().numpy()

    def _format_bbox_summary(self) -> str:
        """Format a short bbox summary string for the progress bar."""
        if self._bbox_tensor is None:
            return "bbox=None"
        size = (self._bbox_tensor[1] - self._bbox_tensor[0]).cpu().numpy()
        return f"bbox=[{size[0]:.2f},{size[1]:.2f},{size[2]:.2f}]"

    def _format_progress(
        self,
        loss_dict: Dict[str, torch.Tensor],
        bbox_summary: str,
    ) -> str:
        """Assemble the per-step progress-bar message."""
        # Keep this deliberately compact: tqdm reserves terminal width for the
        # description before drawing the bar, so a full metric dump can hide
        # the percentage and bar entirely on ordinary terminals.
        parts = [f"loss={loss_dict['total'].item():.4f}"]
        for key in ("proj_l1", "edge"):
            if key in loss_dict:
                parts.append(f"{key}={loss_dict[key].item():.4f}")
        parts.extend(
            [
                f"N={self.model.num_gaussians}",
                f"lr_m={self.optimizers['means'].param_groups[0]['lr']:.2e}",
            ]
        )
        return "|".join(parts)

    @torch.no_grad()
    def _compute_eval_outputs(
        self,
    ) -> Tuple[Dict[str, Optional[float]], List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Compute validation scalars and example render pairs for logging."""
        metrics: Dict[str, Optional[float]] = {
            "psnr2d": None,
            "ssim2d": None,
            "psnr3d": None,
            "ssim3d": None,
        }
        image_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []

        if self.val_loader is None:
            return metrics, image_pairs

        was_training = self.model.training
        self.model.eval()

        try:
            gaussian_state = self.model.prepare_gaussians()
            metrics, image_pairs = self._eval_2d_outputs(
                metrics,
                image_pairs,
                gaussian_state=gaussian_state,
            )
            metrics = self._eval_3d_metrics(
                metrics,
                gaussian_state=gaussian_state,
            )
        except Exception as exc:
            print(f"Failed to compute eval outputs ({type(exc).__name__}: {exc})")
        finally:
            if was_training:
                self.model.train()

        return metrics, image_pairs

    def _eval_2d_outputs(
        self,
        metrics: Dict[str, Optional[float]],
        image_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
        *,
        gaussian_state=None,
    ) -> Tuple[Dict[str, Optional[float]], List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Compute 2D validation metrics and collect render/target pairs."""
        psnr_vals: List[float] = []
        ssim_vals: List[float] = []
        vis_indices = set(
            np.linspace(0, len(self.valset) - 1, 6, dtype=int).tolist()[:-1]
        )

        for idx, raw_batch in enumerate(self.val_loader):
            batch = self._prepare_train_batch(raw_batch)
            pred_hw, _ = self.rasterize_splats(
                batch.camtoworlds,
                batch.intrinsics,
                batch.image_width,
                batch.image_height,
                viewmats=batch.viewmats,
                gaussian_state=gaussian_state,
            )
            pred = self._format_renders(pred_hw)

            if idx in vis_indices:
                image_pairs.append(
                    (pred[0, :, :, 0].cpu(), batch.target_images[0, :, :, 0].cpu())
                )

            metric = metric_proj(
                pred.squeeze(-1),
                batch.target_images.squeeze(-1),
                window_size=self.cfg.trainer.ssim_window,
                data_range=1.0,
            )
            psnr_vals.append(float(metric["psnr"].item()))
            ssim_vals.append(float(metric["ssim"].item()))

        if psnr_vals:
            metrics["psnr2d"] = float(np.mean(psnr_vals))
        if ssim_vals:
            metrics["ssim2d"] = float(np.mean(ssim_vals))
        return metrics, image_pairs

    def _eval_3d_metrics(
        self,
        metrics: Dict[str, Optional[float]],
        *,
        gaussian_state=None,
    ) -> Dict[str, Optional[float]]:
        """Compute 3D validation metrics by voxelizing the current Gaussian field."""
        volumes = self._get_validation_volumes(gaussian_state=gaussian_state)
        if volumes is None:
            return metrics

        pred_vol, gt_vol = volumes
        metric = metric_vol(
            pred_vol,
            gt_vol,
            window_size=self.cfg.trainer.ssim_window,
        )
        metrics["psnr3d"] = float(metric["psnr"].item())
        metrics["ssim3d"] = float(metric["ssim"].item())
        return metrics

    @torch.no_grad()
    def _get_validation_volumes(
        self,
        *,
        gaussian_state=None,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Voxelize and align prediction/GT volumes for evaluation and visualization."""
        if self.vol_gt is None or self._bbox_tensor is None or self.volume_shape_wdh is None:
            return None

        pred_vol = self.voxelize_splats(
            self.volume_shape_wdh,
            self._bbox_tensor,
            gaussian_state=gaussian_state,
        )
        if pred_vol is None:
            return None
        if pred_vol.ndim >= 3:
            pred_vol = pred_vol.squeeze()
        if pred_vol.ndim != 3:
            return None

        pred_vol = pred_vol.permute(2, 1, 0).contiguous()
        gt_vol = self.vol_gt
        if pred_vol.shape != gt_vol.shape:
            slices = tuple(
                slice(0, min(pred_size, gt_size))
                for pred_size, gt_size in zip(pred_vol.shape, gt_vol.shape)
            )
            pred_vol = pred_vol[slices]
            gt_vol = gt_vol[slices]

        return pred_vol, gt_vol

    @torch.no_grad()
    def _create_validation_image_grid(
        self,
        image_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
        step: int,
    ) -> Optional[Any]:
        """Build a visualization grid showing prediction, target, and absolute error."""
        if not image_pairs:
            return None

        preds = [pred.numpy() for pred, _ in image_pairs]
        gts = [gt.numpy() for _, gt in image_pairs]
        diffs = [np.abs(pred - gt) for pred, gt in zip(preds, gts)]

        fig = plt.figure(figsize=(14, 3 * len(image_pairs) + 0.8))
        grid = GridSpec(
            len(image_pairs),
            4,
            figure=fig,
            width_ratios=[1, 1, 1, 0.04],
            wspace=0.1,
            hspace=0.3,
        )
        col_titles = ["Prediction", "Ground Truth", "Difference"]

        for i in range(len(image_pairs)):
            vmin = float(min(preds[i].min(), gts[i].min(), diffs[i].min()))
            vmax = float(max(preds[i].max(), gts[i].max(), diffs[i].max()))
            axes = [fig.add_subplot(grid[i, col]) for col in range(3)]
            cbar_ax = fig.add_subplot(grid[i, 3])

            axes[0].imshow(preds[i], cmap="viridis", vmin=vmin, vmax=vmax, aspect="equal")
            axes[1].imshow(gts[i], cmap="viridis", vmin=vmin, vmax=vmax, aspect="equal")
            image = axes[2].imshow(
                diffs[i],
                cmap="viridis",
                vmin=diffs[i].min(),
                vmax=diffs[i].max(),
                aspect="equal",
            )
            fig.colorbar(image, cax=cbar_ax)

            axes[0].set_ylabel(f"View {i}", fontsize=10)
            for axis in axes:
                axis.set_xticks([])
                axis.set_yticks([])
            if i == 0:
                for axis, title in zip(axes, col_titles):
                    axis.set_title(title, fontsize=11)

        fig.suptitle(f"Validation Rasterization (Step {step})", fontsize=13, y=1.01)

        try:
            image = wandb.Image(fig)
        except Exception:
            plt.close(fig)
            return None
        plt.close(fig)
        return image

    @torch.no_grad()
    def _create_validation_volume_grid(self, step: int) -> Optional[Any]:
        """Build a volume slice grid showing prediction, target, and absolute error."""
        volumes = self._get_validation_volumes()
        if volumes is None:
            return None

        pred_vol, gt_vol = volumes
        width, depth, height = pred_vol.shape
        mid = (width // 2, depth // 2, height // 2)

        def extract_slice(volume: torch.Tensor, axis: int, index: int) -> np.ndarray:
            if axis == 0:
                image = volume[index, :, :]
            elif axis == 1:
                image = volume[:, index, :]
            else:
                image = volume[:, :, index]
            return image.detach().cpu().numpy()

        pred_slices = [
            extract_slice(pred_vol, 0, mid[0]),
            extract_slice(pred_vol, 1, mid[1]),
            extract_slice(pred_vol, 2, mid[2]),
        ]
        gt_slices = [
            extract_slice(gt_vol, 0, mid[0]),
            extract_slice(gt_vol, 1, mid[1]),
            extract_slice(gt_vol, 2, mid[2]),
        ]
        diff_slices = [
            np.abs(pred_slice - gt_slice)
            for pred_slice, gt_slice in zip(pred_slices, gt_slices)
        ]

        vmin = float(min(pred_vol.min().item(), gt_vol.min().item()))
        vmax = float(max(pred_vol.max().item(), gt_vol.max().item()))
        diff_vmax = float(max(diff.max() for diff in diff_slices))
        diff_vmax = max(diff_vmax, 1e-12)

        fig, axes = plt.subplots(3, 3, figsize=(12, 10))
        axis_names = ["W-axis", "D-axis", "H-axis"]
        row_names = ["Prediction", "Ground Truth", "Difference"]
        rows = [pred_slices, gt_slices, diff_slices]

        for row_idx, row_slices in enumerate(rows):
            for col_idx, slice_image in enumerate(row_slices):
                if row_idx < 2:
                    image = axes[row_idx, col_idx].imshow(
                        slice_image,
                        cmap="viridis",
                        vmin=vmin,
                        vmax=vmax,
                        aspect="equal",
                    )
                else:
                    image = axes[row_idx, col_idx].imshow(
                        slice_image,
                        cmap="viridis",
                        vmin=0.0,
                        vmax=diff_vmax,
                        aspect="equal",
                    )
                axes[row_idx, col_idx].set_xticks([])
                axes[row_idx, col_idx].set_yticks([])
                if row_idx == 0:
                    axes[row_idx, col_idx].set_title(axis_names[col_idx], fontsize=11)
                if col_idx == 0:
                    axes[row_idx, col_idx].set_ylabel(row_names[row_idx], fontsize=11)
                fig.colorbar(image, ax=axes[row_idx, col_idx], fraction=0.046, pad=0.04)

        fig.suptitle(f"Validation Volume Slices (Step {step})", fontsize=13, y=0.98)
        plt.tight_layout(rect=(0, 0, 1, 0.96))

        try:
            image = wandb.Image(fig)
        except Exception:
            plt.close(fig)
            return None
        plt.close(fig)
        return image

    def _append_validation_metrics(
        self,
        log_data: Dict[str, Any],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Append validation scalar metrics into the log payload."""
        eval_metrics, image_pairs = self._compute_eval_outputs()
        metric_keys = {
            "psnr2d": "test_metric_2d/psnr2d",
            "ssim2d": "test_metric_2d/ssim2d",
            "psnr3d": "test_metric_3d/psnr3d",
            "ssim3d": "test_metric_3d/ssim3d",
        }
        for src_key, dst_key in metric_keys.items():
            if eval_metrics[src_key] is not None:
                log_data[dst_key] = float(eval_metrics[src_key])
        return image_pairs

    def _append_validation_visualization(
        self,
        log_data: Dict[str, Any],
        step: int,
        image_pairs: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> None:
        """Append validation images into the log payload."""
        if image_pairs is None:
            _, image_pairs = self._compute_eval_outputs()
        grid_img = self._create_validation_image_grid(image_pairs, step)
        if grid_img is not None:
            log_data["test_visualization/projection_views"] = grid_img
        volume_img = self._create_validation_volume_grid(step)
        if volume_img is not None:
            log_data["test_visualization/volume_slices"] = volume_img
