"""Student distillation trainer built on top of the default training flow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
from tqdm import tqdm

from src.data.ctproj import CTImageDataset, CTImageParser
from src.models.binary_tree_gaussian_model import BinaryTreeGaussianModel
from src.models.student_gaussian_model import StudentGaussianModel
from src.strategies import DistillStrategy
from src.trainers.default import DefaultTrainer
from src.utils.loss import LossComputer
from src.utils.math import invert_rigid_transform


@dataclass
class DistillProjectionDataset(torch.utils.data.Dataset):
    images: torch.Tensor
    camtoworlds: torch.Tensor
    K: torch.Tensor
    viewmats: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        if self.viewmats is None:
            self.viewmats = invert_rigid_transform(self.camtoworlds)

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "K": self.K,
            "camtoworld": self.camtoworlds[idx],
            "viewmat": self.viewmats[idx],
            "image": self.images[idx],
            "image_id": idx,
        }

    def get_vol_gt(self):
        raise RuntimeError("Distillation dataset does not provide a ground-truth volume.")


class DistillTrainer(DefaultTrainer):
    """Trainer that distills a student Gaussian model from a binary-tree teacher."""

    def __init__(self, cfg: Any, device: str):
        self.cfg = cfg
        self.device = device

        self.model = None
        self.teacher_model = None
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
            model_name="StudentGaussianModel",
            strategy_name="DistillStrategy",
            loss_name="LossComputer",
        )
        self._build_distillation_trainset()
        self._validate_components()
        self._init_tv_geometry()
        self._init_wandb()

    def build_training(
        self,
        *,
        model_name: str = "StudentGaussianModel",
        strategy_name: Optional[str] = "DistillStrategy",
        loss_name: str = "LossComputer",
    ) -> None:
        model_scale_meta = self._resolve_model_scale_meta()
        if not self.cfg.trainer.teacher_path:
            raise ValueError("trainer.teacher_path must point to a teacher checkpoint.")

        self.teacher_model = BinaryTreeGaussianModel(device=self.device, **model_scale_meta)
        self._load_teacher_checkpoint(self.cfg.trainer.teacher_path)
        self.teacher_model.eval()

        if model_name != "StudentGaussianModel":
            raise ValueError(f"Unknown model name: {model_name}")
        self.model = StudentGaussianModel(device=self.device, **model_scale_meta)
        self.model.initialize_from_teacher(
            self.teacher_model,
            selector=self.cfg.trainer.slice_selector,
            target_ratio=self.cfg.trainer.slice_target_ratio,
            min_target_count=self.cfg.trainer.slice_min_target_count,
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

        if strategy_name != "DistillStrategy":
            raise ValueError(f"Unknown strategy type: {strategy_name}")
        self.strategy = DistillStrategy(self.cfg.strategy)
        self.strategy_state = self.strategy.initialize_state(scene_scale=self.scene_scale)

        if loss_name != "LossComputer":
            raise ValueError(f"Unknown loss name: {loss_name}")
        self.loss_computer = LossComputer(
            w_l1=self.cfg.trainer.proj_l1_weight,
            w_ssim=self.cfg.trainer.proj_ssim_weight,
            w_tv=self.cfg.trainer.vol_tv_weight,
            ssim_window=self.cfg.trainer.ssim_window,
        )
        self.strategy.check_sanity(self.model, self.optimizers)

    def _load_teacher_checkpoint(self, ckpt_path: str) -> None:
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        model_state = ckpt["model"]
        history_state = ckpt.get("history", {})

        self.teacher_model.means = torch.nn.Parameter(
            model_state["means"].detach().to(self.device).clone()
        )
        self.teacher_model.scales = torch.nn.Parameter(
            model_state["scales"].detach().to(self.device).clone()
        )
        self.teacher_model.quats = torch.nn.Parameter(
            model_state["quats"].detach().to(self.device).clone()
        )
        self.teacher_model.opacities = torch.nn.Parameter(
            model_state["opacities"].detach().to(self.device).clone()
        )

        self.teacher_model.trainable_param_keys = ["means", "scales", "quats", "opacities"]
        self.teacher_model.parent = history_state.get(
            "parent",
            torch.full(
                (self.teacher_model.num_gaussians,),
                -1,
                dtype=torch.int32,
                device=self.device,
            ),
        ).to(self.device, dtype=torch.int32)
        self.teacher_model.level = history_state.get("level")
        self.teacher_model.history_means = history_state.get(
            "history_means", torch.empty((0, 3), device=self.device)
        ).to(self.device)
        self.teacher_model.history_scales = history_state.get(
            "history_scales", torch.empty((0, 3), device=self.device)
        ).to(self.device)
        self.teacher_model.history_quats = history_state.get(
            "history_quats", torch.empty((0, 4), device=self.device)
        ).to(self.device)
        self.teacher_model.history_opacities = history_state.get(
            "history_opacities", torch.empty((0,), device=self.device)
        ).to(self.device)
        self.teacher_model.history_parent = history_state.get(
            "history_parent",
            torch.empty((0,), dtype=torch.int32, device=self.device),
        ).to(self.device, dtype=torch.int32)
        self.teacher_model.history_level = history_state.get("history_level")

    def _build_distillation_trainset(self) -> None:
        image_source = str(
            getattr(self.cfg.trainer, "distill_image_source", "teacher")
        ).lower()

        if image_source not in {"teacher", "dataset"}:
            raise ValueError(
                "trainer.distill_image_source must be one of {'teacher', 'dataset'}, "
                f"got {self.cfg.trainer.distill_image_source!r}."
            )

        if image_source == "dataset":
            self.trainset = self._train_geometry_dataset
            print(
                "Distillation image source: dataset "
                f"({len(self.trainset)} original training projections)."
            )
            self._print_student_slice_info()
            self._release_teacher_model()
            return

        projection_count = int(self.cfg.trainer.teacher_projection_count)
        render_batch_size = int(self.cfg.trainer.teacher_render_batch_size)
        teacher_images, camtoworlds, intrinsics = self._render_teacher_projections(
            projection_count=projection_count,
            render_batch_size=render_batch_size,
        )
        self.trainset = DistillProjectionDataset(
            images=teacher_images,
            camtoworlds=camtoworlds,
            K=intrinsics,
        )
        print(
            "Distillation image source: teacher "
            f"({len(self.trainset)} rendered projections)."
        )

        self._print_student_slice_info()
        self._release_teacher_model()

    def _print_student_slice_info(self) -> None:
        slice_info = getattr(self.model, "slice_info", None)
        if slice_info is not None:
            print(
                "Student slice selection: "
                f"selector={slice_info.selector}, "
                f"teacher_active={slice_info.teacher_active_count}, "
                f"teacher_total={slice_info.teacher_total_count}, "
                f"target={slice_info.target_count}, "
                f"selected={slice_info.selected_count}, "
                f"depth={slice_info.slice_depth}"
            )

    def _release_teacher_model(self) -> None:
        if self.teacher_model is None:
            return
        self.teacher_model = self.teacher_model.cpu()
        del self.teacher_model
        self.teacher_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
    def _render_teacher_projections(
        self,
        *,
        projection_count: int,
        render_batch_size: int,
    ):
        width = int(self.parser.scanner["nDetector"][1])
        height = int(self.parser.scanner["nDetector"][0])
        camtoworlds = self._build_teacher_camtoworlds(projection_count)
        intrinsics = self._train_geometry_dataset.K.float()
        images = []
        progress = tqdm(
            range(0, projection_count, render_batch_size),
            desc="Rendering teacher projections",
        )
        for start in progress:
            end = min(start + render_batch_size, projection_count)
            batch_c2w = camtoworlds[start:end].to(self.device)
            batch_K = intrinsics.unsqueeze(0).repeat(end - start, 1, 1).to(self.device)
            renders_hw, _ = self.teacher_model.rasterize(
                camtoworlds=batch_c2w,
                Ks=batch_K,
                width=width,
                height=height,
                packed=self.cfg.trainer.packed,
                sparse_grad=False,
                absgrad=False,
            )
            images.append(self._format_renders(renders_hw).cpu())
        return torch.cat(images, dim=0), camtoworlds.float(), intrinsics

    def _build_teacher_camtoworlds(self, projection_count: int) -> torch.Tensor:
        sampling_mode = str(
            getattr(self.cfg.trainer, "teacher_view_sampling", "ct")
        ).lower()

        if sampling_mode == "ct":
            angles = np.linspace(0.0, 2.0 * np.pi, num=projection_count, endpoint=False)
            return torch.stack(
                [
                    self._train_geometry_dataset._get_camtoworld(float(angle))
                    for angle in angles
                ],
                dim=0,
            )

        if sampling_mode == "free":
            return self._build_free_sphere_camtoworlds(projection_count)

        raise ValueError(
            "trainer.teacher_view_sampling must be one of {'ct', 'free'}, "
            f"got {self.cfg.trainer.teacher_view_sampling!r}."
        )

    def _build_free_sphere_camtoworlds(self, projection_count: int) -> torch.Tensor:
        radius = float(self.parser.scanner["DSO"])
        target = self._get_distill_target_center()
        positions = self._sample_points_on_sphere(
            count=projection_count,
            radius=radius,
            center=target,
        )
        return torch.stack(
            [self._look_at_camtoworld(position, target) for position in positions],
            dim=0,
        )

    def _get_distill_target_center(self) -> torch.Tensor:
        bbox = self.parser.get_bbox()
        if bbox is None:
            return torch.zeros(3, dtype=torch.float32)

        bbox_tensor = torch.as_tensor(bbox, dtype=torch.float32)
        return bbox_tensor.mean(dim=0)

    def _sample_points_on_sphere(
        self,
        *,
        count: int,
        radius: float,
        center: torch.Tensor,
    ) -> torch.Tensor:
        if count <= 0:
            raise ValueError(f"projection_count must be positive, got {count}.")

        indices = torch.arange(count, dtype=torch.float32)
        phi = (1.0 + np.sqrt(5.0)) / 2.0
        theta = 2.0 * np.pi * indices / phi
        z = 1.0 - (2.0 * indices + 1.0) / count
        xy_radius = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
        x = xy_radius * torch.cos(theta)
        y = xy_radius * torch.sin(theta)
        directions = torch.stack([x, y, z], dim=-1)
        return center.unsqueeze(0) + directions * radius

    def _look_at_camtoworld(
        self,
        position: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        forward = target - position
        forward = forward / torch.linalg.norm(forward)

        world_up = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)
        if torch.abs(torch.dot(forward, world_up)) > 0.99:
            world_up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)

        right = torch.cross(world_up, forward, dim=0)
        right = right / torch.linalg.norm(right)
        up = torch.cross(forward, right, dim=0)
        up = up / torch.linalg.norm(up)

        transform = torch.eye(4, dtype=torch.float32)
        transform[:3, 0] = right
        transform[:3, 1] = up
        transform[:3, 2] = forward
        transform[:3, 3] = position
        return transform

    def _load_datasets(self) -> None:
        self.parser = CTImageParser(self.cfg.exp.data_dir)
        self._train_geometry_dataset = CTImageDataset(self.parser, split="train")
        self.trainset = self._train_geometry_dataset

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
