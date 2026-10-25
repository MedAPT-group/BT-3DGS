"""Level-of-Detail trainer: distil multiple LoD student models from a binary-tree teacher."""

from __future__ import annotations

import glob
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from src.data.ctproj import CTImageDataset, CTImageParser
from src.models.binary_tree_gaussian_model import BinaryTreeGaussianModel
from src.models.student_gaussian_model import StudentGaussianModel
from src.strategies import DistillStrategy
from src.trainers.default import DefaultTrainer
from src.trainers.distill import DistillProjectionDataset
from src.utils.loss import LossComputer


LOD_LEVELS: List[int] = []


class LoDTrainer(DefaultTrainer):
    """Train six LoD student models (2^10 .. 2^15 Gaussians) from one teacher."""

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

        self._load_teacher()

    # ------------------------------------------------------------------
    # Teacher loading (shared across all LoD levels)
    # ------------------------------------------------------------------

    def _load_teacher(self) -> None:
        model_scale_meta = self._resolve_model_scale_meta()
        if not self.cfg.trainer.teacher_path:
            raise ValueError("trainer.teacher_path must point to a teacher checkpoint.")
        self.teacher_model = BinaryTreeGaussianModel(device=self.device, **model_scale_meta)
        self._load_teacher_checkpoint(self.cfg.trainer.teacher_path)
        self.teacher_model.eval()
        self._teacher_scale_meta = model_scale_meta
        self._lod_checkpoint_dir = self._infer_lod_checkpoint_dir(
            self.cfg.trainer.teacher_path
        )

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
        self.teacher_model.trainable_param_keys = [
            "means", "scales", "quats", "opacities",
        ]
        self.teacher_model.parent = history_state.get(
            "parent",
            torch.full(
                (self.teacher_model.num_gaussians,), -1,
                dtype=torch.int32, device=self.device,
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

    # ------------------------------------------------------------------
    # Distillation data (rendered once, shared across levels)
    # ------------------------------------------------------------------

    def _build_shared_trainset(self) -> None:
        image_source = str(
            getattr(self.cfg.trainer, "distill_image_source", "teacher")
        ).lower()

        if image_source not in {"teacher", "dataset"}:
            raise ValueError(
                "trainer.distill_image_source must be one of {'teacher', 'dataset'}, "
                f"got {self.cfg.trainer.distill_image_source!r}."
            )

        if image_source == "dataset":
            self._shared_trainset = self._train_geometry_dataset
            print(
                f"LoD distillation image source: dataset "
                f"({len(self._shared_trainset)} projections)."
            )
            return

        projection_count = int(self.cfg.trainer.teacher_projection_count)
        render_batch_size = int(self.cfg.trainer.teacher_render_batch_size)
        teacher_images, camtoworlds, intrinsics = self._render_teacher_projections(
            projection_count=projection_count,
            render_batch_size=render_batch_size,
        )
        self._shared_trainset = DistillProjectionDataset(
            images=teacher_images,
            camtoworlds=camtoworlds,
            K=intrinsics,
        )
        print(
            f"LoD distillation image source: teacher "
            f"({len(self._shared_trainset)} rendered projections)."
        )

    @torch.no_grad()
    def _render_teacher_projections(
        self, *, projection_count: int, render_batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        width = int(self.parser.scanner["nDetector"][1])
        height = int(self.parser.scanner["nDetector"][0])
        camtoworlds = self._build_teacher_camtoworlds(projection_count)
        intrinsics = self._train_geometry_dataset.K.float()
        images: List[torch.Tensor] = []
        progress = tqdm(
            range(0, projection_count, render_batch_size),
            desc="Rendering teacher projections",
        )
        for start in progress:
            end = min(start + render_batch_size, projection_count)
            batch_c2w = camtoworlds[start:end].to(self.device)
            batch_K = intrinsics.unsqueeze(0).repeat(end - start, 1, 1).to(self.device)
            renders_hw, _ = self.teacher_model.rasterize(
                camtoworlds=batch_c2w, Ks=batch_K,
                width=width, height=height,
                packed=self.cfg.trainer.packed,
                sparse_grad=False, absgrad=False,
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
                    self._train_geometry_dataset._get_camtoworld(float(a))
                    for a in angles
                ],
                dim=0,
            )
        if sampling_mode == "free":
            return self._build_free_sphere_camtoworlds(projection_count)
        raise ValueError(
            "teacher_view_sampling must be one of {'ct', 'free'}, "
            f"got {sampling_mode!r}."
        )

    def _build_free_sphere_camtoworlds(self, projection_count: int) -> torch.Tensor:
        radius = float(self.parser.scanner["DSO"])
        bbox = self.parser.get_bbox()
        target = (
            torch.as_tensor(bbox, dtype=torch.float32).mean(dim=0)
            if bbox is not None
            else torch.zeros(3, dtype=torch.float32)
        )
        indices = torch.arange(projection_count, dtype=torch.float32)
        phi = (1.0 + np.sqrt(5.0)) / 2.0
        theta = 2.0 * np.pi * indices / phi
        z = 1.0 - (2.0 * indices + 1.0) / projection_count
        xy_r = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
        directions = torch.stack([xy_r * torch.cos(theta), xy_r * torch.sin(theta), z], dim=-1)
        positions = target.unsqueeze(0) + directions * radius
        return torch.stack([self._look_at(pos, target) for pos in positions], dim=0)

    @staticmethod
    def _look_at(position: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        forward = target - position
        forward = forward / torch.linalg.norm(forward)
        world_up = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)
        if torch.abs(torch.dot(forward, world_up)) > 0.99:
            world_up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
        right = torch.cross(world_up, forward, dim=0)
        right = right / torch.linalg.norm(right)
        up = torch.cross(forward, right, dim=0)
        up = up / torch.linalg.norm(up)
        t = torch.eye(4, dtype=torch.float32)
        t[:3, 0] = right
        t[:3, 1] = up
        t[:3, 2] = forward
        t[:3, 3] = position
        return t

    # ------------------------------------------------------------------
    # Per-level student initialisation
    # ------------------------------------------------------------------

    def _build_student_for_level(self, target_count: int) -> StudentGaussianModel:
        student = StudentGaussianModel(device=self.device, **self._teacher_scale_meta)
        ckpt_path = self._resolve_lod_checkpoint_path(target_count)
        self._load_student_from_lod_checkpoint(student, ckpt_path, target_count)
        print(
            f"  LoD level target={target_count}, "
            f"student initialized={student.num_gaussians} Gaussians, "
            f"ckpt={ckpt_path}"
        )
        return student

    @staticmethod
    def _infer_lod_checkpoint_dir(teacher_path: str) -> str:
        teacher_dir = os.path.dirname(teacher_path)
        scene_dir = os.path.dirname(teacher_dir)
        return os.path.join(scene_dir, "lod")

    @staticmethod
    def _lod_checkpoint_sort_key(path: str) -> Tuple[int, str]:
        filename = os.path.basename(path)
        try:
            step = int(filename.rsplit("_step_", maxsplit=1)[1].removesuffix(".pt"))
        except (IndexError, ValueError):
            step = 10**18
        return step, path

    @staticmethod
    def _parse_lod_threshold(path: str) -> Optional[int]:
        filename = os.path.basename(path)
        try:
            return int(
                filename.split("ckpt_lod_", maxsplit=1)[1].split("_step_", maxsplit=1)[0]
            )
        except (IndexError, ValueError):
            return None

    def _discover_lod_levels(self) -> List[int]:
        pattern = os.path.join(self._lod_checkpoint_dir, "ckpt_lod_*_step_*.pt")
        levels = {
            threshold
            for threshold in (
                self._parse_lod_threshold(path) for path in glob.glob(pattern)
            )
            if threshold is not None
        }
        if not levels:
            raise FileNotFoundError(f"No LoD checkpoints found: {pattern}")
        return sorted(levels)

    def _resolve_lod_checkpoint_path(self, target_count: int) -> str:
        pattern = os.path.join(
            self._lod_checkpoint_dir,
            f"ckpt_lod_{target_count}_step_*.pt",
        )
        matches = sorted(glob.glob(pattern), key=self._lod_checkpoint_sort_key)
        if not matches:
            raise FileNotFoundError(
                f"No LoD checkpoint found for target {target_count}: {pattern}"
            )
        return matches[0]

    def _load_student_from_lod_checkpoint(
        self,
        student: StudentGaussianModel,
        ckpt_path: str,
        target_count: int,
    ) -> None:
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        lod_threshold = int(ckpt.get("lod_threshold", target_count))
        if lod_threshold != target_count:
            raise ValueError(
                f"LoD checkpoint threshold mismatch: expected {target_count}, "
                f"got {lod_threshold} in {ckpt_path}"
            )

        model_state = ckpt["model"]
        student.means = torch.nn.Parameter(
            model_state["means"].detach().to(self.device).clone()
        )
        student.scales = torch.nn.Parameter(
            model_state["scales"].detach().to(self.device).clone()
        )
        student.quats = torch.nn.Parameter(
            model_state["quats"].detach().to(self.device).clone()
        )
        student.opacities = torch.nn.Parameter(
            model_state["opacities"].detach().to(self.device).clone()
        )

    def _build_level_training(self, student: StudentGaussianModel) -> None:
        """Wire up optimizers, strategy, loss for a single LoD level."""
        self.model = student
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
        self.strategy = DistillStrategy(self.cfg.strategy)
        self.strategy_state = self.strategy.initialize_state(scene_scale=self.scene_scale)
        self.loss_computer = LossComputer(
            w_l1=self.cfg.trainer.proj_l1_weight,
            w_ssim=self.cfg.trainer.proj_ssim_weight,
            w_tv=self.cfg.trainer.vol_tv_weight,
            ssim_window=self.cfg.trainer.ssim_window,
        )
        self.strategy.check_sanity(self.model, self.optimizers)

    # ------------------------------------------------------------------
    # Per-level output dirs
    # ------------------------------------------------------------------

    def _level_result_dir(self, target_count: int) -> str:
        return os.path.join(self.cfg.exp.result_dir, f"lod_{target_count}")

    def _setup_level_output(self, target_count: int) -> None:
        level_dir = self._level_result_dir(target_count)
        os.makedirs(level_dir, exist_ok=True)
        self.ckpt_dir = os.path.join(level_dir, "ckpts")
        self.ply_dir = os.path.join(level_dir, "ply")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.ply_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Level validation: filter out invalid LoD levels
    # ------------------------------------------------------------------

    def _validate_lod_levels(
        self, lod_levels: List[int],
    ) -> Tuple[List[int], Dict[int, StudentGaussianModel]]:
        """Pre-build students for all levels and discard invalid ones.

        A level is removed when:
        - Its actual Gaussian count is outside [1000, leaf_count].
        - Its actual Gaussian count is not strictly greater than the previous
          valid level's count (i.e. levels must be monotonically increasing).
        """
        leaf_count = self.teacher_model.num_gaussians
        min_count = 1000
        print(f"\nValidating LoD levels (leaf_count={leaf_count}, min_count={min_count})...")

        candidates: List[Tuple[int, StudentGaussianModel]] = []
        for target in lod_levels:
            student = self._build_student_for_level(target)
            n = student.num_gaussians
            candidates.append((target, student))
            print(f"  target={target}, actual={n}")

        valid_levels: List[int] = []
        valid_students: Dict[int, StudentGaussianModel] = {}
        prev_count = 0

        for target, student in candidates:
            n = student.num_gaussians
            if n < min_count:
                print(f"  [SKIP] lod_{target}: actual {n} < min_count {min_count}")
                del student
                continue
            if n > leaf_count:
                print(f"  [SKIP] lod_{target}: actual {n} > leaf_count {leaf_count}")
                del student
                continue
            if n <= prev_count:
                print(
                    f"  [SKIP] lod_{target}: actual {n} <= previous level count {prev_count}"
                )
                del student
                continue

            valid_levels.append(target)
            valid_students[target] = student
            prev_count = n

        removed = set(lod_levels) - set(valid_levels)
        if removed:
            print(f"\nRemoved levels: {sorted(removed)}")
        print(f"Final LoD levels to train: {valid_levels}\n")
        return valid_levels, valid_students

    # ------------------------------------------------------------------
    # Main entry: train all LoD levels
    # ------------------------------------------------------------------

    def train(self) -> None:
        self._limit_threads_if_needed()

        configured_lod_levels = getattr(self.cfg.trainer, "lod_levels", None)
        lod_levels = (
            list(configured_lod_levels)
            if configured_lod_levels
            else self._discover_lod_levels()
        )
        print(f"LoD levels from checkpoints: {lod_levels}")

        self._build_shared_trainset()
        self.trainset = self._shared_trainset

        valid_levels, valid_students = self._validate_lod_levels(lod_levels)

        if not valid_levels:
            print("No valid LoD levels remain after validation. Exiting.")
            self._release_teacher_model()
            return

        for level_idx, target_count in enumerate(valid_levels):
            print(f"\n{'='*60}")
            print(
                f"LoD level {level_idx+1}/{len(valid_levels)}: "
                f"target {target_count} Gaussians"
            )
            print(f"{'='*60}")

            student = valid_students.pop(target_count)
            self._setup_level_output(target_count)
            self._build_level_training(student)
            self._validate_components()
            self._init_tv_geometry()

            self._train_single_level(target_count)

            del self.model, self.optimizers, self.strategy, self.strategy_state
            self.model = None
            self.optimizers = None
            self.strategy = None
            self.strategy_state = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        for leftover in valid_students.values():
            del leftover
        valid_students.clear()

        self._release_teacher_model()
        print(f"\nAll {len(valid_levels)} LoD levels trained successfully.")

    def _train_single_level(self, target_count: int) -> None:
        """Run the standard training loop for a single LoD level."""
        train_loader = self._create_train_loader()
        train_iterator = iter(train_loader)
        bbox_summary = self._format_bbox_summary()
        max_steps = self.cfg.trainer.max_steps

        progress_bar = tqdm(
            range(max_steps),
            desc=f"LoD-{target_count}",
            mininterval=1.0,
            miniters=50,
        )

        for step in progress_bar:
            self._update_learning_rates(step, max_steps)
            raw_batch, train_iterator = self._fetch_next_batch(train_iterator, train_loader)
            batch = self._prepare_train_batch(raw_batch)
            gaussian_state = self.model.prepare_gaussians(explicit=True)
            renders_hw, render_info = self.rasterize_splats(
                batch.camtoworlds, batch.intrinsics,
                batch.image_width, batch.image_height,
                viewmats=batch.viewmats,
                gaussian_state=gaussian_state,
            )
            renders = self._format_renders(renders_hw)
            volume_pred = self._sample_tv_volume(step, gaussian_state=gaussian_state)
            loss_dict = self._compute_loss(batch, renders, volume_pred)

            self.strategy.step_pre_backward(
                model=self.model, optimizers=self.optimizers,
                state=self.strategy_state, step=step, info=render_info,
            )
            loss_dict["total"].backward()
            self.model.backward_explicit(gaussian_state)
            self.strategy.step_post_backward(
                model=self.model, optimizers=self.optimizers,
                state=self.strategy_state, step=step, info=render_info,
                packed=self.cfg.trainer.packed,
            )

            for optimizer in self.optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            self._save_outputs(step)
            if step % 250 == 0 or step + 1 == max_steps:
                progress_bar.set_description(
                    f"LoD-{target_count}|"
                    + self._format_progress(loss_dict, bbox_summary),
                    refresh=False,
                )

    # ------------------------------------------------------------------
    # Teacher cleanup
    # ------------------------------------------------------------------

    def _release_teacher_model(self) -> None:
        if self.teacher_model is None:
            return
        self.teacher_model = self.teacher_model.cpu()
        del self.teacher_model
        self.teacher_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Dataset loading (same as DistillTrainer)
    # ------------------------------------------------------------------

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
                self.valset, batch_size=1, shuffle=False,
                num_workers=self.cfg.trainer.num_workers,
                pin_memory=self.cfg.trainer.pin_memory,
                persistent_workers=(
                    self.cfg.trainer.num_workers > 0
                    and self.cfg.trainer.persistent_workers
                ),
            )
