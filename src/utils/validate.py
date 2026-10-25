"""
Validation utilities for CT 3DGS models.
"""

import csv
from dataclasses import dataclass
import json
import os
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
import torch
from tqdm import tqdm

from src.utils.checkpoint_config import load_checkpoint_config

try:
    import imageio.v2 as imageio
except Exception:
    import imageio


@dataclass
class ValidatorConfig:
    """Configuration for validation."""

    data_dir: str
    checkpoint_path: str
    result_dir: Optional[str] = None
    eval_save_images_n: int = 100
    ssim_window: int = 11
    num_workers: int = 0
    pin_memory: bool = True
    scale_min: Optional[float] = None
    scale_max: Optional[float] = None
    scale_eps: float = 1e-5
    volume_shape: Optional[Tuple[int, int, int]] = None
    seed: int = 42


class Validator:
    """Validator class for evaluating trained 3D Gaussian models."""

    def __init__(self, cfg: ValidatorConfig, device: str):
        from src.data.ctproj import CTImageDataset, CTImageParser
        from src.models.gaussian_model import GaussianModel
        from src.utils.metric import metric_proj, metric_vol

        self._CTProjParser = CTImageParser
        self._CTProjDataset = CTImageDataset
        self._GaussianModel = GaussianModel
        self._metric_proj = metric_proj
        self._metric_vol = metric_vol

        self.cfg = cfg
        self.device = device

        if cfg.result_dir is None:
            cfg.result_dir = os.path.join(cfg.data_dir, "validation_results")

        os.makedirs(cfg.result_dir, exist_ok=True)
        print(f"📁 Results will be saved to: {cfg.result_dir}")

        print("\n[1/4] Loading metadata...")
        self.parser = self._CTProjParser(cfg.data_dir)
        print(f"✅ Metadata loaded from: {cfg.data_dir}")

        print("\n[2/4] Creating datasets...")
        self.trainset = self._CTProjDataset(self.parser, split="train")
        print(f"   Train projections: {len(self.trainset)}")
        try:
            self.testset = self._CTProjDataset(self.parser, split="test")
            print(f"   Test projections: {len(self.testset)}")
        except Exception as e:
            self.testset = None
            print(f"   Test set not available: {e}")

        print("\n[3/4] Loading ground truth volume...")
        self.vol_gt = None
        try:
            self.vol_gt = self.trainset.get_vol_gt().to(device)
            print(f"✅ GT volume loaded: shape={tuple(self.vol_gt.shape)} (W,D,H)")
        except Exception as e:
            print(f"⚠️  GT volume not available: {e}")

        print("\n[4/4] Setting up voxelization parameters...")
        if self.vol_gt is not None:
            self.volume_shape_wdh = tuple(self.vol_gt.shape)
        else:
            self.volume_shape_wdh = tuple(self.parser.meta.get("nVoxel", cfg.volume_shape))
        print(f"   Volume shape (W,D,H): {self.volume_shape_wdh}")

        if getattr(self.parser, "bbox", None) is not None:
            self.bbox = torch.tensor(self.parser.bbox, dtype=torch.float32, device=device)
            print(f"   BBox min: {self.bbox[0].cpu().numpy()}")
            print(f"   BBox max: {self.bbox[1].cpu().numpy()}")
        else:
            self.bbox = None
            print("   BBox: Not available")

        print("\n[Loading] Loading Gaussian parameters from pickle file...")
        print(f"   Path: {cfg.checkpoint_path}")
        training_model_cfg = self._load_training_model_config(cfg.checkpoint_path)
        model_kwargs = {
            key: training_model_cfg[key]
            for key in (
                "eps2d",
                "eps3d",
                "img_tile_size",
                "vol_tile_size",
                "img_radius_clip",
                "vol_radius_clip",
                "backward_mode",
            )
            if key in training_model_cfg
        }
        resolved_scale_min = (
            cfg.scale_min
            if cfg.scale_min is not None
            else training_model_cfg.get("scale_min")
        )
        resolved_scale_max = (
            cfg.scale_max
            if cfg.scale_max is not None
            else training_model_cfg.get("scale_max")
        )
        resolved_scale_eps = float(
            training_model_cfg.get("scale_eps", cfg.scale_eps)
        )
        if (resolved_scale_min is None) ^ (resolved_scale_max is None):
            raise ValueError(
                "Validation scale_min and scale_max must both be set or both be null."
            )
        if resolved_scale_min is not None:
            model_kwargs.update(
                min_scale=resolved_scale_min,
                max_scale=resolved_scale_max,
                scale_eps=resolved_scale_eps,
            )

        self.model = self._GaussianModel(device=device, **model_kwargs)
        self.model.load_ply(cfg.checkpoint_path)
        # Older pickle exports may not contain scale-bound metadata.  Explicit
        # CLI values and the adjacent training config remain authoritative.
        if resolved_scale_min is not None:
            self.model._configure_scale_bounds(
                min_scale=resolved_scale_min,
                max_scale=resolved_scale_max,
                scale_eps=resolved_scale_eps,
            )

        print("\n✅ Successfully loaded Gaussian model:")
        print(f"   Number of Gaussians: {self.model.num_gaussians}")
        print(f"   Means shape: {tuple(self.model.means.shape)}")
        print(f"   Scales shape: {tuple(self.model.scales.shape)}")
        print(f"   Quats shape: {tuple(self.model.quats.shape)}")
        print(f"   Opacities shape: {tuple(self.model.opacities.shape)}")
        if self.model._use_sigmoid_scale:
            print(
                f"   Scale bounds: [{float(self.model._scale_min.item()):.6f}, "
                f"{float(self.model._scale_max.item()):.6f}]"
            )
        else:
            print("   Scale activation: exp (no bounds)")
        print(
            "   Operator config: "
            f"eps2d={self.model.eps2d:g}, eps3d={self.model.eps3d:g}, "
            f"img_tile={self.model.img_tile_size}, vol_tile={self.model.vol_tile_size}, "
            f"backward={self.model.backward_mode}"
        )

    @staticmethod
    def _load_training_model_config(checkpoint_path: str) -> dict:
        """Load model/operator settings saved beside a standard run checkpoint."""
        try:
            config_path, saved_cfg = load_checkpoint_config(checkpoint_path)
        except FileNotFoundError:
            return {}
        model_cfg = saved_cfg.get("model")
        if model_cfg is None:
            return {}
        resolved = OmegaConf.to_container(model_cfg, resolve=True)
        if not isinstance(resolved, dict):
            return {}
        print(f"   Operator config loaded from: {config_path}")
        return resolved

    @torch.no_grad()
    def _evaluate_on_dataset(self, dataset, split_name: str):
        cfg = self.cfg
        device = self.device
        eval_dir = os.path.join(cfg.result_dir, split_name)
        proj_dir = os.path.join(eval_dir, "projections")
        stats_dir = os.path.join(eval_dir, "projection_stats")
        os.makedirs(proj_dir, exist_ok=True)
        os.makedirs(stats_dir, exist_ok=True)

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
        )

        proj_l1_list = []
        proj_psnr_list = []
        proj_ssim_list = []
        pred_stats_list = []
        gt_stats_list = []
        gaussian_state = self.model.prepare_gaussians()

        print(f"\n   Rendering {len(dataset)} projections...")
        for i, batch in enumerate(tqdm(loader, desc=f"   Evaluating {split_name}")):
            camtoworlds = batch["camtoworld"].to(device)
            viewmats = batch["viewmat"].to(device)
            Ks = batch["K"].to(device)
            gt = batch["image"].to(device)
            width, height = gt.shape[1:3]

            pred_hw, _ = self.model.rasterize(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=height,
                height=width,
                packed=False,
                sparse_grad=False,
                absgrad=False,
                viewmats=viewmats,
                gaussian_state=gaussian_state,
            )
            pred = pred_hw.permute(0, 2, 1, 3).contiguous()

            metric_vals = self._metric_proj(
                pred.squeeze(-1),
                gt.squeeze(-1),
                window_size=cfg.ssim_window,
                data_range=1.0,
            )
            proj_l1_list.append(float(metric_vals["l1"].item()))
            proj_psnr_list.append(float(metric_vals["psnr"].item()))
            proj_ssim_list.append(float(metric_vals["ssim"].item()))

            pred_np = pred[0, ..., 0].cpu().numpy()
            gt_np = gt[0, ..., 0].cpu().numpy()
            pred_stats = {
                "projection_id": i,
                "min": float(np.min(pred_np)),
                "max": float(np.max(pred_np)),
                "mean": float(np.mean(pred_np)),
                "median": float(np.median(pred_np)),
                "q25": float(np.percentile(pred_np, 25)),
                "q75": float(np.percentile(pred_np, 75)),
                "std": float(np.std(pred_np)),
            }
            gt_stats = {
                "projection_id": i,
                "min": float(np.min(gt_np)),
                "max": float(np.max(gt_np)),
                "mean": float(np.mean(gt_np)),
                "median": float(np.median(gt_np)),
                "q25": float(np.percentile(gt_np, 25)),
                "q75": float(np.percentile(gt_np, 75)),
                "std": float(np.std(gt_np)),
            }
            pred_stats_list.append(pred_stats)
            gt_stats_list.append(gt_stats)

            if i < cfg.eval_save_images_n:
                diff_np = pred_np - gt_np
                vmin_global = min(gt_np.min(), pred_np.min())
                vmax_global = max(gt_np.max(), pred_np.max())
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                im0 = axes[0].imshow(gt_np, cmap="viridis", vmin=vmin_global, vmax=vmax_global)
                axes[0].set_title("Ground Truth", fontsize=12, fontweight="bold")
                axes[0].axis("off")
                plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

                im1 = axes[1].imshow(pred_np, cmap="viridis", vmin=vmin_global, vmax=vmax_global)
                axes[1].set_title("Prediction", fontsize=12, fontweight="bold")
                axes[1].axis("off")
                plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

                diff_absmax = max(abs(diff_np.min()), abs(diff_np.max()))
                im2 = axes[2].imshow(diff_np, cmap="viridis", vmin=-diff_absmax, vmax=diff_absmax)
                axes[2].set_title("Difference (Pred - GT)", fontsize=12, fontweight="bold")
                axes[2].axis("off")
                plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

                plt.tight_layout()
                plt.savefig(
                    os.path.join(proj_dir, f"proj_{i:04d}_gt_pred_diff.png"),
                    dpi=100,
                    bbox_inches="tight",
                )
                plt.close()

        pred_stats_path = os.path.join(stats_dir, "predicted_projections_stats.csv")
        with open(pred_stats_path, "w", newline="") as f:
            if pred_stats_list:
                fieldnames = pred_stats_list[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(pred_stats_list)

        gt_stats_path = os.path.join(stats_dir, "ground_truth_projections_stats.csv")
        with open(gt_stats_path, "w", newline="") as f:
            if gt_stats_list:
                fieldnames = gt_stats_list[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(gt_stats_list)

        if pred_stats_list:
            all_pred_mins = [s["min"] for s in pred_stats_list]
            all_pred_maxs = [s["max"] for s in pred_stats_list]
            all_pred_means = [s["mean"] for s in pred_stats_list]
            all_pred_medians = [s["median"] for s in pred_stats_list]
            all_gt_mins = [s["min"] for s in gt_stats_list]
            all_gt_maxs = [s["max"] for s in gt_stats_list]
            all_gt_means = [s["mean"] for s in gt_stats_list]
            all_gt_medians = [s["median"] for s in gt_stats_list]
            overall_stats = {
                "predicted": {
                    "global_min": float(np.min(all_pred_mins)),
                    "global_max": float(np.max(all_pred_maxs)),
                    "mean_of_means": float(np.mean(all_pred_means)),
                    "median_of_medians": float(np.median(all_pred_medians)),
                    "num_projections": len(pred_stats_list),
                },
                "ground_truth": {
                    "global_min": float(np.min(all_gt_mins)),
                    "global_max": float(np.max(all_gt_maxs)),
                    "mean_of_means": float(np.mean(all_gt_means)),
                    "median_of_medians": float(np.median(all_gt_medians)),
                    "num_projections": len(gt_stats_list),
                },
            }
            overall_stats_path = os.path.join(stats_dir, "overall_statistics.json")
            with open(overall_stats_path, "w") as f:
                json.dump(overall_stats, f, indent=2)

        proj_l1_mean = float(sum(proj_l1_list) / max(1, len(proj_l1_list)))
        proj_psnr_mean = float(sum(proj_psnr_list) / max(1, len(proj_psnr_list)))
        proj_ssim_mean = float(sum(proj_ssim_list) / max(1, len(proj_ssim_list)))
        metrics = {
            "split": split_name,
            "proj_l1_mean": proj_l1_mean,
            "proj_psnr_mean": proj_psnr_mean,
            "proj_ssim_mean": proj_ssim_mean,
            "num_projections": len(proj_l1_list),
        }
        with open(os.path.join(eval_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"\n   ✅ {split_name.capitalize()} evaluation completed:")
        print(f"      L1: {proj_l1_mean:.6f}")
        print(f"      PSNR: {proj_psnr_mean:.2f} dB")
        print(f"      SSIM: {proj_ssim_mean:.6f}")
        print(
            f"      Saved {min(len(proj_l1_list), cfg.eval_save_images_n)} comparison images to: {proj_dir}"
        )
        return metrics

    @torch.no_grad()
    def _voxelize_and_evaluate(self):
        if self.vol_gt is None:
            print("\n⚠️  Ground truth volume not available, skipping voxelization evaluation")
            return {}

        print("\n[Voxelization] Generating 3D volume...")
        params_before = {
            "opacities_sum": self.model.opacities.sum().item(),
            "opacities_mean": self.model.opacities.mean().item(),
        }
        pred_vol = self.model.voxelize(self.volume_shape_wdh, self.bbox)
        params_after = {
            "opacities_sum": self.model.opacities.sum().item(),
            "opacities_mean": self.model.opacities.mean().item(),
        }
        diff_sum = abs(params_after["opacities_sum"] - params_before["opacities_sum"])
        diff_mean = abs(params_after["opacities_mean"] - params_before["opacities_mean"])
        if diff_sum > 1e-6 or diff_mean > 1e-9:
            print("   ⚠️  [VALIDATION] Voxelization modified opacities!")
            print(
                f"       Before: sum={params_before['opacities_sum']:.6f}, "
                f"mean={params_before['opacities_mean']:.9f}"
            )
            print(
                f"       After:  sum={params_after['opacities_sum']:.6f}, "
                f"mean={params_after['opacities_mean']:.9f}"
            )
            print(f"       Diff:   sum={diff_sum:.6f}, mean={diff_mean:.9f}")

        if pred_vol is None:
            print("❌ Voxelization failed (returned None)")
            return {}

        if pred_vol.ndim >= 3:
            pred_vol = pred_vol.squeeze()
        if pred_vol.ndim == 3:
            pred_vol = pred_vol.permute(2, 1, 0).contiguous()

        print(f"   Predicted volume shape: {tuple(pred_vol.shape)} (W,D,H)")
        print(f"   GT volume shape: {tuple(self.vol_gt.shape)} (W,D,H)")

        gt_vol = self.vol_gt.to(self.device)
        if pred_vol.shape != gt_vol.shape:
            minW = min(pred_vol.shape[0], gt_vol.shape[0])
            minD = min(pred_vol.shape[1], gt_vol.shape[1])
            minH = min(pred_vol.shape[2], gt_vol.shape[2])
            pred_vol = pred_vol[:minW, :minD, :minH]
            gt_vol = gt_vol[:minW, :minD, :minH]
            print(f"   Aligned shape: {tuple(pred_vol.shape)}")

        vol_metric_vals = self._metric_vol(
            pred_vol,
            gt_vol,
            window_size=self.cfg.ssim_window,
        )
        l1v = float(vol_metric_vals["l1"].item())
        mse = float(vol_metric_vals["mse"].item())
        psnr = float(vol_metric_vals["psnr"].item())
        ssim = float(vol_metric_vals["ssim"].item())
        data_range = float((gt_vol.max() - gt_vol.min()).clamp_min(1e-6).item())
        vol_metrics = {
            "vol_l1": l1v,
            "vol_mse": mse,
            "vol_psnr": psnr,
            "vol_ssim": ssim,
            "data_range": data_range,
        }

        vol_dir = os.path.join(self.cfg.result_dir, "volume")
        os.makedirs(vol_dir, exist_ok=True)
        width, depth, height = pred_vol.shape
        mid = (width // 2, depth // 2, height // 2)

        def extract_slice(arr, axis, idx):
            if axis == 0:
                return arr[idx, :, :].cpu().numpy()
            if axis == 1:
                return arr[:, idx, :].cpu().numpy()
            return arr[:, :, idx].cpu().numpy()

        slices_gt = [extract_slice(gt_vol, 0, mid[0]), extract_slice(gt_vol, 1, mid[1]), extract_slice(gt_vol, 2, mid[2])]
        slices_pred = [
            extract_slice(pred_vol, 0, mid[0]),
            extract_slice(pred_vol, 1, mid[1]),
            extract_slice(pred_vol, 2, mid[2]),
        ]
        slices_diff = [slices_pred[i] - slices_gt[i] for i in range(3)]
        vmin_global = min(gt_vol.min().item(), pred_vol.min().item())
        vmax_global = max(gt_vol.max().item(), pred_vol.max().item())
        diff_absmax = max([abs(s.min()) for s in slices_diff] + [abs(s.max()) for s in slices_diff])

        fig, axes = plt.subplots(3, 3, figsize=(18, 16))
        axis_names = ["W-axis (Sagittal)", "D-axis (Coronal)", "H-axis (Axial)"]
        row_names = ["Ground Truth", "Prediction", "Difference (Pred - GT)"]
        for col_idx in range(3):
            im0 = axes[0, col_idx].imshow(
                slices_gt[col_idx], cmap="viridis", vmin=vmin_global, vmax=vmax_global
            )
            axes[0, col_idx].set_title(axis_names[col_idx], fontsize=12, fontweight="bold")
            axes[0, col_idx].axis("off")
            plt.colorbar(im0, ax=axes[0, col_idx], fraction=0.046, pad=0.04)

            im1 = axes[1, col_idx].imshow(
                slices_pred[col_idx], cmap="viridis", vmin=vmin_global, vmax=vmax_global
            )
            axes[1, col_idx].axis("off")
            plt.colorbar(im1, ax=axes[1, col_idx], fraction=0.046, pad=0.04)

            im2 = axes[2, col_idx].imshow(
                slices_diff[col_idx], cmap="viridis", vmin=-diff_absmax, vmax=diff_absmax
            )
            axes[2, col_idx].axis("off")
            plt.colorbar(im2, ax=axes[2, col_idx], fraction=0.046, pad=0.04)

        for row_idx, row_name in enumerate(row_names):
            axes[row_idx, 0].set_ylabel(row_name, fontsize=14, fontweight="bold", rotation=90, labelpad=20)

        fig.suptitle(f"Volume Reconstruction (PSNR: {psnr:.2f} dB)", fontsize=16, fontweight="bold", y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(os.path.join(vol_dir, "volume_3x3_comparison.png"), dpi=150, bbox_inches="tight")
        plt.close()

        def save_individual_slice(arr, axis, idx, name):
            if axis == 0:
                img = arr[idx, :, :].cpu().numpy()
            elif axis == 1:
                img = arr[:, idx, :].cpu().numpy()
            else:
                img = arr[:, :, idx].cpu().numpy()
            m, M = img.min(), img.max()
            img_n = (img - m) / (M - m) if M - m > 1e-8 else img * 0
            imageio.imwrite(os.path.join(vol_dir, name), (img_n * 255).astype("uint8"))

        save_individual_slice(pred_vol, 0, mid[0], "pred_slice_w.png")
        save_individual_slice(pred_vol, 1, mid[1], "pred_slice_d.png")
        save_individual_slice(pred_vol, 2, mid[2], "pred_slice_h.png")
        save_individual_slice(gt_vol, 0, mid[0], "gt_slice_w.png")
        save_individual_slice(gt_vol, 1, mid[1], "gt_slice_d.png")
        save_individual_slice(gt_vol, 2, mid[2], "gt_slice_h.png")

        with open(os.path.join(vol_dir, "volume_metrics.json"), "w") as f:
            json.dump(vol_metrics, f, indent=2)

        print("\n   ✅ Volume evaluation completed:")
        print(f"      L1: {l1v:.6f}")
        print(f"      MSE: {mse:.6f}")
        print(f"      PSNR: {psnr:.2f} dB")
        print(f"      SSIM: {ssim:.6f}")
        print(f"      Saved 3×3 volume comparison to: {vol_dir}/volume_3x3_comparison.png")
        print(f"      Saved individual slices to: {vol_dir}")
        return vol_metrics

    def validate(self):
        print("\n" + "=" * 80)
        print("STARTING VALIDATION")
        print("=" * 80)
        print("\n[Step 1/3] Evaluating on training set...")
        train_metrics = self._evaluate_on_dataset(self.trainset, "train")

        test_metrics = {}
        if self.testset is not None:
            print("\n[Step 2/3] Evaluating on test set...")
            test_metrics = self._evaluate_on_dataset(self.testset, "test")
        else:
            print("\n[Step 2/3] Test set not available, skipping...")

        print("\n[Step 3/3] Voxelization and volume evaluation...")
        vol_metrics = self._voxelize_and_evaluate()

        final_report = {
            "checkpoint": self.cfg.checkpoint_path,
            "data_dir": self.cfg.data_dir,
            "num_gaussians": self.model.num_gaussians,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "volume_metrics": vol_metrics,
        }
        report_path = os.path.join(self.cfg.result_dir, "validation_report.json")
        with open(report_path, "w") as f:
            json.dump(final_report, f, indent=2)

        print("\n" + "=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)
        print("\n📊 Model Information:")
        print(f"   Checkpoint: {self.cfg.checkpoint_path}")
        print(f"   Number of Gaussians: {self.model.num_gaussians}")
        print("\n📈 Projection Metrics:")
        print(f"   Train L1:   {train_metrics['proj_l1_mean']:.6f}")
        print(f"   Train PSNR: {train_metrics['proj_psnr_mean']:.2f} dB")
        print(f"   Train SSIM: {train_metrics['proj_ssim_mean']:.6f}")
        if test_metrics:
            print(f"   Test L1:    {test_metrics['proj_l1_mean']:.6f}")
            print(f"   Test PSNR:  {test_metrics['proj_psnr_mean']:.2f} dB")
            print(f"   Test SSIM:  {test_metrics['proj_ssim_mean']:.6f}")
        if vol_metrics:
            print("\n🧊 Volume Metrics:")
            print(f"   Volume L1:   {vol_metrics['vol_l1']:.6f}")
            print(f"   Volume MSE:  {vol_metrics['vol_mse']:.6f}")
            print(f"   Volume PSNR: {vol_metrics['vol_psnr']:.2f} dB")
            print(f"   Volume SSIM: {vol_metrics['vol_ssim']:.6f}")
        print("\n📁 Results Directory:")
        print(f"   {self.cfg.result_dir}")
        print("\n   - train/projections/          (projection comparison images with viridis colormap)")
        print("   - train/projection_stats/     (detailed statistics)")
        if test_metrics:
            print("   - test/projections/           (projection comparison images with viridis colormap)")
            print("   - test/projection_stats/      (detailed statistics)")
        if vol_metrics:
            print(
                "   - volume/volume_3x3_comparison.png  "
                "(3×3 grid: GT/Pred/Diff × W/D/H axes, PSNR annotated)"
            )
            print("   - volume/*.png                (individual slices)")
        print("   - validation_report.json      (comprehensive report)")
        print("\n" + "=" * 80)
        print("✅ VALIDATION COMPLETED SUCCESSFULLY!")
        print("=" * 80 + "\n")
        return final_report
