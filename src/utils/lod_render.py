"""Render LoD (Level-of-Detail) models and save comparison images locally.

Usage:
    python -m src.utils.lod_render \
        --lod_dir  output/synthetic_dataset/cone_ntrain_25_angle_360_lod/0_chest_cone \
        --teacher  output/synthetic_dataset/cone_ntrain_25_angle_360/0_chest_cone/ckpts/ckpt_29999.pt \
        --data     /path/to/scene_data \
        --out      output/lod_render_results \
        --ckpt_step 29999 \
        --view_indices 0 25 50 75

    Initial, untrained LoD checkpoints are loaded from the teacher scene's lod/
    directory by default, so each level is rendered as init_lod_* and final_lod_*.
    Model settings are restored from the nearest ancestor config_used.yaml
    saved by Hydra for each checkpoint, including nested lod_*/ckpts/ outputs.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data.ctproj import CTImageDataset, CTImageParser
from src.models.gaussian_model import GaussianModel
from src.models.binary_tree_gaussian_model import BinaryTreeGaussianModel
from src.utils.checkpoint_config import load_model_kwargs
from src.utils.metric import metric_proj, metric_vol


def parse_level(name: str, prefix: str) -> int | None:
    if not name.startswith(prefix):
        return None
    try:
        return int(name.removeprefix(prefix))
    except ValueError:
        return None


def discover_lod_levels(lod_dir: Path, ckpt_step: int) -> list[int]:
    if not lod_dir.is_dir():
        return []

    levels = []
    for lod_path in lod_dir.iterdir():
        level = parse_level(lod_path.name, "lod_")
        ckpt_path = lod_path / "ckpts" / f"point_cloud_{ckpt_step}.pickle"
        if level is not None and ckpt_path.exists():
            levels.append(level)
    return sorted(levels)


def infer_initial_lod_dir(teacher_path: str | None) -> Path | None:
    if not teacher_path:
        return None
    return Path(teacher_path).parent.parent / "lod"


def parse_initial_lod_level(path: str) -> int | None:
    return parse_level(Path(path).name.split("_step_", maxsplit=1)[0], "ckpt_lod_")


def initial_lod_sort_key(path: Path) -> tuple[int, str]:
    try:
        step = int(path.stem.rsplit("_step_", maxsplit=1)[1])
    except (IndexError, ValueError):
        step = 10**18
    return step, str(path)


def discover_initial_lod_levels(initial_lod_dir: Path | None) -> list[int]:
    if not initial_lod_dir or not initial_lod_dir.is_dir():
        return []
    levels = {
        level
        for level in (
            parse_initial_lod_level(path)
            for path in initial_lod_dir.glob("ckpt_lod_*_step_*.pt")
        )
        if level is not None
    }
    return sorted(levels)


def resolve_initial_lod_checkpoint(initial_lod_dir: Path | None, level: int) -> Path | None:
    if not initial_lod_dir:
        return None
    matches = sorted(
        initial_lod_dir.glob(f"ckpt_lod_{level}_step_*.pt"),
        key=initial_lod_sort_key,
    )
    return matches[0] if matches else None


# ------------------------------------------------------------------
# Model loading
# ------------------------------------------------------------------

def load_lod_model(ckpt_path: str, device: torch.device) -> GaussianModel:
    model_kwargs = load_model_kwargs(ckpt_path)
    model = GaussianModel(device=str(device), **model_kwargs)
    model.load_ply(ckpt_path)
    # Pickle exports may contain scale bounds; they must describe the same raw
    # scales as the training config. Older exports may omit this metadata.
    saved_bounds = model._get_scale_bound()
    expected_bounds = (
        np.asarray([model_kwargs["min_scale"], model_kwargs["max_scale"]])
        if model_kwargs["min_scale"] is not None else None
    )
    if saved_bounds is not None and (
        expected_bounds is None
        or np.shape(saved_bounds) != expected_bounds.shape
        or not np.allclose(saved_bounds, expected_bounds, rtol=1e-6, atol=1e-12)
    ):
        raise ValueError(
            f"Checkpoint scale bounds disagree with its Hydra config: {ckpt_path} "
            f"(checkpoint={saved_bounds}, config={expected_bounds})"
        )
    model._configure_scale_bounds(
        min_scale=model_kwargs["min_scale"],
        max_scale=model_kwargs["max_scale"],
        scale_eps=model_kwargs["scale_eps"],
    )
    model.eval()
    return model


def load_lod_checkpoint_model(ckpt_path: str, device: torch.device) -> GaussianModel:
    model_kwargs = load_model_kwargs(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = ckpt["model"]
    model = GaussianModel(device=str(device), **model_kwargs)
    for field in ("means", "scales", "quats", "opacities"):
        setattr(model, field, torch.nn.Parameter(state[field].detach().to(device).clone()))
    model.eval()
    return model


def load_teacher_model(ckpt_path: str, device: torch.device) -> BinaryTreeGaussianModel:
    model_kwargs = load_model_kwargs(ckpt_path)
    model = BinaryTreeGaussianModel(device=str(device), **model_kwargs)
    model.load_ckpt(ckpt_path)
    model.eval()
    return model


# ------------------------------------------------------------------
# Rendering utilities
# ------------------------------------------------------------------

@torch.no_grad()
def render_projection(model: GaussianModel, sample: dict) -> tuple[np.ndarray, np.ndarray]:
    c2w = sample["camtoworld"].float().unsqueeze(0).to(model.device)
    K = sample["K"].float().unsqueeze(0).to(model.device)
    image = sample["image"].float()
    width, height = int(image.shape[1]), int(image.shape[0])

    renders_hw, _ = model.rasterize(
        camtoworlds=c2w, Ks=K,
        width=width, height=height,
        packed=False, sparse_grad=False, absgrad=False,
    )
    render = renders_hw.permute(0, 2, 1, 3).contiguous()
    pred = render[0, :, :, 0].cpu().numpy()
    gt = image[:, :, 0].cpu().numpy() if image.ndim == 3 else image.cpu().numpy()
    return pred, gt


@torch.no_grad()
def voxelize_model(model: GaussianModel, volume_shape, bbox) -> np.ndarray | None:
    vol = model.voxelize(volume_shape, bbox.to(model.device) if bbox is not None else None)
    if vol is None:
        return None
    return vol.squeeze().cpu().numpy()


# ------------------------------------------------------------------
# Image saving helpers
# ------------------------------------------------------------------

def save_single_projection(pred, path, model_name, num_gaussians):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(pred, cmap="gray", vmin=pred.min(), vmax=pred.max())
    ax.set_title(f"{model_name} (N={num_gaussians})")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def volume_to_numpy(vol) -> np.ndarray:
    if torch.is_tensor(vol):
        return vol.detach().cpu().numpy()
    return np.asarray(vol)


def align_model_volume_to_gt(vol: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred = vol.transpose(2, 1, 0) if vol.ndim == 3 else vol
    slices = tuple(slice(0, min(p, g)) for p, g in zip(pred.shape, gt.shape))
    return pred[slices], gt[slices]


def center_slice(vol: np.ndarray, axis: int) -> np.ndarray:
    index = vol.shape[axis] // 2
    if axis == 0:
        return vol[index, :, :]
    if axis == 1:
        return vol[:, index, :]
    return vol[:, :, index]


def ordered_volume_comparison_names(volume_cache: Mapping[str, object]) -> list[str]:
    levels = sorted(
        {
            level
            for name in volume_cache
            for level in (
                parse_level(name, "init_lod_"),
                parse_level(name, "final_lod_"),
            )
            if level is not None
        }
    )
    names = []
    for level in levels:
        names.extend(
            name
            for name in (f"init_lod_{level}", f"final_lod_{level}")
            if name in volume_cache
        )
    if "teacher_leaf" in volume_cache:
        names.append("teacher_leaf")
    return names


def display_limits_from_volumes(
    vol_gt,
    volume_paths: dict[str, Path],
) -> tuple[float | None, float | None]:
    if vol_gt is not None:
        gt = volume_to_numpy(vol_gt)
        vmin, vmax = float(np.nanmin(gt)), float(np.nanmax(gt))
    else:
        mins, maxs = [], []
        for path in volume_paths.values():
            vol = np.load(path, mmap_mode="r")
            mins.append(float(np.nanmin(vol)))
            maxs.append(float(np.nanmax(vol)))
        if not mins:
            return None, None
        vmin, vmax = min(mins), max(maxs)

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return None, None
    return vmin, vmax


def load_volume_for_plot(path: Path) -> np.ndarray:
    return np.load(path, mmap_mode="r")


def save_volume_slice_comparison(
    volume_paths: dict[str, Path],
    vol_gt,
    model_counts: dict[str, int],
    path,
    axis: int,
    axis_name: str,
) -> bool:
    if vol_gt is None:
        return False

    gt = volume_to_numpy(vol_gt)
    names = [
        name
        for name in ordered_volume_comparison_names(volume_paths)
        if volume_paths.get(name) is not None
    ]
    if not names:
        return False

    rows = len(names)
    fig = plt.figure(figsize=(10, 4 * rows))
    gs = fig.add_gridspec(rows, 2)
    vmin, vmax = gt.min(), gt.max()

    for row, name in enumerate(names):
        pred_volume = load_volume_for_plot(volume_paths[name])
        pred, aligned_gt = align_model_volume_to_gt(pred_volume, gt)
        pred_slice = center_slice(pred, axis)
        gt_slice = center_slice(aligned_gt, axis)
        title = "Leaf Model" if name == "teacher_leaf" else name
        count = model_counts.get(name)
        count_suffix = f" (N={count})" if count is not None else ""

        pred_ax = fig.add_subplot(gs[row, 0])
        pred_ax.imshow(pred_slice.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        pred_ax.set_title(f"{title}{count_suffix}")
        pred_ax.axis("off")

        gt_ax = fig.add_subplot(gs[row, 1])
        gt_ax.imshow(gt_slice.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        gt_ax.set_title("GT")
        gt_ax.axis("off")

    fig.suptitle(f"Volume Slice Comparison - {axis_name.upper()} Axis", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def save_volume_slices(
    vol,
    output_prefix,
    vmin: float | None = None,
    vmax: float | None = None,
) -> list[str]:
    centers = [s // 2 for s in vol.shape]
    slices = [vol[centers[0], :, :], vol[:, centers[1], :], vol[:, :, centers[2]]]

    saved_paths = []
    for axis_name, s in zip(("x", "y", "z"), slices):
        path = f"{output_prefix}_{axis_name}.png"
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.imshow(s.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        saved_paths.append(path)

    return saved_paths


def save_volume_slices_from_npy(
    npy_path: Path,
    output_prefix,
    vmin: float | None = None,
    vmax: float | None = None,
) -> list[str]:
    vol = load_volume_for_plot(npy_path)
    return save_volume_slices(vol, output_prefix, vmin=vmin, vmax=vmax)


def collect_metric_curve_names(results, metric_key: str) -> dict[str, list[str]]:
    return {
        "Initial LoD": sorted(
            (
                name for name in results
                if name.startswith("init_lod_") and metric_key in results[name]
            ),
            key=lambda name: int(name.removeprefix("init_lod_")),
        ),
        "Final LoD": sorted(
            (
                name for name in results
                if name.startswith("final_lod_") and metric_key in results[name]
            ),
            key=lambda name: int(name.removeprefix("final_lod_")),
        ),
    }


def save_metric_curve(results, grouped_names, path, metric_key, ylabel, title, marker) -> bool:
    if not any(grouped_names.values()):
        print(f"No {metric_key} metrics available; skipping curve.")
        return False

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    for label, names in grouped_names.items():
        if not names:
            continue
        n_gaussians = [results[n]["num_gaussians"] for n in names]
        values = [results[n][metric_key] for n in names]

        ax.plot(n_gaussians, values, marker, markersize=8, linewidth=2, label=label)
        for i, name in enumerate(names):
            ax.annotate(
                name,
                (n_gaussians[i], values[i]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=8,
            )

    ax.set_xlabel("Number of Gaussians")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xscale("log", base=2)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


METRIC_CURVE_SPECS = (
    ("psnr2d", "PSNR2D (dB)", "2D PSNR vs. Gaussian Count", "o-", "lod_metrics_psnr2d.png"),
    ("ssim2d", "SSIM2D", "2D SSIM vs. Gaussian Count", "s-", "lod_metrics_ssim2d.png"),
    ("psnr3d", "PSNR3D (dB)", "3D PSNR vs. Gaussian Count", "o-", "lod_metrics_psnr3d.png"),
    ("ssim3d", "SSIM3D", "3D SSIM vs. Gaussian Count", "s-", "lod_metrics_ssim3d.png"),
)


def save_metrics_curves(results, metrics_dir) -> list:
    saved_paths = []
    for metric_key, ylabel, title, marker, fname in METRIC_CURVE_SPECS:
        grouped_names = collect_metric_curve_names(results, metric_key)
        if not any(grouped_names.values()):
            continue
        path = metrics_dir / fname
        if save_metric_curve(results, grouped_names, path, metric_key, ylabel, title, marker):
            saved_paths.append(path)
    return saved_paths


def save_metrics_table(results, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "num_gaussians",
                "psnr2d",
                "ssim2d",
                "psnr3d",
                "ssim3d",
            ],
        )
        writer.writeheader()
        for name, r in results.items():
            writer.writerow(
                {
                    "model": name,
                    "num_gaussians": r["num_gaussians"],
                    "psnr2d": f"{r['psnr2d']:.4f}" if "psnr2d" in r else "",
                    "ssim2d": f"{r['ssim2d']:.6f}" if "ssim2d" in r else "",
                    "psnr3d": f"{r['psnr3d']:.4f}" if "psnr3d" in r else "",
                    "ssim3d": f"{r['ssim3d']:.6f}" if "ssim3d" in r else "",
                }
            )
    print(f"Metrics CSV saved to {path}")


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

@torch.no_grad()
def evaluate_model(model, vol, vol_gt=None, ssim_window=11, dataset=None):
    result = {
        "num_gaussians": model.num_gaussians,
    }

    if dataset is not None and len(dataset) > 0:
        try:
            psnr_vals: list[float] = []
            ssim_vals: list[float] = []
            for idx in range(len(dataset)):
                pred, gt = render_projection(model, dataset[idx])
                m2d = metric_proj(
                    torch.from_numpy(pred).unsqueeze(0),
                    torch.from_numpy(gt).unsqueeze(0),
                    window_size=ssim_window,
                    data_range=1.0,
                )
                psnr_vals.append(float(m2d["psnr"].item()))
                ssim_vals.append(float(m2d["ssim"].item()))
            if psnr_vals:
                result["psnr2d"] = float(np.mean(psnr_vals))
                result["ssim2d"] = float(np.mean(ssim_vals))
        except Exception as e:
            print(f"  2D metrics failed: {e}")
    else:
        print("  2D metrics skipped: no test dataset provided")

    if vol_gt is None:
        print("  3D metrics skipped: ground-truth volume is not available")
        return result

    if vol is None:
        print("  3D metrics skipped: voxelization failed")
        return result

    try:
        gt_vol = vol_gt.cpu() if torch.is_tensor(vol_gt) else torch.from_numpy(vol_gt)
        pred_vol = torch.from_numpy(vol).permute(2, 1, 0).contiguous()
        if pred_vol.shape != gt_vol.shape:
            slices = tuple(
                slice(0, min(p, g))
                for p, g in zip(pred_vol.shape, gt_vol.shape)
            )
            pred_vol = pred_vol[slices]
            gt_vol = gt_vol[slices]
        m3d = metric_vol(pred_vol, gt_vol, window_size=ssim_window)
        result["psnr3d"] = m3d["psnr"].item()
        result["ssim3d"] = m3d["ssim"].item()
    except Exception as e:
        print(f"  3D metrics failed: {e}")

    return result


def evaluate_models_only(models, volume_shape, bbox, vol_gt, dataset, metrics_dir):
    """Recompute complete metrics, keeping only one model volume in memory."""
    results = {}
    for name, model in models.items():
        print(f"  Evaluating {name}...")
        vol = voxelize_model(model, volume_shape, bbox)
        metrics = evaluate_model(model, vol, vol_gt=vol_gt, dataset=dataset)
        del vol
        required = ("psnr2d", "ssim2d", "psnr3d", "ssim3d")
        missing = [key for key in required if key not in metrics or np.isnan(metrics[key])]
        if missing:
            raise RuntimeError(
                f"Incomplete evaluation for {name}: {', '.join(missing)}. "
                "Existing metrics CSV has not been replaced."
            )
        results[name] = metrics
        print(
            f"    N={metrics['num_gaussians']}, "
            f"PSNR3D={metrics['psnr3d']:.4f}, SSIM3D={metrics['ssim3d']:.6f}"
        )

    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    save_metrics_table(results, metrics_dir / "lod_metrics.csv")
    return results


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Render and evaluate LoD models")
    parser.add_argument("--lod_dir", type=str, required=True,
                        help="LoD training output dir (contains lod_*/ subdirs)")
    parser.add_argument("--teacher", type=str, default=None,
                        help="Teacher checkpoint (.pt) for leaf-node comparison")
    parser.add_argument("--data", type=str, required=True,
                        help="Scene data directory")
    parser.add_argument("--out", type=str, required=True,
                        help="Output directory for rendered images")
    parser.add_argument("--ckpt_step", type=int, default=29999,
                        help="Checkpoint step number (default: 29999)")
    parser.add_argument("--lod_levels", type=int, nargs="+", default=None,
                        help="LoD levels to load; default: auto-discover existing lod_* dirs")
    parser.add_argument("--init_lod_dir", type=str, default=None,
                        help="Initial LoD checkpoint dir; default: infer from --teacher")
    parser.add_argument("--view_indices", type=int, nargs="+", default=[0],
                        help="Test view indices to render")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Skip quantitative evaluation")
    parser.add_argument("--eval_only", action="store_true",
                        help="Recompute 2D/3D metrics CSV without saving images or volumes")
    args = parser.parse_args()
    if args.eval_only and args.skip_eval:
        parser.error("--eval_only and --skip_eval cannot be used together")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    lod_dir = Path(args.lod_dir)
    out_dir = Path(args.out)
    teacher_path = Path(args.teacher) if args.teacher else None
    print(f"Device: {device}")

    # --- Load scene data ---
    print("\nLoading scene data...")
    scene_parser = CTImageParser(args.data)
    test_dataset = CTImageDataset(scene_parser, split="test")
    bbox = (
        torch.tensor(scene_parser.bbox, dtype=torch.float32)
        if scene_parser.bbox is not None else None
    )

    vol_gt = None
    try:
        vol_gt = test_dataset.get_vol_gt()
        volume_shape = tuple(vol_gt.shape)
    except Exception:
        volume_shape = tuple(scene_parser.meta.get("nVoxel", [128, 128, 128]))
    print(f"  Test views: {len(test_dataset)}, Volume: {volume_shape}")

    # --- Load models ---
    print("\nLoading LoD models...")
    lod_models: dict[str, GaussianModel] = {}
    initial_lod_dir = Path(args.init_lod_dir) if args.init_lod_dir else infer_initial_lod_dir(args.teacher)
    final_levels = discover_lod_levels(lod_dir, args.ckpt_step)
    initial_levels = discover_initial_lod_levels(initial_lod_dir)
    lod_levels = args.lod_levels or sorted(set(final_levels) | set(initial_levels))
    print(f"  LoD levels: {lod_levels}")
    print(f"  Initial LoD dir: {initial_lod_dir or 'not available'}")

    for level in lod_levels:
        init_ckpt_path = resolve_initial_lod_checkpoint(initial_lod_dir, level)
        if init_ckpt_path and init_ckpt_path.exists():
            name = f"init_lod_{level}"
            lod_models[name] = load_lod_checkpoint_model(str(init_ckpt_path), device)
            print(f"  {name}: {lod_models[name].num_gaussians} Gaussians ({init_ckpt_path})")
        else:
            print(f"  init_lod_{level}: not found")

        ckpt_path = lod_dir / f"lod_{level}" / "ckpts" / f"point_cloud_{args.ckpt_step}.pickle"
        if ckpt_path.exists():
            name = f"final_lod_{level}"
            lod_models[name] = load_lod_model(str(ckpt_path), device)
            print(f"  {name}: {lod_models[name].num_gaussians} Gaussians ({ckpt_path})")
        else:
            print(f"  final_lod_{level}: not found ({ckpt_path})")

    if teacher_path and teacher_path.exists():
        teacher = load_teacher_model(str(teacher_path), device)
        lod_models["teacher_leaf"] = teacher
        print(f"  teacher_leaf: {teacher.num_gaussians} Gaussians")

    if not lod_models:
        print("No models loaded. Exiting.")
        return

    print(f"\nLoaded models: {list(lod_models.keys())}")

    if args.eval_only:
        evaluate_models_only(
            lod_models, volume_shape, bbox, vol_gt, test_dataset, out_dir / "metrics",
        )
        print(f"Done. Metrics saved to {out_dir / 'metrics' / 'lod_metrics.csv'}")
        return

    # --- Create output dirs ---
    proj_dir = out_dir / "projections"
    compare_dir = out_dir / "comparisons"
    vol_dir = out_dir / "volumes"
    metrics_dir = out_dir / "metrics"
    for d in [proj_dir, compare_dir, vol_dir, metrics_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # --- 1. Per-model per-view projection rendering ---
    print("\n--- Rendering projections ---")
    for vi in args.view_indices:
        if vi < 0 or vi >= len(test_dataset):
            print(f"  Skipping invalid view index {vi}")
            continue

        sample = test_dataset[vi]

        for name, model in lod_models.items():
            pred, _ = render_projection(model, sample)
            out_path = proj_dir / f"{name}_view_{vi:03d}.png"
            save_single_projection(pred, out_path, name, model.num_gaussians)
            print(f"  Saved {out_path}")

    # --- 2. Volume slices ---
    print("\n--- Rendering volume slices ---")
    volume_cache = {}
    volume_paths: dict[str, Path] = {}
    for name, model in lod_models.items():
        vol = voxelize_model(model, volume_shape, bbox)
        volume_cache[name] = vol
        if vol is not None:
            npy_path = vol_dir / f"{name}_volume.npy"
            np.save(npy_path, vol)
            volume_paths[name] = npy_path
            print(f"  Saved {npy_path}")
        else:
            print(f"  {name}: voxelization failed")

    display_vmin, display_vmax = display_limits_from_volumes(vol_gt, volume_paths)
    if display_vmin is not None and display_vmax is not None:
        print(f"  Volume display range: [{display_vmin:.6g}, {display_vmax:.6g}]")
    else:
        print("  Volume display range: auto")

    for name, npy_path in volume_paths.items():
        output_prefix = vol_dir / f"{name}_volume_slice"
        slice_paths = save_volume_slices_from_npy(
            npy_path,
            output_prefix,
            vmin=display_vmin,
            vmax=display_vmax,
        )
        for slice_path in slice_paths:
            print(f"  Saved {slice_path}")

    print("\n--- Rendering volume comparisons ---")
    model_counts = {name: model.num_gaussians for name, model in lod_models.items()}
    for axis, axis_name in enumerate(("z", "y", "x")):
        compare_path = compare_dir / f"all_levels_volume_slice_{axis_name}.png"
        if save_volume_slice_comparison(
            volume_paths, vol_gt, model_counts, compare_path, axis, axis_name
        ):
            print(f"  Saved {compare_path}")
        else:
            print(f"  Skipped {compare_path}: ground-truth or voxelized volumes unavailable")

    # --- 3. Quantitative evaluation ---
    if not args.skip_eval:
        print("\n--- Evaluating models ---")
        results = {}
        for name, model in lod_models.items():
            print(f"  Evaluating {name}...")
            results[name] = evaluate_model(
                model, volume_cache.get(name), vol_gt=vol_gt,
                dataset=test_dataset,
            )
            r = results[name]
            msg = f"    N={r['num_gaussians']}"
            if "psnr2d" in r:
                msg += f", PSNR2D={r['psnr2d']:.2f}, SSIM2D={r['ssim2d']:.4f}"
            if "psnr3d" in r:
                msg += f", PSNR3D={r['psnr3d']:.2f}, SSIM3D={r['ssim3d']:.4f}"
            print(msg)

        for curve_path in save_metrics_curves(results, metrics_dir):
            print(f"  Saved {curve_path}")

        table_path = metrics_dir / "lod_metrics.csv"
        save_metrics_table(results, table_path)

    print(f"\nDone. All results saved to {args.out}")


if __name__ == "__main__":
    main()
