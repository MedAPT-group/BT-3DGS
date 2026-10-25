from math import exp
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F


_WINDOW_CACHE: Dict[
    Tuple[int, int, str, Optional[int], torch.dtype], torch.Tensor
] = {}


def _gaussian(window_size: int, sigma: float) -> torch.Tensor:
    gauss = torch.tensor(
        [exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2)) for x in range(window_size)]
    )
    return gauss / gauss.sum()


def _create_window(window_size: int, channel: int, device, dtype) -> torch.Tensor:
    device = torch.device(device)
    cache_key = (window_size, channel, device.type, device.index, dtype)
    cached = _WINDOW_CACHE.get(cache_key)
    if cached is not None:
        return cached

    _1d = _gaussian(window_size, 1.5).unsqueeze(1)
    _2d = _1d @ _1d.t()
    window = _2d.float().unsqueeze(0).unsqueeze(0)
    window = window.expand(channel, 1, window_size, window_size).contiguous()
    window = window.to(device=device, dtype=dtype)
    _WINDOW_CACHE[cache_key] = window
    return window


def ssim2d(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """Compute mean SSIM for a batch of images shaped as (N, C, H, W)."""
    assert img1.shape == img2.shape, "Input shapes must match for SSIM"
    _, channel, _, _ = img1.shape
    window = _create_window(window_size, channel, img1.device, img1.dtype)
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    c1 = 0.01**2
    c2 = 0.03**2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean()


def ssim3d(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
) -> torch.Tensor:
    """Approximate 3D SSIM following multi-axis slice averaging.

    Supports input shape:
    - single volume: (W, D, H)
    - batched volume: (N, W, D, H)

    For each sample, compute 2D SSIM on slices from all 3 axes, skip empty
    (all-zero) GT slices, then average per axis and across axes.
    """
    assert pred.shape == target.shape, "Input shapes must match for SSIM3D"
    assert pred.ndim in (3, 4), "Volume tensors must be 3D or 4D"

    if pred.ndim == 3:
        pred_b = pred.unsqueeze(0)
        target_b = target.unsqueeze(0)
    else:
        pred_b = pred
        target_b = target

    sample_scores = []
    for b in range(pred_b.shape[0]):
        pred_v = pred_b[b]
        target_v = target_b[b]
        axis_scores = []
        for axis in (0, 1, 2):
            slice_scores = []
            for i in range(target_v.shape[axis]):
                if axis == 0:
                    target_slice = target_v[i, :, :]
                    pred_slice = pred_v[i, :, :]
                elif axis == 1:
                    target_slice = target_v[:, i, :]
                    pred_slice = pred_v[:, i, :]
                else:
                    target_slice = target_v[:, :, i]
                    pred_slice = pred_v[:, :, i]

                # Keep the same behavior as image_utils: ignore empty GT slices.
                if target_slice.max() > 0:
                    score = ssim2d(
                        pred_slice.unsqueeze(0).unsqueeze(0),
                        target_slice.unsqueeze(0).unsqueeze(0),
                        window_size=window_size,
                    )
                    slice_scores.append(score)

            if slice_scores:
                axis_scores.append(torch.stack(slice_scores).mean())

        if axis_scores:
            sample_scores.append(torch.stack(axis_scores).mean())
        else:
            sample_scores.append(torch.zeros((), device=pred_v.device, dtype=pred_v.dtype))

    return torch.stack(sample_scores).mean()


def psnr_from_mse(
    mse: Union[float, torch.Tensor],
    data_range: float = 1.0,
    eps: float = 1e-13,
) -> torch.Tensor:
    """Compute PSNR from MSE value."""
    mse_tensor = mse if torch.is_tensor(mse) else torch.tensor(float(mse), dtype=torch.float32)
    mse_tensor = torch.clamp(mse_tensor, min=eps)
    max_i_sq = float(data_range) ** 2
    return 10.0 * torch.log10(
        torch.tensor(max_i_sq, device=mse_tensor.device, dtype=mse_tensor.dtype) / mse_tensor
    )


def psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    eps: float = 1e-13,
) -> torch.Tensor:
    """Compute PSNR directly from prediction and target tensors."""
    mse = torch.mean((pred - target) ** 2)
    return psnr_from_mse(mse=mse, data_range=data_range, eps=eps)


def metric_proj(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    data_range: float = 1.0,
    axis: int = 0,
) -> Dict[str, torch.Tensor]:
    """Compute projection metrics for batched images.

    Supports input shape:
    - projections: (N, W, H)
    """
    assert pred.shape == target.shape, "Input shapes must match"

    assert pred.ndim == 3, "image_utils compatibility expects 3D volume input"
    assert axis in (0, 1, 2), "axis must be 0, 1, or 2"

    psnr_list = []
    ssim_list = []
    for i in range(target.shape[axis]):
        if axis == 0:
            target_slice = target[i, :, :]
            pred_slice = pred[i, :, :]
        elif axis == 1:
            target_slice = target[:, i, :]
            pred_slice = pred[:, i, :]
        else:
            target_slice = target[:, :, i]
            pred_slice = pred[:, :, i]

        if target_slice.max() <= 0:
            continue

        target_max = target_slice.max().clamp_min(1e-12)
        pred_max = pred_slice.max().clamp_min(1e-12)
        target_n = target_slice / target_max
        pred_n = pred_slice / pred_max
        target_n = target_n.unsqueeze(0).unsqueeze(0)
        pred_n = pred_n.unsqueeze(0).unsqueeze(0)

        psnr_list.append(psnr(pred_n, target_n, data_range=data_range, eps=0.0))
        ssim_list.append(ssim2d(pred_n, target_n, window_size=window_size))

    if psnr_list:
        psnr_val = torch.stack(psnr_list).mean()
        ssim_val = torch.stack(ssim_list).mean()
    else:
        zero = torch.zeros((), device=pred.device, dtype=pred.dtype)
        psnr_val = zero
        ssim_val = zero

    l1v = torch.abs(pred - target).mean()
    mse = torch.mean((pred - target) ** 2)
    return {
        "l1": l1v,
        "mse": mse,
        "psnr": psnr_val,
        "ssim": ssim_val,
    }


def metric_vol(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    data_range: float = 1.0,
    image_utils_compat: bool = False,
) -> Dict[str, torch.Tensor]:
    """Compute volume metrics for batched volumes.

    Supports input shape:
    - single volume: (W, D, H)
    - batched volume: (N, W, D, H)

    3D SSIM is approximated by averaging 2D SSIM over every axis slices.
    """
    assert pred.shape == target.shape, "Input shapes must match"
    assert pred.ndim in (3, 4), "Volume tensors must be 3D or 4D"

    if pred.ndim == 3:
        pred_b = pred.unsqueeze(0)
        target_b = target.unsqueeze(0)
    else:
        pred_b = pred
        target_b = target

    l1v = torch.abs(pred_b - target_b).mean()
    mse = torch.mean((pred_b - target_b) ** 2)
    if data_range is None:
        # image_utils behavior: pixel_max is img1.max() when pixel_max=None.
        data_range_value = float(target_b.max().clamp_min(1e-12).item())
    else:
        data_range_value = float(data_range)
    psnr_val = psnr_from_mse(mse=mse, data_range=data_range_value, eps=(0.0 if image_utils_compat else 1e-12))

    ssim_val = ssim3d(pred_b, target_b, window_size=window_size)
    return {
        "l1": l1v,
        "mse": mse,
        "psnr": psnr_val,
        "ssim": ssim_val,
    }
