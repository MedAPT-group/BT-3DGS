from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from src.models.gaussian_model import GaussianModel

from .math import normalized_quat_to_rotmat


def compute_relocation(
    opacities: Tensor,
    scales: Tensor,
    ratios: Tensor,
    binoms: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Pure-PyTorch implementation of the MCMC relocation equation."""
    n = opacities.shape[0]
    n_max = int(binoms.shape[0])
    if scales.shape != (n, 3):
        raise ValueError(f"Expected scales {(n, 3)}, got {tuple(scales.shape)}")
    if ratios.shape != (n,):
        raise ValueError(f"Expected ratios {(n,)}, got {tuple(ratios.shape)}")
    if binoms.shape != (n_max, n_max):
        raise ValueError("binoms must be a square matrix")

    ratios = ratios.clamp(min=1, max=n_max).to(dtype=torch.long)
    new_opacities = 1.0 - torch.pow(
        1.0 - opacities, 1.0 / ratios.to(dtype=opacities.dtype)
    )
    denom_sum = torch.zeros_like(opacities)
    max_ratio = int(ratios.max().item()) if ratios.numel() else 0
    for i in range(1, max_ratio + 1):
        active = (ratios >= i).to(dtype=opacities.dtype)
        for k in range(i):
            coefficient = binoms[i - 1, k].to(
                device=opacities.device, dtype=opacities.dtype
            )
            term = ((-1.0) ** k) / ((k + 1.0) ** 0.5)
            denom_sum.add_(
                active * coefficient * term * new_opacities.pow(k + 1)
            )
    new_scales = scales * (opacities / denom_sum).unsqueeze(-1)
    return new_opacities, new_scales


@torch.no_grad()
def _multinomial_sample(weights: Tensor, n: int, replacement: bool = True) -> Tensor:
    num_elements = weights.size(0)
    if num_elements <= 2**24:
        return torch.multinomial(weights, n, replacement=replacement)
    weights = weights / weights.sum()
    sampled_idxs_np = np.random.choice(
        num_elements, size=n, p=weights.detach().cpu().numpy(), replace=replacement
    )
    return torch.from_numpy(sampled_idxs_np).to(weights.device)


@torch.no_grad()
def _update_param_with_optimizer(
    param_fn: Callable[[str, Tensor], Tensor],
    optimizer_fn: Callable[[str, Tensor], Tensor],
    model: torch.nn.Module,
    optimizers: Dict[str, torch.optim.Optimizer],
    names: Optional[List[str]] = None,
):
    """Apply *param_fn* / *optimizer_fn* to model parameters and their optimizer states.

    Reads each named parameter from *model* via ``getattr``, transforms it
    with *param_fn*, writes the result back via ``setattr``, and updates the
    corresponding optimizer state with *optimizer_fn*.

    Args:
        param_fn: ``(name, old_param) -> new_param`` transform.
        optimizer_fn: ``(state_key, state_value) -> new_state_value`` transform.
        model: The Gaussian model whose attributes are modified in-place.
        optimizers: Per-parameter optimizers keyed by parameter name.
        names: Parameter names to process.  Defaults to ``list(optimizers.keys())``.
    """
    if names is None:
        names = list(optimizers.keys())

    for name in names:
        param = getattr(model, name)
        new_param = param_fn(name, param)
        setattr(model, name, new_param)
        if name not in optimizers:
            assert not param.requires_grad, (
                f"Optimizer for {name} is not found, but parameter is trainable "
                f"(requires_grad={param.requires_grad})."
            )
            continue
        optimizer = optimizers[name]
        for i in range(len(optimizer.param_groups)):
            param_state = optimizer.state[param]
            del optimizer.state[param]
            for key in list(param_state.keys()):
                if key != "step":
                    param_state[key] = optimizer_fn(key, param_state[key])
            optimizer.param_groups[i]["params"] = [new_param]
            optimizer.state[new_param] = param_state


@torch.no_grad()
def duplicate(
    model: torch.nn.Module,
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, Tensor],
    mask: Tensor,
):
    device = mask.device
    sel = torch.where(mask)[0]

    def param_fn(name: str, p: Tensor) -> Tensor:
        return torch.nn.Parameter(torch.cat([p, p[sel]]), requires_grad=p.requires_grad)

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        return torch.cat([v, torch.zeros((len(sel), *v.shape[1:]), device=device)])

    _update_param_with_optimizer(param_fn, optimizer_fn, model, optimizers)
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = torch.cat((v, v[sel]))


@torch.no_grad()
def split(
    model: torch.nn.Module,
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, Tensor],
    mask: Tensor,
    revised_opacity: bool = False,
):
    device = mask.device
    sel = torch.where(mask)[0]
    rest = torch.where(~mask)[0]

    scales = model.activated_scales[sel]
    quats = F.normalize(model.quats[sel], dim=-1)
    rotmats = normalized_quat_to_rotmat(quats)
    samples = torch.einsum(
        "nij,nj,bnj->bni",
        rotmats,
        scales,
        torch.randn(2, len(scales), 3, device=device),
    )

    def param_fn(name: str, p: Tensor) -> Tensor:
        repeats = [2] + [1] * (p.dim() - 1)
        if name == "means":
            p_split = (p[sel] + samples).reshape(-1, 3)
        elif name == "scales":
            target_scales = (scales / 1.6).clamp_min(1e-12)
            p_split = model.inverse_activate_scale(target_scales).repeat(2, 1)
        elif name == "opacities" and revised_opacity:
            base = GaussianModel.activate_opacity(p[sel]).clamp(max=0.9999)
            new_opacities = 1.0 - torch.sqrt(1.0 - base)
            p_split = torch.log(torch.expm1(new_opacities)).repeat(repeats)
        else:
            p_split = p[sel].repeat(repeats)
        p_new = torch.cat([p[rest], p_split])
        return torch.nn.Parameter(p_new, requires_grad=p.requires_grad)

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        return torch.cat([v[rest], torch.zeros((2 * len(sel), *v.shape[1:]), device=device)])

    _update_param_with_optimizer(param_fn, optimizer_fn, model, optimizers)
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            repeats = [2] + [1] * (v.dim() - 1)
            state[k] = torch.cat((v[rest], v[sel].repeat(repeats)))


@torch.no_grad()
def remove(
    model: torch.nn.Module,
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, Tensor],
    mask: Tensor,
):
    sel = torch.where(~mask)[0]

    def param_fn(name: str, p: Tensor) -> Tensor:
        return torch.nn.Parameter(p[sel], requires_grad=p.requires_grad)

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        return v[sel]

    _update_param_with_optimizer(param_fn, optimizer_fn, model, optimizers)
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v[sel]


@torch.no_grad()
def reset_opa(
    model: torch.nn.Module,
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, Tensor],
    value: float,
):
    def param_fn(name: str, p: Tensor) -> Tensor:
        if name != "opacities":
            raise ValueError(f"Unexpected parameter name: {name}")
        inv_sp = torch.log(torch.expm1(torch.tensor(value, device=p.device)))
        opacities = torch.clamp(p, max=inv_sp.item())
        return torch.nn.Parameter(opacities, requires_grad=p.requires_grad)

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        return torch.zeros_like(v)

    _update_param_with_optimizer(param_fn, optimizer_fn, model, optimizers, names=["opacities"])


@torch.no_grad()
def relocate(
    model: torch.nn.Module,
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, Tensor],
    mask: Tensor,
    binoms: Tensor,
    min_opacity: float = 0.005,
):
    opacities = GaussianModel.activate_opacity(model.opacities)
    dead_indices = mask.nonzero(as_tuple=True)[0]
    alive_indices = (~mask).nonzero(as_tuple=True)[0]
    n = len(dead_indices)
    probs = opacities[alive_indices].flatten()
    sampled_idxs = alive_indices[_multinomial_sample(probs, n, replacement=True)]
    new_opacities, new_scales = compute_relocation(
        opacities=opacities[sampled_idxs],
        scales=model.activated_scales[sampled_idxs],
        ratios=torch.bincount(sampled_idxs)[sampled_idxs] + 1,
        binoms=binoms,
    )
    eps = torch.finfo(torch.float32).eps
    new_opacities = torch.clamp(new_opacities, max=1.0 - eps, min=min_opacity)

    def param_fn(name: str, p: Tensor) -> Tensor:
        if name == "opacities":
            p[sampled_idxs] = torch.log(torch.expm1(new_opacities))
        elif name == "scales":
            p[sampled_idxs] = model.inverse_activate_scale(new_scales)
        p[dead_indices] = p[sampled_idxs]
        return torch.nn.Parameter(p, requires_grad=p.requires_grad)

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        v[sampled_idxs] = 0
        return v

    _update_param_with_optimizer(param_fn, optimizer_fn, model, optimizers)
    for _, v in state.items():
        if isinstance(v, torch.Tensor):
            v[sampled_idxs] = 0


@torch.no_grad()
def sample_add(
    model: torch.nn.Module,
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, Tensor],
    n: int,
    binoms: Tensor,
    min_opacity: float = 0.005,
):
    opacities = GaussianModel.activate_opacity(model.opacities)
    sampled_idxs = _multinomial_sample(opacities.flatten(), n, replacement=True)
    new_opacities, new_scales = compute_relocation(
        opacities=opacities[sampled_idxs],
        scales=model.activated_scales[sampled_idxs],
        ratios=torch.bincount(sampled_idxs)[sampled_idxs] + 1,
        binoms=binoms,
    )
    eps = torch.finfo(torch.float32).eps
    new_opacities = torch.clamp(new_opacities, max=1.0 - eps, min=min_opacity)

    def param_fn(name: str, p: Tensor) -> Tensor:
        if name == "opacities":
            p[sampled_idxs] = torch.log(torch.expm1(new_opacities))
        elif name == "scales":
            p[sampled_idxs] = model.inverse_activate_scale(new_scales)
        return torch.nn.Parameter(torch.cat([p, p[sampled_idxs]]), requires_grad=p.requires_grad)

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        return torch.cat([v, torch.zeros((len(sampled_idxs), *v.shape[1:]), device=v.device)])

    _update_param_with_optimizer(param_fn, optimizer_fn, model, optimizers)
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = torch.cat(
                [v, torch.zeros((len(sampled_idxs), *v.shape[1:]), device=v.device)]
            )


@torch.no_grad()
def inject_noise_to_position(
    model: torch.nn.Module,
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, Tensor],
    scaler: float,
):
    opacities_sp = GaussianModel.activate_opacity(model.opacities.flatten())
    op_prob = opacities_sp.clamp(max=1.0)
    scales = model.activated_scales
    rotations = normalized_quat_to_rotmat(F.normalize(model.quats, dim=-1))
    covars = (
        rotations
        @ torch.diag_embed(scales.square())
        @ rotations.transpose(-1, -2)
    )

    def atten_fn(x, k=100, x0=0.995):
        return 1 / (1 + torch.exp(-k * (x - x0)))

    attenuation = atten_fn(1 - op_prob).unsqueeze(-1)
    noise = torch.randn_like(model.means) * attenuation * scaler
    noise = torch.einsum("bij,bj->bi", covars, noise)
    model.means.add_(noise)
