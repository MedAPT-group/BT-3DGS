from typing import Any, Dict

import torch


class Strategy:
    """Base class for Gaussian densification strategies."""

    def __init__(self, cfg: Any):
        pass

    def check_sanity(
        self,
        model: torch.nn.Module,
        optimizers: Dict[str, torch.optim.Optimizer],
    ):
        params = model.param_dict()
        trainable_params = {name for name, param in params.items() if param.requires_grad}
        assert trainable_params == set(optimizers.keys()), (
            "trainable parameters and optimizers must have the same keys, "
            f"but got {trainable_params} and {optimizers.keys()}"
        )
        for optimizer in optimizers.values():
            assert len(optimizer.param_groups) == 1, (
                "Each optimizer must have exactly one param_group, "
                f"but got {len(optimizer.param_groups)}"
            )

    def step_pre_backward(self, *args, **kwargs):
        pass

    def step_post_backward(self, *args, **kwargs):
        pass
