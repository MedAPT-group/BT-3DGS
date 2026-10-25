from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from .binary_tree_gaussian_model import BinaryTreeGaussianModel
from .gaussian_model import GaussianModel


@dataclass
class StudentSliceInfo:
    selector: str
    target_count: int
    selected_count: int
    teacher_active_count: int
    teacher_total_count: int
    slice_depth: Optional[int]


class StudentGaussianModel(GaussianModel):
    """Student Gaussian model initialized from a teacher binary-tree slice."""
    _selected_slice_depth: Optional[int] = None

    def initialize_from_teacher(
        self,
        teacher_model: BinaryTreeGaussianModel,
        *,
        selector: str = "depth",
        target_ratio: float = 0.25,
        min_target_count: int = 1,
    ) -> "StudentGaussianModel":
        merged = self._merge_teacher_tree(teacher_model)
        teacher_active_count = int(teacher_model.num_gaussians)
        resolved_target = self._resolve_target_count(
            teacher_active_count=teacher_active_count,
            target_ratio=target_ratio,
            min_target_count=min_target_count,
        )
        selected = self._select_slice_indices(
            parents=merged["parents"],
            opacities=merged["opacities"],
            selector=selector,
            target_count=resolved_target,
        )
        resolved_slice_depth = self._selected_slice_depth
        if selected.numel() == 0:
            raise ValueError("Student slice selection produced an empty Gaussian set.")

        self.means = torch.nn.Parameter(merged["means"].index_select(0, selected).clone())
        self.scales = torch.nn.Parameter(merged["scales"].index_select(0, selected).clone())
        self.quats = torch.nn.Parameter(merged["quats"].index_select(0, selected).clone())
        self.opacities = torch.nn.Parameter(
            merged["opacities"].index_select(0, selected).clone()
        )

        self.slice_info = StudentSliceInfo(
            selector=selector,
            target_count=resolved_target,
            selected_count=int(selected.numel()),
            teacher_active_count=teacher_active_count,
            teacher_total_count=int(merged["parents"].numel()),
            slice_depth=resolved_slice_depth,
        )
        self.selected_teacher_indices = selected
        return self

    def _merge_teacher_tree(
        self,
        teacher_model: BinaryTreeGaussianModel,
    ) -> Dict[str, torch.Tensor]:
        return {
            "means": torch.cat([teacher_model.history_means, teacher_model.means], dim=0),
            "scales": torch.cat([teacher_model.history_scales, teacher_model.scales], dim=0),
            "quats": torch.cat([teacher_model.history_quats, teacher_model.quats], dim=0),
            "opacities": torch.cat(
                [teacher_model.history_opacities, teacher_model.opacities], dim=0
            ),
            "parents": torch.cat([teacher_model.history_parent, teacher_model.parent], dim=0)
            .to(device=self.device, dtype=torch.long),
        }

    @staticmethod
    def _resolve_target_count(
        *,
        teacher_active_count: int,
        target_ratio: float,
        min_target_count: int,
    ) -> int:
        resolved = int(round(float(teacher_active_count) * float(target_ratio)))
        resolved = max(int(min_target_count), resolved)
        return min(teacher_active_count, resolved)

    def _select_slice_indices(
        self,
        *,
        parents: torch.Tensor,
        opacities: torch.Tensor,
        selector: str,
        target_count: int,
    ) -> torch.Tensor:
        left, right, level = self._build_tree(parents)
        if selector == "frontier":
            selected = self._select_frontier_slice(
                left=left,
                right=right,
                level=level,
                opacities=opacities,
                target_count=target_count,
            )
        elif selector == "interior":
            selected = self._select_interior_slice(
                parents=parents,
                left=left,
                right=right,
                level=level,
                opacities=opacities,
                target_count=target_count,
            )
        elif selector == "depth":
            selected = self._select_depth_slice(
                left=left,
                right=right,
                level=level,
                opacities=opacities,
                target_count=target_count,
            )
        else:
            raise ValueError(f"Unsupported slice selector: {selector}")
        return selected.to(device=self.device, dtype=torch.long)

    @staticmethod
    def _build_tree(parents: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        left = -torch.ones_like(parents)
        right = -torch.ones_like(parents)
        level = torch.zeros_like(parents)
        for idx in range(int(parents.numel())):
            parent = int(parents[idx].item())
            if parent < 0:
                continue
            if left[parent] < 0:
                left[parent] = idx
            else:
                right[parent] = idx
            level[idx] = level[parent] + 1
        return left, right, level

    def _select_frontier_slice(
        self,
        *,
        left: torch.Tensor,
        right: torch.Tensor,
        level: torch.Tensor,
        opacities: torch.Tensor,
        target_count: int,
    ) -> torch.Tensor:
        frontier = set(torch.where(level == 0)[0].tolist())

        while len(frontier) < target_count:
            expanded = False
            for node in sorted(frontier):
                children = self._children_of(node, left, right)
                if not children:
                    continue
                expanded_size = len(frontier) - 1 + len(children)
                if expanded_size > target_count:
                    continue
                frontier.remove(node)
                frontier.update(children)
                expanded = True
                break
            if not expanded:
                break

        selected = torch.tensor(frontier, device=self.device, dtype=torch.long)
        if selected.numel() > target_count:
            selected = self._downsample_by_opacity(selected, opacities, target_count)
        return selected

    def _select_interior_slice(
        self,
        *,
        parents: torch.Tensor,
        left: torch.Tensor,
        right: torch.Tensor,
        level: torch.Tensor,
        opacities: torch.Tensor,
        target_count: int,
    ) -> torch.Tensor:
        leaf_mask = (left < 0) & (right < 0)
        frontier = set(torch.where(leaf_mask)[0].tolist())
        parent_count = int(parents.numel())

        while len(frontier) > target_count:
            merged = False
            for parent in range(parent_count):
                children = self._children_of(parent, left, right)
                if not children:
                    continue
                if any(child not in frontier for child in children):
                    continue
                merged_size = len(frontier) - len(children) + 1
                if merged_size < target_count:
                    continue
                for child in children:
                    frontier.remove(child)
                frontier.add(parent)
                merged = True
                break
            if not merged:
                break

        selected = torch.tensor(sorted(frontier), device=self.device, dtype=torch.long)
        if selected.shape[0] > target_count:
            selected = self._downsample_by_opacity(selected, opacities, target_count)
        return selected

    def _select_depth_slice(
        self,
        *,
        left: torch.Tensor,
        right: torch.Tensor,
        level: torch.Tensor,
        opacities: torch.Tensor,
        target_count: int,
    ) -> torch.Tensor:
        if level.numel() == 0:
            self._selected_slice_depth = 0
            return torch.empty(0, device=self.device, dtype=torch.long)

        leaf_mask = (left < 0) & (right < 0)
        best_depth = 0
        best_frontier = torch.empty(0, device=level.device, dtype=torch.long)
        best_fallback_key = None

        for depth in range(int(level.max().item()) + 1):
            frontier_mask = (level == depth) | (leaf_mask & (level < depth))
            frontier = torch.where(frontier_mask)[0]
            frontier_count = int(frontier.numel())

            if frontier_count > target_count:
                best_depth = depth
                best_frontier = frontier
                break
            else:
                key = (-frontier_count, -depth)
                if best_fallback_key is None or key < best_fallback_key:
                    best_fallback_key = key
                    best_depth = depth
                    best_frontier = frontier

        self._selected_slice_depth = best_depth
        selected = best_frontier.to(device=self.device, dtype=torch.long)
        if selected.shape[0] > target_count:
            selected = self._downsample_by_opacity(selected, opacities, target_count)
        return selected

    @staticmethod
    def _children_of(
        node: int,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> List[int]:
        children: List[int] = []
        left_child = int(left[node].item())
        right_child = int(right[node].item())
        if left_child >= 0:
            children.append(left_child)
        if right_child >= 0:
            children.append(right_child)
        return children

    @staticmethod
    def _downsample_by_opacity(
        indices: torch.Tensor,
        opacities: torch.Tensor,
        target_count: int,
    ) -> torch.Tensor:
        if indices.shape[0] <= target_count:
            return indices
        values = GaussianModel.activate_opacity(opacities.index_select(0, indices))
        topk = torch.topk(values, k=target_count, largest=True).indices
        return indices.index_select(0, topk)
