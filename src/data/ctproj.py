import json
import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

import cv2

from src.utils.math import invert_rigid_transform


class CTImageParser:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        meta_path = os.path.join(data_dir, "meta_data.json")
        assert os.path.exists(meta_path), f"meta_data.json not found in {data_dir}"
        with open(meta_path, "r") as f:
            self.meta = json.load(f)

        self.vol_gt_path = os.path.join(data_dir, self.meta["vol"])
        self.scanner = self.meta["scanner"]
        self.bbox = self.meta["bbox"] if "bbox" in self.meta else None
        self.proj_train = self.meta["proj_train"]
        self.proj_test = self.meta["proj_test"]

    def get_proj_info(self, split: str = "train") -> List[Dict[str, Any]]:
        if split == "train":
            return self.proj_train
        if split == "test":
            return self.proj_test
        raise ValueError(f"Unknown split: {split}")

    def get_vol_gt(self) -> torch.Tensor:
        """Return GT volume in its original stored order (W, D, H)."""
        vol_gt_np = np.load(self.vol_gt_path)  # expected shape (W, D, H)
        if vol_gt_np.ndim != 3:
            raise ValueError(f"Expected 3D volume, got shape {vol_gt_np.shape}")
        return torch.from_numpy(vol_gt_np).float()

    def get_scanner_params(self) -> Dict[str, Any]:
        return self.scanner

    def get_bbox(self) -> Optional[List[List[float]]]:
        return self.bbox


class CTImageDataset(torch.utils.data.Dataset):
    def __init__(self, parser: CTImageParser, split: str = "train"):
        self.parser = parser
        self.split = split
        self.proj_info = parser.get_proj_info(split)
        self.data_dir = parser.data_dir

        # load images
        self.images = []
        self.camtoworlds = []
        self.viewmats = []
        self.edge_masks = []
        for info in self.proj_info:
            proj_path = os.path.join(self.data_dir, info["file_path"])
            angle = info["angle"]
            image = np.load(proj_path)
            if image.ndim == 2:  # (W,H)
                image = image[..., None]
            elif image.ndim == 3 and image.shape[-1] == 1:  # (W,H,1)
                pass
            else:
                raise ValueError(
                    f"Unexpected projection shape {image.shape}; expected 2D or (W,H,1)"
                )
            image = torch.from_numpy(image).float()
            self.images.append(image)

            edge_mask = self.get_edge_mask(image)
            self.edge_masks.append(edge_mask)

            camtoworld = self._get_camtoworld(angle)
            self.camtoworlds.append(camtoworld)
            self.viewmats.append(invert_rigid_transform(camtoworld))

        self.K = self._get_K()
        self._preload_device: Optional[torch.device] = None
        self._images_tensor: Optional[torch.Tensor] = None
        self._camtoworlds_tensor: Optional[torch.Tensor] = None
        self._viewmats_tensor: Optional[torch.Tensor] = None
        self._edge_masks_tensor: Optional[torch.Tensor] = None
        self._K_tensor: Optional[torch.Tensor] = None

    def __len__(self):
        return len(self.proj_info)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Load projection keeping original order (W, H) or (W, H, 1)."""
        if self._preload_device is not None:
            return {
                "K": self._K_tensor,
                "camtoworld": self._camtoworlds_tensor[idx],
                "viewmat": self._viewmats_tensor[idx],
                "image": self._images_tensor[idx],
                "image_id": idx,
                "edge_mask": self._edge_masks_tensor[idx],
            }
        camtoworld = self.camtoworlds[idx]
        viewmat = self.viewmats[idx]
        image = self.images[idx]
        intrinsic_mat = self.K
        edge = self.edge_masks[idx]

        return {
            "K": intrinsic_mat,
            "camtoworld": camtoworld,
            "viewmat": viewmat,
            "image": image,
            "image_id": idx,
            "edge_mask": edge,
        }

    def preload_to(self, device: Union[str, torch.device]) -> None:
        """Stack the small projection dataset on one device for fast training."""
        target = torch.device(device)
        if self._preload_device == target:
            return
        self._images_tensor = torch.stack(self.images, dim=0).to(target).contiguous()
        self._camtoworlds_tensor = (
            torch.stack(self.camtoworlds, dim=0).to(target).contiguous()
        )
        self._viewmats_tensor = (
            torch.stack(self.viewmats, dim=0).to(target).contiguous()
        )
        self._edge_masks_tensor = (
            torch.stack(self.edge_masks, dim=0).to(target).contiguous()
        )
        self._K_tensor = self.K.to(target).contiguous()
        self._preload_device = target

    def _get_K(self) -> torch.Tensor:
        scanner = self.parser.scanner
        nDetector = scanner.get("nDetector")
        sDetector = scanner.get("sDetector")
        DSD = scanner.get("DSD")
        offDetector = scanner.get("offDetector", [0, 0])  # [v, u] physical units

        # nDetector / sDetector / offDetector are all [v, u] order
        width, height = nDetector[1], nDetector[0]

        pixel_size_x = sDetector[1] / nDetector[1]
        pixel_size_y = sDetector[0] / nDetector[0]
        focal_x = DSD / pixel_size_x
        focal_y = DSD / pixel_size_y

        center_x = width / 2 + offDetector[1] / pixel_size_x
        center_y = height / 2 + offDetector[0] / pixel_size_y

        K = torch.tensor(
            [[focal_x, 0, center_x], [0, focal_y, center_y], [0, 0, 1]],
            dtype=torch.float32,
        )
        return K

    def _get_camtoworld(
        self,
        angle: float,
        yaw: float = 0.0,
        roll: float = 0.0,
    ) -> torch.Tensor:
        """Construct a 4x4 camera-to-world matrix for cone-beam CT geometry.

                        +Z_w
                        ^
                        |
                        |
                        O ---------> +Y_w
                       /
                      /
                    v
                    +X_w

                            +Z_c
                            ^
                            /
                        /
                        O ---------> +X_c
                        |
                        |
                        v
                        +Y_c
        Parameters
        ----------
        angle : float
            Orbital angle (radians). Camera orbits around world Z (上) in the
            XY plane. angle=0 places the source on +X (后), looking toward
            origin along -X (前).
        dso : float
            Distance from source to origin (orbit radius).
        yaw : float, optional
            Elevation / pitch angle (radians, default 0). Positive yaw moves
            the camera BELOW the XY plane; the look direction tilts upward
            toward the origin. 下为正.
        roll : float, optional
            Roll angle (radians, default 0). Positive roll tilts the camera
            to the LEFT around its forward axis. 左为正.

        Returns
        -------
        np.ndarray
            4x4 camera-to-world homogeneous transformation matrix.

        Rotation chain (extrinsic, all around fixed world axes):
            rot = R_angle @ R_yaw @ R_roll @ R_base

            R_base  = Rz(+π/2) @ Rx(-π/2)  — align camera axes to angle=0 pose
            R_roll  = Rx(roll)              — tilt camera sideways
            R_yaw   = Ry(yaw)              — pitch camera up/down (elevation)
            R_angle = Rz(angle)            — orbit camera around Z axis

        R_angle is leftmost so it carries yaw/roll perturbations along the
        orbit, ensuring consistent physical effects at every angle.
        """

        # R_base = Rz(+π/2) @ Rx(-π/2), precomputed:
        #   cam X -> world +Y (right)
        #   cam Y -> world -Z (down)
        #   cam Z -> world -X (forward, toward origin)
        DSO = self.parser.scanner.get("DSO")

        R_base = np.array([
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ])

        # Rx(roll): roll the camera; positive = left tilt
        R_roll = np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(roll), -np.sin(roll)],
            [0.0, np.sin(roll), np.cos(roll)],
        ])

        # Ry(yaw): pitch the camera; positive = look upward (camera moves down)
        R_yaw = np.array([
            [np.cos(yaw), 0.0, np.sin(yaw)],
            [0.0, 1.0, 0.0],
            [-np.sin(yaw), 0.0, np.cos(yaw)],
        ])

        # Rz(angle): orbit camera around world Z axis
        R_angle = np.array([
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle),  np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])

        rot = R_angle @ R_yaw @ R_roll @ R_base

        # Camera position on a sphere of radius dso (spherical coordinates)
        trans = np.array([
            DSO * np.cos(yaw) * np.cos(angle),
            DSO * np.cos(yaw) * np.sin(angle),
            -DSO * np.sin(yaw),
        ])

        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = rot
        T[:3, 3] = trans
        T = torch.from_numpy(T).float()
        return T

    def get_vol_gt(self):
        return self.parser.get_vol_gt()

    def get_scanner_params(self):
        return self.parser.get_scanner_params()

    def get_bbox(self):
        return self.parser.get_bbox()

    def get_edge_mask(self, img: torch.Tensor) -> torch.Tensor:
        img = img.detach().float().cpu().numpy()
        img = (img * 255.0).clip(0, 255).astype(np.uint8)

        if img.shape[-1] == 1:
            img = img.squeeze(-1)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        img = cv2.GaussianBlur(img, (5, 5), 1.4)
        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(sobel_x**2 + sobel_y**2)
        max_edge = float(np.max(edges))
        if np.isfinite(max_edge) and max_edge > 0.0:
            edges = edges / max_edge
        else:
            edges = np.zeros_like(edges, dtype=np.float64)
        edges = np.nan_to_num(edges, nan=0.0, posinf=1.0, neginf=0.0)
        edges = np.clip(edges, 0.0, 1.0)[..., np.newaxis]

        edges = torch.from_numpy(edges).float()
        return edges
