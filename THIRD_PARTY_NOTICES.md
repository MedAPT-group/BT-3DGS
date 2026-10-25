# Third-party notices

## gsplat lineage

Existing comments in the bundled `r2splat` CUDA sources identify code adapted
from or following patterns from [gsplat](https://github.com/nerfstudio-project/gsplat).
These comments are retained. Examples include:

- `r2splat/cuda/csrc/IsectTiles.h` and `IsectTiles3D.h`: adaptations of tile intersection interfaces.
- `r2splat/cuda/csrc/IsectTiles3DKernel.cu`: refactoring of the original 3D tile intersection code.
- `r2splat/cuda/csrc/VoxelizeToVoxelsFwd.cu`: adaptation of voxel accumulation code.
- `r2splat/cuda/include/types.cuh` and `common.h`: type definitions and common CUDA patterns.

The exact upstream revisions are not recorded in the inherited source tree.
These notices describe known source-level attribution and do not claim a
complete provenance history. The repository retains its
[Apache License 2.0](LICENSE).

The Gaussian backend is included directly in this repository as `r2splat`;
no separate backend checkout or package installation is required.

## Optional simple_knn integration

`src/models/gaussian_model.py` optionally imports `simple_knn._C.distCUDA2` for
nearest-neighbor scale initialization. `simple_knn` source and binaries are not
vendored, and the package is not installed as a required dependency. When it is
unavailable, the model uses a PyTorch fallback. Any separately installed copy
retains its own license and distribution terms.

## External dependencies

PyTorch, NumPy, Hydra/OmegaConf, OpenCV, Matplotlib, imageio, tqdm, Ninja, pytest,
PyYAML, and optional Weights & Biases are external Python/build dependencies and
retain their respective licenses. The CUDA toolkit and host compiler are also
external prerequisites. Their source code is not vendored in this repository.

Datasets and pretrained models are not included. Their owners' terms apply if
you obtain or distribute them separately.
