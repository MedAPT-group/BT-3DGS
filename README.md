# BT-3DGS

BT-3DGS reconstructs a 3D attenuation field from CT projections using Gaussian
splatting and binary-tree densification. The main trainer starts with one
Gaussian at the scene center, preserves the parent history as it splits, and
optimizes projection losses together with volume and hierarchy regularization.
The tree also supports smaller student models through distillation and
level-of-detail (LoD) training.

This repository contains the core models, trainers, Hydra configurations, dataset
loader, validation tools, and the Gaussian-only `r2splat` CUDA backend. Datasets,
trained weights, experiment outputs, and benchmark results are not bundled.

## Installation

Use Python 3.10 or newer (the minimum supported version is 3.9), an NVIDIA GPU,
a compatible CUDA toolkit with `nvcc`, and a C++ compiler. Training, projection
rendering, voxelization, and validation require the compiled CUDA extension.

The development baseline was PyTorch 2.5.1 with CUDA 12.1. This records the
original environment; it is not a fresh GPU compatibility or performance test
of this release. From the repository root, a corresponding setup is:

```bash
conda create -n bt3dgs python=3.10
conda activate bt3dgs
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
conda install -c nvidia/label/cuda-12.1.1 cuda-toolkit
pip install -e '.[dev]' --no-build-isolation
```

Install PyTorch before the editable install because the extension build imports
its build helpers. The last command compiles `r2splat.csrc`; repeat it after
changing CUDA/C++ sources. Set `MAX_JOBS`, for example `MAX_JOBS=4`, to limit
parallel compilation when memory is constrained.

For Python development and CPU tests without compiling CUDA:

```bash
BUILD_NO_CUDA=1 pip install -e '.[dev]' --no-build-isolation
```

This mode does not provide CPU training, rendering, or reconstruction validation.

Weights & Biases logging is disabled by default. To use it, install the optional
dependency with `pip install -e '.[logging]' --no-build-isolation`, configure your
own account, and pass `wandb.enabled=true` and your desired `wandb.project` and
`wandb.entity` overrides when training.

## Prepare a scene

Provide a scene directory containing `meta_data.json`, projection arrays, and a
reference volume. The loader expects prepared cone-beam CT geometry and `.npy`
arrays; it does not ingest raw scanner/DICOM files or perform intensity
calibration. See [the dataset format](docs/DATASET.md) for fields, axis order,
units, and the additional initialization file used by the default trainer.

The binary-tree trainer initializes a single root Gaussian and does not require
an initial point cloud. The `exp` configuration names describe presets; the
number of views actually used comes from the metadata projection lists.

## Train and validate

Run commands from the repository root. Supply your own dataset path and use a
separate output directory for each run.

```bash
python -m scripts.train.train_binary \
  exp.data_dir=/path/to/scene_dir \
  exp.result_dir=output/teacher
```

The standard binary-tree profile runs 30,000 steps and saves at step `29999`
(steps are zero-based). Hydra overrides can change the GPU, duration, and save
schedule together:

```bash
python -m scripts.train.train_binary \
  exp.data_dir=/path/to/scene_dir \
  exp.result_dir=output/custom_teacher \
  trainer.device=cuda:0 \
  trainer.max_steps=10000 \
  'trainer.save_steps=[9999]' \
  'trainer.ply_steps=[9999]'
```

After training, validate the Gaussian parameter export:

```bash
python -m scripts.train.validate \
  data_dir=/path/to/scene_dir \
  checkpoint_path=output/teacher/ckpts/point_cloud_29999.pickle \
  result_dir=output/teacher/valid_output
```

Validation writes projection metrics, comparison images, volume metrics when a
reference volume is available, and `validation_report.json`. It uses the same
CUDA backend as training. Preserve `config_used.yaml` with the checkpoint so
that model scale bounds and operator settings can be restored.

The conventional Gaussian trainer is also available:

```bash
python -m scripts.train.train_default \
  exp.data_dir=/path/to/scene_dir \
  exp.result_dir=output/default
```

This entrypoint additionally requires `init_<scene_dir_name>.npy` inside the
scene directory, as described in [the dataset documentation](docs/DATASET.md).

## Distillation and LoD

Distillation requires a binary-tree teacher **`.pt` checkpoint with tree
history**, rather than the `.pickle` parameter export used by validation:

```bash
python -m scripts.train.train_distill \
  exp.data_dir=/path/to/scene_dir \
  exp.result_dir=output/student \
  trainer.teacher_path=output/teacher/ckpts/ckpt_29999.pt \
  trainer.slice_target_ratio=0.1
```

The ratio requests a target relative to the teacher's active Gaussian count;
the minimum target and tree selection can affect the final count. The default
profile generates teacher projections with free camera sampling. Use
`trainer.teacher_view_sampling=ct` for CT-orbit sampling, or
`trainer.distill_image_source=dataset` to supervise from the provided training
projections.

To train multiple LoD students:

```bash
python -m scripts.train.train_lod \
  exp.data_dir=/path/to/scene_dir \
  exp.result_dir=output/lod_students \
  trainer.teacher_path=output/teacher/ckpts/ckpt_29999.pt
```

Keep the teacher's sibling `lod/` directory: this entrypoint discovers
`lod/ckpt_lod_<count>_step_<step>.pt` snapshots saved as the teacher grows. A
teacher `.pt` alone is insufficient for this workflow. The available snapshots
determine the candidate levels, and invalid levels are skipped. Student runs
are written beneath `output/lod_students/lod_<count>/`.

When the teacher used custom `model.*` settings, pass the same settings to
distillation and LoD training. These trainers construct the teacher using their
current model configuration.

## Saved files

| Path within a run | Contents |
| --- | --- |
| `config_used.yaml` | Hydra configuration used for the run; retain with checkpoints |
| `ckpts/ckpt_<step>.pt` | Model and optimizer states; binary-tree checkpoints also include tree history |
| `ckpts/point_cloud_<step>.pickle` | Gaussian parameters and scale metadata for validation |
| `ply/point_cloud_<step>.ply` | Optional ASCII point cloud with grayscale colors |
| `lod/ckpt_lod_<count>_step_<step>.pt` | Intermediate binary-tree snapshots for LoD initialization |

The historical method name `save_ply()` writes the **pickle** parameter export.
`export_ply()` writes the separate ASCII PLY file. The ASCII PLY is a point-cloud
preview and cannot replace the parameter export in validation. The training
entrypoints do not currently expose a general command-line resume option.

## Code and configuration

| Location | Purpose |
| --- | --- |
| `src/models/` | Gaussian, binary-tree, and student models |
| `src/strategies/` | Densification, pruning, and distillation strategies |
| `src/trainers/` | Training loops and output management |
| `src/data/ctproj.py` | CT metadata, arrays, and camera geometry |
| `r2splat/` | Gaussian projection/voxelization CUDA operators and Python interfaces |
| `configs/` | Hydra profiles and model, strategy, optimizer, and trainer settings |
| `scripts/train/` | Single-run training and validation entrypoints |

The backend supports Gaussian projection and voxelization with first-order
gradients. Training uses explicit output gradients and a CUDA vector-Jacobian
product. Packed rendering, sparse gradients, non-classic rasterization, and
non-pinhole cameras are not supported.

## Tests and contributions

Run the CPU suite with the CUDA test module explicitly excluded:

```bash
pytest --ignore=tests/test_cuda_gaussian_ops.py
```

After building the extension in a CUDA environment, opt into the operator tests:

```bash
pytest --run-cuda tests/test_cuda_gaussian_ops.py
```

CPU checks exercise configuration, Python behavior, and mocked rendering paths;
they do not establish CUDA numerical correctness or end-to-end reconstruction
quality. See [CONTRIBUTING.md](CONTRIBUTING.md) for reporting changes and test
coverage.

## License

This project is distributed under the [Apache License 2.0](LICENSE). See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for inherited-code attribution
and external dependency notes.
