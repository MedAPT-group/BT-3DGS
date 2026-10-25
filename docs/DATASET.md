# Scene data format

The loader in `src/data/ctproj.py` consumes one prepared CT scene at a time.
Data is not included in this repository. The layout and numbers below are an
illustrative schema, not a downloadable dataset or a validated reconstruction
example.

```text
scene_dir/
  meta_data.json
  volume.npy
  projections/
    train_000.npy
    test_000.npy
  init_scene_dir.npy       # required only by train_default
```

Paths in metadata are relative to the scene directory. `proj_train` and
`proj_test` explicitly list the views used; selecting a Hydra preset named
`15_views` or `25_views` does not subsample these lists.

## Metadata

```json
{
  "vol": "volume.npy",
  "bbox": [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
  "scanner": {
    "DSO": 4.0,
    "DSD": 8.0,
    "nDetector": [128, 160],
    "sDetector": [4.0, 5.0],
    "offDetector": [0.0, 0.0]
  },
  "proj_train": [
    {"file_path": "projections/train_000.npy", "angle": 0.0}
  ],
  "proj_test": [
    {"file_path": "projections/test_000.npy", "angle": 0.5}
  ]
}
```

Replace all illustrative dimensions, geometry, angles, and paths with your
actual acquisition parameters.

| Field | Meaning |
| --- | --- |
| `vol` | Path to a numeric 3D NumPy reference volume |
| `bbox` | Lower and upper scene corners, `[[xmin, ymin, zmin], [xmax, ymax, zmax]]` |
| `scanner.DSO` | Source-to-origin distance |
| `scanner.DSD` | Source-to-detector distance |
| `scanner.nDetector` | Detector dimensions in `[vertical, horizontal]` pixel order |
| `scanner.sDetector` | Physical detector size in `[vertical, horizontal]` order |
| `scanner.offDetector` | Physical detector offsets in `[vertical, horizontal]` order; defaults to zero |
| `proj_train`, `proj_test` | Lists of projection records with `file_path` and `angle` |
| `angle` | Orbital angle in radians |

Use positive distances, positive detector sizes, and positive integer detector
dimensions. `bbox`, scanner distances, detector sizes, offsets, and any initial
point positions must use a consistent spatial unit. The loader does not convert
units. The default profiles enable volume TV regularization, which requires
`bbox` with nonzero extents.

The camera follows a circular orbit around world Z in the XY plane. At angle
zero, the source lies on +X and points toward the origin along -X. The current
loader derives camera poses and pinhole intrinsics from this geometry; arbitrary
per-view extrinsics and parallel-beam geometry are not implemented by this
schema.

## Array order and intensity

Projections must be numeric `.npy` arrays with shape `(Nv, Nu)` or `(Nv, Nu, 1)`,
matching `scanner.nDetector = [Nv, Nu]`. The first array axis is the detector row
(image height); the second is the detector column (image width). Arrays are
loaded without a transpose. Some historical loader comments call these axes
`(W, H)`; the training implementation uses their stored order as `(H, W)`.
Use the same dimensions for every view.

The reference volume is stored as `(W, D, H)`, corresponding to the scene's
X, Y, and Z axes. It is loaded without a transpose. The volume shape determines
the grid used for volume evaluation and TV sampling. For the standard workflow,
provide a valid reference volume and keep its axes consistent with `bbox` and
the projection geometry.

The loader casts arrays to `float32` and does not normalize them or apply a
log transform. Supply finite projection values in the line-integral convention
and scale used by your reconstruction, with a consistently scaled attenuation
volume. Raw detector counts require preprocessing outside this repository.
The SSIM implementation uses constants for a unit data range, and edge-mask
construction clips projection values to `[0, 1]`; choose and report a consistent
intensity normalization when comparing experiments.

## Initialization for the default trainer

`scripts.train.train_default` requires `init_<scene_dir_name>.npy` directly
inside the scene directory. For `/datasets/example_scene`, the filename is
`init_example_scene.npy`.

This is a nonempty numeric array with at least four columns:

```text
x  y  z  opacity  [additional columns are ignored]
```

The model initializes positions from the first three columns and opacity from
the fourth. It estimates scales from nearest-neighbor distances. The optional
external `simple_knn` extension accelerates this step if installed; otherwise
the implementation uses chunked `torch.cdist`, which can be expensive for large
point clouds. Points outside `bbox` are removed after initialization.

Binary-tree training initializes a single root Gaussian from the scene bounds
and does not require this file. Distillation and LoD training initialize from
teacher checkpoints instead.

## Reproducibility

Keep the original metadata, projection split, axis conventions, preprocessing
description, and intensity scaling with each dataset. Keep `config_used.yaml`
with each training run. Comparisons should use the same projections, physical
bounds, voxel grid, and metric normalization.
