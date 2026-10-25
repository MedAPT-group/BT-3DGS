# Initial source release

Prepared on 2026-10-19 from the current development working tree.

## Scope

The release contains the binary-tree and conventional Gaussian trainers,
distillation and LoD training, validation, supporting Python utilities,
Hydra configurations, and the self-contained Gaussian `r2splat` backend.
Model implementations, strategies, CUDA sources, and model/strategy/optimizer
hyperparameters match the development snapshot, apart from the backend package
and C++ namespace rename to `r2splat` and the release preparation changes below.

Datasets, checkpoints, run outputs, notebooks, local batch launchers, plotting
experiments, compiled extensions, caches, account settings, and Git history
are excluded.

## Release preparation changes

- Package the command-line modules, Hydra YAML resources, and CUDA sources.
- Name the bundled backend `r2splat`, including Python imports, the
  `r2splat.csrc` extension, C++ namespaces, build paths, and documentation.
- Declare the runtime dependencies and make W&B logging optional and disabled
  by default; remove the development account and absolute dataset paths.
- Require explicit scene and teacher/checkpoint inputs. Reject unsupported
  devices and missing files before creating run outputs.
- Restore saved model settings from the nearest ancestor run configuration,
  including nested LoD student exports.
- Add installation, dataset, checkpoint, and contribution documentation.
- Preserve the existing Apache-2.0 license and inherited source attributions.
- Make CUDA tests opt-in and provide a CPU-only CI workflow.

## Checks performed

- 82 CPU tests passed with CUDA devices hidden, using Python 3.9.25 and
  PyTorch 2.5.1+cu121. Rendering and device checks were stubbed where needed.
- Python 3.9 syntax checks and scans for personal paths, common credential
  patterns, generated binaries, and experiment artifacts passed.
- Built a source archive and Python-only wheel with `BUILD_NO_CUDA=1`.
  Checked that all 106 package source/configuration files were present.
- Installed the wheel into a temporary directory without downloading or
  modifying dependencies. Imported all five entrypoints, composed all five
  Hydra profiles, and loaded both console entrypoints away from the source
  checkout, with W&B absent and CUDA access forbidden.

CUDA compilation, CUDA numerical tests, actual training, reconstruction
validation, and visualization were not run for this release. CPU checks do not
establish GPU correctness, reconstruction quality, or performance. The CI
workflow is supplied but has not yet run on a hosted service.
