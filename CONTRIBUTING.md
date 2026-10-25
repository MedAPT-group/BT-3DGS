# Contributing

Keep changes focused and include a description of the problem, the resulting
behavior, and how it was checked. Follow the surrounding model, strategy, and
Hydra configuration conventions. Avoid committing datasets, checkpoints,
experiment outputs, local environment files, or account credentials.

## Python and CPU checks

Install the development dependencies as described in the README. On a machine
without a CUDA toolchain, the Python package can be installed for CPU tests:

```bash
BUILD_NO_CUDA=1 pip install -e '.[dev]' --no-build-isolation
pytest --ignore=tests/test_cuda_gaussian_ops.py
```

The CPU suite covers Python/configuration behavior and uses stubs for rendering
where appropriate. It does not run a reconstruction or validate GPU kernels.
For a focused edit, run the affected tests before broadening the selection.

## CUDA changes and reconstruction checks

After changing `r2splat/cuda/`, rebuild the extension in a compatible CUDA
environment before testing:

```bash
pip install -e '.[dev]' --no-build-isolation
pytest --run-cuda tests/test_cuda_gaussian_ops.py
```

CUDA tests are opt-in. Report the GPU, PyTorch/CUDA versions, build command, and
tests actually run. For numerical changes, compare forward values and gradients
against an independent reference; use CUDA sanitizer tools when relevant to
memory access or synchronization. Operator tests and end-to-end training cover
different behavior, so report them separately.

Training, validation, and visualization allocate GPU resources and write
outputs. Use an appropriate dataset and a new result directory. For
reconstruction changes, report the scene, projection split, configuration,
checkpoint step, and metric definitions. Preserve `config_used.yaml`; do not
represent CPU checks as evidence of GPU accuracy or reconstruction quality.

If a check could not be run, say so explicitly. Keep upstream attribution and
license notices when adapting third-party code, and identify newly introduced
dependencies in the change description.
