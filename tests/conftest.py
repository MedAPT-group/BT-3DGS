"""Keep CUDA tests opt-in, including on machines with visible GPUs."""


def pytest_addoption(parser):
    parser.addoption(
        "--run-cuda", action="store_true", default=False,
        help="Include tests that execute the compiled CUDA operators.",
    )


def pytest_ignore_collect(collection_path, config):
    if collection_path.name == "test_cuda_gaussian_ops.py":
        return not config.getoption("--run-cuda")
    return None
