import glob
import os
import os.path as osp
from pathlib import Path

from setuptools import find_packages, setup


VERSION = "0.1.0"
BUILD_NO_CUDA = os.getenv("BUILD_NO_CUDA", "0") == "1"
LINE_INFO = os.getenv("LINE_INFO", "0") == "1"

if not os.getenv("MAX_JOBS"):
    os.environ["MAX_JOBS"] = "10"


def get_build_extension():
    from torch.utils.cpp_extension import BuildExtension

    return BuildExtension.with_options(no_python_abi_suffix=True, use_ninja=True)


def get_extensions():
    from torch.utils.cpp_extension import CUDAExtension

    extension_dir = osp.join("r2splat", "cuda")
    sources = sorted(
        glob.glob(osp.join(extension_dir, "csrc", "*.cpp"))
        + glob.glob(osp.join(extension_dir, "csrc", "*.cu"))
    )
    nvcc_flags = ["-O3", "--use_fast_math", "--expt-relaxed-constexpr"]
    if LINE_INFO:
        nvcc_flags.append("-lineinfo")
    extra_nvcc = os.getenv("NVCC_FLAGS", "").split()
    nvcc_flags.extend(extra_nvcc)

    return [
        CUDAExtension(
            "r2splat.csrc",
            sources,
            include_dirs=[osp.abspath(osp.join(extension_dir, "include"))],
            extra_compile_args={
                "cxx": ["-O3", "-Wno-sign-compare"],
                "nvcc": nvcc_flags,
            },
        )
    ]


setup(
    name="bt3dgs",
    version=VERSION,
    description="Binary-tree 3D Gaussian splatting for CT reconstruction",
    long_description=Path(__file__).with_name("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    license_files=["LICENSE", "THIRD_PARTY_NOTICES.md"],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.5",
        "numpy>=1.24",
        "hydra-core>=1.3,<1.4",
        "omegaconf>=2.3,<2.4",
        "matplotlib>=3.7",
        "imageio>=2.31",
        "opencv-python-headless>=4.8",
        "tqdm>=4.65",
        "ninja",
    ],
    extras_require={"dev": ["pytest>=7", "pyyaml", "build"], "logging": ["wandb"]},
    packages=find_packages(include=[
        "r2splat", "r2splat.*", "src", "src.*",
        "scripts", "scripts.*", "configs",
    ]),
    package_data={"configs": ["*.yaml", "*/*.yaml"]},
    entry_points={"console_scripts": [
        "bt3dgs-train=scripts.train.train_binary:main",
        "bt3dgs-validate=scripts.train.validate:main",
    ]},
    include_package_data=True,
    ext_modules=[] if BUILD_NO_CUDA else get_extensions(),
    cmdclass={} if BUILD_NO_CUDA else {"build_ext": get_build_extension()},
)
