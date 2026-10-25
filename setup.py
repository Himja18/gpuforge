"""Builds the optional CUDA extension when a CUDA toolkit is available; otherwise installs pure Python.

Set GPUFORGE_SKIP_EXT=1 to install only the Python package (then build the
extension separately with ``python setup.py build_ext --inplace``).
"""
import os
import shutil

from setuptools import setup


def _have_cuda_toolkit() -> bool:
    home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda"
    return bool(shutil.which("nvcc")) or os.path.exists(os.path.join(home, "bin", "nvcc"))


ext_modules, cmdclass = [], {}
if os.environ.get("GPUFORGE_SKIP_EXT") != "1" and _have_cuda_toolkit():
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    # No --use_fast_math: it swaps tanhf for a ~11-bit hardware approximation, which broke
    # GELU accuracy (caught by the validation suite). Fast intrinsics are used explicitly
    # where they are acceptable (__expf in softmax).
    ext_modules = [CUDAExtension(
        name="gpuforge_kernels",
        sources=["gpuforge/kernels/csrc/binding.cpp", "gpuforge/kernels/csrc/kernels.cu"],
        extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3", "-lineinfo"]},
    )]
    cmdclass = {"build_ext": BuildExtension}

setup(ext_modules=ext_modules, cmdclass=cmdclass)
