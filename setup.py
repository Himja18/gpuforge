"""Builds the optional CUDA extension when nvcc is available; otherwise installs pure Python."""
import os
import shutil

from setuptools import setup

ext_modules, cmdclass = [], {}
if shutil.which("nvcc") or os.environ.get("CUDA_HOME"):
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    ext_modules = [CUDAExtension(
        name="gpuforge_kernels",
        sources=["gpuforge/kernels/csrc/binding.cpp", "gpuforge/kernels/csrc/kernels.cu"],
        extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3", "--use_fast_math", "-lineinfo"]},
    )]
    cmdclass = {"build_ext": BuildExtension}

setup(ext_modules=ext_modules, cmdclass=cmdclass)
