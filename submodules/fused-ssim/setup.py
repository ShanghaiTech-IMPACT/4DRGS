from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
setup(name="fused_ssim", packages=["fused_ssim"], ext_modules=[CUDAExtension(
    name="fused_ssim_cuda", sources=["ssim.cu", "ssim3d.cu", "ext.cpp"],
    extra_compile_args={"cxx": (["/O2"] if os.name == "nt" else ["-O3"]) + ["-DFUSED_SSIM_CUDA"],
                        "nvcc": ["-O3", "-DFUSED_SSIM_CUDA"]})],
    cmdclass={"build_ext": BuildExtension})
