from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


PACKAGE = "bev_vae.models.ops.deformable_aggregation"
ROOT = Path(__file__).resolve().parent


setup(
    name="bev_vae_deformable_aggregation",
    ext_modules=[
        CUDAExtension(
            name=f"{PACKAGE}.deformable_aggregation_ext",
            sources=[
                str(ROOT / "src" / "deformable_aggregation.cpp"),
                str(ROOT / "src" / "deformable_aggregation_cuda.cu"),
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "-D__CUDA_NO_HALF_OPERATORS__",
                    "-D__CUDA_NO_HALF_CONVERSIONS__",
                    "-D__CUDA_NO_HALF2_OPERATORS__",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
