import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def _detect_arch():
    arch = os.environ.get("MEGAKERNEL_CUDA_ARCH")
    if arch:
        return arch
    try:
        import torch
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            if major == 12 and minor in (0, 1):
                return f"sm_{major}{minor}a"
            return f"sm_{major}{minor}"
    except Exception:
        pass
    return "sm_86"


def _int_env(name, default):
    return str(int(os.environ.get(name, default)))


arch = _detect_arch()
num_blocks = _int_env("MEGAKERNEL_NUM_BLOCKS", 82)
block_size = _int_env("MEGAKERNEL_BLOCK_SIZE", 512)
lm_num_blocks = _int_env("MEGAKERNEL_LM_NUM_BLOCKS", 512)
lm_block_size = _int_env("MEGAKERNEL_LM_BLOCK_SIZE", 256)

setup(
    name="qwen35_megakernel_bf16",
    ext_modules=[
        CUDAExtension(
            name="qwen35_megakernel_bf16_C",
            sources=[
                "torch_bindings.cpp",
                "kernel.cu",
                "prefill.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    f"-arch={arch}",
                    "--use_fast_math",
                    "-std=c++17",
                    f"-DNUM_BLOCKS={num_blocks}",
                    f"-DBLOCK_SIZE={block_size}",
                    f"-DLM_NUM_BLOCKS={lm_num_blocks}",
                    f"-DLM_BLOCK_SIZE={lm_block_size}",
                ],
            },
            libraries=["cublas"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
