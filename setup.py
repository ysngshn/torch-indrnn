from setuptools import setup
from torch.utils.cpp_extension import (
    BuildExtension, CppExtension, CUDAExtension)

setup(
    name='torch-indrnn',
    ext_modules=[
        CppExtension('torch_indrnn.indrnn_cpp', [
            'torch_indrnn/indrnn_cpp.cpp']),
        CUDAExtension('torch_indrnn.indrnn_cuda', [
            'torch_indrnn/indrnn_cuda.cpp',
            'torch_indrnn/indrnn_cuda_kernel.cu']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
