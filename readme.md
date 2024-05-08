# [torch-indrnn](https://github.com/ysngshn/torch-indrnn)

This is a PyTorch implementation of the IndRNN architecture with custom C++/CUDA extensions for efficiency. 

IndRNN is a simple and effective recurrent neural network structure proposed by Li et. al. at CVPR 2018:

["Independently Recurrent Neural Network (IndRNN): Building a Longer and Deeper RNN". Shuai Li, Wanqing Li, Chris Cook, Ce Zhu, Yanbo Gao. CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Li_Independently_Recurrent_Neural_CVPR_2018_paper.html)

The authors have also provided an official PyTorch implementation:

[github.com/Sunnydreamrain/IndRNN_pytorch](https://github.com/Sunnydreamrain/IndRNN_pytorch)

Compared to the official implementation, this package is a clean implementation that follows the standard [PyTorch C++/CUDA API](https://pytorch.org/docs/stable/cpp_extension.html). This should result in better compatibility with PyTorch, supporting for instance computation at different precision. Also, this implementation has no dependency on `CuPy` or `pynvrtc`.

## Installation

To use this package, make sure you have installed `PyTorch` and have a compatible `cuda` version with `nvcc` compiler.

Then you can install this library locally via

```commandline
pip install -e .
```
