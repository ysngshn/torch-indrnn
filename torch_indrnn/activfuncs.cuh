#pragma once

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>


// macros to dispatch activation function "af_t" according to flag
#define PRIVATE_ACTIVFUNC_TYPE(flag, type, ...)    \
  case flag: {                                     \
    using af_t = type<scalar_t>;                   \
    return __VA_ARGS__();                          \
  }

#define DISPATCH_ACTIVATION_FUNCTION(FLAG, ...)                             \
  [&] {                                                                     \
    switch (FLAG) {                                                         \
      PRIVATE_ACTIVFUNC_TYPE(0, af_relu, __VA_ARGS__)                       \
      PRIVATE_ACTIVFUNC_TYPE(1, af_tanh, __VA_ARGS__)                       \
      default:                                                              \
        AT_ERROR("Unknown flag for activation function: ", #FLAG);          \
    }                                                                       \
  }()


// generic activation functions "af_t", forward & backward
template <typename scalar_t>
struct af_tanh
{
    __device__ static scalar_t forward(scalar_t x)
    {
        return tanh(x);
    }

    __device__ static scalar_t backward(scalar_t x)
    {
        const auto t = tanh(x);
        return 1 - (t * t);
    }
};

template <typename scalar_t>
struct af_relu
{
    __device__ static scalar_t forward(scalar_t x)
    {
        return (x > 0.0) ? x : 0.0;
    }

    __device__ static scalar_t backward(scalar_t x)
    {
        return (x > 0.0) ? 1.0 : 0.0;
    }
};