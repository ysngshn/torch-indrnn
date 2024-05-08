#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include "activfuncs.cuh"

/* NOTE:
replace "torch::PackedTensorAccessor32" to "torch::PackedTensorAccessor64" and
".packed_accessor32" to ".packed_accessor64" if index overflow might occur.
Might be slower though. See:
https://pytorch.org/cppdocs/notes/tensor_basics.html
*/

namespace {

    // forward kernel
    template <typename scalar_t, class AF>
    __launch_bounds__(1024)
    __global__ void indrnn_forward_kernel(
        const torch::PackedTensorAccessor32<scalar_t,3> t,
        const torch::PackedTensorAccessor32<scalar_t,1> whh,
        const torch::PackedTensorAccessor32<scalar_t,2> h0,
        torch::PackedTensorAccessor32<scalar_t,3> out)
    {
        const auto batch_size = t.size(1);
        const auto hidden_size = t.size(2);
        // column index
        const int c = blockIdx.x * blockDim.x + threadIdx.x;
        // return if out of range
        if (c >= batch_size * hidden_size) return;
        // get current batch and state index numbers
        const int bs = c / hidden_size;
        const int idx = c % hidden_size;
        scalar_t w = whh[idx];
        scalar_t h = h0[bs][idx];
        // run indrnn for this thread
        for (int i = 0; i < t.size(0); ++i)
        {
            h = AF::forward(t[i][bs][idx] + h * w);
            out[i][bs][idx] = h;
        }
    }


    // backward kernel
    template <typename scalar_t, class AF>
    __launch_bounds__(1024)
    __global__ void indrnn_backward_kernel(
        const torch::PackedTensorAccessor32<scalar_t,3> t,
        const torch::PackedTensorAccessor32<scalar_t,1> whh,
        const torch::PackedTensorAccessor32<scalar_t,3> out_,
        const torch::PackedTensorAccessor32<scalar_t,3> dout,
        torch::PackedTensorAccessor32<scalar_t,3> dt,
        torch::PackedTensorAccessor32<scalar_t,2> dwhhb,
        torch::PackedTensorAccessor32<scalar_t,2> dh0)
    {
        const auto batch_size = t.size(1);
        const auto hidden_size = t.size(2);
        // column index
        const int c = blockIdx.x * blockDim.x + threadIdx.x;
        // return if out of range
        if (c >= batch_size * hidden_size) return;
        // get current batch and state index numbers
        const int bs = c / hidden_size;
        const int idx = c % hidden_size;
        scalar_t w = whh[idx];
        scalar_t dw_this = 0.0;
        scalar_t dh = 0.0;
        // run indrnn for this thread
        for (int i = t.size(0) - 1; i > -1; --i)
        {
            scalar_t h = out_[i][bs][idx];
            scalar_t doutt = dout[i][bs][idx] + dh;
            scalar_t dpreact = AF::backward(t[i][bs][idx] + h * w) * doutt;
            dt[i][bs][idx] = dpreact;
            dh = dpreact * w;
            dw_this += dpreact * h;
        }
        dh0[bs][idx] = dh;
        dwhhb[bs][idx] = dw_this;
    }


    // forward seqcat kernel
    template <typename scalar_t, class AF>
    __launch_bounds__(1024)
    __global__ void indrnn_forward_kernel_seqcat(
        const torch::PackedTensorAccessor32<scalar_t,2> scdata,
        const torch::PackedTensorAccessor32<int64_t,1> scpos,
        const torch::PackedTensorAccessor32<scalar_t,1> whh,
        const torch::PackedTensorAccessor32<scalar_t,2> h0,
        torch::PackedTensorAccessor32<scalar_t,2> out,
        torch::PackedTensorAccessor32<scalar_t,2> out_)
    {
        const auto batch_size = scpos.size(0);
        const auto totalsize = scdata.size(0);
        const auto hidden_size = scdata.size(1);
        // column index
        const int c = blockIdx.x * blockDim.x + threadIdx.x;
        // return if out of range
        if (c >= batch_size * hidden_size) return;
        // get current batch and state index numbers
        const int bs = c / hidden_size;
        const int idx = c % hidden_size;
        scalar_t w = whh[idx];
        scalar_t h = h0[bs][idx];
        // run indrnn for this thread
        auto istart = scpos[bs];
        auto iend = bs == batch_size - 1 ? totalsize : scpos[bs+1];
        for (auto i = istart; i < iend; ++i)
        {
            out_[i][idx] = h;
            h = AF::forward(scdata[i][idx] + h * w);
            out[i][idx] = h;
        }
    }


    // backward seqcat kernel
    template <typename scalar_t, class AF>
    __launch_bounds__(1024)
    __global__ void indrnn_backward_kernel_seqcat(
        const torch::PackedTensorAccessor32<scalar_t,2> scdata,
        const torch::PackedTensorAccessor32<int64_t,1> scpos,
        const torch::PackedTensorAccessor32<scalar_t,1> whh,
        const torch::PackedTensorAccessor32<scalar_t,2> out_,
        const torch::PackedTensorAccessor32<scalar_t,2> dout,
        torch::PackedTensorAccessor32<scalar_t,2> dscdata,
        torch::PackedTensorAccessor32<scalar_t,2> dwhhb,
        torch::PackedTensorAccessor32<scalar_t,2> dh0)
    {
        const auto batch_size = scpos.size(0);
        const auto totalsize = scdata.size(0);
        const auto hidden_size = scdata.size(1);
        // column index
        const int c = blockIdx.x * blockDim.x + threadIdx.x;
        // return if out of range
        if (c >= batch_size * hidden_size) return;
        // get current batch and state index numbers
        const int bs = c / hidden_size;
        const int idx = c % hidden_size;
        scalar_t w = whh[idx];
        scalar_t dw_this = 0.0;
        scalar_t dh = 0.0;
        // run indrnn for this thread
        auto istart = scpos[bs];
        auto iend = bs == batch_size - 1 ? totalsize : scpos[bs+1];
        for (auto i = iend - 1; i > istart - 1; --i)
        {
            scalar_t h = out_[i][idx];
            scalar_t doutt = dout[i][idx] + dh;
            scalar_t dpreact = AF::backward(scdata[i][idx] + h * w) * doutt;
            dscdata[i][idx] = dpreact;
            dh = dpreact * w;
            dw_this += dpreact * h;
        }
        dh0[bs][idx] = dh;
        dwhhb[bs][idx] = dw_this;
    }
}  // namespace


// cuda host functions for indrnn

torch::Tensor indrnn_cuda_forward(
    torch::Tensor t,
    torch::Tensor whh,
    const int activ,
    torch::Tensor h0)
{
    const auto batch_size = t.size(1);
    const auto hidden_size = t.size(2);
    auto out = torch::empty_like(t);
    const int threads = 1024;
    const int blocks = (batch_size * hidden_size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(t.type(), "indrnn_forward", ([&] {
        DISPATCH_ACTIVATION_FUNCTION(activ, ([&] {
            indrnn_forward_kernel<scalar_t, af_t><<<blocks, threads>>>(
                t.packed_accessor32<scalar_t,3>(),
                whh.packed_accessor32<scalar_t,1>(),
                h0.packed_accessor32<scalar_t,2>(),
                out.packed_accessor32<scalar_t,3>());
        }));
    }));

    return out;
}

std::vector<torch::Tensor> indrnn_cuda_backward(
    torch::Tensor t,
    torch::Tensor whh,
    const int activ,
    torch::Tensor out_,
    torch::Tensor dout)
{
    const auto batch_size = t.size(1);
    const auto hidden_size = t.size(2);
    auto dt = torch::empty_like(t);
    auto dwhhb = torch::zeros_like(t[0]);
    auto dh0 = torch::zeros_like(t[0]);
    const int threads = 1024;
    const int blocks = (batch_size * hidden_size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(t.type(), "indrnn_backward", ([&] {
        DISPATCH_ACTIVATION_FUNCTION(activ, ([&] {
            indrnn_backward_kernel<scalar_t, af_t><<<blocks, threads>>>(
                t.packed_accessor32<scalar_t,3>(),
                whh.packed_accessor32<scalar_t,1>(),
                out_.packed_accessor32<scalar_t,3>(),
                dout.packed_accessor32<scalar_t,3>(),
                dt.packed_accessor32<scalar_t,3>(),
                dwhhb.packed_accessor32<scalar_t,2>(),
                dh0.packed_accessor32<scalar_t,2>());
        }));
    }));

    auto dwhh = torch::sum(dwhhb, 0);
    return {dt, dwhh, dh0};
}


// cuda host functions for indrnn on SequenceCat

std::vector<torch::Tensor> indrnn_cuda_forward_seqcat(
    torch::Tensor scdata,
    torch::Tensor scpos,
    torch::Tensor whh,
    const int activ,
    torch::Tensor h0)
{
    const auto batch_size = scpos.size(0);
    const auto hidden_size = scdata.size(1);
    auto out = torch::empty_like(scdata);
    auto out_ = torch::empty_like(scdata);
    const int threads = 1024;
    const int blocks = (batch_size * hidden_size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(scdata.type(), "indrnn_forward_seqcat", ([&] {
        DISPATCH_ACTIVATION_FUNCTION(activ, ([&] {
            indrnn_forward_kernel_seqcat<scalar_t, af_t><<<blocks, threads>>>(
                scdata.packed_accessor32<scalar_t,2>(),
                scpos.packed_accessor32<int64_t,1>(),
                whh.packed_accessor32<scalar_t,1>(),
                h0.packed_accessor32<scalar_t,2>(),
                out.packed_accessor32<scalar_t,2>(),
                out_.packed_accessor32<scalar_t,2>());
        }));
    }));

    return {out, out_};
}

std::vector<torch::Tensor> indrnn_cuda_backward_seqcat(
    torch::Tensor scdata,
    torch::Tensor scpos,
    torch::Tensor whh,
    const int activ,
    torch::Tensor out_,
    torch::Tensor dout)
{
    const auto batch_size = scpos.size(0);
    const auto hidden_size = scdata.size(1);
    auto dscdata = torch::empty_like(scdata);
    auto dwhhb = torch::zeros({batch_size, hidden_size}, torch::TensorOptions(
        ).dtype(scdata.dtype()).device(scdata.device()));
    auto dh0 = torch::zeros({batch_size, hidden_size}, torch::TensorOptions(
        ).dtype(scdata.dtype()).device(scdata.device()));
    const int threads = 1024;
    const int blocks = (batch_size * hidden_size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(scdata.type(), "indrnn_backward_seqcat", ([&] {
        DISPATCH_ACTIVATION_FUNCTION(activ, ([&] {
            indrnn_backward_kernel_seqcat<scalar_t, af_t><<<blocks, threads>>>(
                scdata.packed_accessor32<scalar_t,2>(),
                scpos.packed_accessor32<int64_t,1>(),
                whh.packed_accessor32<scalar_t,1>(),
                out_.packed_accessor32<scalar_t,2>(),
                dout.packed_accessor32<scalar_t,2>(),
                dscdata.packed_accessor32<scalar_t,2>(),
                dwhhb.packed_accessor32<scalar_t,2>(),
                dh0.packed_accessor32<scalar_t,2>());
        }));
    }));

    auto dwhh = torch::sum(dwhhb, 0);
    return {dscdata, dwhh, dh0};
}
