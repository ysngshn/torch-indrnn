#include <torch/extension.h>
#include <vector>

// CUDA forward declarations

torch::Tensor indrnn_cuda_forward(
    torch::Tensor t,
    torch::Tensor whh,
    const int activ,
    torch::Tensor h0);

std::vector<torch::Tensor> indrnn_cuda_backward(
    torch::Tensor t,
    torch::Tensor whh,
    const int activ,
    torch::Tensor out_,
    torch::Tensor dout);

std::vector<torch::Tensor> indrnn_cuda_forward_seqcat(
    torch::Tensor scdata,
    torch::Tensor scpos,
    torch::Tensor whh,
    const int activ,
    torch::Tensor h0);

std::vector<torch::Tensor> indrnn_cuda_backward_seqcat(
    torch::Tensor scdata,
    torch::Tensor scpos,
    torch::Tensor whh,
    const int activ,
    torch::Tensor out_,
    torch::Tensor dout);

// C++ interface

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


// -- indrnn on tensor

torch::Tensor indrnn_forward(
    torch::Tensor t,
    torch::Tensor whh,
    const int activ,
    torch::Tensor h0)
{
    CHECK_INPUT(t);
    CHECK_INPUT(whh);
    CHECK_INPUT(h0);

    return indrnn_cuda_forward(t, whh, activ, h0);
}

std::vector<torch::Tensor> indrnn_backward(
    torch::Tensor t,
    torch::Tensor whh,
    const int activ,
    torch::Tensor out_,
    torch::Tensor dout)
{
  CHECK_INPUT(t);
  CHECK_INPUT(whh);
  CHECK_INPUT(out_);
  CHECK_INPUT(dout);

  return indrnn_cuda_backward(t, whh, activ, out_, dout);
}


// -- indrnn on SequenceCat

std::vector<torch::Tensor> indrnn_forward_seqcat(
    torch::Tensor scdata,
    torch::Tensor scpos,
    torch::Tensor whh,
    const int activ,
    torch::Tensor h0)
{
    CHECK_INPUT(scdata);
    CHECK_INPUT(scpos);
    AT_ASSERTM(scpos.dtype() == torch::kInt64, "scpos must be a Long tensor");
    CHECK_INPUT(whh);
    CHECK_INPUT(h0);

    return indrnn_cuda_forward_seqcat(scdata, scpos, whh, activ, h0);
}

std::vector<torch::Tensor> indrnn_backward_seqcat(
    torch::Tensor scdata,
    torch::Tensor scpos,
    torch::Tensor whh,
    const int activ,
    torch::Tensor out_,
    torch::Tensor dout)
{
    CHECK_INPUT(scdata);
    CHECK_INPUT(scpos);
    AT_ASSERTM(scpos.dtype() == torch::kInt64, "scpos must be a Long tensor");
    CHECK_INPUT(whh);
    CHECK_INPUT(out_);
    CHECK_INPUT(dout);

    return indrnn_cuda_backward_seqcat(scdata, scpos, whh, activ, out_, dout);
}


// pybind magic

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &indrnn_forward, "indrnn forward (CUDA)");
    m.def("backward", &indrnn_backward, "indrnn backward (CUDA)");
    m.def("forward_seqcat", &indrnn_forward_seqcat, "indrnn forward (seqcat CUDA)");
    m.def("backward_seqcat", &indrnn_backward_seqcat, "indrnn backward (seqcat CUDA)");
}
