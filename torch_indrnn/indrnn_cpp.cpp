#include <torch/extension.h>
#include <vector>
#include <utility>
#include <map>
#include <functional>

namespace {
torch::Tensor d_relu(torch::Tensor t)
{
    auto mask = t > 0;
    return mask.type_as(t);
}

torch::Tensor d_tanh(torch::Tensor t)
{
    auto tt = torch::tanh(t);
    return 1 - tt * tt;
}

using PtWiseFunc = std::function<torch::Tensor(torch::Tensor)>;
const std::vector<std::pair<PtWiseFunc, PtWiseFunc>> activations = {
    std::make_pair(torch::relu, d_relu),
    std::make_pair(torch::tanh, d_tanh),
};
} // namespace

torch::Tensor indrnn_forward(
    torch::Tensor t,
    torch::Tensor whh,
    int activ,
    torch::Tensor h0)
{
    auto out = torch::empty_like(t);
    auto h = h0;

    for (int i = 0; i < t.size(0); ++i)
    {
        h = activations[activ].first(t[i] + h * whh);
        out[i] = h;
    }

    return out;
}

std::vector<torch::Tensor> indrnn_backward(
    torch::Tensor t,
    torch::Tensor whh,
    int activ,
    torch::Tensor out_,
    torch::Tensor dout)
{
    auto dt = torch::empty_like(t);
    auto dwhh = torch::zeros_like(whh);
    auto dh = torch::zeros_like(t[0]);

    for (int i = t.size(0)-1; i > -1; --i)
    {
      auto h = out_[i];
      auto doutt = dout[i] + dh;
      auto dpreact = activations[activ].second(t[i] + h * whh) * doutt;
      dt[i] = dpreact;
      dh = dpreact * whh;
      dwhh += torch::sum(dpreact * h, 0);
    }

    return {dt, dwhh, dh};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &indrnn_forward, "indrnn forward");
    m.def("backward", &indrnn_backward, "indrnn backward");
}
