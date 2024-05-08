from typing import Optional, Union, Tuple
import torch
from torch import nn, Tensor
from torch.autograd import Function
from . import indrnn_cpp
from . import indrnn_cuda
from .common import SequenceCat, Activation, ACTIVFUNS


# # JIT loading of cpp and cuda modules
# from torch.utils.cpp_extension import load
# indrnn_cpp = load(
#     name='indrnn_cpp', sources=['indrnn_cpp.cpp'], verbose=True)
# if torch.cuda.is_available():
#     indrnn_cuda = load(name='indrnn_cuda',
#                        sources=['indrnn_cuda.cpp', 'indrnn_cuda_kernel.cu'],
#                        verbose=True)
# else:
#     import warnings
#     warnings.warn("cuda acceleration not available for indrnn")
#     indrnn_cuda = None


# Python baselines


def _indrnn_py_fwd(t: Tensor, whh: Tensor, activ: int, h0: Tensor) -> Tensor:
    out = torch.empty_like(t)
    h = h0
    func = ACTIVFUNS[activ][0]
    for i in range(t.size(0)):
        h = func(t[i] + h * whh)
        out[i] = h
    return out


def _indrnn_py_bwd(t: Tensor, whh: Tensor, activ: int, out_: Tensor,
                   dout: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    dt, dwhh = torch.empty_like(t), torch.zeros_like(whh)
    dh = torch.zeros_like(out_[0])
    func = ACTIVFUNS[activ][1]
    for i in range(t.size(0) - 1, -1, -1):
        h = out_[i]
        dh = dout[i] + dh
        dpreact = func(t[i] + h * whh) * dh
        dt[i] = dpreact
        dh = dpreact * whh
        dwhh += torch.sum(dpreact * h, 0)
    return dt, dwhh, dh


class _IndRNNFuncPy(Function):
    @staticmethod
    def forward(ctx, t: Tensor, whh: Tensor, activ: int, h0: Tensor) -> Tensor:
        out = _indrnn_py_fwd(t, whh, activ, h0)
        ctx.save_for_backward(t, whh, torch.cat([h0[None, ...], out[:-1]], 0))
        ctx.saved_nontensors = (activ,)
        return out

    @staticmethod
    def backward(ctx, dout: Tensor):
        t, whh, out_ = ctx.saved_tensors
        activ, = ctx.saved_nontensors
        dt, dwhh, dh = _indrnn_py_bwd(t, whh, activ, out_, dout)
        return dt, dwhh, None, dh


class _IndRNNFuncSeqCatPy(Function):
    @staticmethod
    def forward(ctx, scdata: Tensor, sclens: Tensor, whh: Tensor, activ: int,
                h0: Tensor) -> Tensor:
        # reshape as padded sequence to enable batch (but redundant) compute
        sc_in = SequenceCat(scdata, sclens)
        t, ls = sc_in.to_padded_sequence(), sc_in.lengths
        # do indrnn forward loop through time
        out = _indrnn_py_fwd(t, whh, activ, h0)
        # convert output to seqcat format
        sc_out = SequenceCat.from_sequences(out.unbind(1), ls).data
        ctx.save_for_backward(
            t, whh, torch.cat([h0[None, ...], out[:-1]], 0), sclens)
        ctx.saved_nontensors = (activ,)
        return sc_out

    @staticmethod
    def backward(ctx, dout: Tensor):
        # get saved entries from the context
        t, whh, out_, sclens = ctx.saved_tensors
        activ, = ctx.saved_nontensors
        # convert output gradient to padded sequence
        sc_dout = SequenceCat(dout, sclens)
        dout, ls = sc_dout.to_padded_sequence(), sc_dout.lengths
        # do indrnn backward loop through time
        dt, dwhh, dh = _indrnn_py_bwd(t, whh, activ, out_, dout)
        # convert input gradient to seqcat format
        dscdata = SequenceCat.from_sequences(dt.unbind(1), ls).data
        return dscdata, None, dwhh, None, dh


# C++ implementations


class _IndRNNFuncCPP(Function):
    @staticmethod
    def forward(ctx, t: Tensor, whh: Tensor, activ: int, h0: Tensor) -> Tensor:
        out = indrnn_cpp.forward(t, whh, activ, h0)
        ctx.save_for_backward(t, whh, torch.cat([h0[None, ...], out[:-1]], 0))
        ctx.saved_nontensors = (activ,)
        return out

    @staticmethod
    def backward(ctx, dout: Tensor):
        t, whh, out_ = ctx.saved_tensors
        activ, = ctx.saved_nontensors
        dt, dwhh, dh0 = indrnn_cpp.backward(t, whh, activ, out_, dout)
        return dt, dwhh, None, dh0


class _IndRNNFuncSeqCatCPP(Function):
    @staticmethod
    def forward(ctx, scdata: Tensor, sclens: Tensor, whh: Tensor, activ: int,
                h0: Tensor) -> Tensor:
        # reshape as padded sequence to enable batch (but redundant) compute
        sc_in = SequenceCat(scdata, sclens)
        t, ls = sc_in.to_padded_sequence(), sc_in.lengths
        # do indrnn forward loop through time
        out = indrnn_cpp.forward(t, whh, activ, h0)
        # convert output to seqcat format
        sc_out = SequenceCat.from_sequences(out.unbind(1), ls).data
        ctx.save_for_backward(
            t, whh, torch.cat([h0[None, ...], out[:-1]], 0), sclens)
        ctx.saved_nontensors = (activ,)
        return sc_out

    @staticmethod
    def backward(ctx, dout: Tensor):
        # get saved entries from the context
        t, whh, out_, sclens = ctx.saved_tensors
        activ, = ctx.saved_nontensors
        # convert output gradient to padded sequence
        sc_dout = SequenceCat(dout, sclens)
        dout, ls = sc_dout.to_padded_sequence(), sc_dout.lengths
        # do indrnn backward loop through time
        dt, dwhh, dh0 = indrnn_cpp.backward(t, whh, activ, out_, dout)
        # convert input gradient to seqcat format
        dscdata = SequenceCat.from_sequences(dt.unbind(1), ls).data
        return dscdata, None, dwhh, None, dh0


# Custom CUDA implementations


class _IndRNNFuncCUDA(Function):
    @staticmethod
    def forward(ctx, t: Tensor, whh: Tensor, activ: int, h0: Tensor) -> Tensor:
        out = indrnn_cuda.forward(t, whh, activ, h0)
        ctx.save_for_backward(t, whh, torch.cat([h0[None, ...], out[:-1]], 0))
        ctx.saved_nontensors = (activ,)
        return out

    @staticmethod
    def backward(ctx, dout: Tensor):
        t, whh, out_ = ctx.saved_tensors
        activ, = ctx.saved_nontensors
        dt, dwhh, dh0 = indrnn_cuda.backward(t, whh, activ, out_, dout)
        return dt, dwhh, None, dh0


class _IndRNNFuncSeqCatCUDA(Function):
    @staticmethod
    def forward(ctx, scdata: Tensor, sclens: Tensor, whh: Tensor, activ: int,
                h0: Tensor) -> Tensor:
        scpos = SequenceCat(scdata, sclens).positions[:-1]
        out, out_ = indrnn_cuda.forward_seqcat(scdata, scpos, whh, activ, h0)
        ctx.save_for_backward(scdata, scpos, whh, out_)
        ctx.saved_nontensors = (activ,)
        return out

    @staticmethod
    def backward(ctx, dout: Tensor):
        scdata, scpos, whh, out_ = ctx.saved_tensors
        activ, = ctx.saved_nontensors
        dt, dwhh, dh0 = indrnn_cuda.backward_seqcat(
            scdata, scpos, whh, activ, out_, dout)
        return dt, None, dwhh, None, dh0


# public API


# nn.functional style implementation of indrnn for tensor
# t (flattened): shape (seq_len, batch_size, hidden_size);
# whh: shape (hidden_size,);
# (If specified) h0: shape (batch_size, hidden_size).
# if t has more than one feature dimensions, use flatten=True
def _indrnn(t: Tensor, whh: Tensor, activation: Activation = Activation.relu,
           h0: Optional[Tensor] = None, flatten: bool = False) -> Tensor:
    if flatten:
        tshape = t.size()
        t = t.view(*tshape[:2], -1)
    if h0 is None:
        h0 = torch.zeros_like(t[0])
    elif flatten:
        h0 = h0.view(h0.size(0), -1)
    activ = activation.value
    if t.is_cuda:
        t_out = _IndRNNFuncCUDA.apply(t, whh, activ, h0)
    else:
        t_out = _IndRNNFuncCPP.apply(t, whh, activ, h0)
    return t_out.view(tshape) if flatten else t_out


# nn.functional style implementation of indrnn for SequenceCat
# whh: shape (hidden_size,);
# (If specified) h0: shape (batch_size, hidden_size).
# if t has more than one feature dimensions, use flatten=True
def _indrnn_seqcat(
        sc: SequenceCat, whh: Tensor, activation: Activation = Activation.relu,
        h0: Optional[Tensor] = None, flatten: bool = False) -> SequenceCat:
    t, ls = sc.data, sc.lengths
    if flatten:
        tshape = t.size()
        t = t.view(tshape[0], -1)
    if h0 is None:
        h0 = torch.zeros_like(t[0])
    elif flatten:
        h0 = h0.view(h0.size(0), -1)
    activ = activation.value
    if t.is_cuda:
        t_out = _IndRNNFuncSeqCatCUDA.apply(t, ls, whh, activ, h0)
    else:
        t_out = _IndRNNFuncSeqCatCPP.apply(t, ls, whh, activ, h0)
    t_out = t_out.view(tshape) if flatten else t_out
    return SequenceCat(t_out, ls)


# nn.functional style implementation of indrnn that handles Tensor + SeqCat
def indrnn(t: Union[Tensor, SequenceCat], whh: Tensor,
           activation: Activation = Activation.relu,
           h0: Optional[Tensor] = None, flatten: bool = False
           ) -> Union[Tensor, SequenceCat]:
    if isinstance(t, Tensor):
        return _indrnn(t, whh, activation, h0, flatten)
    elif isinstance(t, SequenceCat):
        return _indrnn_seqcat(t, whh, activation, h0, flatten)
    else:
        raise TypeError('t should be of type torch.Tensor or SequenceCat, '
                        'found {}'.format(type(t)))


# independently RNN module, to be used together with e.g., Linear or Conv*d
# layers wrapped by a "common.SeqWrap" module.
class IndRNN(nn.Module):
    def __init__(self, hidden_size: int,
                 activation: Activation = Activation.relu):
        assert hidden_size > 0
        super().__init__()
        self.hidden_size = hidden_size
        self.activation = activation
        self.weight_hh = nn.Parameter(
            torch.empty(hidden_size), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self, seq_len: int = 0, epsilon: float = 0.0,
                         gamma: float = 1.0) -> None:
        if seq_len <= 0:
            nn.init.uniform_(self.weight_hh, a=0, b=1)
        else:
            nn.init.uniform_(self.weight_hh, a=epsilon**(1.0/seq_len),
                             b=gamma**(1.0/seq_len))

    def forward(self, t_input: Union[Tensor, SequenceCat],
                h0: Optional[Tensor] = None, flatten: bool = False
                ) -> Union[Tensor, SequenceCat]:
        """t_input (flattened): shape (seq_len, batch_size, hidden_size).
        If specified, h0: shape (batch_size, hidden_size).
        If t_input has more than one feature dimensions, use flatten=True."""
        return indrnn(t_input, self.weight_hh, self.activation, h0, flatten)
