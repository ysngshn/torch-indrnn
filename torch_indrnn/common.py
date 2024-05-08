from typing import Optional, Union, Sequence, Tuple
from enum import Enum, unique
import torch
from torch import nn, Tensor, LongTensor


# supported activation functions
@unique
class Activation(int, Enum):
    relu = 0
    tanh = 1


def _d_relu(t):
    return (t > 0).type_as(t)


def _d_tanh(t):
    tt = torch.tanh(t)
    return 1 - tt * tt


ACTIVFUNS = [(torch.relu, _d_relu), (torch.tanh, _d_tanh)]


# Data type for handling of variable length sequences
# similar to PackedSequence but simply concat sequences
# custom CUDA acceleration for indrnn available
class SequenceCat(object):
    def __init__(self, data: Tensor, lengths: LongTensor):
        super().__init__()
        self.data = data  # scalar_t, (t_1+...+t_batchsize, *featureshape)
        self.lengths = lengths  # int64, (batchsize,)

    @property
    def positions(self) -> LongTensor:  # (batchsize + 1,)
        lengths = self.lengths
        positions = torch.zeros(
            len(lengths)+1, dtype=torch.int64, device=lengths.device)
        positions[1:] = lengths
        return torch.cumsum(positions, dim=0)

    @classmethod
    def from_sequences(cls, seqs: Sequence[Tensor],
                       lengths: Optional[Union[Sequence[int], Tensor]] = None):
        assert len(seqs) > 0
        if lengths is None:
            ls = torch.tensor([len(s) for s in seqs],
                              dtype=torch.int64, device=seqs[0].device)
            data = torch.cat(tuple(seqs), dim=0)
        else:
            assert len(seqs) == len(lengths)
            ls = torch.as_tensor(
                lengths, dtype=torch.int64, device=seqs[0].device)
            data = torch.cat([s[:l] for s, l in zip(seqs, lengths)], dim=0)
        return cls(data, ls)

    def to_sequences(self) -> Tuple[Tensor, ...]:
        ps = self.positions
        return tuple([self.data[a:b] for a, b in zip(ps[:-1], ps[1:])])

    def to_padded_sequence(self, batch_first=False, padding_value=0) -> Tensor:
        return nn.utils.rnn.pad_sequence(
            list(self.to_sequences()), batch_first, padding_value)

    def pin_memory(self):
        return type(self)(self.data.pin_memory(), self.lengths.pin_memory())

    def cuda(self, *args, **kwargs):
        # Tests to see if 'cuda' should be added to kwargs
        ex = torch.tensor((), dtype=self.data.dtype,
                          device=self.data.device).to(*args, **kwargs)
        if ex.is_cuda:
            return self.to(*args, **kwargs)
        return self.to(*args, device='cuda', **kwargs)

    def cpu(self, *args, **kwargs):
        # Tests to see if 'cpu' should be added to kwargs
        ex = torch.tensor((), dtype=self.data.dtype,
                          device=self.data.device).to(*args, **kwargs)
        if ex.device.type == 'cpu':
            return self.to(*args, **kwargs)
        return self.to(*args, device='cpu', **kwargs)

    def double(self):
        return self.to(dtype=torch.double)

    def float(self):
        return self.to(dtype=torch.float)

    def half(self):
        return self.to(dtype=torch.half)

    def long(self):
        return self.to(dtype=torch.long)

    def int(self):
        return self.to(dtype=torch.int)

    def short(self):
        return self.to(dtype=torch.short)

    def char(self):
        return self.to(dtype=torch.int8)

    def byte(self):
        return self.to(dtype=torch.uint8)

    def to(self, *args, **kwargs):
        data = self.data.to(*args, **kwargs)
        if data is self.data:
            return self
        else:
            kwargs = {k: v for k, v in
                      filter(lambda t: t[0] != 'device' and t[0] != 'dtype',
                             kwargs.items())}
            lengths = self.lengths.to(data.device, **kwargs)
            return type(self)(data, lengths)

    @property
    def is_cuda(self):
        return self.data.is_cuda

    def is_pinned(self):
        return self.data.is_pinned()


# wrap time unaware module to process sequential data by merging time dimension
# with batch dimension, assuming they occupy the first 2 dimensions.
# Also enables processing of SequenceCat data type.
class SeqWrap(nn.Module):
    def __init__(self, m: nn.Module) -> None:
        super().__init__()
        self.m = m

    def _forward_tensor(self, t: Tensor, *args, **kwargs) -> Tensor:
        headdims = t.size()[:2]
        t = t.reshape(-1, *t.size()[2:])
        t = self.m(t, *args, **kwargs)
        return t.reshape(*headdims, *t.size()[1:])

    def _forward_seqcat(self, t: SequenceCat, *args, **kwargs) -> SequenceCat:
        td = self.m(t.data, *args, **kwargs)
        return SequenceCat(td, t.lengths)

    def forward(self, t: Union[Tensor, SequenceCat], *args, **kwargs
                ) -> Union[Tensor, SequenceCat]:
        if isinstance(t, Tensor):
            return self._forward_tensor(t, *args, **kwargs)
        elif isinstance(t, SequenceCat):
            return self._forward_seqcat(t, *args, **kwargs)
        else:
            raise TypeError('t should be of type torch.Tensor or SequenceCat, '
                            'found {}'.format(type(t)))


# simply take the data at the last timestep
class TakeLast(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _forward_tensor(t: Tensor) -> Tensor:
        return t[-1]

    @staticmethod
    def _forward_seqcat(sc: SequenceCat) -> Tensor:
        return sc.data[sc.positions[1:]-1]

    def forward(self, t: Union[Tensor, SequenceCat]) -> Tensor:
        if isinstance(t, Tensor):
            return self._forward_tensor(t)
        elif isinstance(t, SequenceCat):
            return self._forward_seqcat(t)
        else:
            raise TypeError('t should be of type torch.Tensor or SequenceCat, '
                            'found {}'.format(type(t)))
