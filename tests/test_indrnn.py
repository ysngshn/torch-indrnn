import unittest
import numpy as np
import torch
from torch.autograd import gradcheck
from torch_indrnn.indrnn import (
    _IndRNNFuncPy, _IndRNNFuncCPP, _IndRNNFuncCUDA, Activation, SequenceCat,
    indrnn, _IndRNNFuncSeqCatPy, _IndRNNFuncSeqCatCPP, _IndRNNFuncSeqCatCUDA)


# test case for python baseline
class TestIndRNNFuncPy(unittest.TestCase):
    def setUp(self):
        self.t = torch.randn(
            3, 4, 5, dtype=torch.double, requires_grad=True)
        self.whh = torch.randn(
            5, dtype=torch.double, requires_grad=True)
        self.h0 = torch.zeros(
            4, 5, dtype=torch.double, requires_grad=True)

    def test_gradcheck_relu(self):
        self.assertTrue(gradcheck(
            _IndRNNFuncPy.apply, [self.t, self.whh, Activation.relu, self.h0]))

    def test_gradcheck_tanh(self):
        self.assertTrue(gradcheck(
            _IndRNNFuncPy.apply, [self.t, self.whh, Activation.tanh, self.h0]))


# test case for cpp baseline
class TestIndRNNFuncCPP(unittest.TestCase):
    def setUp(self):
        self.t = torch.randn(
            3, 4, 5, dtype=torch.double, requires_grad=True)
        self.whh = torch.randn(
            5, dtype=torch.double, requires_grad=True)
        self.h0 = torch.zeros(
            4, 5, dtype=torch.double, requires_grad=True)

    def test_baselinecheck_relu(self):
        pyres = _IndRNNFuncPy.apply(
            self.t, self.whh, Activation.relu, self.h0).detach()
        cppres = _IndRNNFuncCPP.apply(
            self.t, self.whh, Activation.relu, self.h0).detach()
        np.testing.assert_allclose(pyres, cppres)

    def test_baselinecheck_tanh(self):
        pyres = _IndRNNFuncPy.apply(
            self.t, self.whh, Activation.tanh, self.h0).detach()
        cppres = _IndRNNFuncCPP.apply(
            self.t, self.whh, Activation.tanh, self.h0).detach()
        np.testing.assert_allclose(pyres, cppres)

    def test_gradcheck_relu(self):
        self.assertTrue(gradcheck(
            _IndRNNFuncCPP.apply, [self.t, self.whh, Activation.relu, self.h0])
        )

    def test_gradcheck_tanh(self):
        self.assertTrue(gradcheck(
            _IndRNNFuncCPP.apply, [self.t, self.whh, Activation.tanh, self.h0])
        )


if torch.cuda.is_available():
    # test case for cuda baseline
    class TestIndRNNFuncCUDA(unittest.TestCase):
        def setUp(self):
            self.t = torch.randn(
                3, 4, 5, dtype=torch.double, requires_grad=True, device='cuda')
            self.whh = torch.randn(
                5, dtype=torch.double, requires_grad=True, device='cuda')
            self.h0 = torch.zeros(
                4, 5, dtype=torch.double, requires_grad=True, device='cuda')

        def test_baselinecheckcpp_relu(self):
            pyres = _IndRNNFuncPy.apply(
                self.t, self.whh, Activation.relu, self.h0).detach().cpu()
            cppres = _IndRNNFuncCPP.apply(
                self.t, self.whh, Activation.relu, self.h0).detach().cpu()
            np.testing.assert_allclose(pyres, cppres)

        def test_baselinecheckcpp_tanh(self):
            pyres = _IndRNNFuncPy.apply(
                self.t, self.whh, Activation.tanh, self.h0).detach().cpu()
            cppres = _IndRNNFuncCPP.apply(
                self.t, self.whh, Activation.tanh, self.h0).detach().cpu()
            np.testing.assert_allclose(pyres, cppres)

        def test_baselinecheckcuda_relu(self):
            pyres = _IndRNNFuncPy.apply(
                self.t, self.whh, Activation.relu, self.h0).detach().cpu()
            cures = _IndRNNFuncCUDA.apply(
                self.t, self.whh, Activation.relu, self.h0).detach().cpu()
            np.testing.assert_allclose(pyres, cures)

        def test_baselinecheckcuda_tanh(self):
            pyres = _IndRNNFuncPy.apply(
                self.t, self.whh, Activation.tanh, self.h0).detach().cpu()
            cures = _IndRNNFuncCUDA.apply(
                self.t, self.whh, Activation.tanh, self.h0).detach().cpu()
            np.testing.assert_allclose(pyres, cures)

        def test_gradcheckpy_relu(self):
            self.assertTrue(gradcheck(
                _IndRNNFuncPy.apply,
                (self.t, self.whh, Activation.relu, self.h0)))

        def test_gradcheckpy_tanh(self):
            self.assertTrue(gradcheck(
                _IndRNNFuncPy.apply,
                (self.t, self.whh, Activation.tanh, self.h0)))

        def test_gradcheckcpp_relu(self):
            self.assertTrue(gradcheck(
                _IndRNNFuncCPP.apply,
                (self.t, self.whh, Activation.relu, self.h0)))

        def test_gradcheckcpp_tanh(self):
            self.assertTrue(gradcheck(
                _IndRNNFuncCPP.apply,
                (self.t, self.whh, Activation.tanh, self.h0)))

        def test_gradcheckcuda_relu(self):
            self.assertTrue(gradcheck(
                _IndRNNFuncCUDA.apply,
                (self.t, self.whh, Activation.relu, self.h0)))

        def test_gradcheckcuda_tanh(self):
            self.assertTrue(gradcheck(
                _IndRNNFuncCUDA.apply,
                (self.t, self.whh, Activation.tanh, self.h0)))


class TestIndRNN(unittest.TestCase):
    def test_gradcheck_relu_cpu(self):
        t = torch.randn(
            3, 4, 5, dtype=torch.double, requires_grad=True)
        whh = torch.randn(
            5, dtype=torch.double, requires_grad=True)
        h0 = torch.zeros(
            4, 5, dtype=torch.double, requires_grad=True)
        self.assertTrue(gradcheck(indrnn, (t, whh, Activation.relu, h0)))

    def test_gradcheck_tanh_cpu(self):
        t = torch.randn(
            3, 4, 5, dtype=torch.double, requires_grad=True)
        whh = torch.randn(
            5, dtype=torch.double, requires_grad=True)
        h0 = torch.zeros(
            4, 5, dtype=torch.double, requires_grad=True)
        self.assertTrue(gradcheck(indrnn, (t, whh, Activation.tanh, h0)))

    def test_gradcheck_relu_cuda(self):
        if torch.cuda.is_available():
            t = torch.randn(
                3, 4, 5, dtype=torch.double, requires_grad=True, device='cuda')
            whh = torch.randn(
                5, dtype=torch.double, requires_grad=True, device='cuda')
            h0 = torch.zeros(
                4, 5, dtype=torch.double, requires_grad=True, device='cuda')
            self.assertTrue(gradcheck(indrnn, (t, whh, Activation.relu, h0)))

    def test_gradcheck_tanh_cuda(self):
        if torch.cuda.is_available():
            t = torch.randn(
                3, 4, 5, dtype=torch.double, requires_grad=True, device='cuda')
            whh = torch.randn(
                5, dtype=torch.double, requires_grad=True, device='cuda')
            h0 = torch.zeros(
                4, 5, dtype=torch.double, requires_grad=True, device='cuda')
            self.assertTrue(gradcheck(indrnn, (t, whh, Activation.tanh, h0)))

    def test_gradcheck_flatten(self):
        t = torch.randn(
            3, 4, 2, 3, dtype=torch.double, requires_grad=True)
        whh = torch.randn(
            6, dtype=torch.double, requires_grad=True)
        h0 = torch.zeros(
            4, 2, 3, dtype=torch.double, requires_grad=True)
        self.assertTrue(gradcheck(
            indrnn, (t, whh, Activation.relu, h0, True)))


class TestSequenceCat(unittest.TestCase):
    def setUp(self):
        self.t1 = torch.randn(4, 5, dtype=torch.double, requires_grad=True)
        self.t2 = torch.randn(1, 5, dtype=torch.double, requires_grad=True)
        self.t3 = torch.randn(3, 5, dtype=torch.double, requires_grad=True)
        self.seqcat = SequenceCat.from_sequences((self.t1, self.t2, self.t3))
        self.whh = torch.randn(
            5, dtype=torch.double, requires_grad=True)
        self.h0 = torch.zeros(
            3, 5, dtype=torch.double, requires_grad=True)

    def test_toseqs(self):
        t1, t2, t3 = self.seqcat.to_sequences()
        self.assertAlmostEqual(torch.sum(abs(t1 - self.t1)), 0)
        self.assertAlmostEqual(torch.sum(abs(t2 - self.t2)), 0)
        self.assertAlmostEqual(torch.sum(abs(t3 - self.t3)), 0)

    def test_topadseq(self):
        ps, ls = self.seqcat.to_padded_sequence(), self.seqcat.lengths
        sc = SequenceCat.from_sequences(ps.unbind(1), ls)
        self.assertAlmostEqual(torch.sum(abs(sc.data - self.seqcat.data)), 0)
        self.assertAlmostEqual(torch.sum(abs(
            sc.lengths - self.seqcat.lengths)), 0)

    def test_gradcheck_py(self):
        self.assertTrue(gradcheck(
            _IndRNNFuncSeqCatPy.apply,
            [self.seqcat.data, self.seqcat.lengths, self.whh,
             Activation.relu, self.h0]))

    def test_gradcheck_cpp(self):
        self.assertTrue(gradcheck(
            _IndRNNFuncSeqCatCPP.apply,
            [self.seqcat.data, self.seqcat.lengths, self.whh,
             Activation.tanh, self.h0]))

    def test_gradcheck_indrnn(self):
        def indrnngc(sc, whh, activ, h0):
            return indrnn(sc, whh, activ, h0).data
        self.assertTrue(gradcheck(indrnngc, (
            self.seqcat, self.whh, Activation.relu, self.h0)))

    def test_baselinecheckcpp_relu(self):
        pyres = _IndRNNFuncSeqCatPy.apply(
            self.seqcat.data, self.seqcat.lengths, self.whh,
            Activation.relu, self.h0).detach().cpu()
        cppres = _IndRNNFuncSeqCatCPP.apply(
            self.seqcat.data, self.seqcat.lengths, self.whh,
            Activation.relu, self.h0).detach().cpu()
        np.testing.assert_allclose(pyres, cppres)

    def test_baselinecheckcpp_tanh(self):
        pyres = _IndRNNFuncSeqCatPy.apply(
            self.seqcat.data, self.seqcat.lengths, self.whh,
            Activation.tanh, self.h0).detach().cpu()
        cppres = _IndRNNFuncSeqCatCPP.apply(
            self.seqcat.data, self.seqcat.lengths, self.whh,
            Activation.tanh, self.h0).detach().cpu()
        np.testing.assert_allclose(pyres, cppres)


if torch.cuda.is_available():
    class TestSequenceCatCUDA(unittest.TestCase):
        def setUp(self):
            self.t1 = torch.randn(
                4, 5, dtype=torch.double, requires_grad=True, device='cuda')
            self.t2 = torch.randn(
                1, 5, dtype=torch.double, requires_grad=True, device='cuda')
            self.t3 = torch.randn(
                3, 5, dtype=torch.double, requires_grad=True, device='cuda')
            self.seqcat = SequenceCat.from_sequences(
                (self.t1, self.t2, self.t3))
            self.whh = torch.randn(
                5, dtype=torch.double, requires_grad=True, device='cuda')
            self.h0 = torch.zeros(
                3, 5, dtype=torch.double, requires_grad=True, device='cuda')

        def test_gradcheck_cuda_tanh(self):
            self.assertTrue(gradcheck(
                _IndRNNFuncSeqCatCUDA.apply,
                [self.seqcat.data, self.seqcat.lengths, self.whh,
                 Activation.tanh, self.h0]))

        def test_gradcheck_cuda_relu(self):
            self.assertTrue(gradcheck(
                _IndRNNFuncSeqCatCUDA.apply,
                [self.seqcat.data, self.seqcat.lengths, self.whh,
                 Activation.relu, self.h0]))

        def test_baselinecheckcpp_relu(self):
            pyres = _IndRNNFuncSeqCatPy.apply(
                self.seqcat.data, self.seqcat.lengths, self.whh,
                Activation.relu, self.h0).detach().cpu()
            cppres = _IndRNNFuncSeqCatCPP.apply(
                self.seqcat.data, self.seqcat.lengths, self.whh,
                Activation.relu, self.h0).detach().cpu()
            np.testing.assert_allclose(pyres, cppres)

        def test_baselinecheckcpp_tanh(self):
            pyres = _IndRNNFuncSeqCatPy.apply(
                self.seqcat.data, self.seqcat.lengths, self.whh,
                Activation.tanh, self.h0).detach().cpu()
            cppres = _IndRNNFuncSeqCatCPP.apply(
                self.seqcat.data, self.seqcat.lengths, self.whh,
                Activation.tanh, self.h0).detach().cpu()
            np.testing.assert_allclose(pyres, cppres)

        def test_baselinecheckcuda_relu(self):
            pyres = _IndRNNFuncSeqCatPy.apply(
                self.seqcat.data, self.seqcat.lengths, self.whh,
                Activation.relu, self.h0).detach().cpu()
            cudares = _IndRNNFuncSeqCatCUDA.apply(
                self.seqcat.data, self.seqcat.lengths, self.whh,
                Activation.relu, self.h0).detach().cpu()
            np.testing.assert_allclose(pyres, cudares)

        def test_baselinecheckcuda_tanh(self):
            pyres = _IndRNNFuncSeqCatPy.apply(
                self.seqcat.data, self.seqcat.lengths, self.whh,
                Activation.tanh, self.h0).detach().cpu()
            cudares = _IndRNNFuncSeqCatCUDA.apply(
                self.seqcat.data, self.seqcat.lengths, self.whh,
                Activation.tanh, self.h0).detach().cpu()
            np.testing.assert_allclose(pyres, cudares)


if __name__ == '__main__':
    unittest.main()
