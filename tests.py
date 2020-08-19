""" tests for torch_lfilter """

import torch
import numpy
import pytest
from scipy.signal import lfilter as scipy_lfilter, butter
from torch_lfilter import lfilter as torch_lfilter


@pytest.fixture
def gen():
    """ default pytorch random number generator """
    return torch.Generator(device="cpu").manual_seed(37)


@pytest.fixture
def stream(gen):
    """ default bit stream to be lowpass filtered """
    num_bits = 10
    num_samples_per_bit = 8  # eg: bitrate=20e9, samplerate=160e9
    stream = (
        (torch.randn(num_bits, generator=gen) > 0)[:, None]
        .expand(num_bits, num_samples_per_bit)
        .flatten()
        .float()
        .requires_grad_()
    )
    return stream


@pytest.fixture
def a():
    normal_cutoff = 0.1  # eg: bitrate=20e9, samplerate=160e9
    _, a = butter(N=4, Wn=normal_cutoff, btype="lowpass", analog=False)
    return a


@pytest.fixture
def b():
    normal_cutoff = 0.1  # eg: bitrate=20e9, samplerate=160e9
    b, _ = butter(N=4, Wn=normal_cutoff, btype="lowpass", analog=False)
    return b


def test_scipy_vs_torch(b, a, stream):
    """ compare lowpass filtered result of scipy_lfilter vs torch_lfiler """
    scipy_stream = scipy_lfilter(b, a, stream.data.cpu().numpy(), axis=0)
    torch_stream = torch_lfilter(b, a, stream).data.cpu().numpy()
    numpy.testing.assert_almost_equal(scipy_stream, torch_stream)


def test_gradcheck(b, a, stream):
    """ compare lowpass filtered result of scipy_lfilter vs torch_lfiler """
    torch.autograd.gradcheck(torch_lfilter, [b, a, stream.double()])


if __name__ == "__main__":
    pytest.main([__file__])
