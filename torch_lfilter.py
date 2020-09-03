""" Bring lowpass filtering to PyTorch! """

## Metadata -------------------------------------------------------------------

__version__ = "0.0.3"
__author__ = "Floris Laporte"
__all__ = ["lfilter"]


## Imports --------------------------------------------------------------------

import torch
import warnings


## Constants ------------------------------------------------------------------

WARNING_MSG = (
    "no efficient C++ lfilter implementation for %s-tensors found. "
    "falling back to a (much slower) pure python implementation.\n"
    "So far, only an efficient implementation for cpu-tensors exists. "
    "Consider placing your '%s'-tensor on the CPU."
)


## C++ Extension Imports ------------------------------------------------------

try:
    from torch_lfilter_cpp import _lfilter_cpu_forward, _lfilter_cpu_backward
except ImportError:
    _lfilter_cpu_forward = None
    _lfilter_cpu_backward = None
    warnings.warn(
        "no efficient C++ lfilter implementation for cpu-tensors found. "
        "falling back to a (much slower) pure python implementation.\n\n"
        "Maybe something went wrong during the compilation of torch_lfilter? "
        "Please check out the installation instructions at "
        "https://github.com/flaport/torch_lfilter."
    )

try:
    from torch_lfilter_cpp import _lfilter_cuda_forward, _lfilter_cuda_backward
except ImportError:
    _lfilter_cuda_forward = None
    _lfilter_cuda_backward = None


## lfilter --------------------------------------------------------------------


def lfilter(b, a, x):
    """PyTorch lfilter

    Args:
        b (torch.Tensor): The numerator coefficient vector in a 1-D sequence.
        a (torch.Tensor): The denominator coefficient vector in a 1-D sequence.
            if ``a[0]`` is not 1, then both ``a`` and ``b`` are normalized by ``a[0]``.
        x (torch.Tensor): An N-dimensional input tensor.

    Note:
        The filtering happens along dimension (axis) 0.

    """
    y = _LFilter.apply(
        b, a, x.reshape(x.shape[0], -1).to(dtype=torch.float64, device=x.device)
    )
    return y.reshape(*x.shape).to(device=x.device, dtype=x.dtype)


## LFilter --------------------------------------------------------------------


def _lfilter_general_forward(x, y, b, a, order, num_timesteps):
    """ general lfilter implementation valid for all devices """
    y[0] += b[-1] * x[0]
    for n in range(1, order, 1):
        y[n] += (b[-1 - n :] * x[: n + 1]).sum(0)
        y[n] -= (a[-n:] * y[:n]).sum(0)

    for n in range(order, num_timesteps, 1):
        y[n] += (b * x[n - order + 1 : n + 1]).sum(0)
        y[n] -= (a * y[n - order + 1 : n]).sum(0)


def _lfilter_general_backward(dL_dx, dL_dy, b, a, order, num_timesteps):
    """ general lfilter backward implementation valid for all devices """
    for n in range(num_timesteps - 1, order - 1, -1):
        dL_dy[n - order + 1 : n] -= a * dL_dy[n : n + 1]
        dL_dx[n - order + 1 : n + 1] += b * dL_dy[n : n + 1]

    for n in range(order - 1, 0, -1):
        dL_dy[:n] -= a[-n:] * dL_dy[n : n + 1]
        dL_dx[: n + 1] += b[-n - 1 :] * dL_dy[n : n + 1]
    dL_dx[0] += b[-1] * dL_dy[0]


class _LFilter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, b, a, x):
        if not (b.ndim == a.ndim == 1):
            raise ValueError("filter vectors b and a should be 1D.")
        b = torch.tensor(
            [float(bb) / float(a[0]) for bb in b], dtype=x.dtype, device=x.device
        )[:, None]
        a = torch.tensor(
            [float(aa) / float(a[0]) for aa in reversed(a[1:])],
            dtype=x.dtype,
            device=x.device,
        )[:, None]
        order = b.shape[0]
        num_timesteps = x.shape[0]
        ctx.save_for_backward(b, a)

        y = torch.zeros_like(x)

        if x.device == torch.device("cpu"):
            if _lfilter_cpu_forward is not None:
                _lfilter_forward = _lfilter_cpu_forward
            else:
                warnings.warn(WARNING_MSG % (x.device, x.device))
                _lfilter_forward = _lfilter_general_forward
        elif x.device == torch.device("cuda"):
            if _lfilter_cuda_forward is not None:
                _lfilter_forward = _lfilter_cuda_forward
            else:
                warnings.warn(WARNING_MSG % (x.device, x.device))
                _lfilter_forward = _lfilter_general_forward
        else:
            warnings.warn(WARNING_MSG % (x.device, x.device))
            _lfilter_forward = _lfilter_general_forward

        _lfilter_forward(x, y, b, a, order, num_timesteps)

        return y

    @staticmethod
    def backward(ctx, dL_dy):
        b, a = ctx.saved_tensors
        order = b.shape[0]
        num_timesteps = dL_dy.shape[0]

        dL_dy = dL_dy.clone()  # allow inplace operations on dL_dy
        dL_dx = torch.zeros_like(dL_dy)

        if dL_dy.device == torch.device("cpu"):
            if _lfilter_cpu_backward is not None:
                _lfilter_backward = _lfilter_cpu_backward
            else:
                _lfilter_backward = _lfilter_general_backward
        elif dL_dy.device == torch.device("cuda"):
            if _lfilter_cuda_backward is not None:
                _lfilter_backward = _lfilter_cuda_backward
            else:
                _lfilter_backward = _lfilter_general_backward
        else:
            _lfilter_backward = _lfilter_general_backward

        _lfilter_backward(dL_dx, dL_dy, b, a, order, num_timesteps)

        return None, None, dL_dx
