# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn

__all__ = []


def _golden_function(input_tensor_a, *args, **kwargs):
    import torch

    return torch.real(input_tensor_a)


ttnn.attach_golden_function(ttnn.real, golden_function=_golden_function)


def _golden_function(input_tensor_a, *args, **kwargs):
    import torch

    return torch.imag(input_tensor_a)


ttnn.attach_golden_function(ttnn.imag, golden_function=_golden_function)


def _golden_function(input_tensor_a, *args, **kwargs):
    import torch

    return torch.angle(input_tensor_a)


ttnn.attach_golden_function(ttnn.angle, golden_function=_golden_function)


def _golden_function(input_tensor_a, *args, **kwargs):
    import torch

    # ttnn.is_imag is eqz(real part): true where the value is purely imaginary.
    return torch.real(input_tensor_a) == 0


ttnn.attach_golden_function(ttnn.is_imag, golden_function=_golden_function)


def _golden_function(input_tensor_a, *args, **kwargs):
    import torch

    # ttnn.is_real is eqz(imag part): true where the value is purely real.
    return torch.isreal(input_tensor_a)


ttnn.attach_golden_function(ttnn.is_real, golden_function=_golden_function)


def _golden_function(input_tensor_a, *args, **kwargs):
    import torch

    return torch.abs(input_tensor_a)


ttnn.attach_golden_function(ttnn.abs, golden_function=_golden_function)


def _golden_function(input_tensor_a, *args, **kwargs):
    import torch

    return torch.conj(input_tensor_a)


ttnn.attach_golden_function(ttnn.conj, golden_function=_golden_function)


def _golden_function(input_tensor_a, *args, **kwargs):
    import torch

    return torch.polar(input_tensor_a)


ttnn.attach_golden_function(ttnn.polar, golden_function=_golden_function)


def _golden_function(input_tensor_a, *args, **kwargs):
    import torch

    return torch.reciprocal(input_tensor_a)


ttnn.attach_golden_function(ttnn.reciprocal, golden_function=_golden_function)


__all__ = []
