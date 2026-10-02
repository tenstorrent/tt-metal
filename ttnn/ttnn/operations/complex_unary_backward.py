# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from ttnn.operations.golden_common import (
    golden_compute_gradients,
    golden_pack_complex_gradient,
    golden_prepare_grad_inputs,
)


def _complex_unary_backward_reference(forward, grad_tensor, input_tensor):
    (input_tensor,) = golden_prepare_grad_inputs(input_tensor)
    output = forward(input_tensor)
    (input_gradient,) = golden_compute_gradients(output, (input_tensor,), grad_tensor)
    return [golden_pack_complex_gradient(input_gradient)]


def _make_complex_unary_backward(forward):
    def golden_function(grad_tensor, input_tensor, *args, **kwargs):
        import torch

        forward_function = getattr(torch, forward) if isinstance(forward, str) else forward
        return _complex_unary_backward_reference(forward_function, grad_tensor, input_tensor)

    return golden_function


def _polar_forward(input_tensor):
    import torch

    return torch.polar(input_tensor.real, input_tensor.imag)


ttnn.attach_golden_function(ttnn.real_bw, golden_function=_make_complex_unary_backward("real"))
ttnn.attach_golden_function(ttnn.imag_bw, golden_function=_make_complex_unary_backward("imag"))
ttnn.attach_golden_function(ttnn.angle_bw, golden_function=_make_complex_unary_backward("angle"))
ttnn.attach_golden_function(ttnn.conj_bw, golden_function=_make_complex_unary_backward("conj"))
ttnn.attach_golden_function(ttnn.polar_bw, golden_function=_make_complex_unary_backward(_polar_forward))

__all__ = []
