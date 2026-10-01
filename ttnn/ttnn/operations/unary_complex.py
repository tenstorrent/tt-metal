# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn

__all__ = []


def _preprocess_complex_golden_inputs(function_args, function_kwargs):
    """Convert ComplexTensor arguments into Torch complex tensors before default preprocessing."""

    import torch

    def convert(value):
        if isinstance(value, ttnn._ttnn.operations.complex.ComplexTensor):
            # Default preprocessing only converts ttnn.Tensor, so the two-component wrapper would reach Torch as-is.
            real = ttnn.decorators.to_torch_for_comparison(value.real).float()
            imag = ttnn.decorators.to_torch_for_comparison(value.imag).float()
            return torch.complex(real, imag)
        return value

    function_args = tuple(convert(arg) for arg in function_args)
    function_kwargs = {key: convert(value) for key, value in function_kwargs.items()}
    return ttnn.decorators.default_preprocess_golden_function_inputs(function_args, function_kwargs)


def _golden_function(input_tensor_a, *args, **kwargs):
    import torch

    return torch.real(input_tensor_a)


ttnn.attach_golden_function(
    ttnn.real, golden_function=_golden_function, preprocess_golden_function_inputs=_preprocess_complex_golden_inputs
)


def _golden_function(input_tensor_a, *args, **kwargs):
    import torch

    return torch.imag(input_tensor_a)


ttnn.attach_golden_function(
    ttnn.imag, golden_function=_golden_function, preprocess_golden_function_inputs=_preprocess_complex_golden_inputs
)


def _golden_function(input_tensor_a, *args, **kwargs):
    import torch

    return torch.angle(input_tensor_a)


ttnn.attach_golden_function(
    ttnn.angle, golden_function=_golden_function, preprocess_golden_function_inputs=_preprocess_complex_golden_inputs
)


def _golden_function(input_tensor_a, *args, **kwargs):
    import torch

    # ttnn.is_imag is eqz(real part): true where the value is purely imaginary.
    return torch.real(input_tensor_a) == 0


ttnn.attach_golden_function(
    ttnn.is_imag, golden_function=_golden_function, preprocess_golden_function_inputs=_preprocess_complex_golden_inputs
)


def _golden_function(input_tensor_a, *args, **kwargs):
    import torch

    # ttnn.is_real is eqz(imag part): true where the value is purely real.
    return torch.isreal(input_tensor_a)


ttnn.attach_golden_function(
    ttnn.is_real, golden_function=_golden_function, preprocess_golden_function_inputs=_preprocess_complex_golden_inputs
)


def _golden_function(input_tensor_a, *args, **kwargs):
    import torch

    return torch.abs(input_tensor_a)


ttnn.attach_golden_function(
    ttnn.abs, golden_function=_golden_function, preprocess_golden_function_inputs=_preprocess_complex_golden_inputs
)


def _golden_function(input_tensor_a, *args, **kwargs):
    import torch

    return torch.conj(input_tensor_a)


ttnn.attach_golden_function(
    ttnn.conj, golden_function=_golden_function, preprocess_golden_function_inputs=_preprocess_complex_golden_inputs
)


def _golden_function(input_tensor_a, *args, **kwargs):
    import torch

    # The ComplexTensor stores radius in its real component and angle in its imaginary component.
    return torch.polar(input_tensor_a.real, input_tensor_a.imag)


ttnn.attach_golden_function(
    ttnn.polar, golden_function=_golden_function, preprocess_golden_function_inputs=_preprocess_complex_golden_inputs
)


def _golden_function(input_tensor_a, *args, **kwargs):
    import torch

    # Keep reciprocal's signed infinities and NaNs intact when complex registrations are loaded last.
    # Replacing them with finite extrema would undo the real unary golden's special-value contract.
    return torch.reciprocal(input_tensor_a)


ttnn.attach_golden_function(
    ttnn.reciprocal,
    golden_function=_golden_function,
    preprocess_golden_function_inputs=_preprocess_complex_golden_inputs,
)


def _golden_function_complex_tensor(real, imag, *args, **kwargs):
    import torch

    # The op returns a ComplexTensor wrapping (real, imag); the golden is the equivalent complex-valued tensor.
    return torch.complex(real, imag)


ttnn.attach_golden_function(ttnn.complex_tensor, golden_function=_golden_function_complex_tensor)


__all__ = []
