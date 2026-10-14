# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


def make_aliasing_preprocess(argument_aliases):
    """Return a golden input preprocess that renames keyword aliases to their canonical golden parameter names.
    It records the aliases so global comparison applies the same renaming to its cached inputs.
    """

    def preprocess_golden_function_inputs(function_args, function_kwargs):
        golden_args, golden_kwargs = ttnn.decorators.default_preprocess_golden_function_inputs(
            function_args, function_kwargs
        )
        for alias, canonical_name in argument_aliases.items():
            if alias in golden_kwargs and canonical_name not in golden_kwargs:
                golden_kwargs[canonical_name] = golden_kwargs.pop(alias)
        golden_kwargs["_ttnn_golden_argument_aliases"] = argument_aliases
        return golden_args, golden_kwargs

    return preprocess_golden_function_inputs


def golden_to_output_dtype(tensor, dtype):
    """Return the values TTNN stores for a golden result written in the requested output dtype."""

    import torch

    if dtype is None or not isinstance(tensor, torch.Tensor):
        return tensor
    if dtype in (ttnn.bfloat8_b, ttnn.bfloat4_b):
        # Block floats share one exponent per 16 values, so small values beside large ones lose precision;
        # round-trip through host packing to model the stored values.
        return ttnn.Tensor(tensor=tensor.contiguous(), data_type=dtype, layout=ttnn.TILE_LAYOUT).to_torch()
    return tensor.to(ttnn.ttnn_dtype_to_torch_dtype(dtype))


def golden_compute_gradients(output, inputs, grad_output):
    """Compute ordered gradients, preserving None for inputs unused by the output."""

    import torch

    return list(torch.autograd.grad(output, inputs, grad_outputs=grad_output, allow_unused=True))


def golden_prepare_grad_inputs(*inputs):
    """Enable autograd for golden inputs without detaching an existing graph."""

    prepared_inputs = []
    for input_tensor in inputs:
        if not input_tensor.requires_grad:
            input_tensor.requires_grad_(True)
        prepared_inputs.append(input_tensor)
    return tuple(prepared_inputs)


def golden_pack_complex_gradient(gradient):
    """Pack a complex gradient as concatenated real and imaginary halves."""

    if gradient is None:
        return None

    import torch

    return torch.cat((torch.real(gradient), torch.imag(gradient)), dim=-1)


def golden_select_optional_outputs(values, required):
    """Preserve optional output positions, using None for unrequested values.
    A required of None requests every output.
    """

    if required is None:
        return values
    if len(values) != len(required):
        raise ValueError("Output values and requirements must have equal length")

    return [value if is_required else None for value, is_required in zip(values, required)]
