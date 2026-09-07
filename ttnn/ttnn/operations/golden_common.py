# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


_MISSING = object()


def golden_identity(value=_MISSING, *_, **kwargs):
    """Return the logical value unchanged."""

    if value is _MISSING:
        for argument_name in ("input_tensor", "tensor", "input"):
            if argument_name in kwargs:
                return kwargs[argument_name]
        raise TypeError("golden_identity requires a value")
    return value


def golden_torch_dtype_for_ttnn(dtype, *, default=None):
    """Translate a TTNN dtype to its Torch counterpart, with an optional default."""

    if dtype is None:
        return default

    import ttnn

    return ttnn.ttnn_dtype_to_torch_dtype(dtype)


def golden_normalize_shape(shape):
    """Normalize a TTNN, list, or tuple shape to a tuple."""

    return tuple(shape)


def golden_apply_fused_activations(tensor, activation=None, *, program_config=None):
    """Apply the program-config fused activation or the explicitly requested activation."""

    if program_config is not None:
        program_config_activation = getattr(program_config, "fused_activation", None)
        if program_config_activation:
            activation = program_config_activation

    if activation is None:
        return tensor

    from ttnn.operations.activations import get_golden_function_for_activation

    return get_golden_function_for_activation(activation)(tensor)


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
    """Preserve optional output positions, using None for unrequested values."""

    if len(values) != len(required):
        raise ValueError("Output values and requirements must have equal length")

    return [value if is_required else None for value, is_required in zip(values, required)]


def golden_assemble_conditional_result(primary, *conditional_parts):
    """Return a primary result alone or append enabled values in a tuple."""

    result = [primary]
    result.extend(value for include, value in conditional_parts if include)
    return result[0] if len(result) == 1 else tuple(result)
