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
    if dtype == ttnn.bfloat2_b:
        return _bfloat2_b_device_pack(tensor)
    if dtype in (ttnn.bfloat8_b, ttnn.bfloat4_b):
        # Block floats share one exponent per 16 values, so small values beside large ones lose precision;
        # round-trip through host packing to model the stored values.
        return ttnn.Tensor(tensor=tensor.contiguous(), data_type=dtype, layout=ttnn.TILE_LAYOUT).to_torch()
    return tensor.to(ttnn.ttnn_dtype_to_torch_dtype(dtype))


def _bfloat2_b_device_pack(tensor):
    """Model the device's BFLOAT16 -> BFLOAT2_B pack on the host.

    BFLOAT2_B keeps one sign bit and one magnitude bit per element with one 8-bit exponent shared by each 16-element
    face row, so every element stores 0 or +/- 2^(shared exponent). The host packer (bfp_simd::row_scalar, reached by
    ttnn.Tensor(..., data_type=ttnn.bfloat2_b)) rounds the significand straight to one bit, keeping anything above half
    the block maximum. The device rounds to BFP8 first and then truncates the 7-bit magnitude to one bit, so only
    elements in the block maximum's power-of-two range survive. For a block [1.0, 0.75, 0, ...] the host packer stores
    1.0 and the device stores 0 for the second element, so packing through the host here would compare a device result
    against values the hardware cannot produce.

    The mantissa is taken from the bfloat16 the tensor would be unpacked from, and the 16-element blocks follow the
    tile layout: each 32x32 tile holds four 16x16 faces, so each face row is one block. Float32 results packed from a
    32-bit Dest can still differ from the device in rare elements.
    """
    import torch

    block = 16
    tile = 32
    source = tensor.detach().to(torch.bfloat16)
    if source.dim() < 2:
        source = source.reshape(1, -1)

    height, width = source.shape[-2], source.shape[-1]
    pad_h, pad_w = (-height) % tile, (-width) % tile
    padded = source.reshape(-1, height, width)
    if pad_h or pad_w:
        padded = torch.nn.functional.pad(padded, (0, pad_w, 0, pad_h))
    padded_h, padded_w = height + pad_h, width + pad_w

    # Each 32x32 tile holds four 16x16 faces, and every face row is one 16-element block with its own exponent.
    flat = padded.reshape(-1, padded_h // tile, tile, padded_w // tile, tile).permute(0, 1, 3, 2, 4).reshape(-1)
    bits = flat.view(torch.int16).to(torch.int32) & 0xFFFF
    signs = bits >> 15
    exponents = (bits >> 7) & 0xFF
    # Implicit leading one, dropping the two mantissa bits BFP8 does not keep.
    mantissas = ((bits & 0x7F) >> 1) | 0x40

    num_blocks = flat.numel() // block
    signs = signs.view(num_blocks, block)
    exponents = exponents.view(num_blocks, block)
    mantissas = mantissas.view(num_blocks, block)

    shared_exponents = exponents.max(dim=1, keepdim=True).values
    deltas = shared_exponents - exponents
    shifts = deltas.clamp(max=8)
    guard_shifts = (deltas - 1).clamp(min=0, max=8)
    guard = torch.where(deltas > 0, (mantissas >> guard_shifts) & 1, torch.zeros_like(mantissas))
    # Round to BFP8 (round to nearest, ties away from zero), then truncate to the single BFP2 magnitude bit.
    bfp8_mantissas = ((mantissas >> shifts) + guard) & 0x7F
    # Exponent-zero inputs (zeros, denormals) store 0; their implicit one would otherwise survive an all-zero block.
    magnitudes = torch.where(exponents == 0, torch.zeros_like(bfp8_mantissas), bfp8_mantissas >> 6)
    values = magnitudes.float() * torch.exp2((shared_exponents - 127).float())
    values = torch.where(signs.bool(), -values, values).flatten()

    untiled = (
        values.reshape(-1, padded_h // tile, padded_w // tile, tile, tile)
        .permute(0, 1, 3, 2, 4)
        .reshape(-1, padded_h, padded_w)[:, :height, :width]
    )
    return untiled.reshape(tensor.shape).to(torch.float32)


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
