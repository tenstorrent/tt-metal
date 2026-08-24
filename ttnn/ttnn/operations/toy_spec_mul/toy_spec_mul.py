# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn

from .toy_spec_mul_program_artifacts import create_program_artifacts


def toy_spec_mul(
    a: ttnn.Tensor,
    b: ttnn.Tensor,
    *,
    out: ttnn.Tensor | None = None,
    tile_limit: int | None = None,
) -> ttnn.Tensor:
    """Elementwise multiply via a Metal 2.0 ProgramSpec.

    `tile_limit` stops after that many output tiles, leaving the rest of `out` untouched. It
    varies only runtime arg values, so it exercises the cache-hit refresh path.
    """
    if a.shape != b.shape:
        raise NotImplementedError(f"toy_spec_mul requires matching shapes, got {a.shape} and {b.shape}")
    if a.layout != ttnn.TILE_LAYOUT or b.layout != ttnn.TILE_LAYOUT:
        raise NotImplementedError("toy_spec_mul requires TILE_LAYOUT inputs")
    if a.dtype != ttnn.bfloat16 or b.dtype != ttnn.bfloat16:
        raise NotImplementedError(f"toy_spec_mul requires bfloat16, got {a.dtype} and {b.dtype}")

    if out is None:
        out = ttnn.allocate_tensor_on_device(a.spec, a.device())
    # The factory owns every name it declares, so it returns the tensor bindings alongside the
    # spec and run args. Nothing here needs to know what those names are.
    io_tensors = [a, b, out]
    spec, run_args, tensor_indices = create_program_artifacts(a, b, out, tile_limit=tile_limit)
    return ttnn.generic_op(io_tensors, spec, run_args, tensor_indices)
