# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn

from .toy_spec_mul_program_spec import TP_A, TP_B, TP_OUT, create_program_spec


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
    spec, run_args = create_program_spec(a, b, out, tile_limit=tile_limit)
    return ttnn.generic_op([a, b, out], spec, run_args, {TP_A: 0, TP_B: 1, TP_OUT: 2})
