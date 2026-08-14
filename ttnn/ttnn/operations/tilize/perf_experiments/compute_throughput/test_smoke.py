# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Smoke: every arm compiles and arm 0 reproduces the op.

Tiny shape, correctness only (no perf claim) — this is the cheap gate before the
measured runs.
"""
import pytest
import ttnn

from ._harness import VARIANTS, run

SHAPE = [1, 1, 128, 128]
CORES = 4


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_smoke_bf16(device, variant):
    ns, exact = run(device, variant, SHAPE, cores=CORES, dtype=ttnn.bfloat16, measure=False, label="smoke/bf16")
    if variant != 3:  # arm 3 is the payload floor: wrong output by construction
        assert exact, f"arm {variant} not bit-exact"


@pytest.mark.parametrize("variant", [0, 4, 5, 6, 7])
def test_smoke_uint32(device, variant):
    """uint32 takes the REGULAR tilize path — where arms 4/5/6/7 actually differ."""
    ns, exact = run(device, variant, SHAPE, cores=CORES, dtype=ttnn.uint32, measure=False, label="smoke/uint32")
    assert exact, f"arm {variant} not bit-exact"


@pytest.mark.parametrize("variant", [0, 4, 5])
def test_smoke_tiny_tile(device, variant):
    """tile_h=8 also takes the regular path (fast tilize needs 32x32 output tiles)."""
    ns, exact = run(
        device, variant, SHAPE, cores=CORES, dtype=ttnn.bfloat16, tile_h=8, measure=False, label="smoke/tile_h8"
    )
    assert exact, f"arm {variant} not bit-exact"
