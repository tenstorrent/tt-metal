# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision gate of the tensix_untilize_bf16 candidate: bf16 TILE -> ROW_MAJOR must be bit-identical.

Every case compares int16 views of the output and the logical input. Inputs:
  all65536_*   every bf16 bit pattern, randomly permuted (several copies when the shape is larger),
               spread over many tiles and all four faces.
  pad_*        non-tile-aligned shapes holding all patterns cyclically, tile padding poisoned with
               1000.0 or NaN (padding must never reach the output).
Env G6TU_PRESERVE=0 runs the unpinned variant (expected to fail on exponent-0 patterns); the
mismatch census is printed either way (grep G6TU_EXACT).
"""

import os
import sys
from pathlib import Path

import pytest

# Test-only dependency; the ttnn package forbids a module-level torch import.
torch = pytest.importorskip("torch")
import ttnn

sys.path.insert(0, str(Path(__file__).parent))
import g6tu_program as g6tu  # noqa: E402


def _all_patterns(numel, seed):
    gen = torch.Generator()
    gen.manual_seed(seed)
    copies = -(-numel // 65536)
    chunks = [torch.randperm(65536, generator=gen) for _ in range(copies)]
    bits = torch.cat(chunks)[:numel].to(torch.int32)
    bits = torch.where(bits >= 0x8000, bits - 0x10000, bits).to(torch.int16)
    return bits


def _census(got_bits, exp_bits):
    bad = got_bits != exp_bits
    n = int(bad.sum())
    if n == 0:
        return 0, ""
    e = exp_bits[bad].to(torch.int32) & 0xFFFF
    g = got_bits[bad].to(torch.int32) & 0xFFFF
    exp_field = (e >> 7) & 0xFF
    classes = {
        "neg_zero": int(((e == 0x8000)).sum()),
        "subnormal": int(((exp_field == 0) & ((e & 0x7F) != 0)).sum()),
        "nan": int(((exp_field == 0xFF) & ((e & 0x7F) != 0)).sum()),
        "inf": int(((e & 0x7FFF) == 0x7F80).sum()),
        "normal": int(((exp_field != 0) & (exp_field != 0xFF)).sum()),
        "pos_zero": int((e == 0).sum()),
    }
    sample = [(hex(int(a)), hex(int(b))) for a, b in zip(e[:8], g[:8])]
    return n, f"classes={classes} sample(exp->got)={sample}"


CASES = {
    # name: (shape, max_block_cols, poison)
    "all65536_c2036": ((1, 1024, 2560), 0, None),
    "all65536_c2036_k8": ((1, 1024, 2560), 8, None),
    "all65536_c2036_k4": ((1, 1024, 2560), 4, None),
    "all65536_256x256": ((1, 256, 256), 0, None),
    "all65536_c295": ((1, 8192, 128), 0, None),
    "all65536_c2006": ((1, 64, 2560), 0, None),
    "all65536_c297": ((1, 128, 256, 32), 0, None),
    "pad_c290_1000": ((1, 196), 0, 1000.0),
    "pad_c290_nan": ((1, 196), 0, "nan"),
    "pad_c309_nan": ((1, 185), 0, "nan"),
    "pad_c311_1000": ((1, 207), 0, 1000.0),
    "pad_c2026_nan": ((8, 1, 2560), 0, "nan"),
    "pad_c1993_1000": ((16, 1, 2560), 0, 1000.0),
    "pad_3x50x1000_nan": ((3, 50, 1000), 0, "nan"),
    "pad_3x50x1000_1000_k8": ((3, 50, 1000), 8, 1000.0),
    "pad_c294_nan": ((1, 131072), 0, "nan"),
    "pad_2x33x77_nan_k4": ((2, 33, 77), 4, "nan"),
    # non-aligned, no poison: host tilize pads with zeros; every live pattern must survive
    "nonaligned_3x50x1000": ((3, 50, 1000), 0, None),
    "nonaligned_c311": ((1, 207), 0, None),
    "nonaligned_c2026": ((8, 1, 2560), 0, None),
    "nonaligned_c294": ((1, 131072), 0, None),
    "nonaligned_2x33x77_k4": ((2, 33, 77), 4, None),
}


@pytest.mark.parametrize("name", list(CASES))
def test_exact(device, name):
    shape, max_cols, poison = CASES[name]
    preserve = int(os.environ.get("G6TU_PRESERVE", "1"))
    buf = os.environ.get("G6TU_BUF", "DRAM")
    mem = ttnn.L1_MEMORY_CONFIG if buf == "L1" else ttnn.DRAM_MEMORY_CONFIG
    numel = 1
    for d in shape:
        numel *= d
    bits = _all_patterns(numel, seed=hash(name) & 0xFFFF).reshape(shape)
    x = bits.view(torch.bfloat16)
    tt_x = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem, preserve_nan_values=True
    )
    if poison is not None:
        tt_x = ttnn.fill_implicit_tile_padding(tt_x, float("nan") if poison == "nan" else float(poison))
    # What the device input holds (host untilize of the TILE tensor): fill_implicit_tile_padding is a
    # device op and must not have touched a live datum either.
    held = ttnn.to_torch(tt_x).reshape(shape).view(torch.int16)
    n_in, detail_in = _census(held.reshape(-1), bits.reshape(-1))
    print(f"\nG6TU_INPUT {name} poison={poison} input-vs-host mismatches={n_in} {detail_in}")
    knobs = g6tu.Knobs(
        max_block_cols=max_cols, preserve_zero_flag=preserve, cb_format=os.environ.get("G6TU_FMT", "uint16")
    )
    plan = g6tu.plan_for(tt_x, mem, knobs)
    out = g6tu.run(tt_x, mem, knobs, plan=plan)
    got = ttnn.to_torch(out).reshape(shape).view(torch.int16)
    # Reference: what the device input holds. With poison, ttnn.fill_implicit_tile_padding (a device op on
    # the Tensix float path) has already flushed live exponent-0 datums and NaN payloads of the input
    # itself (G6TU_INPUT); without poison the device input equals the host bits (asserted below).
    n, detail = _census(got.reshape(-1), held.reshape(-1))
    if poison is None:
        assert n_in == 0, f"{name}: host -> device input altered {n_in} datums {detail_in}"
    print(
        f"\nG6TU_EXACT {name} shape={shape} fmt={knobs.cb_format} preserve={preserve} {g6tu.describe(plan)} mismatches={n} {detail}"
    )
    assert n == 0, f"{name}: {n} mismatching datums {detail}"


def _tile_bytes_to_rows(out_bits, shape):
    """Diag modes 2/3 write each tile's bytes in face order as its 32 rows: undo that (all-live shapes)."""
    h, w = shape[-2], shape[-1]
    t = out_bits.reshape(h // 32, 32, w // 32, 32).permute(0, 2, 1, 3).reshape(h // 32, w // 32, 1024)
    r = torch.arange(32).view(32, 1)
    c = torch.arange(32).view(1, 32)
    idx = (2 * (r // 16) + c // 16) * 256 + (r % 16) * 16 + (c % 16)
    rows = t[:, :, idx.reshape(-1)].reshape(h // 32, w // 32, 32, 32).permute(0, 2, 1, 3).reshape(h, w)
    return rows


@pytest.mark.parametrize("mode", [1, 2, 3])
def test_diag(device, mode):
    """Which stage alters a pattern: 1 copy+pack_untilize_dest, 2 copy+pack_tile, 3 as 2 with flush flag."""
    shape = (256, 256)
    bits = _all_patterns(65536, seed=7).reshape(shape)
    x = bits.view(torch.bfloat16)
    mem = ttnn.DRAM_MEMORY_CONFIG
    tt_x = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem, preserve_nan_values=True
    )
    knobs = g6tu.Knobs(max_block_cols=1, diag_mode=mode)
    plan = g6tu.plan_for(tt_x, mem, knobs)
    out = g6tu.run(tt_x, mem, knobs, plan=plan)
    got = ttnn.to_torch(out).reshape(shape).view(torch.int16)
    if mode in (2, 3):
        got = _tile_bytes_to_rows(got, shape)
    n, detail = _census(got.reshape(-1), bits.reshape(-1))
    print(f"\nG6TU_DIAG mode={mode} {g6tu.describe(plan)} mismatches={n} {detail}")
