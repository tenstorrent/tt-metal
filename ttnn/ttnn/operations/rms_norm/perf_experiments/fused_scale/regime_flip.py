# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only: what would DELETING cb_normed buy the op's blocking plan?

The fusion's non-perf payoff is L1: `cb_normed` is BLOCK_HT * WT_SCALE_BLOCK
pages of the working set that `blocking_plan`'s `fits` predicate tests, and the
Regime A/B choice (ONE DRAM read of x vs TWO) hangs off that predicate.  This
re-runs the op's OWN plan with `_cb_layout` patched to the fused CB set
(cb_normed gone, cb_rms_full = BLOCK_HT pages added) and diffs regime, W-chunk,
buffer depths and BLOCK_HT.  Read-only: it patches a copy in this process, never
the op on disk.
"""

from __future__ import annotations

from types import SimpleNamespace

import ttnn

from ttnn.operations.rms_norm import rms_norm_program_descriptor as opd

SHAPES = [
    ((1, 1, 32, 7168), ttnn.bfloat16, "focus (goal shape)"),
    ((1, 1, 32, 5120), ttnn.bfloat16, "decode_5120"),
    ((1, 1, 32, 4096), ttnn.bfloat16, "decode_4096"),
    ((1, 1, 32, 2304), ttnn.bfloat16, "decode_2304"),
    ((1, 1, 32, 1024), ttnn.bfloat16, "decode_1024"),
    ((1, 1, 8192, 7168), ttnn.bfloat16, "prefill_7168"),
    ((1, 1, 8192, 1024), ttnn.bfloat16, "prefill_1024"),
    ((1, 1, 32, 4095), ttnn.bfloat16, "w_nonalign"),
    ((1, 1, 32, 7168), ttnn.bfloat8_b, "focus bf8b"),
    ((1, 1, 8192, 7168), ttnn.bfloat8_b, "prefill_7168 bf8b"),
    ((1, 1, 32, 16384), ttnn.bfloat16, "very wide"),
]


def cfg():
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = ttnn.MathFidelity.HiFi2
    c.fp32_dest_acc_en = False
    c.math_approx_mode = False
    return c


_ORIG_LAYOUT = opd._cb_layout


def shrunk_cb_layout(pages_fn):
    """The op's CB set with cb_normed's page count replaced by `pages_fn(kw)`.

    pages_fn(kw) = kw['block_ht']         -> the FUSED set (cb_normed deleted, one
                                             materialised 1/rms tile per row)
    pages_fn(kw) = block_ht * min(s, ws)  -> the SUB-CHUNKED two-mul set
    """

    def layout(**kw):
        return [
            (e[0], pages_fn(kw), e[2], e[3]) if e[0] == opd.CB_NORMED else e for e in _ORIG_LAYOUT(**kw)
        ]

    return layout


def plan(shape, dtype, pages_fn=None):
    x = SimpleNamespace(shape=list(shape), layout=ttnn.TILE_LAYOUT, dtype=dtype)
    g = SimpleNamespace(shape=[1, 1, 1, shape[-1]], layout=ttnn.TILE_LAYOUT, dtype=dtype)
    opd._cb_layout = _ORIG_LAYOUT if pages_fn is None else shrunk_cb_layout(pages_fn)
    try:
        return opd.blocking_plan(x, g, None, DEVICE, cfg())
    finally:
        opd._cb_layout = _ORIG_LAYOUT


# (label, cb_normed page-count function).  None = the op as it is today.
ARMS = [
    ("baseline", None),
    ("fused", lambda kw: kw["block_ht"]),
    ("subchunk8", lambda kw: kw["block_ht"] * min(8, kw["ws"])),
    ("subchunk16", lambda kw: kw["block_ht"] * min(16, kw["ws"])),
    ("subchunk32", lambda kw: kw["block_ht"] * min(32, kw["ws"])),
    ("subchunk56", lambda kw: kw["block_ht"] * min(56, kw["ws"])),
]


def report(device):
    global DEVICE
    DEVICE = device
    print(f"{'shape':<22} {'arm':<11} {'regime':>6} {'WT_SC':>6} {'BHT':>4} {'IN_D':>5} {'OUT_D':>6} {'ws_bytes':>10}")
    for shape, dtype, name in SHAPES:
        for tag, fn in ARMS:
            p = plan(shape, dtype, fn)
            print(
                f"{name:<22} {tag:<11} {p.regime:>6} {p.WT_SCALE_BLOCK:>6} {p.BLOCK_HT:>4} "
                f"{p.IN_BUF_DEPTH:>5} {p.OUT_BUF_DEPTH:>6} {p.working_set_bytes():>10}  budget={p.l1_cb_budget}"
            )
