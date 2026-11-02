# SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Whole-op linear_gelu with the packer GELU swapped for a gst_gelu.h variant (op kernels copied, compute kernel
gets `#define GST_VARIANT n` + `#include "gst_gelu.h"` first). Default config (HiFi2, fp32 DEST, approx off),
x L1, W / bias DRAM, out L1. Prints PCC / rel RMS vs torch fp32 and the bf16-ULP diff vs the unmodified op.
Perf: run with --profile, then parse_perf.py <order>.
"""
import math
import os
import shutil
from pathlib import Path

import pytest
import ttnn

import ttnn.operations.linear_gelu.linear_gelu_program_descriptor as pd
from ttnn.operations.linear_gelu import linear_gelu

HERE = Path(__file__).parent
SRC = Path(pd.__file__).parent / "kernels"
GEN = HERE / "gen"
VARIANTS = {"op": None, "magic4x2w": 10, "magic3x2w": 11, "magic4x2": 4, "magic3x2": 6, "base_nr": 1}
SEL = os.environ.get("GST_VARIANTS", "op,magic4x2w,magic3x2w").split(",")
SHAPES = [tuple(int(v) for v in s.split("x")) for s in os.environ.get("GST_SHAPES", "8x512").split(",")]
_REF_OUT = {}


def _make(name):
    if VARIANTS[name] is None:
        return SRC
    d = GEN / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    for f in SRC.glob("*.cpp"):
        text = f.read_text()
        if f.name == "linear_gelu_compute.cpp":
            text = f'#define GST_VARIANT {VARIANTS[name]}\n#include "../../gst_gelu.h"\n' + text
        (d / f.name).write_text(text)
    return d


def _inputs(batch, seq):
    import torch

    torch.manual_seed(0)
    x = torch.randn((batch, 1, seq, 768)).to(torch.bfloat16)
    w = (torch.randn((768, 3072)) / math.sqrt(768)).to(torch.bfloat16)
    b = torch.randn((3072,)).to(torch.bfloat16)
    return x, w, b


def _ordered(bf16):
    import torch

    i = bf16.view(torch.int16).to(torch.int32)
    return torch.where(i < 0, -(i & 0x7FFF), i)


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"{s[0]}x{s[1]}")
@pytest.mark.parametrize("variant", SEL)
def test_e2e(device, variant, shape, monkeypatch):
    import torch

    batch, seq = shape
    monkeypatch.setattr(pd, "KERNEL_DIR", _make(variant))
    device.disable_and_clear_program_cache()
    x, w, b = _inputs(batch, seq)
    tile = dict(dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    xd = ttnn.from_torch(x, memory_config=ttnn.L1_MEMORY_CONFIG, **tile)
    wd = ttnn.from_torch(w, memory_config=ttnn.DRAM_MEMORY_CONFIG, **tile)
    bd = ttnn.from_torch(b, memory_config=ttnn.DRAM_MEMORY_CONFIG, **tile)
    out = linear_gelu(xd, wd, bd, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.synchronize_device(device)
    y = ttnn.to_torch(out)
    ref = torch.nn.functional.gelu(x.float() @ w.float() + b.float(), approximate="tanh")
    yf, rf = y.float().flatten(), ref.flatten()
    pcc = torch.corrcoef(torch.stack([yf, rf]))[0, 1].item()
    rel_rms = ((yf - rf).pow(2).mean().sqrt() / rf.pow(2).mean().sqrt()).item()
    mean_err = (yf - rf).mean().item()
    msg = f"GST_E2E {variant} {batch}x{seq} pcc={pcc:.6f} rel_rms={rel_rms:.4e} mean_err={mean_err:.3e}"
    if variant == "op":
        _REF_OUT[shape] = y
    elif shape in _REF_OUT:
        d = (_ordered(y.to(torch.bfloat16)) - _ordered(_REF_OUT[shape].to(torch.bfloat16))).abs()
        msg += f" vs_op_max_ulp={d.max().item()} vs_op_frac_diff={(d > 0).double().mean().item():.4g}"
    print(msg)
    assert pcc > 0.9999
