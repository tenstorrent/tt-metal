# SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Whole-op linear_gelu with the packer GELU body swapped for a gelu_loadmacro variant (kernels copied from the op
at run time into gen/v<V>/, compute kernel's GeluTanhActivation routed through e2e/gelu_e2e_spec.h).
Variant 0 = the unmodified op kernels.  Env: GV_VARIANTS, GV_SHAPES (BxS,...), GV_OUT (L1|DRAM)."""
import math
import os
from pathlib import Path

import pytest
import ttnn

import ttnn.operations.linear_gelu.linear_gelu_program_descriptor as pd
from ttnn.operations.linear_gelu import linear_gelu

HERE = Path(__file__).parent
SRC = Path(pd.__file__).parent / "kernels"
GEN = HERE / "gen"
VARIANTS = [int(v) for v in os.environ.get("GV_VARIANTS", "0,6").split(",")]
SHAPES = [tuple(int(t) for t in s.split("x")) for s in os.environ.get("GV_SHAPES", "8x512").split(",")]
OUT = os.environ.get("GV_OUT", "L1")
_BASE = {}


def _make(variant):
    d = GEN / f"v{variant}"
    d.mkdir(parents=True, exist_ok=True)
    for kind in ("reader", "compute", "writer"):
        text = (SRC / f"linear_gelu_{kind}.cpp").read_text()
        if kind == "compute" and variant != 0:
            inc = '#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"\n'
            assert inc in text
            spec = (HERE / "e2e" / "gelu_e2e_spec.h").read_text().replace("GV_KERNELS_DIR", str(HERE / "kernels"))
            (d / "gelu_e2e_spec.h").write_text(spec)
            text = text.replace(
                inc, inc + f'#define GV_V {variant}\n#define GV_P0 {100 + variant}\n#include "gelu_e2e_spec.h"\n'
            )
            old = "using Gelu = GeluTanhActivation<gelu_fast>;"
            assert old in text
            text = text.replace(old, "using Gelu = GeluTanhActivation<GV_P0>;")
        (d / f"linear_gelu_{kind}.cpp").write_text(text)
    return d


def _inputs(batch, seq):
    """The model's distributions (pre-activation ~ N(-2.8, 1.7^2), mostly GELU's negative tail), as the op's
    precision baseline test uses."""
    from eval.golden_tests.linear_gelu.helpers import make_inputs

    return make_inputs(batch, seq)


def _golden(x, w, b):
    import torch

    y = x.double() @ w.double() + b.double()
    return torch.nn.functional.gelu(y, approximate="tanh")


def _ulp_diff(a, b):
    """|a - b| in bf16 ULPs of b (both bf16-representable)."""
    import torch

    m = b.abs().clamp_min(1e-30)
    return (a - b).abs() / torch.pow(2.0, torch.floor(torch.log2(m)) - 7)


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"{s[0]}x{s[1]}")
@pytest.mark.parametrize("variant", VARIANTS)
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
    mem = ttnn.L1_MEMORY_CONFIG if OUT == "L1" else ttnn.DRAM_MEMORY_CONFIG
    out = linear_gelu(xd, wd, bd, memory_config=mem)
    ttnn.synchronize_device(device)
    print(f"E2E variant={variant} shape={batch}x{seq} out={OUT}")
    got = ttnn.to_torch(out).double()
    ref = _golden(x, w, b)
    pcc = torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1].item()
    rel_rms = ((got - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
    key = (batch, seq, OUT)
    line = f"PRECISION e2e variant={variant} shape={batch}x{seq} pcc={pcc:.6f} rel_rms={rel_rms:.4e}"
    if variant == 0:
        _BASE[key] = got
    elif key in _BASE:
        u = _ulp_diff(got, _BASE[key])
        mask = _BASE[key].abs() >= 1e-3
        line += (
            f" vs_op: max_ulp={u[mask].max().item():.0f} frac_diff={(got != _BASE[key]).double().mean().item():.2e}"
            f" mean_signed={(got - _BASE[key]).mean().item():.2e}"
        )
    print(line)
    assert pcc > 0.9999, line
