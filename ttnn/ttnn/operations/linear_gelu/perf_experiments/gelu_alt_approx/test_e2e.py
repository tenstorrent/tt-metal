# SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Whole-op linear_gelu with the packer GELU swapped (kernel copies + KERNEL_DIR monkeypatch).

  GA_E2E=prod,t21sh GA_SHAPES=8x512,2x512 ./run.sh --profile test_e2e.py
`prod` = the op's own kernels unchanged (honest baseline); others = copies whose compute kernel names
GeluTanhActivation<ID> routed to a gelu_alt variant by gelu_alt_spec.h. Prints PCC / rel RMS vs torch.
"""

import math
import os
from pathlib import Path

import pytest
import ttnn

import ttnn.operations.linear_gelu.linear_gelu_program_descriptor as pd
from ttnn.operations.linear_gelu import linear_gelu

HERE = Path(__file__).parent
SRC = Path(pd.__file__).parent / "kernels"
IDS = {"base": 100, "t21": 101, "t21s": 102, "t21sh": 103, "magb": 104}


def _kernel_dir(variant):
    if variant == "prod":
        return SRC
    out = HERE / "gen_kernels" / variant
    out.mkdir(parents=True, exist_ok=True)
    for name in ("linear_gelu_reader.cpp", "linear_gelu_writer.cpp"):
        (out / name).write_text((SRC / name).read_text())
    src = (SRC / "linear_gelu_compute.cpp").read_text()
    anchor = '#include "ttnn/cpp/ttnn/kernel_lib/sfpu_activation_helpers.hpp"\n'
    old = "using Gelu = GeluTanhActivation<gelu_fast>;"
    assert anchor in src and old in src
    src = src.replace(anchor, anchor + '#include "../../gelu_alt_spec.h"\n')
    src = src.replace(old, f"using Gelu = GeluTanhActivation<{IDS[variant]}>;")
    (out / "linear_gelu_compute.cpp").write_text(src)
    return out


VARIANTS = os.environ.get("GA_E2E", "prod,t21sh").split(",")
SHAPES = [tuple(int(v) for v in s.split("x")) for s in os.environ.get("GA_SHAPES", "8x512").split(",")]


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"{s[0]}x{s[1]}")
@pytest.mark.parametrize("variant", VARIANTS)
def test_e2e(device, variant, shape, monkeypatch):
    import torch

    batch, seq = shape
    monkeypatch.setattr(pd, "KERNEL_DIR", _kernel_dir(variant))
    device.disable_and_clear_program_cache()
    torch.manual_seed(0)
    x = torch.randn((batch, 1, seq, 768)).to(torch.bfloat16)
    w = (torch.randn((768, 3072)) / math.sqrt(768)).to(torch.bfloat16)
    b = torch.randn((3072,)).to(torch.bfloat16)
    tile = dict(dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    xd = ttnn.from_torch(x, memory_config=ttnn.L1_MEMORY_CONFIG, **tile)
    wd = ttnn.from_torch(w, memory_config=ttnn.DRAM_MEMORY_CONFIG, **tile)
    bd = ttnn.from_torch(b, memory_config=ttnn.DRAM_MEMORY_CONFIG, **tile)
    out = linear_gelu(xd, wd, bd, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.synchronize_device(device)
    got = ttnn.to_torch(out).double()
    ref = torch.nn.functional.gelu(x.double() @ w.double() + b.double(), approximate="tanh")
    pcc = torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1].item()
    rel_rms = ((got - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
    mean_err = (got - ref).mean().item()
    print(f"E2E variant={variant} shape={batch}x{seq} pcc={pcc:.6f} rel_rms={rel_rms:.4e} mean_err={mean_err:+.3e}")
    (HERE / "results").mkdir(exist_ok=True)
    with open(HERE / "results" / "e2e_order.txt", "a") as f:
        f.write(f"{variant} {batch}x{seq} pcc={pcc:.6f} rel_rms={rel_rms:.4e} mean_err={mean_err:+.3e}\n")
    assert pcc > 0.999
