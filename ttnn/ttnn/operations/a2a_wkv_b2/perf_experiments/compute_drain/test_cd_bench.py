# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""compute_drain bench: correctness (PCC vs fp64 torch, bit-identity vs base) per variant, one run each.
Env: CD_VARIANTS (comma list, default all), CD_ROW_GAP_NS (reader landing interval per row, default 1900)."""
import os
from pathlib import Path

import pytest
import ttnn

from cd_bench import VARIANTS, run

RES = Path(__file__).parent / "results"
SEL = [v for v in os.environ.get("CD_VARIANTS", ",".join(VARIANTS)).split(",") if v]
GAP_NS = int(os.environ.get("CD_ROW_GAP_NS", "1900"))


def _pcc(a, b):
    import torch

    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


@pytest.mark.parametrize("variant", SEL)
def test_cd(device, variant):
    import torch

    torch.manual_seed(0)
    attn = torch.randn(1, 1, 160, 512).to(torch.bfloat16)
    w = (torch.randn(1, 1, 512, 128) * 0.05).to(torch.bfloat16)
    attn_tt = ttnn.from_torch(attn, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    w_tt = ttnn.from_torch(w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    w_dev = ttnn.to_torch(w_tt)
    out = ttnn.to_torch(run(device, attn_tt, w_tt, variant, int(GAP_NS * 1.35)))
    ref = torch.matmul(attn.double(), w_dev.double())
    pcc = _pcc(out, ref)
    RES.mkdir(exist_ok=True)
    torch.save(out, RES / f"out_{variant}.pt")
    base = RES / "out_base.pt"
    ident = "n/a"
    if base.exists() and variant != "base":
        b = torch.load(base)
        ident = f"bit-identical={torch.equal(b, out)} maxdiff={(b.float() - out.float()).abs().max().item():.4g}"
    ttnn.ReadDeviceProfiler(device)
    print(f"\nCD_RESULT variant={variant} gap_ns={GAP_NS} pcc={pcc:.6f} {ident}")
    assert pcc >= 0.9995
