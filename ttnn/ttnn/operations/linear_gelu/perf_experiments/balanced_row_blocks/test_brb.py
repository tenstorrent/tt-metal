# balanced_row_blocks bake-off: whole op, baseline (op as is) vs balanced row-block split (descriptor_balanced.py).
# BRB_VARIANTS: baseline, first (block 0 = block_h, rest balanced), full (all blocks balanced).
# BRB_CASES: keys of CASES. BRB_BH: override BLOCK_H_MAX / BLOCK_H_MAX_DRAM_OUT (correctness of other splits).
import math
import os
import sys
from pathlib import Path

import pytest
import ttnn

import ttnn.operations.linear_gelu.linear_gelu_program_descriptor as pd
from ttnn.operations.linear_gelu import linear_gelu

# The package re-exports the function under the submodule's name.
lg_mod = sys.modules["ttnn.operations.linear_gelu.linear_gelu"]

sys.path.insert(0, str(Path(__file__).parent))
import descriptor_balanced as bd  # noqa: E402

L1, DRAM = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
CASES = {
    "8x512": (8, 512, L1, L1),
    "16x256": (16, 256, L1, L1),
    "32x128": (32, 128, L1, L1),
    "8x384": (8, 384, L1, L1),
    "4x512": (4, 512, L1, L1),
    "8x256": (8, 256, L1, L1),
    "2x512": (2, 512, L1, L1),
    "4x256": (4, 256, L1, L1),
    "8x512_dramx": (8, 512, DRAM, L1),
    "8x512_dramout": (8, 512, L1, DRAM),
    "8x500": (8, 500, L1, L1),
    "1x32": (1, 32, L1, L1),
    "7x511": (7, 511, L1, L1),
    "13x300": (13, 300, L1, L1),
    "2x37": (2, 37, L1, L1),
    "16x8": (16, 8, L1, L1),
    "13x300_dramout": (13, 300, L1, DRAM),
}
VARIANTS = os.environ.get("BRB_VARIANTS", "baseline,first").split(",")
SELECTED = os.environ.get("BRB_CASES", "8x512").split(",")
BH = os.environ.get("BRB_BH")


@pytest.mark.parametrize("case", SELECTED)
@pytest.mark.parametrize("variant", VARIANTS)
def test_brb(device, variant, case, monkeypatch):
    import torch

    batch, seq, x_mem, out_mem = CASES[case]
    mod = pd
    if variant != "baseline":
        monkeypatch.setattr(bd, "BALANCE", variant)
        monkeypatch.setattr(lg_mod, "create_program_descriptor", bd.create_program_descriptor)
        mod = bd
    if BH:
        monkeypatch.setattr(mod, "BLOCK_H_MAX", int(BH))
        monkeypatch.setattr(mod, "BLOCK_H_MAX_DRAM_OUT", int(BH))
    device.disable_and_clear_program_cache()
    torch.manual_seed(0)
    x = torch.randn((batch, 1, seq, 768)).to(torch.bfloat16)
    w = (torch.randn((768, 3072)) / math.sqrt(768)).to(torch.bfloat16)
    b = torch.randn((3072,)).to(torch.bfloat16)
    tile = dict(dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    xd = ttnn.from_torch(x, memory_config=x_mem, **tile)
    wd = ttnn.from_torch(w, memory_config=DRAM, **tile)
    bdev = ttnn.from_torch(b, memory_config=DRAM, **tile)
    out = linear_gelu(xd, wd, bdev, memory_config=out_mem)
    ttnn.synchronize_device(device)
    ref = torch.nn.functional.gelu(x.double().reshape(batch, seq, 768) @ w.double() + b.double(), approximate="tanh")
    act = ttnn.to_torch(out).reshape(batch, seq, 3072).double()
    pcc = torch.corrcoef(torch.stack([act.flatten(), ref.flatten()]))[0, 1].item()
    bitwise = ""
    if os.environ.get("BRB_BITWISE") and variant != "baseline":
        assert lg_mod.create_program_descriptor is bd.create_program_descriptor
        monkeypatch.setattr(lg_mod, "create_program_descriptor", pd.create_program_descriptor)
        device.disable_and_clear_program_cache()
        base = ttnn.to_torch(linear_gelu(xd, wd, bdev, memory_config=out_mem)).reshape(batch, seq, 3072).double()
        bitwise = f" bitwise_eq_baseline={torch.equal(base, act)}"
    print(f"BRB_PCC {variant} {case} bh={BH} {pcc:.6f}{bitwise}")
    assert pcc >= 0.9999
