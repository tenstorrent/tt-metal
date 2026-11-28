"""Bit-exact correctness of the two_mover_split variants vs torch.reshape, incl. wide tile_element_run shapes
(the production INPUTS have none with two face-column halves) and partial last tile columns."""
import importlib.util
import sys
from pathlib import Path

import pytest

torch = __import__("torch")  # experiment bench: global torch import is fine here (not part of ttnn)
import ttnn

_spec = importlib.util.spec_from_file_location("tms_op", Path(__file__).resolve().parent / "tms_op.py")
TMS = importlib.util.module_from_spec(_spec)
sys.modules["tms_op"] = TMS
_spec.loader.exec_module(TMS)

CASES = [
    ((1, 1, 6, 256), (1, 1, 1536), torch.bfloat16),
    ((1, 8, 64), (8, 1, 64), torch.bfloat16),
    ((1, 1, 40, 72), (1, 1, 72, 40), torch.bfloat16),  # element run, partial last column (8 wide)
    ((3, 50), (5, 30), torch.bfloat16),  # element run, one core per atom
    ((2, 7, 48), (1, 14, 48), torch.float32),  # face_row, last column 16 wide (one half)
    ((1, 5, 100), (10, 50), torch.float32),  # element run, last column 18 wide (two halves)
    ((4, 33, 20), (1, 66, 40), torch.int32),
    ((1, 256), (1, 1, 256, 1), torch.float32),
    ((16, 1, 5120), (1, 1, 16, 5120), torch.bfloat16),
    ((1, 1, 16, 4608), (16, 1, 4608), torch.bfloat16),
    ((1, 3, 40, 72), (1, 1, 3, 2880), torch.bfloat16),  # element run, H < atom rows... multi-image
]
DT = {torch.bfloat16: ttnn.bfloat16, torch.float32: ttnn.float32, torch.int32: ttnn.int32}


@pytest.mark.parametrize(
    "variant",
    ["baseline", "half_single", "half_auto", "half_tiles", "half_all", "drop_idle", "solo_brisc", "cores_half"],
)
@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c[0]}->{c[1]}-{c[2]}")
def test_correct(case, variant, device, monkeypatch):
    monkeypatch.setenv("TMS_VARIANT", variant)
    ins, out, tdt = case
    torch.manual_seed(0)
    n = 1
    for d in ins:
        n *= d
    x = (torch.arange(n, dtype=torch.int64) % 30011 - 15000).reshape(ins).to(tdt)
    t = ttnn.from_torch(x, dtype=DT[tdt], layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    t = ttnn.fill_implicit_tile_padding(t, 7.0)
    y = ttnn.to_torch(TMS.g6_reshape(t, list(out)))
    exp = x.reshape(out)
    assert tuple(y.shape) == tuple(exp.shape)
    assert torch.equal(
        y.view(torch.int16 if tdt == torch.bfloat16 else torch.int32),
        exp.view(torch.int16 if tdt == torch.bfloat16 else torch.int32),
    )
