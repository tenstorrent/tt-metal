# fill_first_block bake-off: one op run per selected variant, in FILL_VARIANTS order; correctness-gated vs torch and
# bitwise vs the baseline output. Perf: run.sh (profile log run IDs map to the variants in order).
import importlib
import math
import os
import sys
from pathlib import Path

import pytest
import ttnn

import ttnn.operations.linear_gelu.linear_gelu_program_descriptor as pd
from ttnn.operations.linear_gelu import linear_gelu

# The package re-exports the function under the submodule's name, so fetch the module itself.
lg_mod = sys.modules["ttnn.operations.linear_gelu.linear_gelu"]

HERE = Path(__file__).parent
OUTS = HERE / "_out"

# variant -> (kernel dir, descriptor module or None for the op's own, W_CHUNKS_BEFORE_X override or None,
#             reader defines)
VARIANTS = {
    "baseline": ("kernels_baseline", None, None, {}),
    "fillzones": ("kernels_fillzones", None, None, {}),
    "direct_x0": ("kernels_direct_x0", None, None, {}),
    "wfirst": ("kernels_baseline", None, 1, {}),
    "xsplit3_zones": ("kernels_xsplit_zones", None, None, {"X0_PIECES": "3"}),
    "ragged": ("kernels_ragged", None, None, {}),
    "early": ("kernels_early", None, None, {"EARLY_ISSUE": "1"}),
    "dualnoc": ("kernels_early", None, None, {"X0_DUAL_NOC": "1"}),
    "early_dualnoc": ("kernels_early", None, None, {"EARLY_ISSUE": "1", "X0_DUAL_NOC": "1"}),
    "wnohs": ("kernels_wnohs", "wnohs", None, {}),
    "ragged_wnohs": ("kernels_ragged_wnohs", "wnohs", None, {}),
    "wnohs_zones": ("kernels_wnohs_zones", "wnohs", None, {}),
    "wnohs_of": ("kernels_wnohs", "wnohs", None, {"W0_OWNER_FIRST": "1"}),
    "wnohs_of_zones": ("kernels_wnohs_zones", "wnohs", None, {"W0_OWNER_FIRST": "1"}),
    "direct_w0": ("kernels_direct_w0", None, None, {}),
    "shift0_wfirst": ("kernels_baseline", None, {"W_CHUNKS_BEFORE_X": 1, "W_ROUND_SHIFT": 0}, {}),
    "shift0_wfirst_zones": ("kernels_fillzones", None, {"W_CHUNKS_BEFORE_X": 1, "W_ROUND_SHIFT": 0}, {}),
    "shift0": ("kernels_baseline", None, {"W_ROUND_SHIFT": 0}, {}),
    "xsplit2": ("kernels_xsplit", None, None, {"X0_PIECES": "2"}),
    "xsplit3": ("kernels_xsplit", None, None, {"X0_PIECES": "3"}),
    "xsplit6": ("kernels_xsplit", None, None, {"X0_PIECES": "6"}),
    "xsplit3_wfirst": ("kernels_xsplit", None, 1, {"X0_PIECES": "3"}),
    "xsplit6_wfirst": ("kernels_xsplit", None, 1, {"X0_PIECES": "6"}),
}

SELECTED = os.environ.get("FILL_VARIANTS", "baseline").split(",")
SHAPE = tuple(int(v) for v in os.environ.get("FILL_SHAPE", "8x512").split("x"))
X_BUF = os.environ.get("FILL_X", "L1")


def _pcc(a, b):
    a = a.double().flatten()
    b = b.double().flatten()
    a = a - a.mean()
    b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


@pytest.mark.parametrize("variant", SELECTED)
def test_fill(device, variant, monkeypatch):
    import torch

    kdir, desc_mod, w_first, reader_defines = VARIANTS[variant]
    monkeypatch.setattr(pd, "KERNEL_DIR", HERE / kdir)
    create = pd.create_program_descriptor
    if desc_mod:
        create = importlib.import_module(desc_mod).wrap(create)
    # w_first: int (W_CHUNKS_BEFORE_X for both placements) or dict of descriptor-module attribute overrides.
    patches = w_first if isinstance(w_first, dict) else ({} if w_first is None else {"W_CHUNKS_BEFORE_X": w_first})
    for attr, value in patches.items():
        if attr == "W_CHUNKS_BEFORE_X":
            for buf in (ttnn.BufferType.L1, ttnn.BufferType.DRAM):
                monkeypatch.setitem(pd.W_CHUNKS_BEFORE_X, buf, value)
        else:
            monkeypatch.setattr(pd, attr, value)

    def create_with_defines(*args, **kwargs):
        desc = create(*args, **kwargs)
        if reader_defines:
            kernels = list(desc.kernels)
            kernels[0].defines = list(kernels[0].defines) + list(reader_defines.items())
            desc.kernels = kernels
        return desc

    monkeypatch.setattr(lg_mod, "create_program_descriptor", create_with_defines)
    device.disable_and_clear_program_cache()

    batch, seq = SHAPE
    torch.manual_seed(0)
    x = torch.randn((batch, 1, seq, 768)).to(torch.bfloat16)
    w = (torch.randn((768, 3072)) / math.sqrt(768)).to(torch.bfloat16)
    b = torch.randn((3072,)).to(torch.bfloat16)
    tile = dict(dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    x_mem = ttnn.L1_MEMORY_CONFIG if X_BUF == "L1" else ttnn.DRAM_MEMORY_CONFIG
    xd = ttnn.from_torch(x, memory_config=x_mem, **tile)
    wd = ttnn.from_torch(w, memory_config=ttnn.DRAM_MEMORY_CONFIG, **tile)
    bd = ttnn.from_torch(b, memory_config=ttnn.DRAM_MEMORY_CONFIG, **tile)
    out = linear_gelu(xd, wd, bd, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.synchronize_device(device)

    tag = f"{batch}x{seq}_{X_BUF}_{variant}"

    got = ttnn.to_torch(out).float()
    ref = torch.nn.functional.gelu(x.float() @ w.float() + b.float(), approximate="tanh")
    pcc = _pcc(got, ref)
    OUTS.mkdir(exist_ok=True)
    base_path = OUTS / f"{batch}x{seq}_{X_BUF}_baseline.pt"
    bitwise = None
    if math.isnan(pcc):
        return  # up-front collect pass (no real device output)
    if variant == "baseline":
        torch.save(got.to(torch.bfloat16), base_path)
    elif base_path.exists():
        base = torch.load(base_path).float()
        bitwise = bool(torch.equal(base, got))
        if not bitwise:
            diff = (base - got).abs()
            print(f"FILL_DIFF {tag} max={diff.max().item()} count={(diff > 0).sum().item()} of {diff.numel()}")
    print(f"FILL_RESULT {tag} pcc={pcc:.6f} bitwise_vs_baseline={bitwise}")
    assert pcc > 0.9999
    if bitwise is not None:
        assert bitwise, "output differs from baseline"
