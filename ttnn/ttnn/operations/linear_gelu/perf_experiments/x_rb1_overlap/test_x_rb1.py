# x_rb1_overlap bake-off: one op run per (variant, case), parametrize order = ops CSV order.
# baseline = the op's own kernels; other variants = kernels/ here with reader defines X1_ISSUE_AT / X1_FINISH_AT
# (W chunk index before which x row-block 1's sender reads are issued / its row multicast is finished).
# Correctness: PCC vs torch >= 0.9999 and bitwise vs the baseline output of the same case (pure DM reorder).
import math
import os
import sys
from pathlib import Path

import pytest
import ttnn

import ttnn.operations.linear_gelu.linear_gelu_program_descriptor as pd
from ttnn.operations.linear_gelu import linear_gelu

lg_mod = sys.modules["ttnn.operations.linear_gelu.linear_gelu"]
HERE = Path(__file__).parent
OUTS = HERE / "_out"
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
}


def _variant(name):
    import torch

    # baseline | i<A>f<B>[t][n][r<R>] (issue before chunk A, finish before chunk B; 'e' = after the last chunk)
    if name == "baseline":
        return None, {}
    if name == "graduate":  # op kernels + graduate_reader.diff
        return HERE / "kernels_graduate", {}
    defines = {}
    if "r" in name:  # r<R>: R rows of row-block 1 issued per W chunk
        name, rows = name.split("r")
        defines["X1_ROWS_PER_CHUNK"] = rows
    if name.endswith("n"):  # trailing n: x reads of row-blocks >= 1 on NoC1
        defines["X1_NOC1"] = "1"
        name = name[:-1]
    if name.endswith("t"):  # trailing t: per-stream NoC transaction ids
        defines["X1_TRID"] = "1"
        name = name[:-1]
    if name == "half":  # issue row-block 1 before the middle W chunk, multicast after the last
        return HERE / "kernels", {"X1_ISSUE_HALF": "1"}
    issue, finish = name[1:].split("f")
    if issue != "e":
        defines["X1_ISSUE_AT"] = issue
    if finish != "e":
        defines["X1_FINISH_AT"] = finish
    return HERE / "kernels", defines


VARIANTS = os.environ.get("X1_VARIANTS", "baseline").split(",")
SELECTED = os.environ.get("X1_CASES", "8x512").split(",")


def _pcc(a, b):
    import torch

    a = a.double().flatten() - a.double().mean()
    b = b.double().flatten() - b.double().mean()
    return float((a @ b) / (a.norm() * b.norm()))


@pytest.mark.parametrize("case", SELECTED)
@pytest.mark.parametrize("variant", VARIANTS)
def test_x_rb1(device, variant, case, monkeypatch):
    import torch

    kdir, defines = _variant(variant)
    if kdir is not None:
        monkeypatch.setattr(pd, "KERNEL_DIR", kdir)
    create = pd.create_program_descriptor

    def create_with_defines(*args, **kwargs):
        desc = create(*args, **kwargs)
        if defines:
            kernels = list(desc.kernels)
            assert "reader" in kernels[0].kernel_source
            kernels[0].defines = list(kernels[0].defines) + list(defines.items())
            desc.kernels = kernels
        return desc

    monkeypatch.setattr(lg_mod, "create_program_descriptor", create_with_defines)
    device.disable_and_clear_program_cache()

    batch, seq, x_mem, out_mem = CASES[case]
    torch.manual_seed(0)
    x = torch.randn((batch, 1, seq, 768)).to(torch.bfloat16)
    w = (torch.randn((768, 3072)) / math.sqrt(768)).to(torch.bfloat16)
    b = torch.randn((3072,)).to(torch.bfloat16)
    tile = dict(dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    xd = ttnn.from_torch(x, memory_config=x_mem, **tile)
    wd = ttnn.from_torch(w, memory_config=DRAM, **tile)
    bd = ttnn.from_torch(b, memory_config=DRAM, **tile)
    out = linear_gelu(xd, wd, bd, memory_config=out_mem)
    ttnn.synchronize_device(device)

    got = ttnn.to_torch(out).float()
    ref = torch.nn.functional.gelu(x.float() @ w.float() + b.float(), approximate="tanh")
    pcc = _pcc(got, ref)
    if math.isnan(pcc):
        return  # precompile pass, no real device output
    OUTS.mkdir(exist_ok=True)
    base_path = OUTS / f"{case}_baseline.pt"
    bitwise = None
    if variant == "baseline":
        torch.save(got.to(torch.bfloat16), base_path)
    elif base_path.exists():
        bitwise = bool(torch.equal(torch.load(base_path).float(), got))
    print(f"X1_RESULT {variant} {case} pcc={pcc:.6f} bitwise_vs_baseline={bitwise}")
    assert pcc >= 0.9999
    if bitwise is not None:
        assert bitwise, "output differs from baseline"
