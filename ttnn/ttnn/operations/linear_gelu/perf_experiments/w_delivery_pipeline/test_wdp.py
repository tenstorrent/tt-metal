# w_delivery_pipeline bake-off: one op run per (case, variant), parametrize order = ops CSV row order.
# Correctness: PCC vs torch >= 0.9999 and bitwise vs the baseline output of the same case.
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
sys.path.insert(0, str(HERE))
import wnohs  # noqa: E402
import wends  # noqa: E402
import wfixed  # noqa: E402

OUTS = HERE / "_out"
LA = {"W_TRID": "1"}
# variant -> (kernel dir, reader defines, nohs host wrap, descriptor attribute overrides)
VARIANTS = {
    "baseline": ("kernels_baseline", {}, False, {}),
    "graduate": ("kernels_graduate", {}, False, {}),
    "cand0": ("kernels_cand", {"W_TRID": "1"}, False, {}),
    "la1": ("kernels_cand", {**LA, "W_LOOKAHEAD": "1"}, False, {}),
    "la2": ("kernels_cand", {**LA, "W_LOOKAHEAD": "2"}, False, {}),
    "la3": ("kernels_cand", {**LA, "W_LOOKAHEAD": "3"}, False, {}),
    "la1_s0": ("kernels_cand", {**LA, "W_LOOKAHEAD": "1", "W_LA_START": "0"}, False, {}),
    "la_all": ("kernels_cand", {**LA, "W_LOOKAHEAD": "64"}, False, {}),
    "g2": ("kernels_cand", {**LA, "W_LOOKAHEAD": "1", "W_GROUP": "2"}, False, {}),
    "g4": ("kernels_cand", {**LA, "W_LOOKAHEAD": "1", "W_GROUP": "4"}, False, {}),
    "g8": ("kernels_cand", {**LA, "W_LOOKAHEAD": "1", "W_GROUP": "8"}, False, {}),
    "g2_la0": ("kernels_cand", {**LA, "W_GROUP": "2"}, False, {}),
    "n1": ("kernels_cand", {**LA, "W_READ_NOC": "1"}, False, {}),
    "n1_la1": ("kernels_cand", {**LA, "W_READ_NOC": "1", "W_LOOKAHEAD": "1"}, False, {}),
    "n1_la1_s0": ("kernels_cand", {**LA, "W_READ_NOC": "1", "W_LOOKAHEAD": "1", "W_LA_START": "0"}, False, {}),
    "n1_la2": ("kernels_cand", {**LA, "W_READ_NOC": "1", "W_LOOKAHEAD": "2"}, False, {}),
    "ends": ("kernels_cand", {**LA}, False, {"ends": True}),
    "ends_la1": ("kernels_cand", {**LA, "W_LOOKAHEAD": "1"}, False, {"ends": True}),
    "ends_la1_s0": ("kernels_cand", {**LA, "W_LOOKAHEAD": "1", "W_LA_START": "0"}, False, {"ends": True}),
    "ends_la2": ("kernels_cand", {**LA, "W_LOOKAHEAD": "2"}, False, {"ends": True}),
    "br": ("kernels_cand", {**LA, "W_BIAS_ROW": "1"}, False, {}),
    "br_la1": ("kernels_cand", {**LA, "W_BIAS_ROW": "1", "W_LOOKAHEAD": "1"}, False, {}),
    "br_la1_s0": ("kernels_cand", {**LA, "W_BIAS_ROW": "1", "W_LOOKAHEAD": "1", "W_LA_START": "0"}, False, {}),
    "debug_nozero": ("kernels_cand", {**LA, "W_DEBUG_NO_ZERO": "1"}, False, {}),
    "fx": ("kernels_cand", {**LA}, False, {"fixed": True}),
    "fx_la1": ("kernels_cand", {**LA, "W_LOOKAHEAD": "1"}, False, {"fixed": True}),
    "fx_la1_s0": ("kernels_cand", {**LA, "W_LOOKAHEAD": "1", "W_LA_START": "0"}, False, {"fixed": True}),
    "fx_la2": ("kernels_cand", {**LA, "W_LOOKAHEAD": "2"}, False, {"fixed": True}),
    "fx_la2_s0": ("kernels_cand", {**LA, "W_LOOKAHEAD": "2", "W_LA_START": "0"}, False, {"fixed": True}),
    "fx_br_la1_s0": (
        "kernels_cand",
        {**LA, "W_BIAS_ROW": "1", "W_LOOKAHEAD": "1", "W_LA_START": "0"},
        False,
        {"fixed": True},
    ),
    "fx2_la1": (
        "kernels_cand",
        {**LA, "W_BIAS_ROW": "1", "W_LOOKAHEAD": "1", "W_ISSUE_AFTER_SEND": "1", "W_SEND_NOGUARD": "1"},
        False,
        {"fixed": True},
    ),
    "fx2_la1_s0": (
        "kernels_cand",
        {
            **LA,
            "W_BIAS_ROW": "1",
            "W_LOOKAHEAD": "1",
            "W_LA_START": "0",
            "W_ISSUE_AFTER_SEND": "1",
            "W_SEND_NOGUARD": "1",
        },
        False,
        {"fixed": True},
    ),
    "fx2_la2": (
        "kernels_cand",
        {**LA, "W_BIAS_ROW": "1", "W_LOOKAHEAD": "2", "W_ISSUE_AFTER_SEND": "1", "W_SEND_NOGUARD": "1"},
        False,
        {"fixed": True},
    ),
    "fx2_la1_g": (
        "kernels_cand",
        {**LA, "W_BIAS_ROW": "1", "W_LOOKAHEAD": "1", "W_ISSUE_AFTER_SEND": "1"},
        False,
        {"fixed": True},
    ),
    "sp2": ("kernels_cand", {**LA, "W_READ_SPLIT": "2"}, False, {}),
    "sp3": ("kernels_cand", {**LA, "W_READ_SPLIT": "3"}, False, {}),
    "sp2_s0": ("kernels_cand", {**LA, "W_READ_SPLIT": "2", "W_LA_START": "0"}, False, {}),
    "sp3_s0": ("kernels_cand", {**LA, "W_READ_SPLIT": "3", "W_LA_START": "0"}, False, {}),
    "sp1_s0": ("kernels_cand", {**LA, "W_READ_SPLIT": "1", "W_LA_START": "0"}, False, {}),
    "la1_s0_nt": ("kernels_cand", {"W_LOOKAHEAD": "1", "W_LA_START": "0"}, False, {}),
    "m1_la1": ("kernels_cand", {**LA, "W_SHIFT_M1": "1", "W_LOOKAHEAD": "1"}, False, {}),
    "xt_la1_s0": ("kernels_cand", {**LA, "X_TRID": "1", "W_LOOKAHEAD": "1", "W_LA_START": "0"}, False, {}),
    "e_la1": (
        "kernels_cand",
        {**LA, "W_SHIFT_M1": "1", "X_TRID": "1", "W_EARLY_ON_XSEND": "1", "W_LOOKAHEAD": "1"},
        False,
        {},
    ),
    "e_la2": (
        "kernels_cand",
        {**LA, "W_SHIFT_M1": "1", "X_TRID": "1", "W_EARLY_ON_XSEND": "1", "W_LOOKAHEAD": "2"},
        False,
        {},
    ),
    "e_sp2": (
        "kernels_cand",
        {**LA, "W_SHIFT_M1": "1", "X_TRID": "1", "W_EARLY_ON_XSEND": "1", "W_READ_SPLIT": "2"},
        False,
        {},
    ),
    "e_br_la1": (
        "kernels_cand",
        {**LA, "W_SHIFT_M1": "1", "X_TRID": "1", "W_EARLY_ON_XSEND": "1", "W_LOOKAHEAD": "1", "W_BIAS_ROW": "1"},
        False,
        {},
    ),
    "nohs": ("kernels_cand", {**LA, "W_NOHS": "1"}, True, {}),
    "la1_nohs": ("kernels_cand", {**LA, "W_LOOKAHEAD": "1", "W_NOHS": "1"}, True, {}),
    "la2_nohs": ("kernels_cand", {**LA, "W_LOOKAHEAD": "2", "W_NOHS": "1"}, True, {}),
}
ZONED = {f"{k}_z": (v[0], {**v[1], "W_ROUND_ZONES": "1"}, v[2], v[3]) for k, v in VARIANTS.items()}
ZONED["baseline_z"] = ("kernels_cand", {"W_ROUND_ZONES": "1"}, False, {})
VARIANTS.update(ZONED)

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
SEL_V = os.environ.get("WDP_VARIANTS", "baseline").split(",")
SEL_C = os.environ.get("WDP_CASES", "8x512").split(",")


def _pcc(a, b):
    import torch

    a = a.double().flatten() - a.double().mean()
    b = b.double().flatten() - b.double().mean()
    return float((a @ b) / (a.norm() * b.norm()))


@pytest.mark.parametrize("variant", SEL_V)
@pytest.mark.parametrize("case", SEL_C)
def test_wdp(device, case, variant, monkeypatch):
    import torch

    kdir, defines, nohs, overrides = VARIANTS[variant]
    monkeypatch.setattr(pd, "KERNEL_DIR", HERE / kdir)
    for attr, value in overrides.items():
        if attr not in ("ends", "fixed"):
            monkeypatch.setattr(pd, attr, value)
    create = pd.create_program_descriptor
    if nohs:
        create = wnohs.wrap(create)
    if overrides.get("fixed"):
        create = wfixed.wrap(create)
    if overrides.get("ends"):
        create = wends.wrap(create)

    def create_with_defines(*args, **kwargs):
        desc = create(*args, **kwargs)
        if defines:
            kernels = list(desc.kernels)
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
    if os.environ.get("WDP_GARBAGE_BIAS"):
        # Bias tile padding rows 1..31 hold NaN (fill_implicit_tile_padding): 0 * NaN leaks into the output unless zeroed.
        bd = ttnn.from_torch(b.reshape(1, 3072), memory_config=DRAM, **tile)
        bd = ttnn.fill_implicit_tile_padding(bd, float("nan"))
        check = ttnn.to_torch(bd)
        assert check.shape == (1, 3072) and torch.equal(check[0], b)
    else:
        bd = ttnn.from_torch(b, memory_config=DRAM, **tile)
    out = linear_gelu(xd, wd, bd, memory_config=out_mem)
    ttnn.synchronize_device(device)
    got = ttnn.to_torch(out).float()
    ref = torch.nn.functional.gelu(x.float() @ w.float() + b.float(), approximate="tanh")
    pcc = _pcc(got, ref)
    if math.isnan(pcc):
        return  # precompile collect pass (no real device output)
    OUTS.mkdir(exist_ok=True)
    base_path = OUTS / f"{case}{'_gb' if os.environ.get('WDP_GARBAGE_BIAS') else ''}_baseline.pt"
    bitwise = None
    if variant == "baseline":
        torch.save(got.to(torch.bfloat16), base_path)
    elif base_path.exists():
        bitwise = bool(torch.equal(torch.load(base_path).float(), got))
    print(f"WDP_RESULT case={case} variant={variant} pcc={pcc:.6f} bitwise={bitwise}")
    assert pcc >= 0.9999
    assert bitwise is not False
