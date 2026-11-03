# bias_cheap bake-off: whole-op linear_gelu with the op's kernels copied into gen_kernels/ and the compute kernel's
# bias K step swapped. Correctness: PCC vs torch and max bf16 ULP diff vs the base variant's output (same mode /
# shape / fidelity, earlier in the same session). Perf comes from the profiler CSV (parse.py), never asserted.
#
# Env:
#   BC_VARIANTS  comma list: base, fullk_split, halfk, halfk_1unp, nobias_diag
#   BC_MODES     comma list: full, nogelu
#   BC_SHAPES    comma list like 8x512, optional :dramx / :dramout suffix
#   BC_FIDELITY  optional MathFidelity name (default: the op default, HiFi2) - applied to every variant alike
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
ORDER_LOG = HERE / "order.log"
BASE_OUT = {}

pytestmark = pytest.mark.skipif(not os.environ.get("BC_VARIANTS"), reason="set BC_VARIANTS")

GELU_LINE = "using Gelu = GeluTanhActivation<kGeluLoadMacro>;"
CALL_ARGS = "shape, {}, {}, block_w, 0, {}, {}, {}, chunk_offset"
OFFSET_LINE = "const WeightChunkOffset chunk_offset{n_block_idx * w_chunk_tiles};"


def _kernel_dir(variant, mode):
    import torch

    out = HERE / "gen_kernels" / f"{variant}_{mode}"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for f in SRC.iterdir():
        if f.is_file():
            shutil.copy(f, out / f.name)
    shutil.copy(HERE / "bias_half_k.h", out / "bias_half_k.h")
    comp = out / "linear_gelu_compute.cpp"
    text = comp.read_text()
    if mode == "nogelu":
        assert GELU_LINE in text
        text = text.replace(GELU_LINE, "using Gelu = NoneActivation;")
    if variant != "base":
        assert text.count("NoPostCompute,") == 2 and text.count("NoKBlockInnerDimFn,") == 2
        assert text.count(CALL_ARGS) == 2 and OFFSET_LINE in text
        text = text.replace(
            '#include "linear_gelu_gelu_sfpu.h"', '#include "linear_gelu_gelu_sfpu.h"\n#include "bias_half_k.h"'
        )
        text = text.replace("NoKBlockInnerDimFn,", "BiasInnerDim,")
        if variant == "nobias_diag":
            text = text.replace(CALL_ARGS, "shape, {}, {}, block_w, 0, {}, BiasInnerDim{}, {}, chunk_offset")
        else:
            text = text.replace("NoPostCompute,", "BiasPostCompute,")
            text = text.replace(
                OFFSET_LINE,
                OFFSET_LINE + "\n            const BiasPostCompute bias_step{cb_activation_block, cb_weight_resident,"
                " n_block_idx * w_chunk_tiles, cur_sb_h, cur_sb_w, block_w, k_ext_tiles};",
            )
            text = text.replace(CALL_ARGS, "shape, bias_step, {}, block_w, 0, {}, BiasInnerDim{}, {}, chunk_offset")
        if variant == "fullk_split":
            text = text.replace("#include <cstdint>", "#define BIAS_FULL_K 1\n#include <cstdint>", 1)
        if variant == "halfk_1unp":
            text = text.replace("#include <cstdint>", "#define BIAS_ONE_UNPACK 1\n#include <cstdint>", 1)
    comp.write_text(text)
    return out


def _ref(x, w, b, gelu, bias=True):
    import torch

    y = x.float().reshape(-1, 768) @ w.float()
    if bias:
        y = y + b.float()
    if gelu:
        y = torch.nn.functional.gelu(y, approximate="tanh")
    return y.reshape(*x.shape[:-1], 3072)


def _pcc(a, b):
    import torch

    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _ordered(t):
    import torch

    i = t.contiguous().view(torch.int16).to(torch.int32)
    return torch.where(i < 0, -32768 - i, i)


def _cases():
    import torch

    out = []
    for s in os.environ.get("BC_SHAPES", "8x512").split(","):
        for m in os.environ.get("BC_MODES", "full").split(","):
            for v in os.environ.get("BC_VARIANTS", "base").split(","):
                out.append((v, m, s))
    return out


@pytest.mark.parametrize("variant,mode,shape", _cases(), ids=lambda p: str(p))
def test_bench(device, variant, mode, shape, monkeypatch):
    import torch

    parts = shape.split(":")
    batch, seq = (int(v) for v in parts[0].split("x"))
    x_mem = ttnn.DRAM_MEMORY_CONFIG if "dramx" in parts[1:] else ttnn.L1_MEMORY_CONFIG
    out_mem = ttnn.DRAM_MEMORY_CONFIG if "dramout" in parts[1:] else ttnn.L1_MEMORY_CONFIG
    monkeypatch.setattr(pd, "KERNEL_DIR", _kernel_dir(variant, mode))
    device.disable_and_clear_program_cache()
    torch.manual_seed(0)
    x = torch.randn((batch, 1, seq, 768)).to(torch.bfloat16)
    w = (torch.randn((768, 3072)) / math.sqrt(768)).to(torch.bfloat16)
    b = torch.randn((3072,)).to(torch.bfloat16)
    tile = dict(dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    xd = ttnn.from_torch(x, memory_config=x_mem, **tile)
    wd = ttnn.from_torch(w, memory_config=ttnn.DRAM_MEMORY_CONFIG, **tile)
    bd = ttnn.from_torch(b, memory_config=ttnn.DRAM_MEMORY_CONFIG, **tile)
    cfg = None
    fid = os.environ.get("BC_FIDELITY")
    if fid:
        cfg = ttnn.ComputeConfigDescriptor(
            math_fidelity=getattr(ttnn.MathFidelity, fid), fp32_dest_acc_en=True, math_approx_mode=False
        )
    y = linear_gelu(xd, wd, bd, memory_config=out_mem, compute_kernel_config=cfg)
    ttnn.synchronize_device(device)
    got_bf16 = ttnn.to_torch(y)
    got = got_bf16.float()
    ref = _ref(x, w, b, mode == "full", bias=variant != "nobias_diag")
    pcc = _pcc(got, ref)
    key = (mode, shape, fid)
    if variant == "base":
        BASE_OUT[key] = got_bf16.clone()
        ulp = "self"
    elif key in BASE_OUT and variant != "nobias_diag":
        d = (_ordered(got_bf16) - _ordered(BASE_OUT[key])).abs()
        ulp = f"max={d.max().item()},n_diff={(d > 0).sum().item()}"
    else:
        ulp = "na"
    line = f"{variant}|{mode}|{shape}|fid={fid or 'default'}|pcc={pcc:.6f}|ulp_vs_base={ulp}"
    with open(ORDER_LOG, "a") as f:
        f.write(line + "\n")
    print(line)
    assert pcc > 0.9999, line
