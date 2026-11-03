# SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""pack_thread_cost whole-op bench: linear_gelu with matmul_block's pack stage (GELU + pack per subblock) routed
through kernels/pt_pack.hpp (local copy of the matmul helper, mbh_local.*). Variant "op" = the unmodified op kernels.

Correctness: every non-diag variant is bit-compared to the op output on the same inputs (max bf16 ULP diff, mean
signed diff) and PCC-checked vs torch. Inputs: "model" = the op's precision-baseline distribution; "wide" = pre-
activations spread over about +-40 (exercises both exp2 clamps).
Env: PT_VARIANTS (comma list, order = launch order), PT_SHAPES (BxS,...), PT_INPUTS (model,wide), PT_OUT (L1|DRAM),
PT_X (L1|DRAM).
"""
import os
import shutil
from pathlib import Path

import pytest
import ttnn

import ttnn.operations.linear_gelu.linear_gelu_program_descriptor as pd
from ttnn.operations.linear_gelu import linear_gelu

HERE = Path(__file__).parent
SRC_OP = Path(pd.__file__).parent / "kernels"
SRC_LOCAL = HERE / "kernels"
GEN = HERE / "gen"
CALL_LOG = GEN / "calls.txt"

SHARED_INC = '#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"'
LOCAL_INC = '#include "mbh_local.hpp"'

ALL4 = {"PT_NOSTALL": 1, "PT_BATCH": 1, "PT_NORECONFIG": 1, "PT_INTERLEAVE": 1}
VARIANTS = {
    "op": None,
    "base": {},
    "nostall": {"PT_NOSTALL": 1},
    "batch": {"PT_BATCH": 1, "PT_NOSTALL": 1},
    "noreconfig": {"PT_NORECONFIG": 1},
    "interleave": {"PT_INTERLEAVE": 1},
    "all4": ALL4,
    "noclamp0": {"PT_NOCLAMP0": 1},
    "all4_noclamp0": {**ALL4, "PT_NOCLAMP0": 1},
    "il_norec": {"PT_INTERLEAVE": 1, "PT_NORECONFIG": 1},
    "il_nostall": {"PT_INTERLEAVE": 1, "PT_NOSTALL": 1},
    "nr_nostall": {"PT_NORECONFIG": 1, "PT_NOSTALL": 1},
    "nr_batch": {"PT_NORECONFIG": 1, "PT_BATCH": 1, "PT_NOSTALL": 1},
    "nr_c3": {"PT_NORECONFIG": 1, "PT_C3HOIST": 1},
    "nr_nostall_c3": {"PT_NORECONFIG": 1, "PT_NOSTALL": 1, "PT_C3HOIST": 1},
    "diag_nogelu": {"PT_NOGELU": 1},
    "diag_nopack": {"PT_NOPACK": 1},
}

SEL = os.environ.get("PT_VARIANTS", "op,base,all4").split(",")
SHAPES = [tuple(int(v) for v in s.split("x")) for s in os.environ.get("PT_SHAPES", "8x512").split(",")]
INPUTS = os.environ.get("PT_INPUTS", "model").split(",")
OUT = os.environ.get("PT_OUT", "L1")
XMEM = os.environ.get("PT_X", "L1")


def _make(name):
    d = GEN / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    defines = VARIANTS[name]
    for kind in ("reader", "writer", "compute"):
        shutil.copy(SRC_OP / f"linear_gelu_{kind}.cpp", d)
    shutil.copy(SRC_OP / "linear_gelu_gelu_sfpu.h", d)
    if defines is not None:
        for f in ("mbh_local.hpp", "mbh_local.inl", "pt_pack.hpp", "linear_gelu_gelu_sfpu.h"):
            shutil.copy(SRC_LOCAL / f, d)
        p = d / "linear_gelu_compute.cpp"
        text = p.read_text()
        assert SHARED_INC in text
        text = text.replace(SHARED_INC, LOCAL_INC)
        p.write_text("".join(f"#define {k} {v}\n" for k, v in defines.items()) + text)
    return d


def _collecting():
    try:
        import tests.plugins.up_front_collect as upc

        return upc._PASS == "collect"
    except ImportError:
        return False


def _log_call(name, tag):
    if _collecting():
        return
    CALL_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(CALL_LOG, "a") as f:
        f.write(f"{name} {tag}\n")


def _inputs(kind, batch, seq):
    import torch

    from eval.golden_tests.linear_gelu.helpers import make_inputs

    if kind == "model":
        return make_inputs(batch, seq)
    # wide: pre-activation std ~ 20, so |y| reaches ~ 60 (both clamps of log2 in [0, 254] are hit).
    x, w, b = make_inputs(batch, seq, seed=1, bias_mean=0.0, bias_std=4.0)
    return x, (w.float() * 12).to(torch.bfloat16), b


def _run(device, name, x, w, b):
    monkeypatch_dir = _make(name)
    pd.KERNEL_DIR = monkeypatch_dir
    device.disable_and_clear_program_cache()
    tile = dict(dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    xmem = ttnn.L1_MEMORY_CONFIG if XMEM == "L1" else ttnn.DRAM_MEMORY_CONFIG
    xd = ttnn.from_torch(x, memory_config=xmem, **tile)
    wd = ttnn.from_torch(w, memory_config=ttnn.DRAM_MEMORY_CONFIG, **tile)
    bd = ttnn.from_torch(b, memory_config=ttnn.DRAM_MEMORY_CONFIG, **tile)
    mem = ttnn.L1_MEMORY_CONFIG if OUT == "L1" else ttnn.DRAM_MEMORY_CONFIG
    out = linear_gelu(xd, wd, bd, memory_config=mem)
    ttnn.synchronize_device(device)
    return ttnn.to_torch(out)


_ref = {}


@pytest.mark.parametrize("inputs", INPUTS)
@pytest.mark.parametrize("shape", SHAPES, ids=[f"{b}x{s}" for b, s in SHAPES])
@pytest.mark.parametrize("variant", SEL)
def test_e2e(device, variant, shape, inputs, monkeypatch):
    import torch

    monkeypatch.setattr(pd, "KERNEL_DIR", pd.KERNEL_DIR)  # restored after the test
    batch, seq = shape
    tag = f"{batch}x{seq}:{inputs}:x{XMEM}:o{OUT}"
    x, w, b = _inputs(inputs, batch, seq)
    key = (shape, inputs)
    if key not in _ref or _collecting():
        ref = _run(device, "op", x, w, b).clone()
        _log_call("ref_op", tag)
        if not _collecting():
            _ref[key] = ref
    out = _run(device, variant, x, w, b)
    _log_call(variant, tag)
    if _collecting():
        return
    ref = _ref[key]
    golden = torch.nn.functional.gelu(x.double() @ w.double() + b.double(), approximate="tanh")
    got = out.double()
    pcc = torch.corrcoef(torch.stack([got.flatten(), golden.flatten()]))[0, 1].item()
    r = ref.double()
    m = r.abs().clamp_min(1e-30)
    ulp = (got - r).abs() / torch.pow(2.0, torch.floor(torch.log2(m)) - 7)
    mismatches = (out.view(torch.int16) != ref.view(torch.int16)).sum().item()
    line = (
        f"PT_RESULT variant={variant} {tag} pcc={pcc:.6f} bit_mismatches={mismatches} "
        f"max_ulp={ulp.max().item():.0f} mean_signed={(got - r).mean().item():.2e}"
    )
    print(line)
    if not variant.startswith("diag_"):
        assert pcc >= 0.9999, line
        assert torch.isfinite(got).all(), line
