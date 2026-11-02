# gelu_call_overhead bench: whole-op linear_gelu with the packer-thread GELU+pack stage swapped for local
# variants (kernels/gco_gelu_pack.hpp). Correctness: every non-diagnostic variant must be bit-identical to the
# unmodified op output on the same inputs. Perf: run under `run_safe_pytest.sh --profile`, then parse.py.
import math
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

SHARED_INC = '#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"'
LOCAL_INC = '#include "mbh_local.hpp"'

# name -> defines; None = the unmodified op kernels. Names starting with diag_ are perf-only (wrong output).
VARIANTS = {
    "op": None,
    "base": {},
    "diag_empty": {"GCO_EMPTY": 1},
    "diag_body2": {"GCO_BODY_REPS": 2},
    "diag_call2": {"GCO_CALL_REPS": 2},
    "diag_nopack": {"GCO_PACK_REPS": 0},
    "diag_pack2": {"GCO_PACK_REPS": 2},
    "hoist1": {"GCO_HOIST": 1},
    "hoist2": {"GCO_HOIST": 2},
    "hoist3": {"GCO_HOIST": 3},
    "rec": {"GCO_HOIST": 1, "GCO_NORECONFIG": 1},
    "rec_batch": {"GCO_HOIST": 1, "GCO_NORECONFIG": 1, "GCO_BATCH": 1, "GCO_NOSTALL": 1},
    "batch": {"GCO_BATCH": 1},
    "nostall": {"GCO_NOSTALL": 1},
    "batch_nostall": {"GCO_BATCH": 1, "GCO_NOSTALL": 1},
    "noreconfig": {"GCO_NORECONFIG": 1},
    "interleave": {"GCO_INTERLEAVE": 1},
    "unroll4": {"GCO_UNROLL": 4},
    "unroll2": {"GCO_UNROLL": 2},
    "unroll1": {"GCO_UNROLL": 1},
    "all": {"GCO_BATCH": 1, "GCO_NOSTALL": 1, "GCO_NORECONFIG": 1, "GCO_INTERLEAVE": 1},
    "batch_nostall_noreconfig": {"GCO_BATCH": 1, "GCO_NOSTALL": 1, "GCO_NORECONFIG": 1},
}

SEL = os.environ.get("GCO_VARIANTS")
SEL = SEL.split(",") if SEL else list(VARIANTS)
SHAPES = [tuple(int(v) for v in s.split("x")) for s in os.environ.get("GCO_SHAPES", "8x512").split(",")]


def _make(name):
    d = GEN / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    defines = VARIANTS[name]
    for kind in ("reader", "writer", "compute"):
        shutil.copy(SRC_OP / f"linear_gelu_{kind}.cpp", d)
    if defines is not None:
        for f in ("mbh_local.hpp", "mbh_local.inl", "gco_gelu_pack.hpp"):
            shutil.copy(SRC_LOCAL / f, d)
        p = d / "linear_gelu_compute.cpp"
        text = p.read_text()
        assert SHARED_INC in text
        text = text.replace(SHARED_INC, LOCAL_INC)
        head = "".join(f"#define {k} {v}\n" for k, v in defines.items())
        p.write_text(head + text)
    return d


_ref = {}
CALL_LOG = HERE / "gen" / "calls.txt"


def _collecting():
    # run_safe_pytest's inline up-front-collect pass runs every test once with fake outputs, in this process.
    try:
        import tests.plugins.up_front_collect as upc

        return upc._PASS == "collect"
    except ImportError:
        return False


def _log_call(name, shape):
    if _collecting():
        return
    CALL_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(CALL_LOG, "a") as f:
        f.write(f"{name} {shape[0]}x{shape[1]}\n")


@pytest.mark.parametrize("shape", SHAPES, ids=[f"{b}x{s}" for b, s in SHAPES])
@pytest.mark.parametrize("variant", SEL)
def test_gco(device, shape, variant, monkeypatch):
    import torch

    batch, seq = shape
    torch.manual_seed(0)
    x = torch.randn((batch, 1, seq, 768)).to(torch.bfloat16)
    w = (torch.randn((768, 3072)) / math.sqrt(768)).to(torch.bfloat16)
    b = torch.randn((3072,)).to(torch.bfloat16)
    tile = dict(dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    xd = ttnn.from_torch(x, memory_config=ttnn.L1_MEMORY_CONFIG, **tile)
    wd = ttnn.from_torch(w, memory_config=ttnn.DRAM_MEMORY_CONFIG, **tile)
    bd = ttnn.from_torch(b, memory_config=ttnn.DRAM_MEMORY_CONFIG, **tile)

    if shape not in _ref or _collecting():
        monkeypatch.setattr(pd, "KERNEL_DIR", _make("op"))
        device.disable_and_clear_program_cache()
        ref = ttnn.to_torch(linear_gelu(xd, wd, bd, memory_config=ttnn.L1_MEMORY_CONFIG)).clone()
        if not _collecting():
            _ref[shape] = ref
        _log_call("ref_op", shape)

    monkeypatch.setattr(pd, "KERNEL_DIR", _make(variant))
    device.disable_and_clear_program_cache()
    out = ttnn.to_torch(linear_gelu(xd, wd, bd, memory_config=ttnn.L1_MEMORY_CONFIG))
    ttnn.synchronize_device(device)
    _log_call(variant, shape)

    golden = torch.nn.functional.gelu(x.float() @ w.float() + b.float(), approximate="tanh")
    if _collecting():
        return
    ref = _ref[shape]
    pcc = torch.corrcoef(torch.stack([out.float().flatten(), golden.flatten()]))[0, 1].item()
    mismatches = (out.view(torch.int16) != ref.view(torch.int16)).sum().item()
    print(f"GCO_RESULT variant={variant} shape={batch}x{seq} pcc={pcc:.6f} bit_mismatches={mismatches}")
    if not variant.startswith("diag_"):
        assert mismatches == 0, f"{variant}: {mismatches} elements differ from the op output"
