# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""combine_bytes bake-off: shrink the bytes of the regime-R3 cross-core Σx² exchange (rms_norm perf round 1).

Whole-op bench of a COPY of rms_norm (kernels/ + cb_program_descriptor.py in this dir) — the only difference between
variants is how each R3-combine tile crosses the NoC (reader kernel, RMS_COMBINE_FACES):
  base      production: whole 32x32 tiles, one write per contiguous run
  faces     faces 0 and 2 only, two noc_async_writes per tile
  facesws   faces 0 and 2 only, one-packet stateful writes (set_state per destination core)
  *_p       the same + NaN poison in faces 1/3 of every cb_gather / cb_inv_rms page (correctness probe)
Correctness: PCC vs torch AND bit-exact vs the base variant's output of the same case (same session/process).

    CB_CASES=b8192x1024 CB_VARIANTS=base,faces scripts/run_safe_pytest.sh --profile <this file>
    python3 ttnn/ttnn/operations/rms_norm/perf_experiments/combine_bytes/report.py
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")  # ttnn/ forbids a module-level torch import
import ttnn

from eval.sharding import auto_shard_config, shard_config

_HERE = Path(__file__).parent
_spec = importlib.util.spec_from_file_location("cb_program_descriptor", _HERE / "cb_program_descriptor.py")
pd = importlib.util.module_from_spec(_spec)
sys.modules["cb_program_descriptor"] = pd
_spec.loader.exec_module(pd)

ML = ttnn.TensorMemoryLayout

# id -> (shape, memory_layout, shard ([h, w], (gx, gy)) | "auto" | None, layout, gamma, fp32_dest, pcc gate, poison pad)
CASES = {
    "b8192x1024": (
        (1, 1, 8192, 1024),
        ML.BLOCK_SHARDED,
        ([1024, 128], (8, 8)),
        ttnn.TILE_LAYOUT,
        True,
        False,
        0.9995,
        None,
    ),
    "b8192x1024_fp32": (
        (1, 1, 8192, 1024),
        ML.BLOCK_SHARDED,
        ([1024, 128], (8, 8)),
        ttnn.TILE_LAYOUT,
        True,
        True,
        0.9995,
        None,
    ),
    "b8192x1024_nog": (
        (1, 1, 8192, 1024),
        ML.BLOCK_SHARDED,
        ([1024, 128], (8, 8)),
        ttnn.TILE_LAYOUT,
        False,
        False,
        0.9995,
        None,
    ),
    "b7168x2048": (
        (1, 1, 7168, 2048),
        ML.BLOCK_SHARDED,
        ([736, 192], (11, 10)),
        ttnn.TILE_LAYOUT,
        True,
        False,
        0.9995,
        None,
    ),
    "b7168x1024": (
        (1, 1, 7168, 1024),
        ML.BLOCK_SHARDED,
        ([736, 128], (8, 10)),
        ttnn.TILE_LAYOUT,
        True,
        False,
        0.9995,
        None,
    ),
    "b640x7168": (
        (1, 1, 640, 7168),
        ML.BLOCK_SHARDED,
        ([64, 1024], (7, 10)),
        ttnn.TILE_LAYOUT,
        True,
        False,
        0.9995,
        None,
    ),
    "w32x7168": ((1, 1, 32, 7168), ML.WIDTH_SHARDED, ([32, 256], (7, 4)), ttnn.TILE_LAYOUT, True, False, 0.9995, None),
    "i32x7168": ((1, 1, 32, 7168), ML.INTERLEAVED, None, ttnn.TILE_LAYOUT, True, False, 0.9995, None),
    "i640x6144": ((1, 1, 640, 6144), ML.INTERLEAVED, None, ttnn.TILE_LAYOUT, True, False, 0.9995, None),
    # ragged-owner geometries (R6 pins / golden resilience list)
    "b448x384_ragged": (
        (1, 1, 448, 384),
        ML.BLOCK_SHARDED,
        ([224, 128], (3, 2)),
        ttnn.TILE_LAYOUT,
        True,
        False,
        0.999,
        None,
    ),
    "w3232x96": ((1, 1, 3232, 96), ML.WIDTH_SHARDED, "auto", ttnn.TILE_LAYOUT, True, False, 0.995, None),
    "b99991x64": ((99991, 64), ML.BLOCK_SHARDED, "auto", ttnn.TILE_LAYOUT, True, False, 0.995, None),
    # W % 32 != 0: masked W-pad core, poisoned tile padding
    "b256x200_masked": (
        (1, 1, 256, 200),
        ML.BLOCK_SHARDED,
        ([128, 64], (4, 2)),
        ttnn.TILE_LAYOUT,
        True,
        False,
        0.999,
        1000.0,
    ),
    # ROW_MAJOR shards (re-laid locally into the tilize staging)
    "b256x200_rm": (
        (1, 1, 256, 200),
        ML.BLOCK_SHARDED,
        ([128, 64], (4, 2)),
        ttnn.ROW_MAJOR_LAYOUT,
        True,
        False,
        0.999,
        None,
    ),
}

VARIANTS = {
    "base": (0, 0),
    "faces": (1, 0),
    "facesws": (2, 0),
    "faces2cb": (3, 0),
    "rotonly": (8, 0),
    "facesrot": (7, 0),
    "facesrot_p": (7, 1),
    "facesmc": (6, 0),
    "facesmc_p": (6, 1),
    "faceswsl": (4, 0),
    "faceswsl_p": (4, 1),
    "faces2cb_p": (3, 1),
    "faces_p": (1, 1),
    "facesws_p": (2, 1),
    # zone-instrumented runs (RMS_NORM_PERF_ZONES=1) for attribution only
    "base_z": (0, 0),
    "facesws_z": (2, 0),
    # payload ablations (ceiling probes, numerics garbage — no correctness check): a_<variant>_<what>
    "a_base_combine": (0, 0),
    "a_facesws_scatter": (2, 0),
    "a_facesws_allgather": (2, 0),
    "a_facesws_combine": (2, 0),
    # NEGATIVE CONTROL: face 0 only + NaN in faces 1-3 — must FAIL (proves the poison reaches the consumers)
    "negctl": (9, 2),
}


def _select(env, ids):
    want = [w for w in os.environ.get(env, "").split(",") if w]
    return [i for i in ids if not want or i in want]


def _pcc(got, x, g):
    got = got.double().flatten()
    xf = x.double()
    exp = xf / torch.sqrt((xf * xf).mean(-1, keepdim=True) + 1e-6)
    if g is not None:
        exp = exp * g.double()
    return torch.corrcoef(torch.stack([got, exp.flatten()]))[0, 1].item()


def _rms_norm(tx, tg, cfg, out_mc):
    out = ttnn.allocate_tensor_on_device(ttnn.Shape(list(tx.shape)), tx.dtype, tx.layout, tx.device(), out_mc)
    desc = pd.create_program_descriptor(tx, out, gamma=tg, epsilon=1e-6, compute_kernel_config=cfg)
    io = [tx] + ([tg] if tg is not None else []) + [out]
    return ttnn.generic_op(io, desc)


_BASE_OUT = {}


@pytest.mark.parametrize("variant", _select("CB_VARIANTS", list(VARIANTS)))
@pytest.mark.parametrize("case", _select("CB_CASES", list(CASES)))
def test_combine_bytes(device, monkeypatch, case, variant):
    shape, ml, shard, layout, with_gamma, fp32_dest, pcc_gate, poison = CASES[case]
    pd.COMBINE_FACES, pd.POISON_FACES13 = VARIANTS[variant]
    pd.END_ATOMIC_BARRIER = int(os.environ.get("CB_END_ATOMIC_BARRIER", "0"))
    monkeypatch.delenv("RMS_NORM_PERF_ZONES", raising=False)
    monkeypatch.delenv("RMS_NORM_ABLATE", raising=False)
    if variant.endswith("_z"):
        monkeypatch.setenv("RMS_NORM_PERF_ZONES", "1")
    if variant.startswith("a_"):
        monkeypatch.setenv("RMS_NORM_ABLATE", variant.split("_")[-1])
    torch.manual_seed(0)
    x = torch.randn(shape).to(torch.bfloat16)
    W = shape[-1]
    g = torch.randn(W).to(torch.bfloat16) if with_gamma else None
    if ml == ML.INTERLEAVED:
        mc = ttnn.DRAM_MEMORY_CONFIG
    elif shard == "auto":
        mc = auto_shard_config(list(shape), ml, layout=layout, dtype=ttnn.bfloat16, device=device)
    else:
        mc = shard_config(shard[0], shard[1], ml, layout=layout, dtype=ttnn.bfloat16, device=device)
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mc)
    tg = None
    if with_gamma:
        tg = ttnn.from_torch(g.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=layout, device=device)
    if poison is not None:
        tx = ttnn.fill_implicit_tile_padding(tx, poison)
        if tg is not None:
            tg = ttnn.fill_implicit_tile_padding(tg, poison)
    plan = pd.select_plan(tx, tg, fp32_dest)
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=fp32_dest, math_approx_mode=False
    )
    out = _rms_norm(tx, tg, cfg, tx.memory_config())
    log = os.environ.get("CB_BENCH_LOG", "generated/combine_bytes_order.txt")
    with open(log, "a") as fh:
        if variant.endswith("_z"):
            fh.write(f"{case}-{variant}|zones\n")
        else:
            fh.write(
                f"{case}|{variant}|{plan.regime}|G={plan.topo.G}|owners={plan.topo.owners}|R={plan.block_rows}"
                f"|T={plan.geometry.T_interm}\n"
            )
    if variant.startswith("a_"):
        return
    got = ttnn.to_torch(out)
    p = _pcc(got, x, g)
    print(f"CB_RESULT {case} {variant} pcc={p:.6f} regime={plan.regime} G={plan.topo.G} owners={plan.topo.owners}")
    assert not torch.isnan(got.float()).any(), "NaN in output"
    assert p >= pcc_gate, f"PCC {p} < {pcc_gate}"
    if variant == "base":
        _BASE_OUT[case] = got
    elif case in _BASE_OUT:
        assert torch.equal(got, _BASE_OUT[case]), "candidate output differs bitwise from the base variant"
        print(f"CB_RESULT {case} {variant} bit-exact vs base")
