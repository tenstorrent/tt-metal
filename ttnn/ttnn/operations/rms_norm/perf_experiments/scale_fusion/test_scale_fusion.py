# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""scale_fusion bake-off: the normalize stage's two element-wise scalings (x*γ Row-bcast, then *inv_rms Col-bcast)
fused into one DEST window per tile block vs the op's current two-pass chain (x*γ round-trips through cb_x_gamma).

Whole-op A/B: the production descriptor is reused unchanged except that KERNEL_DIR points at this dir's kernel
copies and the compute kernel gets an SF_VARIANT define (monkeypatched per test; the production module is untouched
on disk). Variants (SF_VARIANTS, comma list):
  base  the production op (kernels/ of the op) - the honest baseline
  v0    this dir's kernels with SF_VARIANT=0 (code-path control: must be bit-exact with base)
  v1    fused, γ first: FPU x*γ (Row) -> DEST, DEST-reuse ELWMUL (MOVD2A -> srcA) * inv_rms (Col bcast in srcB)
  v2    fused, inv first: FPU x*inv (Col) -> DEST, DEST-reuse ELWMUL * γ (Row bcast in srcB)
  v3..  see kernels/rms_norm_compute.cpp head
Cases (SF_CASES). Every run is correctness-gated: PCC vs a torch fp64 reference, plus SF_EXACT=1 -> element
mismatches vs the production op on the same input (printed; asserted only when SF_ASSERT_EXACT=1).

    SF_CASES=b8192x1024 SF_VARIANTS=base,v1 scripts/run_safe_pytest.sh --profile <this file>
    python3 ttnn/ttnn/operations/rms_norm/perf_experiments/scale_fusion/sf_report.py
"""

import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import ttnn

from eval.sharding import auto_shard_config, shard_config
from ttnn.operations.rms_norm import rms_norm
from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd

HERE = Path(__file__).parent
ML = ttnn.TensorMemoryLayout

CASES = {
    "b8192x1024": dict(shape=(1, 1, 8192, 1024), ml=ML.BLOCK_SHARDED, shard=([1024, 128], (8, 8))),
    "b8192x1024_fp32": dict(shape=(1, 1, 8192, 1024), ml=ML.BLOCK_SHARDED, shard=([1024, 128], (8, 8)), fp32=True),
    "b8192x1024_nog": dict(shape=(1, 1, 8192, 1024), ml=ML.BLOCK_SHARDED, shard=([1024, 128], (8, 8)), gamma=False),
    "b7168x2048": dict(shape=(1, 1, 7168, 2048), ml=ML.BLOCK_SHARDED, shard=([736, 192], (11, 10))),
    "b7168x1024": dict(shape=(1, 1, 7168, 1024), ml=ML.BLOCK_SHARDED, shard=([736, 128], (8, 10))),
    "b640x7168": dict(shape=(1, 1, 640, 7168), ml=ML.BLOCK_SHARDED, shard=([64, 1024], (7, 10))),
    "w32x7168": dict(shape=(1, 1, 32, 7168), ml=ML.WIDTH_SHARDED, shard=([32, 256], (7, 4))),
    "i32x7168": dict(shape=(1, 1, 32, 7168), ml=ML.INTERLEAVED),
    "i8192x1024": dict(shape=(1, 1, 8192, 1024), ml=ML.INTERLEAVED),
    "i640x6144": dict(shape=(1, 1, 640, 6144), ml=ML.INTERLEAVED),
    "h2048x1024": dict(shape=(1, 1, 2048, 1024), ml=ML.HEIGHT_SHARDED, shard=([32, 1024], (8, 8))),
    # correctness-only geometries (ragged / masked / multi-chunk / RM / streaming γ / tall shards)
    "b8192x1000_mask": dict(shape=(1, 1, 8192, 1000), ml=ML.BLOCK_SHARDED, shard=([1024, 128], (8, 8)), poison=1000.0),
    "b256x3072_2ch": dict(shape=(1, 1, 256, 3072), ml=ML.BLOCK_SHARDED, shard=([128, 1536], (2, 2))),
    "b256x3072_2ch_fp32": dict(shape=(1, 1, 256, 3072), ml=ML.BLOCK_SHARDED, shard=([128, 1536], (2, 2)), fp32=True),
    "ragged448x384": dict(shape=(1, 1, 448, 384), ml=ML.BLOCK_SHARDED, shard=([224, 128], (3, 2))),
    "w3232x96": dict(shape=(1, 1, 3232, 96), ml=ML.WIDTH_SHARDED, pcc=0.995),
    "b99991x64": dict(shape=(99991, 64), ml=ML.BLOCK_SHARDED, pcc=0.995),
    "i8192x5120_rmg": dict(shape=(1, 1, 8192, 5120), ml=ML.INTERLEAVED, gamma_rm=True),
    "i8192x5120": dict(shape=(1, 1, 8192, 5120), ml=ML.INTERLEAVED),
    "i640x6144_rm": dict(shape=(1, 1, 640, 6144), ml=ML.INTERLEAVED, rm=True),
    "i2048x1024_fp32": dict(shape=(1, 1, 2048, 1024), ml=ML.INTERLEAVED, fp32=True),
    "i8192x1000_mask": dict(shape=(1, 1, 8192, 1000), ml=ML.INTERLEAVED),
    "i64x16384": dict(shape=(1, 1, 64, 16384), ml=ML.INTERLEAVED),
    "b8192x1024_stream": dict(
        shape=(1, 1, 8192, 1024), ml=ML.BLOCK_SHARDED, shard=([1024, 128], (8, 8)), stream_gamma=True
    ),
    "i8192x5120_stream": dict(shape=(1, 1, 8192, 5120), ml=ML.INTERLEAVED, stream_gamma=True),
    "b8192x1024_bf8": dict(shape=(1, 1, 8192, 1024), ml=ML.BLOCK_SHARDED, shard=([1024, 128], (8, 8)), bf8=True),
}

VARIANTS = ["base", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12"]


def _select(env, ids):
    want = [w for w in os.environ.get(env, "").split(",") if w]
    return [i for i in (want or ids) if i in ids]


def _ref(x, g):
    xf = x.double()
    y = xf / torch.sqrt((xf * xf).mean(-1, keepdim=True) + 1e-6)
    return y * g.double() if g is not None else y


def _pcc(a, b):
    return torch.corrcoef(torch.stack([a.double().flatten(), b.double().flatten()]))[0, 1].item()


def _use_variant(monkeypatch, variant):
    if variant == "base":
        return
    monkeypatch.setattr(pd, "KERNEL_DIR", HERE / "kernels")
    orig = pd._measurement_defines

    def defines(compute: bool = False):
        d = orig(compute)
        if compute:
            d = d + [("SF_VARIANT", variant[1:])]
        return d

    monkeypatch.setattr(pd, "_measurement_defines", defines)


@pytest.mark.parametrize("variant", _select("SF_VARIANTS", VARIANTS))
@pytest.mark.parametrize("case", _select("SF_CASES", list(CASES)))
def test_scale_fusion(device, monkeypatch, case, variant):
    c = CASES[case]
    shape, ml = c["shape"], c["ml"]
    fp32 = c.get("fp32", False)
    with_gamma = c.get("gamma", True)
    dtype = ttnn.bfloat8_b if c.get("bf8") else ttnn.bfloat16
    layout = ttnn.ROW_MAJOR_LAYOUT if c.get("rm") else ttnn.TILE_LAYOUT
    monkeypatch.delenv("RMS_NORM_PERF_ZONES", raising=False)
    monkeypatch.delenv("RMS_NORM_ABLATE", raising=False)
    if os.environ.get("SF_ZONES", "0") == "1":
        monkeypatch.setenv("RMS_NORM_PERF_ZONES", "1")
    if c.get("stream_gamma"):
        # in-memory only (base and variants alike): the streaming (non-resident) γ path
        monkeypatch.setattr(pd, "GAMMA_RESIDENT", False)

    torch.manual_seed(0)
    x = torch.randn(shape).to(torch.bfloat16)
    W = shape[-1]
    g = torch.randn(W).to(torch.bfloat16) if with_gamma else None
    if ml == ML.INTERLEAVED:
        mc = ttnn.DRAM_MEMORY_CONFIG
    elif c.get("shard") is not None:
        mc = shard_config(c["shard"][0], c["shard"][1], ml, layout=layout, dtype=dtype, device=device)
    else:
        mc = auto_shard_config(list(shape), ml, layout=layout, dtype=dtype, device=device)
    tx = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device, memory_config=mc)
    tg = None
    if with_gamma:
        if c.get("gamma_rm"):
            tg = ttnn.from_torch(
                g.reshape(1, 1, 1, W), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
            )
        else:
            tg = ttnn.from_torch(g.reshape(1, 1, 1, W), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    if c.get("poison") is not None:
        tx = ttnn.fill_implicit_tile_padding(tx, c["poison"])
        if tg is not None:
            tg = ttnn.fill_implicit_tile_padding(tg, c["poison"])
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=fp32, math_approx_mode=False
    )
    kwargs = {} if ml == ML.INTERLEAVED else {"memory_config": tx.memory_config()}
    with monkeypatch.context() as m:
        _use_variant(m, variant)
        out = rms_norm(tx, gamma=tg, compute_kernel_config=cfg, **kwargs)
    with open(os.environ.get("SF_LOG", str(HERE / "bench_order.txt")), "a") as fh:
        zones = "|zones" if os.environ.get("SF_ZONES", "0") == "1" else ""
        fh.write(f"{case}|{variant}{zones}\n")
    got = ttnn.to_torch(out).float()
    ttnn.deallocate(out)
    ref = _ref(x.float(), g.float() if g is not None else None)
    p = _pcc(ref, got)
    maxabs = (ref.double() - got.double()).abs().max().item()
    relrms = ((ref.double() - got.double()).pow(2).mean().sqrt() / ref.double().pow(2).mean().sqrt()).item()
    msg = f"SF_RESULT {case} {variant} pcc={p:.6f} maxabs={maxabs:.4g} relrms={relrms:.5g}"
    if os.environ.get("SF_EXACT", "0") == "1" and variant != "base":
        base_out = ttnn.to_torch(rms_norm(tx, gamma=tg, compute_kernel_config=cfg, **kwargs)).float()
        n_diff = int((base_out != got).sum())
        mx = (base_out.double() - got.double()).abs().max().item()
        msg += f" mismatches_vs_base={n_diff} maxabs_vs_base={mx:.4g}"
        if os.environ.get("SF_ASSERT_EXACT", "0") == "1":
            assert n_diff == 0, msg
    print(msg)
    with open(HERE / "results.txt", "a") as fh:
        fh.write(msg + "\n")
    if variant in os.environ.get("SF_DIAG", "v3,v4,v7,v8,v9,v10").split(","):
        return  # diagnostic (attribution-only) variant: numerics intentionally wrong
    assert p >= c.get("pcc", 0.9995), msg
