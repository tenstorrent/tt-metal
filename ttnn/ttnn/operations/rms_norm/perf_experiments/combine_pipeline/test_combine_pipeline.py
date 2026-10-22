# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""combine_pipeline bake-off: software-pipelined regime-R3 reduce-scatter combine vs the op's one-shot schedule.

Variants (CP_VARIANTS, comma list; default all):
  base     the production op (ttnn.operations.rms_norm.rms_norm) — the honest baseline
  cp0      this dir's descriptor + kernels with CP_WAVES=0 (the one-shot code path; must equal base)
  k<W>p<P> CP_WAVES=W waves per block, CP_GAMMA0_POS=P (0: γ0 before the combines, 1: before the last, 2: after)
Cases (CP_CASES, comma list; default all). Every run is correctness-gated: PCC vs torch, and — with CP_EXACT=1 —
bit-exact equality with the production op's output on the same input (the schedule change keeps the math).

    CP_CASES=b8192x1024 CP_VARIANTS=base,k4p0 scripts/run_safe_pytest.sh --profile <this file>
The run appends "<case>|<variant>" per test to CP_LOG (default: this dir's bench_order.txt) for labelling the CSV.
"""

import importlib.util
import os
import re
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")  # ttnn/ forbids a module-level torch import
import ttnn

from eval.sharding import auto_shard_config, shard_config
from ttnn.operations.rms_norm import rms_norm

HERE = Path(__file__).parent
_spec = importlib.util.spec_from_file_location("cp_descriptor", HERE / "cp_descriptor.py")
cp = importlib.util.module_from_spec(_spec)
sys.modules["cp_descriptor"] = cp  # dataclasses resolve their module through sys.modules
_spec.loader.exec_module(cp)

ML = ttnn.TensorMemoryLayout

# id -> dict(shape, ml, shard (None = auto for sharded), layout, fp32, gamma, pcc, cap)
CASES = {
    "b8192x1024": dict(shape=(1, 1, 8192, 1024), ml=ML.BLOCK_SHARDED, shard=([1024, 128], (8, 8))),
    "b8192x1024_fp32": dict(shape=(1, 1, 8192, 1024), ml=ML.BLOCK_SHARDED, shard=([1024, 128], (8, 8)), fp32=True),
    "b8192x1024_nog": dict(shape=(1, 1, 8192, 1024), ml=ML.BLOCK_SHARDED, shard=([1024, 128], (8, 8)), gamma=False),
    "b8192x1000_mask": dict(shape=(1, 1, 8192, 1000), ml=ML.BLOCK_SHARDED, shard=([1024, 128], (8, 8)), poison=1000.0),
    "b7168x2048": dict(shape=(1, 1, 7168, 2048), ml=ML.BLOCK_SHARDED, shard=([736, 192], (11, 10))),
    "b7168x1024": dict(shape=(1, 1, 7168, 1024), ml=ML.BLOCK_SHARDED, shard=([736, 128], (8, 10))),
    "b640x7168": dict(shape=(1, 1, 640, 7168), ml=ML.BLOCK_SHARDED, shard=([64, 1024], (7, 10))),
    "w32x7168": dict(shape=(1, 1, 32, 7168), ml=ML.WIDTH_SHARDED, shard=([32, 256], (7, 4))),
    "i640x6144": dict(shape=(1, 1, 640, 6144), ml=ML.INTERLEAVED),
    "i32x7168": dict(shape=(1, 1, 32, 7168), ml=ML.INTERLEAVED),
    "w3232x96": dict(shape=(1, 1, 3232, 96), ml=ML.WIDTH_SHARDED, pcc=0.995),
    "w4064x160": dict(shape=(1, 1, 4064, 160), ml=ML.WIDTH_SHARDED, pcc=0.995),
    "b99991x64": dict(shape=(99991, 64), ml=ML.BLOCK_SHARDED, pcc=0.995),
    "ragged448x384": dict(shape=(1, 1, 448, 384), ml=ML.BLOCK_SHARDED, shard=([224, 128], (3, 2))),
    "ragged448x384_cap3": dict(shape=(1, 1, 448, 384), ml=ML.BLOCK_SHARDED, shard=([224, 128], (3, 2)), cap=3),
    "fewer64x1024": dict(shape=(1, 1, 64, 1024), ml=ML.BLOCK_SHARDED, shard=([64, 128], (8, 1))),
    "w256x1024": dict(shape=(1, 1, 256, 1024), ml=ML.WIDTH_SHARDED, shard=([256, 128], (8, 1))),
    # two W-chunks per core (S = 48 tiles > the 32-tile chunk cap): γ(c > 0) passes between the wave loops
    "b256x3072_2ch": dict(shape=(1, 1, 256, 3072), ml=ML.BLOCK_SHARDED, shard=([128, 1536], (2, 2))),
    "b256x3072_2ch_stream": dict(
        shape=(1, 1, 256, 3072), ml=ML.BLOCK_SHARDED, shard=([128, 1536], (2, 2)), stream_gamma=True
    ),
    # streaming (non-resident) γ on the focus geometry: the γ writer pushes per (block, chunk)
    "b8192x1024_stream": dict(
        shape=(1, 1, 8192, 1024), ml=ML.BLOCK_SHARDED, shard=([1024, 128], (8, 8)), stream_gamma=True
    ),
    "mask256x200": dict(shape=(1, 1, 256, 200), ml=ML.BLOCK_SHARDED, shard=([128, 64], (4, 2)), poison=1000.0),
}

VARIANTS = ["base", "auto", "cp0", "k1p0", "k2p0", "k2p1", "k2p2", "k4p0", "k4p1", "k4p2", "k4p3", "k4p4"]


def _select(env, ids):
    want = [w for w in os.environ.get(env, "").split(",") if w]
    if not want:
        return list(ids)
    return [i for i in want if i in ids or (env == "CP_VARIANTS" and re.fullmatch(r"k\d+p(\d+|h)(g\d+)?w?", i))]


def _ref(x, g):
    xf = x.double()
    y = xf / torch.sqrt((xf * xf).mean(-1, keepdim=True) + 1e-6)
    return y * g.double() if g is not None else y


def _pcc(a, b):
    return torch.corrcoef(torch.stack([a.double().flatten(), b.double().flatten()]))[0, 1].item()


def _set_variant(monkeypatch, variant):
    # trailing "w": the all-gather runs on the writer RISC-V (its NoC) instead of the reader's poll loop
    monkeypatch.setenv("CP_AG_WRITER", "1" if variant.endswith("w") else os.environ.get("CP_AG_WRITER", "0"))
    variant = variant.rstrip("w")
    # "g<N>" suffix before w: N waves per inv scaling call
    monkeypatch.setenv("CP_INV_GROUP", "1")
    if "g" in variant:
        variant, grp = variant.split("g")
        monkeypatch.setenv("CP_INV_GROUP", grp)
    if variant == "auto":
        # the recommended option: auto wave count, γ0 half-way, all-gather on the writer RISC-V
        monkeypatch.setenv("CP_WAVES", "-1")
        monkeypatch.setenv("CP_GAMMA0_POS", "255")
        monkeypatch.setenv("CP_AG_WRITER", "1")
        return
    if variant in ("base", "cp0"):
        monkeypatch.setenv("CP_WAVES", "0")
        monkeypatch.setenv("CP_GAMMA0_POS", "0")
        return
    w, p = variant[1:].split("p")
    monkeypatch.setenv("CP_WAVES", w)
    monkeypatch.setenv("CP_GAMMA0_POS", "255" if p == "h" else p)


@pytest.mark.parametrize("variant", _select("CP_VARIANTS", VARIANTS))
@pytest.mark.parametrize("case", _select("CP_CASES", list(CASES)))
def test_combine_pipeline(device, monkeypatch, case, variant):
    c = CASES[case]
    shape, ml = c["shape"], c["ml"]
    fp32 = c.get("fp32", False)
    with_gamma = c.get("gamma", True)
    monkeypatch.delenv("RMS_NORM_PERF_ZONES", raising=False)
    monkeypatch.delenv("RMS_NORM_ABLATE", raising=False)
    if os.environ.get("CP_ZONES", "0") == "1":
        monkeypatch.setenv("RMS_NORM_PERF_ZONES", "1")
    _set_variant(monkeypatch, variant)
    if c.get("stream_gamma"):
        if variant == "base":
            pytest.skip("streaming γ is a cp-descriptor monkeypatch (the production descriptor is untouched)")
        monkeypatch.setattr(cp, "GAMMA_RESIDENT", False)
    if c.get("cap"):
        orig = cp._max_block_rows
        cap = c["cap"]
        monkeypatch.setattr(cp, "_max_block_rows", lambda g, d, k, b: orig(g, d, min(k, cap), b))
        if variant == "base":
            pytest.skip("block cap is a cp-descriptor monkeypatch (the production descriptor is untouched)")

    torch.manual_seed(0)
    x = torch.randn(shape).to(torch.bfloat16)
    W = shape[-1]
    g = torch.randn(W).to(torch.bfloat16) if with_gamma else None
    if ml == ML.INTERLEAVED:
        mc = ttnn.DRAM_MEMORY_CONFIG
    elif c.get("shard") is not None:
        mc = shard_config(c["shard"][0], c["shard"][1], ml, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    else:
        mc = auto_shard_config(list(shape), ml, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    tg = None
    if with_gamma:
        tg = ttnn.from_torch(g.reshape(1, 1, 1, W), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    if c.get("poison") is not None:
        tx = ttnn.fill_implicit_tile_padding(tx, c["poison"])
        if tg is not None:
            tg = ttnn.fill_implicit_tile_padding(tg, c["poison"])
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=fp32, math_approx_mode=False
    )
    kwargs = {} if ml == ML.INTERLEAVED else {"memory_config": tx.memory_config()}
    fn = rms_norm if variant == "base" else cp.rms_norm_cp
    out = fn(tx, gamma=tg, compute_kernel_config=cfg, **kwargs)
    with open(os.environ.get("CP_LOG", str(HERE / "bench_order.txt")), "a") as fh:
        fh.write(f"{case}|{variant}\n")
    got = ttnn.to_torch(out).float()
    ttnn.deallocate(out)  # the bit-exact reference below needs the L1 back (huge tall shards)
    p = _pcc(_ref(x.float(), g.float() if g is not None else None), got)
    print(f"CP_RESULT {case} {variant} pcc={p:.6f}")
    assert p >= c.get("pcc", 0.9995), f"PCC {p}"
    if os.environ.get("CP_EXACT", "0") == "1" and variant != "base":
        if c.get("cap") or c.get("stream_gamma"):
            monkeypatch.setenv("CP_WAVES", "0")
            ref_out = ttnn.to_torch(cp.rms_norm_cp(tx, gamma=tg, compute_kernel_config=cfg, **kwargs)).float()
        else:
            ref_out = ttnn.to_torch(rms_norm(tx, gamma=tg, compute_kernel_config=cfg, **kwargs)).float()
        n_diff = int((ref_out != got).sum())
        print(f"CP_EXACT {case} {variant} mismatches={n_diff}")
        assert n_diff == 0, f"{n_diff} elements differ from the one-shot schedule"
