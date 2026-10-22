# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""bcast_srcb_reuse whole-op A/B: the production rms_norm vs this dir's kernel copy (kernels/) with BSR_MODE.

Variants (BSR_OP_VARIANTS, default all):
  base  production op (ttnn.operations.rms_norm.rms_norm), the honest baseline
  m<N> this dir's kernels with BSR_MODE=N (bit flags, kernels/rms_norm_compute.cpp): 1 γ stock per-tile LLK,
        2 γ HOLD (srcB resident per γ column), 4 inv stock per-tile LLK, 8 inv HOLD; m0 = the op's chains
        (code-path control, must equal base). m5 = both stock, m6 = γ HOLD + inv stock, m10 = both HOLD.
Cases (BSR_OP_CASES). Every non-base run is gated: PCC >= case threshold vs torch fp64 AND bit-exact vs the
production op on the same input (BSR_OP_EXACT=0 to skip). Run under --profile; report.py labels the CSV.

    BSR_OP_CASES=b8192x1024 scripts/run_safe_pytest.sh --profile <this file> --import-mode=prepend
"""

import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import ttnn

from eval.sharding import shard_config
from ttnn.operations.rms_norm import rms_norm
from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd

HERE = Path(__file__).parent
LOG = os.environ.get("BSR_LOG", str(HERE / "bench_order.txt"))
ML = ttnn.TensorMemoryLayout

CASES = {
    "b8192x1024": dict(shape=(1, 1, 8192, 1024), ml=ML.BLOCK_SHARDED, shard=([1024, 128], (8, 8))),
    "b8192x1024_fp32": dict(shape=(1, 1, 8192, 1024), ml=ML.BLOCK_SHARDED, shard=([1024, 128], (8, 8)), fp32=True),
    "b8192x1024_nog": dict(shape=(1, 1, 8192, 1024), ml=ML.BLOCK_SHARDED, shard=([1024, 128], (8, 8)), gamma=False),
    "b8192x1024_lofi": dict(shape=(1, 1, 8192, 1024), ml=ML.BLOCK_SHARDED, shard=([1024, 128], (8, 8)), fid="LoFi"),
    "b8192x1024_hifi4": dict(shape=(1, 1, 8192, 1024), ml=ML.BLOCK_SHARDED, shard=([1024, 128], (8, 8)), fid="HiFi4"),
    "b7168x2048": dict(shape=(1, 1, 7168, 2048), ml=ML.BLOCK_SHARDED, shard=([736, 192], (11, 10))),
    "b7168x1024": dict(shape=(1, 1, 7168, 1024), ml=ML.BLOCK_SHARDED, shard=([736, 128], (8, 10))),
    "w32x7168": dict(shape=(1, 1, 32, 7168), ml=ML.WIDTH_SHARDED, shard=([32, 256], (7, 4))),
    "i32x7168": dict(shape=(1, 1, 32, 7168), ml=ML.INTERLEAVED),
    "i8192x1024": dict(shape=(1, 1, 8192, 1024), ml=ML.INTERLEAVED),
    "b8192x1000_mask": dict(shape=(1, 1, 8192, 1000), ml=ML.BLOCK_SHARDED, shard=([1024, 128], (8, 8)), poison=1e3),
    "b256x3072_2ch": dict(shape=(1, 1, 256, 3072), ml=ML.BLOCK_SHARDED, shard=([128, 1536], (2, 2))),
    "b99991x64": dict(shape=(99991, 64), ml=ML.BLOCK_SHARDED, shard=None, pcc=0.995),
    "b8192x1024_rm": dict(shape=(1, 1, 8192, 1024), ml=ML.BLOCK_SHARDED, shard=([1024, 128], (8, 8)), rm=True),
    "h8192x1024": dict(shape=(1, 1, 8192, 1024), ml=ML.HEIGHT_SHARDED, shard=([128, 1024], (8, 8))),
    "i8192x1024_rm": dict(shape=(1, 1, 8192, 1024), ml=ML.INTERLEAVED, rm=True),
    "h8192x64": dict(shape=(1, 1, 8192, 64), ml=ML.HEIGHT_SHARDED, shard=([256, 64], (8, 4))),
    "h8192x32": dict(shape=(1, 1, 8192, 32), ml=ML.HEIGHT_SHARDED, shard=([256, 32], (8, 4))),
    "h8192x128": dict(shape=(1, 1, 8192, 128), ml=ML.HEIGHT_SHARDED, shard=([256, 128], (8, 4))),
    "h16384x64": dict(shape=(1, 1, 16384, 64), ml=ML.HEIGHT_SHARDED, shard=([512, 64], (8, 4))),
    "i65536x128": dict(shape=(1, 1, 65536, 128), ml=ML.INTERLEAVED),
    "i65536x128_rm": dict(shape=(1, 1, 65536, 128), ml=ML.INTERLEAVED, rm=True),
    "b8192x1024_bf8": dict(shape=(1, 1, 8192, 1024), ml=ML.BLOCK_SHARDED, shard=([1024, 128], (8, 8)), bf8=True),
}
VARIANTS = ["base", "m0", "m1", "m2", "m4", "m5", "m6", "m8", "m10", "m18", "m42", "m74", "m106"]


def _sel(env, ids):
    want = [w for w in os.environ.get(env, "").split(",") if w]
    return [i for i in (want or ids) if i in ids]


def _ref(x, g):
    xf = x.double()
    y = xf / torch.sqrt((xf * xf).mean(-1, keepdim=True) + 1e-6)
    return y * g.double() if g is not None else y


def _pcc(a, b):
    return torch.corrcoef(torch.stack([a.double().flatten(), b.double().flatten()]))[0, 1].item()


def _use_copy(monkeypatch, mode):
    monkeypatch.setattr(pd, "KERNEL_DIR", HERE / "kernels")
    orig = pd._measurement_defines

    def defs(compute=False):
        d = orig(compute=compute)
        return d + [("BSR_MODE", str(mode))] if compute else d

    monkeypatch.setattr(pd, "_measurement_defines", defs)


@pytest.mark.parametrize("variant", _sel("BSR_OP_VARIANTS", VARIANTS))
@pytest.mark.parametrize("case", _sel("BSR_OP_CASES", list(CASES)))
def test_op_ab(device, monkeypatch, case, variant):
    c = CASES[case]
    shape, ml = c["shape"], c["ml"]
    monkeypatch.delenv("RMS_NORM_ABLATE", raising=False)
    if os.environ.get("BSR_ZONES", "0") != "1":
        monkeypatch.delenv("RMS_NORM_PERF_ZONES", raising=False)
    torch.manual_seed(0)
    x = torch.randn(shape).to(torch.bfloat16)
    W = shape[-1]
    g = torch.randn(W).to(torch.bfloat16) if c.get("gamma", True) else None
    dt = ttnn.bfloat8_b if c.get("bf8") else ttnn.bfloat16
    lay = ttnn.ROW_MAJOR_LAYOUT if c.get("rm") else ttnn.TILE_LAYOUT
    if ml == ML.INTERLEAVED:
        mc = ttnn.DRAM_MEMORY_CONFIG
    elif c.get("shard") is not None:
        mc = shard_config(c["shard"][0], c["shard"][1], ml, layout=lay, dtype=dt, device=device)
    else:
        from eval.sharding import auto_shard_config

        mc = auto_shard_config(list(shape), ml, layout=ttnn.TILE_LAYOUT, dtype=dt, device=device)
    tx = ttnn.from_torch(x, dtype=dt, layout=lay, device=device, memory_config=mc)
    tg = None
    if g is not None:
        tg = ttnn.from_torch(g.reshape(1, 1, 1, W), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    if c.get("poison") is not None:
        tx = ttnn.fill_implicit_tile_padding(tx, c["poison"])
        if tg is not None:
            tg = ttnn.fill_implicit_tile_padding(tg, c["poison"])
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=getattr(ttnn.MathFidelity, c.get("fid", "HiFi2")),
        fp32_dest_acc_en=c.get("fp32", False),
        math_approx_mode=False,
    )
    kwargs = {} if ml == ML.INTERLEAVED else {"memory_config": tx.memory_config()}
    if variant != "base":
        _use_copy(monkeypatch, int(variant[1:]))
    out = rms_norm(tx, gamma=tg, compute_kernel_config=cfg, **kwargs)
    with open(LOG, "a") as fh:
        fh.write(f"{case}|{variant}|{'zones' if os.environ.get('BSR_ZONES', '0') == '1' else 1}\n")
    got = ttnn.to_torch(out).float()
    ttnn.deallocate(out)
    xin = ttnn.to_torch(tx).float() if c.get("bf8") else x.float()
    p = _pcc(_ref(xin[..., :W] if xin.shape[-1] != W else xin, g.float() if g is not None else None), got)
    print(f"BSR_OP_RESULT {case} {variant} pcc={p:.6f}")
    assert p >= c.get("pcc", 0.9995 if not c.get("bf8") else 0.99), f"PCC {p}"
    if variant != "base" and os.environ.get("BSR_OP_EXACT", "1") == "1":
        monkeypatch.undo()
        ref = ttnn.to_torch(rms_norm(tx, gamma=tg, compute_kernel_config=cfg, **kwargs)).float()
        with open(LOG, "a") as fh:
            fh.write(f"{case}|ref_base|1\n")
        nd = int((ref != got).sum())
        print(f"BSR_OP_EXACT {case} {variant} mismatches={nd}")
        assert nd == 0, f"{nd} elements differ from the production op"
