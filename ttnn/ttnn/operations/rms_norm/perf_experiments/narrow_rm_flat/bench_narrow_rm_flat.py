# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""narrow_rm_flat bake-off: ROW_MAJOR WIDTH / BLOCK_SHARDED 16 B-wide (w_s | 16 datums) shards computed as flat tiles.

Not collected by a directory run (no `test_` prefix). Run via run.sh:

    NF_CELLS=wid_rm_5x3x928x544 NF_VARIANTS=base,cmp,bc NF_TAG=r1 scripts/run_safe_pytest.sh --profile \
        ttnn/ttnn/operations/rms_norm/perf_experiments/narrow_rm_flat/bench_narrow_rm_flat.py -s --import-mode=prepend

Variants (NF_VARIANTS):
  base   the UNMODIFIED op (honest baseline: stick staging + tilize + owner combine per tile-row + untilize)
  cmp    flat tiles, compressed partials (matmul with S_j, one value per stick), expand matmul in pass 2
  bc     flat tiles, broadcast-form partials (matmul with blockdiag ones), no expand
  grad   the graduation-ready descriptor in graduate/ (cmp form)
Cells whose input is not in the flat domain fall through to the op's own descriptor for every variant.
Each dispatch is checked vs torch (golden tolerance / cell pcc_threshold) and vs the base output; labels + metrics are
appended to runs/<NF_TAG>_labels.jsonl in dispatch order (GenericOp rows of the profiler CSV map 1:1).
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest
import ttnn

from eval.golden_tests.rms_norm.helpers import _TORCH_DTYPE, TOLERANCES, pytorch_rms_norm
from eval.metrics import check_output
from eval.sharding import auto_shard_config, shard_config

HERE = Path(__file__).parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


import ttnn.operations.rms_norm.rms_norm  # noqa: E402,F401

opmod = sys.modules["ttnn.operations.rms_norm.rms_norm"]
BASE_PD = opmod.create_program_descriptor
fpd = _load("nf_flat_pd", HERE / "flat_program_descriptor.py")
fpd1 = _load("nf_flat_pd_v1", HERE / "flat_program_descriptor_v1.py")  # v1: unpipelined, kernels_v1/
fpd1.KERNEL_DIR = HERE / "kernels_v1"
fpd2 = _load("nf_flat_pd_v2", HERE / "flat_program_descriptor_v2.py")  # v2: tile units, unicast return
fpd2.KERNEL_DIR = HERE / "kernels_v2"
_grad_path = HERE / "graduate" / "rms_norm_narrow_flat_program_descriptor.py"
gpd = _load("nf_grad_pd", _grad_path) if _grad_path.exists() else None
_REAL_RECTS = gpd._mcast_rects if gpd else None

ML = ttnn.TensorMemoryLayout
RM = ttnn.ROW_MAJOR_LAYOUT
BF16, F32 = ttnn.bfloat16, ttnn.float32


def _c(shape, ml, *, dtype=BF16, fp32=False, gdtype=BF16, shard=None, fid=ttnn.MathFidelity.HiFi4, pcc=None):
    return dict(shape=shape, ml=ml, dtype=dtype, fp32=fp32, gdtype=gdtype, shard=shard, fid=fid, pcc=pcc)


CELLS = {
    # ---- targets (perf_p2 probe ids, same axes) ----
    "wid_rm_5x3x928x544": _c((5, 3, 928, 544), ML.WIDTH_SHARDED),
    "wid_rm_13x777x1023": _c((13, 777, 1023), ML.WIDTH_SHARDED),
    "wid_rm_1024x1024": _c((1024, 1024), ML.WIDTH_SHARDED, fp32=True, gdtype=F32),
    "wid_rm_f32_4x1x512x512": _c((4, 1, 512, 512), ML.WIDTH_SHARDED, dtype=F32, fp32=True),
    "blk_rm_99991x64": _c((99991, 64), ML.BLOCK_SHARDED),
    # ---- at-risk / guard ----
    "blk_rm_128x4096": _c((1, 1, 128, 4096), ML.BLOCK_SHARDED, gdtype=F32),
    "wid_rm_3x736x5119": _c((3, 1, 736, 5119), ML.WIDTH_SHARDED),
    "wid_rm_2048x256": _c((1, 1, 2048, 256), ML.WIDTH_SHARDED, fp32=True),
    # ---- domain sweep ----
    "wid_rm_nog_2048x256": _c((1, 1, 2048, 256), ML.WIDTH_SHARDED, gdtype=None),
    "wid_rm_fp32d_2048x256": _c((1, 1, 2048, 256), ML.WIDTH_SHARDED, fp32=True, gdtype=F32),
    "wid_rm_sw16_2048x256": _c((1, 1, 2048, 256), ML.WIDTH_SHARDED, shard=([2048, 16], (8, 2))),
    "wid_rm_f32sw8_2048x256": _c((1, 1, 2048, 256), ML.WIDTH_SHARDED, dtype=F32, fp32=True, shard=([2048, 8], (8, 4))),
    "wid_rm_f32sw16_2048x256": _c(
        (1, 1, 2048, 256), ML.WIDTH_SHARDED, dtype=F32, fp32=True, shard=([2048, 16], (8, 2))
    ),
    "wid_rm_small_64x250": _c((1, 1, 64, 250), ML.WIDTH_SHARDED),
    "wid_rm_ragged_2000x249": _c((1, 1, 2000, 249), ML.WIDTH_SHARDED),
    "blk_rm_odd_2x47x97": _c((2, 1, 47, 97), ML.BLOCK_SHARDED),
    "blk_rm_2048x256": _c((1, 1, 2048, 256), ML.BLOCK_SHARDED),
    "wid_rm_hifi2_2048x256": _c((1, 1, 2048, 256), ML.WIDTH_SHARDED, fid=ttnn.MathFidelity.HiFi2),
    "wid_rm_lofi_2048x256": _c((1, 1, 2048, 256), ML.WIDTH_SHARDED, fid=ttnn.MathFidelity.LoFi),
    "wid_rm_gf32_2048x256": _c((1, 1, 2048, 256), ML.WIDTH_SHARDED, gdtype=F32),  # bf16 x, 16-bit DEST, fp32 gamma
    "wid_rm_f32_nog_2048x256": _c((1, 1, 2048, 256), ML.WIDTH_SHARDED, dtype=F32, fp32=True, gdtype=None),
    "wid_rm_f32_gbf_2048x256": _c((1, 1, 2048, 256), ML.WIDTH_SHARDED, dtype=F32, fp32=True, gdtype=BF16),
    "blk_rm_k1_4096x8": _c((1, 1, 4096, 8), ML.BLOCK_SHARDED),  # one shard column: k = 1, no exchange
    "wid_rm_big_16384x128": _c((1, 1, 16384, 128), ML.WIDTH_SHARDED),  # 16 x [16384, 8]: 256 KB shards
    "wid_rm_w7_2048x7": _c((1, 1, 2048, 7), ML.WIDTH_SHARDED),  # one member, W tail 7 of 8
    "blk_rm_sw16_1024x128": _c((1, 1, 1024, 128), ML.BLOCK_SHARDED, shard=([128, 16], (8, 8)), fp32=True),
}

_sel = os.environ.get("NF_CELLS")
IDS = [c for c in CELLS if not _sel or c in _sel.split(",")]
VSEL = os.environ.get("NF_VARIANTS", "base,cmp").split(",")
TAG = os.environ.get("NF_TAG", "dev")
LABELS = HERE / "runs" / f"{TAG}_labels.jsonl"
EXTRA = tuple(d for d in os.environ.get("NF_DEFINES", "").split(",") if d)


def _pd_for(variant, info_sink):
    if variant == "base":
        return BASE_PD

    def make(input_tensor, gamma, output_tensor, *, epsilon, compute_kernel_config):
        if not fpd.flat_applies(input_tensor):
            info_sink["flat"] = False
            return BASE_PD(
                input_tensor, gamma, output_tensor, epsilon=epsilon, compute_kernel_config=compute_kernel_config
            )
        info_sink["flat"] = True
        if variant in ("grad", "grad_uc"):
            if variant == "grad_uc":  # force the leader's unicast fallback (groups of > MAX_RECTS rectangles)
                gpd._mcast_rects = lambda device, cores: []
            else:
                gpd._mcast_rects = _REAL_RECTS
            d = gpd.create_narrow_flat_program_descriptor(
                input_tensor, gamma, output_tensor, epsilon=epsilon, compute_kernel_config=compute_kernel_config
            )
            if d is None:
                info_sink["flat"] = False
                return BASE_PD(
                    input_tensor, gamma, output_tensor, epsilon=epsilon, compute_kernel_config=compute_kernel_config
                )
            return d
        toks = variant.split("_")
        form = toks[0]
        if toks[-1] in ("v1", "v2"):
            mod = fpd1 if toks[-1] == "v1" else fpd2
            kw = {} if toks[-1] == "v1" else {"lag": None}
        else:
            mod = fpd
            kw = dict(
                lag=next((int(t[3:]) for t in toks[1:] if t.startswith("lag")), None),
                q=next((int(t[1:]) for t in toks[1:] if t.startswith("q")), None),
                mcast=False if "uc" in toks[1:] else None,
                ret=0 if "uc" in toks[1:] else (1 if "mc" in toks[1:] else None),
            )
        desc, info = mod.create_flat_program_descriptor(
            input_tensor,
            gamma,
            output_tensor,
            epsilon=epsilon,
            compute_kernel_config=compute_kernel_config,
            form=form,
            extra_defines=EXTRA + (("FLAT_NO_SQ_AHEAD",) if "noahead" in toks else ()),
            **kw,
        )
        info_sink.update(info)
        return desc

    return make


@pytest.mark.parametrize("cell", IDS)
def test_bench(device, cell, monkeypatch):
    import torch

    cfg = CELLS[cell]
    shape, ml, dtype = cfg["shape"], cfg["ml"], cfg["dtype"]
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32).to(_TORCH_DTYPE[dtype])
    W = shape[-1]
    g = None
    tg = None
    if cfg["gdtype"] is not None:
        g = torch.randn(W, dtype=torch.float32).to(_TORCH_DTYPE[cfg["gdtype"]])
        tg = ttnn.from_torch(g.reshape(1, 1, 1, W), dtype=cfg["gdtype"], layout=RM, device=device)
    if cfg["shard"]:
        mc = shard_config(cfg["shard"][0], cfg["shard"][1], ml, layout=RM, dtype=dtype, device=device)
    else:
        mc = auto_shard_config(list(shape), ml, layout=RM, dtype=dtype, device=device)
    tx = ttnn.from_torch(x, dtype=dtype, layout=RM, device=device, memory_config=mc)
    ck = ttnn.ComputeConfigDescriptor()
    ck.math_fidelity = cfg["fid"]
    ck.fp32_dest_acc_en = cfg["fp32"]
    ck.math_approx_mode = False
    expected = pytorch_rms_norm(x, gamma=g, epsilon=1e-6)
    tol = TOLERANCES[dtype]
    if cfg["pcc"] is not None:
        tol = (cfg["pcc"], tol[1])
    ss = tx.memory_config().shard_spec
    print(f"NF_GEOM {cell}: {shape} {ml} shard={list(ss.shape)} cores={ss.grid.num_cores()}")

    base_out = None
    for v in VSEL:
        device.disable_and_clear_program_cache()
        info = {}
        monkeypatch.setattr(opmod, "create_program_descriptor", _pd_for(v, info))
        out = opmod.rms_norm(tx, gamma=tg, epsilon=1e-6, compute_kernel_config=ck, memory_config=tx.memory_config())
        res = ttnn.to_torch(out).float()
        ttnn.deallocate(out)
        ok, err = True, ""
        try:
            check_output(
                ttnn.from_torch(res.to(_TORCH_DTYPE[dtype]), dtype=dtype, layout=RM),
                expected,
                shape=shape,
                dtype=dtype,
                expected_layout=RM,
                tolerance=tol,
            )
        except Exception as e:  # recorded, asserted at the end
            ok, err = False, f"{type(e).__name__}: {str(e)[:200]}"
        exp = expected.float()
        pcc = torch.corrcoef(torch.stack([res.flatten(), exp.flatten()]))[0, 1].item()
        rec = dict(
            cell=cell,
            variant=v,
            ok=ok,
            pcc=pcc,
            max_abs_vs_torch=(res - exp).abs().max().item(),
            **{kk: vv for kk, vv in info.items()},
        )
        if base_out is None and v == "base":
            base_out = res
        elif base_out is not None:
            rec["pcc_vs_base"] = torch.corrcoef(torch.stack([res.flatten(), base_out.flatten()]))[0, 1].item()
            rec["max_abs_vs_base"] = (res - base_out).abs().max().item()
        print("NF " + json.dumps(rec))
        with open(LABELS, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        assert ok, f"{cell} {v}: {err}"
