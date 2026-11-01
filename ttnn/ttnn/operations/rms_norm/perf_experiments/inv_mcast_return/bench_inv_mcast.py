# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""inv_mcast_return bake-off (Perf 1). Not collected by a directory run (no test_ prefix).

Owner-combine inv_rms return: k unicasts (op today) vs multicast. Variants, back to back per cell:
  base       the op as committed (ttnn/ttnn/operations/rms_norm/...)
  uc         the clone in this dir, INV_RETURN="unicast" (= base + the wr_inv_wait_cmb / wr_inv_send / wr_inv_signal
             zones that split the owner's return into "waiting on compute" vs "the transport")
  mc_flag    raw multicast data + per-owner flag slots (no barrier)
  mc_inc     raw multicast data + one write barrier + multicast atomic increment
  mc_helper  kernel_lib ttnn.Mcast, rotating senders, Counter data-ready (one send per owned row)

  MC_CELLS=a,b MC_VARIANTS=base,mc_flag  scripts/run_safe_pytest.sh --profile <this file>::test_perf

Every variant is gated by the golden harness's check_output (PCC vs torch) and compared to base
(`MC <cell> <variant> pcc_vs_base=.. maxabs_vs_base=..`). The executed (cell, variant) order is appended to
labels.txt (GenericOp rows of the profiler CSV come out in that order).
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest
import ttnn

import eval.golden_tests.rms_norm.helpers as H
from eval.golden_tests.rms_norm.helpers import run_rms_norm

HERE = Path(__file__).parent
REPO = HERE.parents[5]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


clone = _load("rms_norm_pd_inv_mc", HERE / "pd_mc.py")
probe = _load("perf_p1_probe_mc", REPO / "tests/ttnn/unit_tests/operations/rms_norm/perf_p1_rms_norm.py")
import ttnn.operations.rms_norm.rms_norm  # noqa: E402,F401

OPMOD = sys.modules["ttnn.operations.rms_norm.rms_norm"]
BASE_PD = OPMOD.create_program_descriptor
ML = ttnn.TensorMemoryLayout


def _p(shape, ml, shard=None, fp32=False, orient=None):
    ex = {"math_fidelity": ttnn.MathFidelity.HiFi2, "pcc_threshold": 0.9995}
    if shard:
        ex["shard_shape"], ex["core_grid"] = shard
    if orient is not None:
        ex["shard_orientation"] = orient
    return dict(
        inputs=(shape,),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_layout=ml,
        fp32_dest_acc_en=fp32,
        gamma_mode="gamma",
        gamma_dtype=ttnn.bfloat16,
        gamma_layout=ttnn.TILE_LAYOUT,
        extras=ex,
    )


C = probe.CELLS
CELLS = {
    # targets
    "wid_perf_32x7168": C["wid_perf_32x7168"],
    "wid_perf_32x5120": C["wid_perf_32x5120"],
    "int_decode_32768": C["int_decode_32768"],
    "blk_perf_7168x1024": C["blk_perf_7168x1024"],
    # at-risk
    "wid_tile_2048x256": C["wid_tile_2048x256"],  # 8 owners x 8 rows each
    "bk2_r2": _p((1, 1, 640, 512), ML.BLOCK_SHARDED, ([64, 256], (2, 10))),  # k = 2 line
    "bk4_r4": _p((1, 1, 1280, 1024), ML.BLOCK_SHARDED, ([128, 256], (4, 10))),  # k = 4
    "bk_ragged_2048x1120": _p((1, 1, 2048, 1120), ML.BLOCK_SHARDED, ([256, 128], (9, 8))),  # ragged last column
    "wid_26of28": _p((1, 1, 32, 6656), ML.WIDTH_SHARDED, ([32, 256], (7, 4))),  # idle cores inside bbox
    "wid_col_32x7168": _p(
        (1, 1, 32, 7168), ML.WIDTH_SHARDED, ([32, 256], (7, 4)), orient=ttnn.ShardOrientation.COL_MAJOR
    ),
    "int_160x11008": _p((1, 1, 160, 11008), ML.INTERLEAVED),  # k = 25, groups span rows -> 3-rectangle covers
    "int_1024x16384": _p((1, 1, 1024, 16384), ML.INTERLEAVED),  # interleaved k = 16, 4-row blocks
    # run 2: single-owner regime boundaries
    "wk8_r1": _p((1, 1, 32, 1024), ML.WIDTH_SHARDED, ([32, 128], (8, 1))),  # k = 8 line, 1 row
    "bk4_r1": _p((1, 1, 320, 1024), ML.BLOCK_SHARDED, ([32, 256], (4, 10))),  # k = 4, 10 concurrent groups, 1 row
    "int_1280x16384": _p((1, 1, 1280, 16384), ML.INTERLEAVED),  # mixed: multi-owner blocks then a 1-row block
}
_sel = os.environ.get("MC_CELLS")
IDS = [c for c in CELLS if not _sel or c in _sel.split(",")]
VARIANTS = os.environ.get("MC_VARIANTS", "base,uc,mc_flag,mc_inc,mc_helper").split(",")
LABELS = HERE / os.environ.get("MC_LABELS", "labels.txt")


def _pcc(a, b):
    import torch  # function-local: the ttnn tree forbids global torch imports

    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _run_variant(device, axes, variant, monkeypatch):
    import torch  # function-local: the ttnn tree forbids global torch imports

    captured = {}
    orig = H.rms_norm

    def wrap(*a, **kw):
        out = orig(*a, **kw)
        captured["out"] = ttnn.to_torch(out).float()
        return out

    monkeypatch.setattr(H, "rms_norm", wrap)
    if variant == "base":
        monkeypatch.setattr(OPMOD, "create_program_descriptor", BASE_PD)
    else:
        monkeypatch.setattr(clone, "INV_RETURN", {"uc": "unicast", "mc_one_f2": "mc_one"}.get(variant, variant))
        monkeypatch.setattr(clone, "INV_MC_FACES", 2 if variant == "mc_one_f2" else 1)
        monkeypatch.setattr(OPMOD, "create_program_descriptor", clone.create_program_descriptor)
    run_rms_norm(device=device, **axes)
    monkeypatch.setattr(H, "rms_norm", orig)
    return captured["out"]


@pytest.mark.parametrize("cell", IDS)
def test_perf(device, cell, monkeypatch):
    axes = dict(CELLS[cell])
    outs = {}
    for v in VARIANTS:
        outs[v] = _run_variant(device, axes, v, monkeypatch)
        with open(LABELS, "a") as f:
            f.write(f"{cell}:{v}\n")
        if v != "base":
            info = dict(clone.LAST_INV_INFO)
            print(
                f"MCINFO {cell} {v} k={info.get('k')} b={info.get('b')} mode={info.get('mode')} rows={info.get('rows')} "
                f"groups={len(info.get('groups', []))} rects0={info.get('groups', [None])[0]}"
            )
    base = outs.get("base")
    for v, o in outs.items():
        if v == "base" or base is None:
            continue
        print(f"MC {cell} {v} pcc_vs_base={_pcc(o, base):.7f} maxabs_vs_base={(o - base).abs().max().item():.5g}")


# Mixed blocks: force block_row_tiles = 2 so a 3-tile-row group runs blocks [2, 1] — multi-owner (unicast) block
# then a single-owner (multicast) block on the same semaphore; plus 2-row-only groups. Clone only (uc = reference).
MIXED = {
    "int_1024x16384_b3": _p((1, 1, 1024, 16384), ML.INTERLEAVED),  # k = 16, 4 rows/group -> blocks [3, 1]
    "int_1280x32768_b3": _p((1, 1, 1280, 32768), ML.INTERLEAVED, fp32=True),
}
MIXED_B = {"int_1024x16384_b3": 3, "int_1280x32768_b3": 3}


@pytest.mark.parametrize("cell", list(MIXED))
def test_mixed(device, cell, monkeypatch):
    import dataclasses

    orig_sel = clone.select_blocking

    def forced(*a, **kw):
        blk = orig_sel(*a, **kw)
        return dataclasses.replace(blk, block_row_tiles=min(MIXED_B[cell], blk.block_row_tiles))

    monkeypatch.setattr(clone, "select_blocking", forced)
    axes = dict(MIXED[cell])
    outs = {}
    for v in ("uc", "mc_one", "mc_one_helper"):
        outs[v] = _run_variant(device, axes, v, monkeypatch)
        info = dict(clone.LAST_INV_INFO)
        print(f"MCINFO {cell} {v} k={info.get('k')} b={info.get('b')} mode={info.get('mode')} rows={info.get('rows')}")
    for v in ("mc_one", "mc_one_helper"):
        o, b = outs[v], outs["uc"]
        print(f"MC {cell} {v} pcc_vs_uc={_pcc(o, b):.7f} maxabs_vs_uc={(o - b).abs().max().item():.5g}")
