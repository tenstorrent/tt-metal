# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""row_partial_mcast bake-off (Perf 2). Not collected by a directory run (no test_ prefix).

Owner-combine exchange: what a member ships to a row's owner, and how inv_rms comes back. Variants, back to back
per cell in one pytest session:
  base        the op as it is (ttnn/ttnn/operations/rms_norm/...): faces 0/2 partial (k slots per owned row) +
              unicast return (single-owner one-row blocks: member 0 multicast)
  clone       this dir's clone with both levers OFF (must equal base: clone sanity)
  row_uc      row partial (ceil(k/32) gather tiles per owned row) + the op's unicast return
  faces_root  faces 0/2 partial + root return (owners -> member 0 -> one multicast)
  row_root    row partial + root return  (the candidate)

  RP_CELLS=a,b RP_VARIANTS=base,row_root  scripts/run_safe_pytest.sh --profile <this file>

Every variant runs the golden harness's check_output (PCC vs torch); RP lines print PCC / max-abs vs torch and vs
base. Executed (cell, variant) order is appended to $RP_LABELS (GenericOp rows of the profiler CSV follow it).
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


clone = _load("rms_norm_pd_row_partial", HERE / "pd_row.py")
grad = _load("rms_norm_pd_row_grad", HERE / "graduate" / "rms_norm_program_descriptor.py")
probe = _load("perf_p2_probe_rp", REPO / "tests/ttnn/unit_tests/operations/rms_norm/perf_p2_rms_norm.py")
import ttnn.operations.rms_norm.rms_norm  # noqa: E402,F401

OPMOD = sys.modules["ttnn.operations.rms_norm.rms_norm"]
BASE_PD = OPMOD.create_program_descriptor
ML = ttnn.TensorMemoryLayout
_cell, _perf, _res = probe._cell, probe._perf, probe._res
F32, BF16, BF8, RM = ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.ROW_MAJOR_LAYOUT
HIFI2 = ttnn.MathFidelity.HiFi2
LOFI = ttnn.MathFidelity.LoFi


def _w(shape, shard, grid, fp32=True, dtype=BF16, fid=None, gdtype=BF16):
    ex = {"shard_shape": shard, "core_grid": grid}
    if fid is not None:
        ex["math_fidelity"] = fid
    return _cell(shape, dtype=dtype, ml=ML.WIDTH_SHARDED, fp32=fp32, gdtype=gdtype, extras=ex)


C = probe.CELLS
CELLS = {
    # ---- targets ----
    "wid_tile_128x4096": C["wid_tile_128x4096"],
    "wid_tile_bf8_128x4096_nog": C["wid_tile_bf8_128x4096_nog"],
    "wid_tile_224x3072": C["wid_tile_224x3072"],
    "wid_tile_160x11008": C["wid_tile_160x11008"],
    "wid_tile_13x777x1023": C["wid_tile_13x777x1023"],
    "wid_rm_1024x1024": C["wid_rm_1024x1024"],
    # ---- at-risk ----
    "wid_perf_32x7168": C["wid_perf_32x7168"],
    "blk_perf_7168x1024": C["blk_perf_7168x1024"],
    "blk_perf_7168x2048": C["blk_perf_7168x2048"],
    "int_decode_32768": _cell((1, 1, 32, 32768)),
    # ---- domain sweep: K x b (WIDTH, one-tile-wide shards, pinned grids) ----
    "wk16_b4": _w((1, 1, 128, 512), [128, 32], (8, 2)),
    "wk32_b4": _w((1, 1, 128, 1024), [128, 32], (8, 4)),
    "wk64_b4": _w((1, 1, 128, 2048), [128, 32], (8, 8)),
    "wk8_b4": _w((1, 1, 128, 256), [128, 32], (8, 1)),
    "wk64_b1": _w((1, 1, 32, 2048), [32, 32], (8, 8)),
    "wk128_b1": _w((1, 1, 32, 4096), [32, 32], (13, 10)),
    "wk128_b16": _w((1, 1, 512, 4096), [512, 32], (13, 10)),
    "wid_tile_2048x256": _cell((1, 1, 2048, 256), ml=ML.WIDTH_SHARDED),
    "wid_tile_256x2048": _cell((1, 1, 256, 2048), ml=ML.WIDTH_SHARDED),
    # precision regimes
    "wk128_b4_f32": _w((1, 1, 128, 4096), [128, 32], (13, 10), dtype=F32, gdtype=F32),
    "wk128_b4_lofi": _w((1, 1, 128, 4096), [128, 32], (13, 10), fid=LOFI),
    "wk128_b4_nofp32": _w((1, 1, 128, 4096), [128, 32], (13, 10), fp32=False),
    "wk115_b5_lofi_nofp32": _w((1, 1, 160, 3680), [160, 32], (13, 10), fp32=False, fid=LOFI),
    # interleaved owner (k > 8), multi-row
    "int_160x11008": _cell((1, 1, 160, 11008), fp32=False),
    "int_1024x16384": _cell((1, 1, 1024, 16384), fp32=False),
    "int_256x32768": _cell((1, 1, 256, 32768), fp32=False),
    "int_2048x16384": _cell((1, 1, 2048, 16384), fp32=False),
    "int_512x32768": _cell((1, 1, 512, 32768), fp32=False),
    "int_1024x16384_f32": _cell((1, 1, 1024, 16384)),
    "int_640x12288": _cell((1, 1, 640, 12288), fp32=False),
    "int_128x65536": _cell((1, 1, 128, 65536)),
    "int_rm_160x11008": _cell((1, 1, 160, 11008), layout=RM, glayout=RM, fp32=False),
    "blk_tile_128x4096": C["blk_tile_128x4096"],
    "blk_rm_128x4096": C["blk_rm_128x4096"],
    "wid_rm_f32_4x1x512x512": C["wid_rm_f32_4x1x512x512"],
    "wid_rm_5x3x928x544": C["wid_rm_5x3x928x544"],
    "wid_rm_13x777x1023": C["wid_rm_13x777x1023"],
    "blk_rm_99991x64": C["blk_rm_99991x64"],
    "wid_perf_32x5120": _perf((1, 1, 32, 5120), ML.WIDTH_SHARDED, ([32, 160], (8, 4))),
}
_sel = os.environ.get("RP_CELLS")
IDS = [c for c in CELLS if not _sel or c in _sel.split(",")]
VARIANTS = os.environ.get("RP_VARIANTS", "base,row_root").split(",")
LABELS = HERE / os.environ.get("RP_LABELS", "runs/labels.txt")
FLAGS = {
    "clone": (False, False),
    "row_uc": (True, False),
    "faces_root": (False, True),
    "row_root": (True, True),
    "row_root_mm": (True, True, True),
    "row_uc_mm": (True, False, True),
    "row_root_mm_sp": (True, True, True, True),
    "row_root_mm_sp_kg": (True, True, True, True, True),  # + combine-pipeline k gate lifted
}


def _stats(a, b):
    import torch  # function-local: the ttnn tree forbids global torch imports

    a, b = a.flatten().double(), b.flatten().double()
    pcc = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
    return pcc, (a - b).abs().max().item()


def _run_variant(device, axes, variant, monkeypatch):
    captured = {}
    orig_check = H.check_output

    def chk(out, expected, **kw):
        import torch  # function-local

        captured["out"] = ttnn.to_torch(out).float()
        captured["exp"] = expected.float() if isinstance(expected, torch.Tensor) else expected
        try:
            orig_check(out, expected, **kw)
            captured["ok"] = True
        except Exception as e:  # report, keep going (correctness is reported per variant)
            captured["ok"] = f"FAIL {type(e).__name__}: {str(e)[:200]}"

    monkeypatch.setattr(H, "check_output", chk)
    if variant == "base":
        monkeypatch.setattr(OPMOD, "create_program_descriptor", BASE_PD)
    elif variant == "grad":
        monkeypatch.setattr(OPMOD, "create_program_descriptor", grad.create_program_descriptor)
    else:
        rp, root, mm, sp, kg = (FLAGS[variant] + (False, False, False))[:5]
        monkeypatch.setattr(clone, "COMBINE_PIPELINE_K_PER_WORK", 10**6 if kg else 2)
        monkeypatch.setattr(clone, "ROW_COLLAPSE_MM", mm)
        monkeypatch.setattr(clone, "OWNER_SPREAD", sp)
        monkeypatch.setattr(clone, "ROW_PARTIAL", rp)
        monkeypatch.setattr(clone, "INV_ROOT_MCAST", root)
        monkeypatch.setattr(OPMOD, "create_program_descriptor", clone.create_program_descriptor)
    run_rms_norm(device=device, **axes)
    monkeypatch.setattr(H, "check_output", orig_check)
    return captured


@pytest.mark.parametrize("cell", IDS)
def test_perf(device, cell, monkeypatch):
    axes = dict(CELLS[cell])
    outs = {}
    LABELS.parent.mkdir(exist_ok=True)
    for v in VARIANTS:
        outs[v] = _run_variant(device, axes, v, monkeypatch)
        with open(LABELS, "a") as f:
            f.write(f"{cell}:{v}\n")
        info = dict(clone.LAST_INFO) if v not in ("base", "grad") else {}
        print(f"RPINFO {cell} {v} {info}")
    base = outs.get("base")
    bad = []
    for v, o in outs.items():
        p_t, m_t = _stats(o["out"], o["exp"])
        line = f"RP {cell} {v} ok={o['ok']} pcc_vs_torch={p_t:.7f} maxabs_vs_torch={m_t:.5g}"
        if base is not None and v != "base":
            p_b, m_b = _stats(o["out"], base["out"])
            line += f" pcc_vs_base={p_b:.7f} maxabs_vs_base={m_b:.5g}"
        print(line)
        if o["ok"] is not True:
            bad.append(v)
    assert not bad, f"{cell}: {bad} failed check_output"
