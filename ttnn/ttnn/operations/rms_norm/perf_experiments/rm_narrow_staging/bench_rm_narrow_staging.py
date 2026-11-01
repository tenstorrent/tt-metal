# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""rm_narrow_staging bake-off: ROW_MAJOR sharded stick staging (reader) + write-back (writer).

    RNS_CELLS=wid_rm_2048x256 RNS_VARIANTS=base,cpu64 scripts/run_safe_pytest.sh --profile \
        ttnn/ttnn/operations/rms_norm/perf_experiments/rm_narrow_staging/bench_rm_narrow_staging.py

Per cell the input is built exactly like the golden harness (seed 0, auto_shard_config or pinned shard geometry,
RM bf16 gamma), then dispatched once per variant. `base` = the real op (unmodified kernels); every other variant =
this dir's kernel clone with named defines (rm_stage_copy.hpp). Correctness gate: each variant's output is
bit-identical to `base` (data movement only — the math is untouched) AND passes the golden PCC vs torch.
One GenericOp row per dispatch, in the printed RNS_ORDER.
"""

import os
from pathlib import Path

import pytest
import ttnn

from eval.golden_tests.rms_norm.helpers import pytorch_rms_norm, create_ttnn_input_tensor, TOLERANCES
from eval.metrics import check_output
from eval.sharding import shard_config
from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd
from ttnn.operations.rms_norm.rms_norm import rms_norm

ML = ttnn.TensorMemoryLayout
RM = ttnn.ROW_MAJOR_LAYOUT
HERE = Path(__file__).resolve().parent  # run via a symlink outside the ttnn package (see README in docstring)
REAL_KDIR = pd.KERNEL_DIR

VARIANTS = {
    "base": None,
    "grad": "GRAD",  # graduation/kernels: the exact diff proposed for the op (unconditional pzstc)
    "clone": (),
    "cpu16": ("RMS_CPU_MAX_16",),
    "cpu32": ("RMS_CPU_MAX_32",),
    "cpu64": ("RMS_CPU_MAX_64",),
    "cpu128": ("RMS_CPU_MAX_128",),
    "cpu256": ("RMS_CPU_MAX_256",),
    "cpu512": ("RMS_CPU_MAX_512",),
    "cpuall": ("RMS_CPU_MAX_4096",),
    "pz": ("RMS_PREZERO",),
    "cpu64pz": ("RMS_CPU_MAX_64", "RMS_PREZERO"),
    "cpu128pz": ("RMS_CPU_MAX_128", "RMS_PREZERO"),
    "cpuallpz": ("RMS_CPU_MAX_4096", "RMS_PREZERO"),
    "contig": ("RMS_CONTIG",),
    "st": ("RMS_STATEFUL",),
    "pzst": ("RMS_PREZERO", "RMS_STATEFUL"),
    "pzstc": ("RMS_PREZERO", "RMS_STATEFUL", "RMS_CONTIG"),
    "cpu64pzc": ("RMS_CPU_MAX_64", "RMS_PREZERO", "RMS_CONTIG"),
    "cpu128pzc": ("RMS_CPU_MAX_128", "RMS_PREZERO", "RMS_CONTIG"),
}


def _c(shape, ml, *, dtype=ttnn.bfloat16, shard=None, gamma=True, fp32=True):
    return dict(shape=shape, ml=ml, dtype=dtype, shard=shard, gamma=gamma, fp32=fp32)


CELLS = {
    # targets
    "wid_rm_2048x256": _c((1, 1, 2048, 256), ML.WIDTH_SHARDED),
    "blk_rm_2048x256": _c((1, 1, 2048, 256), ML.BLOCK_SHARDED),
    "wid_rm_1504x544": _c((1, 1, 1504, 544), ML.WIDTH_SHARDED),
    "blk_rm_1504x544": _c((1, 1, 1504, 544), ML.BLOCK_SHARDED),
    # piece-width sweep: (1,1,2048,256) WIDTH RM pinned shard widths (bytes per piece = 2 * sw)
    "wid_sw16": _c((1, 1, 2048, 256), ML.WIDTH_SHARDED, shard=([2048, 16], (8, 2))),
    "wid_sw32": _c((1, 1, 2048, 256), ML.WIDTH_SHARDED, shard=([2048, 32], (8, 1))),
    "wid_sw64": _c((1, 1, 2048, 256), ML.WIDTH_SHARDED, shard=([2048, 64], (4, 1))),
    "wid_sw128": _c((1, 1, 2048, 256), ML.WIDTH_SHARDED, shard=([2048, 128], (2, 1))),
    # at-risk
    "hgt_rm_grm": _c((128, 8192), ML.HEIGHT_SHARDED),
    "int_rm_128x4096_gf32": _c((1, 1, 128, 4096), ML.INTERLEAVED),
    # correctness: ragged H (valid_sticks), ragged / odd W tail, fp32 RM, ragged last shard
    "wid_rm_h2000": _c((1, 1, 2000, 256), ML.WIDTH_SHARDED),
    "wid_rm_w250": _c((1, 1, 64, 250), ML.WIDTH_SHARDED),
    "wid_rm_w249": _c((1, 1, 64, 249), ML.WIDTH_SHARDED),
    "wid_rm_f32": _c((1, 1, 2048, 256), ML.WIDTH_SHARDED, dtype=ttnn.float32),
    "blk_rm_ragged": _c((1, 1, 1000, 300), ML.BLOCK_SHARDED, shard=([128, 48], (7, 8))),
    "blk_rm_odd": _c((2, 1, 47, 97), ML.BLOCK_SHARDED),
}

_sel = os.environ.get("RNS_CELLS")
IDS = [c for c in CELLS if not _sel or c in _sel.split(",")]
VSEL = os.environ.get("RNS_VARIANTS", "base,cpu64").split(",")


@pytest.mark.parametrize("cell", IDS)
def test_bench(device, cell, monkeypatch):
    import torch  # function-local: the ttnn tree forbids global torch imports

    device.disable_and_clear_program_cache()  # variants differ only in kernel source/defines
    cfg = CELLS[cell]
    shape, ml, dtype = cfg["shape"], cfg["ml"], cfg["dtype"]
    torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32).to(torch_dtype)
    W = shape[-1]
    g = torch.randn(W, dtype=torch.float32).to(torch.bfloat16) if cfg["gamma"] else None
    gdtype = ttnn.bfloat16
    if cell == "int_rm_128x4096_gf32":
        g = g.to(torch.float32)
        gdtype = ttnn.float32
    tg = create_ttnn_input_tensor(g.reshape(1, 1, 1, W), device, dtype=gdtype, layout=RM) if g is not None else None
    mc = None
    if cfg["shard"]:
        mc = shard_config(cfg["shard"][0], cfg["shard"][1], ml, layout=RM, dtype=dtype, device=device)
    tx = create_ttnn_input_tensor(x, device, dtype=dtype, layout=RM, memory_layout=ml, memory_config=mc)
    kw = {} if ml == ML.INTERLEAVED else {"memory_config": tx.memory_config()}
    ck = ttnn.ComputeConfigDescriptor()
    ck.math_fidelity = ttnn.MathFidelity.HiFi4
    ck.fp32_dest_acc_en = cfg["fp32"]
    ck.math_approx_mode = False
    expected = pytorch_rms_norm(x, gamma=g, epsilon=1e-6)
    ss = tx.memory_config().shard_spec
    print(f"RNS_GEOM {cell}: {shape} {ml} shard={None if ss is None else (list(ss.shape), ss.grid.num_cores())}")

    ref_out = None
    for v in VSEL:
        defs = VARIANTS[v]
        kdir = REAL_KDIR if defs is None else (HERE / "graduation" / "kernels" if defs == "GRAD" else HERE / "kernels")
        monkeypatch.setattr(pd, "KERNEL_DIR", kdir)
        monkeypatch.setattr(pd, "PERF_ABLATE", () if defs in (None, "GRAD") else defs)
        out = rms_norm(tx, gamma=tg, epsilon=1e-6, compute_kernel_config=ck, **kw)
        print(f"RNS_ORDER {cell} {v}")
        res = ttnn.to_torch(out)
        check_output(out, expected, shape=shape, dtype=dtype, expected_layout=RM, tolerance=TOLERANCES[dtype])
        ttnn.deallocate(out)  # wide shards: input + two live outputs would not fit L1
        if ref_out is None:
            ref_out = res
        else:
            diff = (res.float() - ref_out.float()).abs().max().item()
            print(f"RNS_DIFF {cell} {v} max_abs_vs_{VSEL[0]}={diff}")
            assert torch.equal(res, ref_out), f"{cell} {v}: output differs from {VSEL[0]} (max abs {diff})"
