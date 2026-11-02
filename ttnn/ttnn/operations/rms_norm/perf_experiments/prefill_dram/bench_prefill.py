# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""prefill_dram bench: the real op (HEAD, baseline) vs the op clone (op_clone/) under knob presets.

    PD_CELLS=int_prefill_7168 PD_VARIANTS=clone,stag_r ./run.sh <tag>

Per cell (PD_CELLS order): one dispatch of the real op, then one per variant (PD_VARIANTS order; `@N` suffix just
repeats a variant) -> the profiler GenericOp rows are [base, v1, v2, ...] per cell. Inputs are built exactly like
eval/golden_tests/rms_norm/helpers.run_rms_norm. Every variant is gated against torch with the cell's tolerance and
must be bit-identical to the HEAD output (data movement only): `PD <cell> <variant> ...`.
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest
import ttnn

from eval.golden_tests.rms_norm.helpers import (
    TOLERANCES,
    _TORCH_DTYPE,
    check_output,
    create_ttnn_input_tensor,
    pytorch_rms_norm,
)
from eval.sharding import shard_config
from ttnn.operations.rms_norm import rms_norm as base_rms_norm
from ttnn.operations.rms_norm import rms_norm_program_descriptor as base_pd

HERE = Path(__file__).parent
ROOT = HERE.parents[5]
sys.path.insert(0, str(ROOT / "tests/ttnn/unit_tests/operations/rms_norm"))
from perf_p2_rms_norm import CELLS, _cell, _perf  # noqa: E402

CELLS = dict(CELLS)
CELLS.update(
    {
        # golden interleaved perf cells (perf cfg: bf16, HiFi2, fp32 DEST off)
        "int_7168x1024": _perf((1, 1, 7168, 1024)),
        "int_7168x2048": _perf((1, 1, 7168, 2048)),
        "int_8192x2304": _perf((1, 1, 8192, 2304)),
        "int_640x7168": _perf((1, 1, 640, 7168)),
        "int_32x7168": _perf((1, 1, 32, 7168)),
        # at-risk: interleaved decode / owner combine (_WIDE loose case: HiFi4 fp32 DEST)
        "int_decode_32768": _cell((1, 1, 32, 32768)),
        # domain sweep
        "int_4096x4096": _perf((1, 1, 4096, 4096)),
        "int_2048x8192": _perf((1, 1, 2048, 8192)),
        "int_16384x2048": _perf((1, 1, 16384, 2048)),
        "int_8192x7168_f32": _cell((1, 1, 8192, 7168), dtype=ttnn.float32, gdtype=ttnn.float32),
        "int_8192x7168_bf8": _cell((1, 1, 8192, 7168), dtype=ttnn.bfloat8_b, gdtype=ttnn.bfloat8_b),
        "int_3x1x736x5119": _cell((3, 1, 736, 5119), fp32=False),
    }
)


def _load(name, sub):
    spec = importlib.util.spec_from_file_location(
        name, HERE / sub / "__init__.py", submodule_search_locations=[str(HERE / sub)]
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod, sys.modules[f"{name}.rms_norm_program_descriptor"]


clone, clone_pd = _load("pd_clone", "op_clone")
# name -> attributes set on the clone's descriptor module (DEFAULTS first). "spacer" (bytes) allocates a DRAM spacer
# between the input and the op's output (address-alignment experiment; not an op change).
DEFAULTS = {
    "STAGGER_R": 0,
    "STAGGER_W": 0,
    "STAGGER_STEP": 1,
    "OVERLAP_BLOCKS": 1,
    "OUT_UNIT_CAP": 16,
    "OUT_DEPTH": 2,
    "NOC_SWAP_COLUMNS": 3,
    "NOC_SWAP_ROWS": 0,
    "NOC_SWAP_PRED": None,
    "WRITE_SPLIT": 0,
    "PERF_ABLATE": (),
    "ROW_WEIGHT": None,
    "ROW_MIN": 1,
    "ROW_MAX": 3,
    "BANK_AFFINE": None,
    "BANK_AFFINE_PRINT": False,
}
VARIANTS = {
    "clone": {},
    # ablations (probe only; output wrong by design -> correctness gate skipped)
    "abl_rd": {"PERF_ABLATE": ("RMS_ABLATE_RD",)},  # x reads dropped: write stream only
    "abl_wr": {"PERF_ABLATE": ("RMS_ABLATE_WR",)},  # out writes dropped: read stream only
    "abl_dm": {"PERF_ABLATE": ("RMS_ABLATE_DM",)},
    # bank-affinity probes (wrong output): compute stubbed, as the copy-only reference abl_cmp
    "aff_near": {"BANK_AFFINE": "near", "PERF_ABLATE": ("CKL_ELTWISE_CHAIN_SKIP_COMPUTE",), "BANK_AFFINE_PRINT": True},
    "aff_rr": {"BANK_AFFINE": "rr", "PERF_ABLATE": ("CKL_ELTWISE_CHAIN_SKIP_COMPUTE",)},
    "aff_near_rd": {"BANK_AFFINE": "near", "PERF_ABLATE": ("CKL_ELTWISE_CHAIN_SKIP_COMPUTE", "RMS_ABLATE_WR")},
    "aff_near_wr": {"BANK_AFFINE": "near", "PERF_ABLATE": ("CKL_ELTWISE_CHAIN_SKIP_COMPUTE", "RMS_ABLATE_RD")},
    "abl_cmp": {"PERF_ABLATE": ("CKL_ELTWISE_CHAIN_SKIP_COMPUTE",)},
    "abl_cmp_wr": {"PERF_ABLATE": ("CKL_ELTWISE_CHAIN_SKIP_COMPUTE", "RMS_ABLATE_WR")},
    "abl_cmp_rd": {"PERF_ABLATE": ("CKL_ELTWISE_CHAIN_SKIP_COMPUTE", "RMS_ABLATE_RD")},
    "stag_r": {"STAGGER_R": 1},
    "stag_rw": {"STAGGER_R": 1, "STAGGER_W": 1},
    "stag_w": {"STAGGER_W": 1},
    "stag_r29": {"STAGGER_R": 1, "STAGGER_STEP": 29},
    "ovl2": {"OVERLAP_BLOCKS": 2},
    "stag_ovl2": {"OVERLAP_BLOCKS": 2, "STAGGER_R": 1},
    "stag_r8": {"STAGGER_R": 1, "STAGGER_STEP": 28},  # rot = 28 i: start bank i % 8 shifted by whole bank sweeps
    "out32": {"OUT_UNIT_CAP": 32},
    "outd4": {"OUT_DEPTH": 4},
    "swap0": {"NOC_SWAP_COLUMNS": 0},
    "ws2": {"WRITE_SPLIT": 2},
    "ws3": {"WRITE_SPLIT": 3},
    "ws4": {"WRITE_SPLIT": 4},
    "ws8": {"WRITE_SPLIT": 8},
    "rows2": {"NOC_SWAP_ROWS": 2},
    "rows3": {"NOC_SWAP_ROWS": 3},
    "rows4": {"NOC_SWAP_ROWS": 4},
    "rows5": {"NOC_SWAP_ROWS": 5},
    "rows3_nocol": {"NOC_SWAP_ROWS": 3, "NOC_SWAP_COLUMNS": 0},
}
for s in (1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32, 64):  # spacer in KB per bank (x 8 banks)
    VARIANTS[f"sp{s}k"] = {"spacer": s * 1024 * 8}
# Measured base rd_read per Tensix core (us, int_prefill_7168, run r1; logical rows 0..9 x cols 0..12; per tile-row:
# the 4 one-row cores at logical (9..12, 9) are doubled).
_RD_MAP = [
    [50, 91, 118, 513, 506, 495, 486, 455, 452, 440, 431, 414, 391],
    [55, 99, 125, 525, 518, 508, 502, 474, 475, 471, 456, 424, 395],
    [56, 99, 127, 500, 497, 476, 465, 416, 421, 411, 393, 373, 352],
    [63, 106, 136, 452, 428, 405, 389, 356, 365, 357, 342, 318, 290],
    [66, 106, 137, 394, 382, 362, 351, 313, 324, 319, 305, 270, 239],
    [66, 95, 133, 343, 334, 311, 292, 250, 252, 240, 227, 205, 187],
    [82, 112, 143, 277, 259, 238, 227, 193, 194, 187, 172, 155, 130],
    [79, 110, 143, 222, 208, 184, 172, 144, 145, 138, 121, 109, 96],
    [71, 101, 140, 170, 176, 153, 138, 115, 117, 105, 92, 80, 67],
    [42, 79, 87, 50, 57, 59, 56, 57, 51, 56, 50, 42, 38],
]


def _w_inv(x, y):
    return 1.0 / _RD_MAP[y][x]


def _w_sqrt(x, y):
    return _RD_MAP[y][x] ** -0.5


VARIANTS.update(
    {
        "bal_inv3": {"ROW_WEIGHT": _w_inv, "ROW_MAX": 3},
        "bal_sqrt3": {"ROW_WEIGHT": _w_sqrt, "ROW_MAX": 3},
        "bal_inv4": {"ROW_WEIGHT": _w_inv, "ROW_MAX": 4},
        "bal_sqrt3_ovl": {"ROW_WEIGHT": _w_sqrt, "ROW_MAX": 3, "OVERLAP_BLOCKS": 3},
    }
)
_cells = [c for c in os.environ.get("PD_CELLS", "int_prefill_1024").split(",") if c]
_vars = [v for v in os.environ.get("PD_VARIANTS", "clone").split(",") if v]


def _build(device, axes):
    import torch  # function-local: the ttnn tree forbids global torch imports

    shape = axes["inputs"][0]
    dtype, layout, ml = axes["dtype"], axes["layout"], axes["memory_layout"]
    extras = axes["extras"]
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32).to(_TORCH_DTYPE[dtype])
    g = gt = None
    if axes["gamma_mode"] == "gamma":
        W = shape[-1]
        g = torch.randn(W, dtype=torch.float32).to(_TORCH_DTYPE[axes["gamma_dtype"]])
        gt = create_ttnn_input_tensor(
            g.reshape(1, 1, 1, W), device, dtype=axes["gamma_dtype"], layout=axes["gamma_layout"]
        )
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = extras.get("math_fidelity", ttnn.MathFidelity.HiFi4)
    cfg.fp32_dest_acc_en = axes["fp32_dest_acc_en"]
    cfg.math_approx_mode = False
    mc = None
    if extras.get("shard_shape") is not None:
        mc = shard_config(extras["shard_shape"], extras["core_grid"], ml, layout=layout, dtype=dtype, device=device)
    xt = create_ttnn_input_tensor(x, device, dtype=dtype, layout=layout, memory_layout=ml, memory_config=mc)
    kw = {}
    if ml != ttnn.TensorMemoryLayout.INTERLEAVED:
        kw["memory_config"] = xt.memory_config()
    expected = pytorch_rms_norm(x, gamma=g, epsilon=1e-6)
    tol = TOLERANCES[dtype]
    if extras.get("pcc_threshold") is not None:
        tol = (extras["pcc_threshold"], tol[1])
    return xt, gt, cfg, kw, expected, tol


def _pcc(a, b):
    import torch

    return float(torch.corrcoef(torch.stack([a.flatten().double(), b.flatten().double()]))[0, 1])


@pytest.mark.parametrize("cell", _cells)
def test_bench(device, cell, monkeypatch):
    import torch

    axes = dict(CELLS[cell])
    if os.environ.get("PD_NOGAMMA") == "1":
        axes.update(gamma_mode="no_gamma", gamma_dtype="none", gamma_layout="none")
    xt, gt, cfg, kw, expected, tol = _build(device, axes)
    shape = axes["inputs"][0]
    base_out = base_rms_norm(xt, gamma=gt, epsilon=1e-6, compute_kernel_config=cfg, **kw)
    base_t = ttnn.to_torch(base_out).float()
    print(
        f"PD {cell} base pcc_vs_torch={_pcc(base_t, expected.float()):.6f} "
        f"x_addr={xt.buffer_address():#x} out_addr={base_out.buffer_address():#x}"
    )
    check_output(base_out, expected, shape=shape, dtype=axes["dtype"], expected_layout=axes["layout"], tolerance=tol)
    ttnn.deallocate(base_out)
    for v in _vars:
        name = v.split("@")[0]
        attrs = dict(DEFAULTS)
        attrs.update(VARIANTS.get(name, {}))
        spacer_bytes = attrs.pop("spacer", 0)
        for attr, val in attrs.items():
            monkeypatch.setattr(clone_pd, attr, val)
        spacer = None
        if spacer_bytes:
            n = spacer_bytes // 2  # bf16 datums, TILE interleaved pages of 2 KB
            spacer = ttnn.from_torch(
                torch.zeros((1, 1, 32, n // 32), dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        if name == "base":
            out = base_rms_norm(xt, gamma=gt, epsilon=1e-6, compute_kernel_config=cfg, **kw)
        else:
            out = clone.rms_norm(xt, gamma=gt, epsilon=1e-6, compute_kernel_config=cfg, **kw)
        t = ttnn.to_torch(out).float()
        ident = bool((t == base_t).all()) or bool(((t == base_t) | (t.isnan() & base_t.isnan())).all())
        print(
            f"PD {cell} {v} pcc_vs_torch={_pcc(t, expected.float()):.6f} bit_identical_vs_head={ident} "
            f"maxabs_vs_head={float((t - base_t).abs().max()):.4g} out_addr={out.buffer_address():#x}"
        )
        if not attrs["PERF_ABLATE"]:
            check_output(out, expected, shape=shape, dtype=axes["dtype"], expected_layout=axes["layout"], tolerance=tol)
            assert ident, f"{cell} {v}: output differs from HEAD"
        ttnn.deallocate(out)
        if spacer is not None:
            ttnn.deallocate(spacer)
