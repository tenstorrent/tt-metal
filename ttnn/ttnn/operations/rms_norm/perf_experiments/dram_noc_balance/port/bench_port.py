# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""dram_noc_balance port onto HEAD: the real op (baseline, HEAD) vs the port clone (op_port/) under presets.

    PB_CELLS=int_prefill_7168,x1024 PB_VARIANTS=port,port_ks scripts/run_safe_pytest.sh --profile \
        ttnn/ttnn/operations/rms_norm/perf_experiments/dram_noc_balance/port/bench_port.py

Per cell (PB_CELLS order): one dispatch of the real op, then one per variant (PB_VARIANTS order; a `@N` suffix just
repeats a variant) -> the profiler GenericOp rows are [base, v1, v2, ...] per cell. Inputs are built exactly like
eval/golden_tests/rms_norm/helpers.run_rms_norm. Every variant is gated against torch with the cell's tolerance and
compared to the HEAD output (bit-identity + max abs diff): `PB <cell> <variant> ...`.
A variant name `base` re-runs the real op (HEAD) — e.g. PB_VARIANTS=port,base gives base/port/base.
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

HERE = Path(__file__).parent
ROOT = HERE.parents[6]
sys.path.insert(0, str(ROOT / "tests/ttnn/unit_tests/operations/rms_norm"))
from perf_p1_rms_norm import CELLS, _cell, _perf  # noqa: E402

CELLS = dict(CELLS)
CELLS.update(
    {
        "x1024": _perf((1, 1, 1024, 1024)),  # (1,1,1024,1024) perf cfg, k = 3, 96 Tensix cores
        "g_bf8": _cell((1, 1, 1024, 4096), gdtype=ttnn.bfloat8_b),  # bf8b TILE gamma, HiFi4, fp32 DEST
        "x32x1024": _perf((1, 1, 32, 1024)),  # k = 8, 8 Tensix cores, one gamma column group
        "own_160x11008": _perf((1, 1, 160, 11008)),  # k = 25 owner, inv multicast, row-spanning groups
        # domain sweep
        "x320x8192": _perf((1, 1, 320, 8192)),  # k = 13 owner: gamma column mcast AND inv mcast together
        "x2048": _perf((1, 1, 2048, 2048)),
        "x4096": _perf((1, 1, 4096, 4096)),
        "x512x256_f32": _cell((1, 1, 512, 256), dtype=ttnn.float32, gdtype=None),
        "x256x4096": _cell((1, 1, 256, 4096)),
        "na_w": _cell((1, 1, 256, 1000)),
        "na_rm": _cell((1, 1, 100, 1000), layout=ttnn.ROW_MAJOR_LAYOUT, glayout=ttnn.ROW_MAJOR_LAYOUT),
        "x8192x4096_rm": _cell(
            (1, 1, 8192, 4096), layout=ttnn.ROW_MAJOR_LAYOUT, fp32=False, glayout=ttnn.ROW_MAJOR_LAYOUT
        ),
        "x2x8x512x2048": _perf((2, 8, 512, 2048)),
        "t_64x256": _cell((1, 1, 64, 256)),
        "f32_256x256": _cell((1, 1, 256, 256), dtype=ttnn.float32, gdtype=None),
        "f32_4x1x128x256": _cell((4, 1, 128, 256), dtype=ttnn.float32, gdtype=None),
        "f32_2x1x128x512": _cell((2, 1, 128, 512), dtype=ttnn.float32, gdtype=None),
        "bf16_2x1x128x256": _cell((2, 1, 128, 256), gdtype=None),
        "c24_bf16_384x256": _cell((1, 1, 384, 256)),
        "c20_f32_320x256": _cell((1, 1, 320, 256), dtype=ttnn.float32, gdtype=None),
        "c24_f32_3x1x128x256": _cell((3, 1, 128, 256), dtype=ttnn.float32, gdtype=None),
        "c24_bf16_96x1024": _cell((1, 1, 96, 1024)),
        "c16_f32_128x512": _cell((1, 1, 128, 512), dtype=ttnn.float32, gdtype=None),
        "c16_f32_64x1024": _cell((1, 1, 64, 1024), dtype=ttnn.float32, gdtype=None),
        "c16_bf16_64x2048": _cell((1, 1, 64, 2048)),
        "bf8_256x4096": _cell((1, 1, 256, 4096), dtype=ttnn.bfloat8_b),
        "bf8_1024x1024": _cell((1, 1, 1024, 1024), dtype=ttnn.bfloat8_b),
        "bf8_64x8192": _cell((1, 1, 64, 8192), dtype=ttnn.bfloat8_b),
        "bf16_128x8192": _cell((1, 1, 128, 8192)),
        "bf16_64x8192": _cell((1, 1, 64, 8192)),
        "bf16_128x4096": _cell((1, 1, 128, 4096)),
        "bf16_256x512": _cell((1, 1, 256, 512)),
        "bf16_512x256": _cell((1, 1, 512, 256)),
        "bf16_1024x256": _cell((1, 1, 1024, 256)),
        "f32_1024x256": _cell((1, 1, 1024, 256), dtype=ttnn.float32, gdtype=None),
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


port, port_pd = _load("nb_port", "op_port")  # the graduation candidate
pvar, pvar_pd = _load("nb_pvar", "op_port_variants")  # gamma-sender variants explored on the way
VARIANTS = {  # name -> (module, descriptor module, attributes)
    "port": (port, port_pd, {"NOC_SWAP_COLUMNS": 3}),
    "port_off": (port, port_pd, {"NOC_SWAP_COLUMNS": 0}),
    "port2": (port, port_pd, {"NOC_SWAP_COLUMNS": 2}),
    "port4": (port, port_pd, {"NOC_SWAP_COLUMNS": 4}),
    # op_port_variants: split = per-writer-NoC gamma channels (k > 1 column groups swapped whole, sender row 0);
    # last = split with the swapped channels sending from the column's last core; ks = every gamma sender unswapped;
    # hyb = k > 1 gamma senders unswapped, k = 1 sender = first unswapped core (== op_port).
    "v_split": (pvar, pvar_pd, {"NOC_SWAP_KEEP_GAMMA_SENDERS": False}),
    "v_last": (pvar, pvar_pd, {"NOC_SWAP_GAMMA_SENDER_LAST": True}),
    "v_ks": (pvar, pvar_pd, {"NOC_SWAP_KEEP_GAMMA_SENDERS": True}),
    "v_hyb": (pvar, pvar_pd, {"NOC_SWAP_KEEP_GAMMA_SENDERS": "k>1"}),
    "v_own": (pvar, pvar_pd, {"NOC_SWAP_KEEP_GAMMA_SENDERS": "k>1", "NOC_SWAP_OWNER_SAME_ROW": True}),
}
_cells = [c for c in os.environ.get("PB_CELLS", "int_prefill_1024").split(",") if c]
_vars = [v for v in os.environ.get("PB_VARIANTS", "port").split(",") if v]


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
    axes = dict(CELLS[cell])
    xt, gt, cfg, kw, expected, tol = _build(device, axes)
    shape = axes["inputs"][0]
    base_out = base_rms_norm(xt, gamma=gt, epsilon=1e-6, compute_kernel_config=cfg, **kw)
    base_t = ttnn.to_torch(base_out).float()
    print(f"PB {cell} base pcc_vs_torch={_pcc(base_t, expected.float()):.6f}")
    check_output(base_out, expected, shape=shape, dtype=axes["dtype"], expected_layout=axes["layout"], tolerance=tol)
    ttnn.deallocate(base_out)
    for v in _vars:
        name = v.split("@")[0]
        if name == "base":
            out = base_rms_norm(xt, gamma=gt, epsilon=1e-6, compute_kernel_config=cfg, **kw)
        else:
            mod, pdm, attrs = VARIANTS[name]
            knobs = {"NOC_SWAP_COLUMNS": 3}
            if pdm is pvar_pd:
                knobs.update(
                    NOC_SWAP_KEEP_GAMMA_SENDERS=False, NOC_SWAP_GAMMA_SENDER_LAST=False, NOC_SWAP_OWNER_SAME_ROW=False
                )
            knobs.update(attrs)
            for attr, val in knobs.items():
                monkeypatch.setattr(pdm, attr, val)
            out = mod.rms_norm(xt, gamma=gt, epsilon=1e-6, compute_kernel_config=cfg, **kw)
        t = ttnn.to_torch(out).float()
        ident = bool((t == base_t).all()) or bool(((t == base_t) | (t.isnan() & base_t.isnan())).all())
        print(
            f"PB {cell} {v} pcc_vs_torch={_pcc(t, expected.float()):.6f} bit_identical_vs_head={ident} "
            f"maxabs_vs_head={float((t - base_t).abs().max()):.4g}"
        )
        check_output(out, expected, shape=shape, dtype=axes["dtype"], expected_layout=axes["layout"], tolerance=tol)
        assert ident, f"{cell} {v}: output differs from HEAD"
        ttnn.deallocate(out)
