# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""dram_noc_balance bake-off: the real op (baseline) vs the experiment clone under VARIANT presets.

    NB_CELLS=int_prefill_7168 NB_VARIANTS=clone,top3,ws4 scripts/run_safe_pytest.sh --profile \
        ttnn/ttnn/operations/rms_norm/perf_experiments/dram_noc_balance/bench_noc_balance.py

Per cell (in NB_CELLS order): one dispatch of the real op, then one per variant (NB_VARIANTS order) -> the profiler
GenericOp rows are [base, v1, v2, ...] per cell. Inputs are built exactly like eval/golden_tests/rms_norm/helpers.
run_rms_norm (same seed, gamma, compute config). Every variant is gated against torch with the cell's tolerance and
compared to the baseline output (PCC + max abs diff, printed as `NB <cell> <variant> ...`).
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
sys.path.insert(0, str(HERE.parent / "tools"))
sys.path.insert(
    0,
    "/localdev/adamjanovic/2026_11_01/1349_adamjanovic_ab-tilize-run-scoreboard-catalog4/clones/rms_norm_run1/tt-metal/tests/ttnn/unit_tests/operations/rms_norm",
)
from perf_p1_rms_norm import CELLS, _cell, _perf  # noqa: E402

CELLS = dict(CELLS)
CELLS.update(
    {  # domain-sweep cells (not in the scoreboard): tiles per Tensix core / active cores / total bytes
        "x512x256_f32": _cell((1, 1, 512, 256), dtype=ttnn.float32, gdtype=None),
        "x256x4096": _cell((1, 1, 256, 4096)),
        "x1024": _perf((1, 1, 1024, 1024)),
        "x2048": _perf((1, 1, 2048, 2048)),
        "x4096": _perf((1, 1, 4096, 4096)),
        "x32x1024": _perf((1, 1, 32, 1024)),
        "na_w": _cell((1, 1, 256, 1000)),
        "t_32x64": _cell((1, 1, 32, 64)),
        "t_64x256": _cell((1, 1, 64, 256)),
        "t_rm_32x64": _cell((1, 1, 32, 64), layout=ttnn.ROW_MAJOR_LAYOUT, glayout=ttnn.ROW_MAJOR_LAYOUT),
        "t_96x2048_bf8": _cell((1, 1, 96, 2048), dtype=ttnn.bfloat8_b),
        "na_h": _cell((1, 1, 100, 4096)),
        "na_rm": _cell((1, 1, 100, 1000), layout=ttnn.ROW_MAJOR_LAYOUT, glayout=ttnn.ROW_MAJOR_LAYOUT),
        "g_bf8": _cell((1, 1, 1024, 4096), gdtype=ttnn.bfloat8_b),
        "ng_2048x4096": _perf((1, 1, 2048, 4096)),
        "own_160x11008": _perf((1, 1, 160, 11008)),
        "x2x8x512x2048": _perf((2, 8, 512, 2048)),
        "x8192x4096_rm": _cell(
            (1, 1, 8192, 4096), layout=ttnn.ROW_MAJOR_LAYOUT, fp32=False, glayout=ttnn.ROW_MAJOR_LAYOUT
        ),
    }
)


def _load_clone(name="nb_clone", sub="op_clone"):
    spec = importlib.util.spec_from_file_location(
        name, HERE / sub / "__init__.py", submodule_search_locations=[str(HERE / sub)]
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod, sys.modules[f"{name}.rms_norm_program_descriptor"]


clone, clone_pd = _load_clone()
grad, grad_pd = _load_clone("nb_grad", "op_grad")  # the exact graduation diff applied to a copy of the real op
GRAD = {  # name -> (NOC_SWAP_COLUMNS, READ_WINDOW_TILES)
    "grad": (3, 32),
    "grad_norw": (3, 0),
    "grad_off": (0, 0),
}

EAST3 = ("cols", [10, 11, 12])
VARIANTS = {
    "clone": {},
    "clone2": {},
    "f3b": {"swap": ("f", 3)},
    "f2": {"swap": ("f", 2)},
    "f3": {"swap": ("f", 3)},
    "f4": {"swap": ("f", 4)},
    "e2": {"swap": ("e", 2)},
    "e4": {"swap": ("e", 4)},
    "east2": {"swap": ("cols", [11, 12])},
    "east3": {"swap": EAST3},
    "top2": {"swap": ("rows", [0, 1])},
    "top3": {"swap": ("rows", [0, 1, 2])},
    "f3top2": {"swap": ("union", [("f", 3), ("rows", [0, 1])])},
    "f3top1": {"swap": ("union", [("f", 3), ("rows", [0])])},
    "ks": {"keep_senders": True},
    "rw16": {"read_window": 16},
    "rw32": {"read_window": 32},
    "rw64": {"read_window": 64},
    "wbar": {"write_barrier": True},
    "ws4": {"write_split": 4},
    "ws2": {"write_split": 2},
    "ws8": {"write_split": 8},
    "ws2east": {"write_split": 2, "write_split_cores": EAST3},
    "ws4east": {"write_split": 4, "write_split_cores": EAST3},
}
_cells = [c for c in os.environ.get("NB_CELLS", "int_prefill_7168").split(",") if c]
_vars = [v for v in os.environ.get("NB_VARIANTS", "clone").split(",") if v]


def _variant(name):
    cfg = {}
    for part in name.split("@")[0].split("+"):
        cfg.update(VARIANTS[part])
    return cfg


def _pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-30))


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


@pytest.mark.parametrize("cell", _cells)
def test_bench(device, cell, monkeypatch):
    import torch  # function-local: the ttnn tree forbids global torch imports

    axes = dict(CELLS[cell])
    if os.environ.get("NB_NOGAMMA") == "1":
        axes.update(gamma_mode="no_gamma", gamma_dtype="none", gamma_layout="none")
    xt, gt, cfg, kw, expected, tol = _build(device, axes)
    shape = axes["inputs"][0]
    nearest = ttnn.device.get_optimal_dram_bank_to_logical_worker_assignment(device, ttnn.NOC.NOC_0)
    print("NB dram-bank nearest logical workers:", [(c.x, c.y) for c in nearest])
    base_out = base_rms_norm(xt, gamma=gt, epsilon=1e-6, compute_kernel_config=cfg, **kw)
    base_t = ttnn.to_torch(base_out).float()
    print(f"NB {cell} base pcc_vs_torch={_pcc(base_t, expected.float()):.6f}")
    check_output(base_out, expected, shape=shape, dtype=axes["dtype"], expected_layout=axes["layout"], tolerance=tol)
    ttnn.deallocate(base_out)  # every variant's output lands at the same DRAM address as the baseline's
    for v in _vars:
        g = GRAD.get(v.split("@")[0])
        if g is not None:
            monkeypatch.setattr(grad_pd, "NOC_SWAP_COLUMNS", g[0])
            monkeypatch.setattr(grad_pd, "READ_WINDOW_TILES", g[1])
            out = grad.rms_norm(xt, gamma=gt, epsilon=1e-6, compute_kernel_config=cfg, **kw)
        else:
            monkeypatch.setattr(clone_pd, "VARIANT", _variant(v))
            out = clone.rms_norm(xt, gamma=gt, epsilon=1e-6, compute_kernel_config=cfg, **kw)
        t = ttnn.to_torch(out).float()
        print(
            f"NB {cell} {v} pcc_vs_torch={_pcc(t, expected.float()):.6f} pcc_vs_base={_pcc(t, base_t):.6f} "
            f"maxabs_vs_base={float((t - base_t).abs().max()):.4g}"
        )
        check_output(out, expected, shape=shape, dtype=axes["dtype"], expected_layout=axes["layout"], tolerance=tol)
        ttnn.deallocate(out)
