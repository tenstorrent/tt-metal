# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-side plan dump (no kernel runs): for every [perf] focus case, the blocking and kw-fold decision of the base op
(ttnn.operations.conv2d_3x3) and of this experiment's copy. ttnn.generic_op is stubbed, so only the descriptor builds.

    scripts/run_safe_pytest.sh --run-all --import-mode=prepend -s \
        ttnn/ttnn/operations/conv2d_3x3/perf_experiments/kwf_ngroup_outer/test_kno_plan.py
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kno_copy import copy_modules  # noqa: E402


def _cases():
    from eval.golden_tests.conv2d_3x3.feature_spec import LOOSE_CASES

    cases = [c for c in LOOSE_CASES if "[perf]" in c["extras"]["layer"]]
    ids = [c["extras"]["layer"].split(" ")[0].replace(".", "_") for c in cases]
    return cases, ids


_CASES, _IDS = _cases()


@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_kno_plan(case, device, monkeypatch):
    import torch
    import ttnn
    import importlib

    base_op = importlib.import_module("ttnn.operations.conv2d_3x3.conv2d_3x3")
    base_pd = importlib.import_module("ttnn.operations.conv2d_3x3.conv2d_3x3_program_descriptor")
    from eval.golden_tests.conv2d_3x3 import helpers

    op_mod, pd_mod = copy_modules()
    (H, W, c_in), c_out, padding, stride, dilation = case["inputs"]
    extras = case["extras"]
    dtype, weight_dtype, layout = case["dtype"], case["weight_dtype"], case["layout"]
    mc = helpers.build_input_memory_config(
        (1, 1, H * W, c_in), case["memory_layout"], layout=layout, dtype=dtype, device=device, extras=extras
    )
    kwargs = dict(
        stride=stride,
        padding=padding,
        dilation=dilation,
        input_dtype=dtype,
        input_layout=layout,
        weight_dtype=weight_dtype,
        input_memory_config=mc,
        device=device,
    )
    has_bias = case["bias_mode"] == "with_bias"
    tw = helpers.make_weight(c_out, c_in, dtype=torch.bfloat16)
    tb = helpers.make_bias(c_out, dtype=torch.bfloat16) if has_bias else None
    tx = helpers.make_activation(H, W, c_in, dtype=torch.bfloat16)
    cfg = helpers.make_compute_kernel_config(case["fp32_dest_acc_en"], extras.get("math_fidelity"))
    monkeypatch.setattr(ttnn, "generic_op", lambda io, desc: io[-1])
    line = [extras["layer"].split(" ")[0]]
    for name, op, pd in (("base", base_op, base_pd), ("copy", op_mod, pd_mod)):
        seen = {}
        real = pd.create_program_descriptor

        def wrapped(*a, **k):
            seen.update(k)
            seen["inputs"] = a
            return real(*a, **k)

        monkeypatch.setattr(op, "create_program_descriptor", wrapped)
        x = ttnn.from_torch(tx, dtype=dtype, layout=layout, device=device, memory_config=mc)
        op.conv2d_3x3(
            x,
            base_op.prepare_conv2d_weights(tw, **kwargs),
            input_height=H,
            input_width=W,
            bias=base_op.prepare_conv2d_bias(tb, **kwargs) if has_bias else None,
            padding=padding,
            stride=stride,
            dilation=dilation,
            activation=None if case["activation"] == "none" else case["activation"],
            compute_kernel_config=cfg,
        )
        g, b = seen["geometry"], seen["blocking"]
        inp = seen["inputs"][0]
        resident = (
            inp.memory_config().shard_spec.shape[0] // 32 * g.c_in_tiles * g.in0_tile_bytes
            + b.core_m_tiles * g.out_n_tiles * g.out_tile_bytes
            if g.sharded_output and g.tile_input
            else 0
        )
        budget = pd.l1_cb_budget_bytes(resident)
        info = f"{name}: core_m {b.core_m_tiles} bm {b.block_m_tiles} mb {b.max_m_blocks_per_core} ct {g.c_in_tiles} n {g.tensor_n_tiles}"
        if g.tile_input and g.sharded_output and not g.k_split and has_bias:
            kwf = pd.kw_fold_mode(g, b, has_bias, budget, cfg.math_fidelity)
            if hasattr(pd, "kw_fold_plan"):
                kb = pd.kw_fold_plan(g, b, has_bias, budget, cfg.math_fidelity)
                bms = [d for d in range(b.core_m_tiles, 0, -1) if b.core_m_tiles % d == 0]
                fits = [(bm, pd._kw_fold_cb_bytes(g, pd._kw_fold_blocking(g, b, bm)) // 1024) for bm in bms]
                info += f" fits {fits[:3]}"
                if kb is not None:
                    info += (
                        f" -> kwf bm {kb.block_m_tiles} mb {kb.max_m_blocks_per_core} ub {kb.untilize_batch_rows}"
                        f" slab {kb.slab_tile_rows} sub {kb.block_k_tiles}x{kb.block_n_tiles} nkb {kb.num_k_blocks}"
                        f" wd {kb.weight_depth} bytes {pd._kw_fold_cb_bytes(g, kb) // 1024}K"
                    )
            else:
                kb = pd._kw_fold_blocking(g, b)
                info += f" kwf_bytes {pd._kw_fold_cb_bytes(g, kb) // 1024}K ub {kb.untilize_batch_rows} slab {kb.slab_tile_rows} ring {b.ring} center {b.center_mode}"
            info += f" budget {budget // 1024}K kwf {kwf}"
        line.append(info)
    print("KNO_PLAN", " | ".join(line))
