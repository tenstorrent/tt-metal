# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""No-dispatch census: which golden sharded gamma cells STREAM gamma under the op vs under compact cb_gamma.

    scripts/run_safe_pytest.sh ttnn/ttnn/operations/rms_norm/perf_experiments/compact_gamma/test_cg_enumerate.py \
        --import-mode=prepend -s

For every golden case (eval/golden_harness parametrize_cases + parametrize_loose_cases) that is sharded, has gamma and
is runnable (no skip / xfail mark), allocate the input + output shards (no data), and run both descriptors' blocking
solve with the same L1 budget the op would see. Prints (and writes runs/census.txt): op streamed? / cg streamed? / fits?
"""

import importlib
import importlib.util
import sys
from pathlib import Path

import ttnn

from eval.golden_harness import parametrize_cases, parametrize_loose_cases
from eval.golden_tests.rms_norm.feature_spec import INPUTS, INVALID, LOOSE_CASES, TARGET
from eval.sharding import auto_shard_config, shard_config

HERE = Path(__file__).parent
_spec = importlib.util.spec_from_file_location("cg_descriptor", HERE / "cg_descriptor.py")
cgd = importlib.util.module_from_spec(_spec)
sys.modules["cg_descriptor"] = cgd
_spec.loader.exec_module(cgd)
real_pd = importlib.import_module("ttnn.operations.rms_norm.rms_norm_program_descriptor")
pkg = importlib.import_module("ttnn.operations.rms_norm")


def _cases():
    out = []
    for p in parametrize_cases(TARGET, INPUTS, pkg.INPUT_TAGGERS, pkg.SUPPORTED, pkg.EXCLUSIONS, INVALID):
        if p.marks:
            continue
        inputs, axes = p.values
        out.append((inputs, axes, {}))
    for p in parametrize_loose_cases(LOOSE_CASES, pkg.INPUT_TAGGERS, pkg.SUPPORTED, pkg.EXCLUSIONS, INVALID):
        if p.marks:
            continue
        inputs, axes, extras = p.values
        out.append((inputs, axes, extras or {}))
    return out


def _plan(mod, xt, gdtype, fp32):
    geo = mod.derive_geometry(xt.shape)
    sz = mod._Sizes(xt.dtype, gdtype, xt.layout == ttnn.ROW_MAJOR_LAYOUT, True, fp32)
    try:
        blk = mod.select_sharded_blocking(geo, sz, xt, mod.cb_l1_budget(xt.device()))
    except RuntimeError:
        return "NOFIT"
    return f"G={blk.opts.gamma_chunk}" if blk.opts.gamma_chunk else "res"


def test_census(device):
    seen = {}
    lines = []
    for inputs, axes, extras in _cases():
        ml = axes["memory_layout"]
        if ml == ttnn.TensorMemoryLayout.INTERLEAVED or axes["gamma_mode"] != "gamma":
            continue
        shape = list(inputs[0])
        key = (
            tuple(shape),
            axes["dtype"],
            axes["layout"],
            ml,
            axes["fp32_dest_acc_en"],
            axes["gamma_dtype"],
            str(extras.get("shard_shape")),
        )
        if key in seen:
            continue
        xt = None
        try:
            if extras.get("shard_shape") is not None:
                mc = shard_config(
                    extras["shard_shape"],
                    extras["core_grid"],
                    ml,
                    layout=axes["layout"],
                    dtype=axes["dtype"],
                    device=device,
                )
            else:
                mc = auto_shard_config(shape, ml, layout=axes["layout"], dtype=axes["dtype"], device=device)
            xt = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), axes["dtype"], axes["layout"], device, mc)
            ot = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), axes["dtype"], axes["layout"], device, mc)
        except Exception:  # infeasible shard (harness skips it)
            if xt is not None:
                ttnn.deallocate(xt)  # an output OOM must not leak the input shard into later budgets
            seen[key] = "SKIP"
            continue
        fp32 = bool(axes["fp32_dest_acc_en"])
        a, b = _plan(real_pd, xt, axes["gamma_dtype"], fp32), _plan(cgd, xt, axes["gamma_dtype"], fp32)
        seen[key] = (a, b)
        ttnn.deallocate(ot)
        ttnn.deallocate(xt)
        if a != "res" or b != "res":
            lines.append(
                f"{shape} {axes['dtype']} {axes['layout']} {ml} fp32={fp32} g={axes['gamma_dtype']} "
                f"pin={extras.get('shard_shape')}: op={a} cg={b}"
            )
    n = sum(1 for v in seen.values() if v != "SKIP")
    op_str = sum(1 for v in seen.values() if v != "SKIP" and v[0] != "res")
    cg_str = sum(1 for v in seen.values() if v != "SKIP" and v[1] != "res")
    summary = f"census: {n} distinct sharded gamma plans; op streams/nofit {op_str}; compact streams/nofit {cg_str}"
    (HERE / "runs" / "census.txt").write_text(summary + "\n" + "\n".join(lines) + "\n")
    print(summary)
    print("\n".join(lines))
