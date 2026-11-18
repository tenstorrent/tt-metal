# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bake-off: per-subblock pack-state reconfig vs. hoisted (per K-block) in the fp32 L1-acc K walk.

    KW_BENCH=1 scripts/run_safe_pytest.sh --run-all \
        ttnn/ttnn/operations/conv2d_3x3/perf_experiments/l1acc_pack_hoist/test_l1acc_pack_hoist.py::test_focus

Env knobs: KW_BENCH=1 (required), KW_SHAPES (comma list of FOCUS/DOMAIN names, or "domain"),
KW_VARIANTS (comma list, baseline first), KW_MITERS. hoist_rowmop is expected to FAIL the exactness gate on
bfloat8_b Out/OutWithRelu targets (a multi-tile pack MOP does not close BFP tiles).
Correctness is the only pass/fail: every exact variant must be bit-identical to the baseline, and the
baseline must match torch (PCC). Perf is printed, never asserted.
"""

import os

# `import ttnn` imports every module of every package under ttnn/operations (pkgutil.walk_packages), so this
# module also loads in unrelated processes: only touch the environment when the bench was asked for.
if os.environ.get("KW_BENCH") == "1":
    os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
    os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
    os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
    os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")

import json
import socket
from dataclasses import replace
from pathlib import Path

import pytest
from loguru import logger

import ttnn

from .kw_bench import (
    FOCUS,
    VARIANTS,
    Shape,
    TILE,
    create_program_descriptor,
    effective_shape,
    shard_cfg,
)

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"


def _pcc(ref, out):
    import torch

    return float(torch.corrcoef(torch.stack([ref.flatten(), out.flatten()]))[0, 1])


RESULTS = Path(__file__).parent / "results.jsonl"

# Domain sweep: other matmul_block instantiations the hoist would cover (targets, K-block counts, owner, h>1).
DOMAIN = {
    "res5_bk1": replace(FOCUS["res5"], name="res5_bk1", bk=1, nkb=18),
    "res5_nkb1": replace(FOCUS["res5"], name="res5_nkb1", nkb=1),
    "res5_nkb2": replace(FOCUS["res5"], name="res5_nkb2", nkb=2),
    "res5_nkb3": replace(FOCUS["res5"], name="res5_nkb3", nkb=3, kb_store=3),
    "sem_res2_out": replace(FOCUS["sem_res2"], name="sem_res2_out", target="Out"),
    "sem_res2_relu": replace(FOCUS["sem_res2"], name="sem_res2_relu", target="OutWithRelu"),
    "sem_res2_relu_nkb2": replace(FOCUS["sem_res2"], name="sem_res2_relu_nkb2", target="OutWithRelu", nkb=2),
    "sem_res2_out_nkb1": replace(FOCUS["sem_res2"], name="sem_res2_out_nkb1", target="Out", nkb=1),
    "sem_res2_sb2x2": replace(FOCUS["sem_res2"], name="sem_res2_sb2x2", bm=6, sbh=2, sbw=2),
    "sem_res2_sb2x2_out": replace(FOCUS["sem_res2"], name="sem_res2_sb2x2_out", bm=6, sbh=2, sbw=2, target="Out"),
    "sem_res2_out_bf16": replace(FOCUS["sem_res2"], name="sem_res2_out_bf16", target="Out", out_dtype="bfloat16"),
    "sem_res2_relu_fp32": replace(
        FOCUS["sem_res2"], name="sem_res2_relu_fp32", target="OutWithRelu", out_dtype="float32"
    ),
}


def _shapes():
    names = os.environ.get("KW_SHAPES", "res5,sem_res2").split(",")
    out = []
    for n in names:
        if n == "domain":
            out += list(DOMAIN.values())
        else:
            out.append(FOCUS.get(n) or DOMAIN[n])
    m_iters = os.environ.get("KW_MITERS")
    return [replace(s, m_iters=int(m_iters)) for s in out] if m_iters else out


def _variants():
    names = os.environ.get("KW_VARIANTS", "baseline,hoist,hoist_rowmop").split(",")
    assert names[0] == "baseline", "baseline must run first (it is the exactness reference)"
    return [VARIANTS[n] for n in names]


def _read_kernel_ns(device):
    ttnn.ReadDeviceProfiler(device)
    total, found = 0.0, False
    for programs in (ttnn.get_latest_programs_perf_data() or {}).values():
        for program in programs:
            entry = (getattr(program, "program_analyses_results", None) or {}).get(_DURATION_KEY)
            if entry is not None:
                total += float(entry.duration)
                found = True
    return total if found else None


def _inputs(device, s: Shape):
    import torch

    torch.manual_seed(1234)
    a = torch.randn(s.kb_store * s.bm * TILE, s.bk * TILE) * 0.5
    b = torch.randn(s.kb_store * s.bk * TILE, s.bn * TILE) * 0.5
    tt_a = ttnn.from_torch(
        a,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=shard_cfg(s.kb_store * s.bm, s.bk),
    )
    tt_b = ttnn.from_torch(
        b,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=shard_cfg(s.kb_store * s.bk, s.bn),
    )
    return tt_a, tt_b, ttnn.to_torch(tt_a).double(), ttnn.to_torch(tt_b).double()


def _reference(s: Shape, a, b):
    """Result of the LAST M-iteration: K-block j of it uses stored block ((m_iters-1)*nkb + j) % kb_store."""
    import torch

    acc = torch.zeros(s.bm * TILE, s.bn * TILE, dtype=torch.float64)
    for j in range(s.nkb):
        idx = ((s.m_iters - 1) * s.nkb + j) % s.kb_store
        acc += a[idx * s.bm * TILE : (idx + 1) * s.bm * TILE] @ b[idx * s.bk * TILE : (idx + 1) * s.bk * TILE]
    if s.target == "OutWithRelu":
        acc = torch.relu(acc)
    return acc


def _run(device, s: Shape, variant, tt_a, tt_b):
    es = effective_shape(s, variant)
    partials = ttnn.allocate_tensor_on_device(
        ttnn.Shape([es.bm * TILE, es.bn * TILE]), ttnn.float32, ttnn.TILE_LAYOUT, device, shard_cfg(es.bm, es.bn)
    )
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([es.bm * TILE, es.bn * TILE]),
        getattr(ttnn, es.out_dtype),
        ttnn.TILE_LAYOUT,
        device,
        shard_cfg(es.bm, es.bn),
    )
    desc = create_program_descriptor(tt_a, tt_b, partials, out, s, variant)
    io = [tt_a, tt_b, partials, out]
    ttnn.generic_op(io, desc)  # compile + correctness run
    result = ttnn.to_torch(partials if s.target == "Interm" else out).double()
    _read_kernel_ns(device)  # flush
    ttnn.generic_op(io, desc)
    ns = _read_kernel_ns(device)
    partials.deallocate()
    out.deallocate()
    return result, ns


@pytest.mark.skipif(os.environ.get("KW_BENCH") != "1", reason="bench: run with KW_BENCH=1")
def test_focus(device):
    import torch

    box, arch = socket.gethostname(), str(device.arch())
    lines = [f"l1acc_pack_hoist  box={box} arch={arch} cores=1 LoFi fp32_dest half-sync bf8 in0/in1 fp32 partials"]
    failures = []
    for s in _shapes():
        tt_a, tt_b, a, b = _inputs(device, s)
        ref = _reference(s, a, b)
        base_out, base_ns = None, None
        subblocks = (s.bm // s.sbh) * (s.bn // s.sbw) * s.nkb * s.m_iters
        lines.append(
            f"  {s.name}: bm={s.bm} bk={s.bk} bn={s.bn} nkb={s.nkb} sb={s.sbh}x{s.sbw} owns={int(s.caller_owns)} "
            f"target={s.target} out={s.out_dtype} m_iters={s.m_iters} ({subblocks} subblock packs, {s.bm*s.bn*s.nkb*s.m_iters} tile packs)"
        )
        for v in _variants():
            out, ns = _run(device, s, v, tt_a, tt_b)
            pcc = _pcc(ref, out)
            if v.name == "baseline":
                base_out, base_ns = out, ns
                exact = "ref"
                if pcc < 0.99:
                    failures.append(f"{s.name}/baseline pcc {pcc:.5f}")
            else:
                same = torch.equal(out, base_out)
                exact = "bit-exact" if same else f"DIFF(max {float((out - base_out).abs().max()):.3e})"
                if v.exact and not same:
                    failures.append(f"{s.name}/{v.name} not bit-identical to baseline")
            speed = f"{base_ns / ns:5.3f}x" if (ns and base_ns) else "-"
            per_sb = ns / ((s.bm // s.sbh) * (s.bn // effective_shape(s, v).sbw) * s.nkb * s.m_iters)
            lines.append(
                f"    {v.name:<24} {ns:>11.0f} ns  {speed}  pcc={pcc:.6f}  {exact:<12} "
                f"({per_sb:6.1f} ns/subblock-pack)"
            )
            with RESULTS.open("a") as f:
                f.write(
                    json.dumps(
                        {
                            "shape": s.name,
                            "cfg": {**s.__dict__, "sbw_eff": effective_shape(s, v).sbw},
                            "variant": v.name,
                            "ns": ns,
                            "pcc": pcc,
                            "exact": exact,
                            "box": box,
                        }
                    )
                    + "\n"
                )
        tt_a.deallocate()
        tt_b.deallocate()
    logger.info("\n" + "\n".join(lines))
    assert not failures, failures
