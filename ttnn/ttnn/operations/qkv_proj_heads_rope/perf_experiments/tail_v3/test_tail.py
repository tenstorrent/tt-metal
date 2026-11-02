# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""tail_v3: end-to-end variants of the post-K-loop tail on a forked qkv_proj_heads_rope (op_fork/).
Env: TAIL_ORDER=base,<variant>,... (variant = QKV_TAIL value, '+'-joined for combos, 'base' = none),
TAIL_CASES=<comma list>. Prints TAIL <case> <variant> ns=<device kernel ns> + per-output pcc / rel-rms and
bit-equality vs the first variant. Correctness (unit tolerance) is the only assert."""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

import importlib.util
import sys
from pathlib import Path

import pytest
import ttnn

from tests.ttnn.unit_tests.operations.qkv_proj_heads_rope.test_qkv_proj_heads_rope import (
    check_outputs,
    make_inputs,
    padded_rows,
    pcc,
    reference,
    relative_rms,
    to_device,
    to_device_padded,
)

_FORK = Path(__file__).parent / os.environ.get("TAIL_FORK", "op_fork")
_spec = importlib.util.spec_from_file_location(
    "qkv_tail3_fork", _FORK / "__init__.py", submodule_search_locations=[str(_FORK)]
)
fork = importlib.util.module_from_spec(_spec)
sys.modules["qkv_tail3_fork"] = fork
_spec.loader.exec_module(fork)

_KEY = "DEVICE KERNEL DURATION [ns]"
L1, DRAM = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
fork_pd = sys.modules["qkv_tail3_fork.qkv_proj_heads_rope_program_descriptor"]


def _cfg(fidelity=ttnn.MathFidelity.HiFi3, fp32=True, full_sync=False):
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=fidelity, fp32_dest_acc_en=fp32, math_approx_mode=False, dst_full_sync_en=full_sync
    )


# Precision-surface cases: (batch, seq, forced heads_per_core or 0, compute config)
CFG_CASES = {
    "2x512_hifi4": (2, 512, 0, _cfg(ttnn.MathFidelity.HiFi4)),
    "2x512_lofi": (2, 512, 0, _cfg(ttnn.MathFidelity.LoFi)),
    "2x512_fullsync": (2, 512, 0, _cfg(full_sync=True)),
    "2x512_bf16dest": (2, 512, 0, _cfg(fp32=False)),
    "4x256_h3": (4, 256, 3, None),
    "1x32_h1": (1, 32, 1, None),
}
# id -> (batch, seq, x, tables, outputs, padded-inf inputs)
CASES = {
    "8x512": (8, 512, L1, L1, L1, False),
    "4x512": (4, 512, L1, L1, L1, False),
    "2x512": (2, 512, L1, L1, L1, False),
    "1x32": (1, 32, L1, L1, L1, False),
    "7x511": (7, 511, L1, L1, L1, True),
    "4096x1": (4096, 1, DRAM, DRAM, DRAM, False),
    "8x256": (8, 256, L1, L1, L1, False),
    "8x384": (8, 384, L1, L1, L1, False),
}
SELECTED = os.environ.get("TAIL_CASES", "8x512").split(",")
ORDER = os.environ.get("TAIL_ORDER", "base").split(",")


def _kernel_ns(device):
    ttnn.ReadDeviceProfiler(device)
    total = None
    for programs in (ttnn.get_latest_programs_perf_data() or {}).values():
        for program in programs:
            entry = (getattr(program, "program_analyses_results", None) or {}).get(_KEY)
            if entry is not None:
                total = (total or 0.0) + float(entry.duration)
    return total


@pytest.mark.parametrize("case", SELECTED)
def test_tail(device, monkeypatch, case):
    kwargs = {}
    loose = False
    if case in CFG_CASES:
        batch, seq, heads, cfg = CFG_CASES[case]
        x_mc = t_mc = o_mc = L1
        padded = False
        if heads:
            monkeypatch.setattr(fork_pd, "HEADS_PER_CORE_CANDIDATES", (heads,))
        if cfg is not None:
            kwargs["compute_kernel_config"] = cfg
        # LoFi / bf16 DEST miss the bf16 tolerance by construction (baseline too): gated on bit-equality.
        loose = case.endswith(("_lofi", "_bf16dest"))
    else:
        batch, seq, x_mc, t_mc, o_mc, padded = CASES[case]
    x, w, b, cos, sin = make_inputs(batch, seq, seq)
    expected = reference(x, w, b, cos, sin)
    if padded:
        inf = float("inf")
        tensors = (
            to_device_padded(x, device, x_mc, inf),
            to_device(w, device, DRAM),
            to_device(b, device, DRAM),
            to_device_padded(cos, device, t_mc, inf),
            to_device_padded(sin, device, t_mc, inf),
        )
    else:
        tensors = (
            to_device(x, device, x_mc),
            to_device(w, device, DRAM),
            to_device(b, device, DRAM),
            to_device(cos, device, t_mc),
            to_device(sin, device, t_mc),
        )
    first = None
    for variant in ORDER:
        items = [] if variant == "base" else variant.split("+")
        abl = [i for i in items if i.startswith("ABL_")]
        tail = ",".join(i for i in items if not i.startswith("ABL_"))
        if abl:
            monkeypatch.setenv("QKV_ABLATE", ",".join(abl))
        else:
            monkeypatch.delenv("QKV_ABLATE", raising=False)
        monkeypatch.setenv("QKV_TAIL", tail)
        outputs = fork.qkv_proj_heads_rope(*tensors, memory_config=o_mc, **kwargs)
        ns = _kernel_ns(device)
        if abl or "abl" in variant:
            for o in outputs:
                ttnn.deallocate(o)
            print(f"TAIL {case} {variant} ns={ns:.0f} (ablated, unchecked)")
            if os.environ.get("TAIL_KEEP_PROF"):
                import shutil

                logs = Path(os.environ["TT_METAL_PROFILER_DIR"]) / ".logs"
                shutil.copy(logs / "profile_log_device.csv", logs / f"device_{case}_{variant}.csv")
            continue
        if not loose:
            check_outputs(outputs, expected, buffer_type=o_mc.buffer_type)
        import torch

        got = [ttnn.to_torch(o) for o in outputs]
        stats = " ".join(
            f"{n}:pcc={pcc(g, r):.6f},rms={relative_rms(g, r):.3e}" for n, g, r in zip("qkv", got, expected)
        )
        if padded:
            for n, o in zip("qkv", outputs):
                assert (padded_rows(o, seq) == 0).all(), f"{variant} {n}: non-zero padding"
        same = "" if first is None else f" bitwise_eq_first={all(torch.equal(a, c) for a, c in zip(first, got))}"
        ref_dir = os.environ.get("TAIL_SAVE")
        if ref_dir:
            # Last write wins, so a precompile pass's outputs are overwritten by the real pass.
            Path(ref_dir).mkdir(parents=True, exist_ok=True)
            torch.save(got, Path(ref_dir) / f"{_FORK.name}_{variant}_{case}.pt")
            other = os.environ.get("TAIL_CMP")
            if other and (Path(ref_dir) / f"{other}_{case}.pt").exists():
                ref_out = torch.load(Path(ref_dir) / f"{other}_{case}.pt")
                same += f" bitwise_eq_{other}={all(torch.equal(a, c) for a, c in zip(ref_out, got))}"
        if loose and first is not None:
            assert all(torch.equal(a, c) for a, c in zip(first, got)), f"{case} {variant}: differs from {ORDER[0]}"
        first = first or got
        for o in outputs:
            ttnn.deallocate(o)
        print(f"TAIL {case} {variant} ns={ns:.0f} {stats}{same}")
