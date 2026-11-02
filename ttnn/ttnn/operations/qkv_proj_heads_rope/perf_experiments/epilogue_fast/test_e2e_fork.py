# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end baseline vs EPI_FAST on a forked copy of qkv_proj_heads_rope (op_fork/).
Env: E2E_ORDER=base,fast (or fast,base), E2E_CASES=<comma list of case ids>.
Prints E2E <case> <variant> ns=<device kernel ns> + per-output pcc/rms; correctness is the only assert."""

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
    TOLERANCES,
    check_outputs,
    device_inputs,
    make_inputs,
    padded_rows,
    pcc,
    reference,
    relative_rms,
    to_device,
    to_device_padded,
)

_FORK = Path(__file__).parent / "op_fork"
_spec = importlib.util.spec_from_file_location(
    "qkv_fork", _FORK / "__init__.py", submodule_search_locations=[str(_FORK)]
)
fork = importlib.util.module_from_spec(_spec)
sys.modules["qkv_fork"] = fork
_spec.loader.exec_module(fork)
fork_pd = sys.modules["qkv_fork.qkv_proj_heads_rope_program_descriptor"]

_KEY = "DEVICE KERNEL DURATION [ns]"
ORDER = os.environ.get("E2E_ORDER", "base,fast").split(",")


def _cfg(fidelity=ttnn.MathFidelity.HiFi3, fp32=True, full_sync=False):
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=fidelity, fp32_dest_acc_en=fp32, math_approx_mode=False, dst_full_sync_en=full_sync
    )


# id -> (batch, seq, forced heads_per_core or 0, compute config or None = default, padded-inf inputs)
CASES = {
    "8x512": (8, 512, 0, None, False),
    "4x512": (4, 512, 0, None, False),
    "2x512": (2, 512, 0, None, False),
    "7x511": (7, 511, 0, None, True),
    "4x256_h3": (4, 256, 3, None, False),
    "1x32_h1": (1, 32, 1, None, False),
    "2x512_hifi4": (2, 512, 0, _cfg(ttnn.MathFidelity.HiFi4), False),
    "2x512_lofi": (2, 512, 0, _cfg(ttnn.MathFidelity.LoFi), False),
    "2x512_bf16dest": (2, 512, 0, _cfg(fp32=False), False),
    "2x512_fullsync": (2, 512, 0, _cfg(full_sync=True), False),
    "32x128": (32, 128, 0, None, False),
    "13x300": (13, 300, 0, None, True),
}
SELECTED = os.environ.get("E2E_CASES", "8x512,4x512,2x512,7x511").split(",")


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
def test_e2e(device, monkeypatch, case):
    import torch

    batch, seq, heads, cfg, padded = CASES[case]
    if heads:
        monkeypatch.setattr(fork_pd, "HEADS_PER_CORE_CANDIDATES", (heads,))
    if "E2E_CB_BUDGET" in os.environ:  # smaller budget -> smaller block_m -> several blocks per core
        monkeypatch.setattr(fork_pd, "CB_BUDGET_BYTES", int(os.environ["E2E_CB_BUDGET"]))
    x, w, b, cos, sin = make_inputs(batch, seq, seq)
    expected = reference(x, w, b, cos, sin)
    l1 = ttnn.L1_MEMORY_CONFIG
    if padded:
        inf = float("inf")
        tensors = (
            to_device_padded(x, device, l1, inf),
            to_device(w, device, ttnn.DRAM_MEMORY_CONFIG),
            to_device(b, device, ttnn.DRAM_MEMORY_CONFIG),
            to_device_padded(cos, device, l1, inf),
            to_device_padded(sin, device, l1, inf),
        )
    else:
        tensors = device_inputs(device, x, w, b, cos, sin)
    kwargs = {} if cfg is None else {"compute_kernel_config": cfg}
    blk = fork_pd.select_blocking(device, batch, seq, cfg or fork.default_compute_kernel_config())
    rows_max = -(-blk.tensor_row_tiles // blk.grid_m)
    print(
        f"E2E {case} blocking hpc={blk.heads_per_core} grid={blk.grid_m}x{blk.grid_n} block_m={blk.block_m} "
        f"blocks_per_core={-(-rows_max // blk.block_m)}"
    )
    results = {}
    for variant in ORDER:
        if variant == "fast":
            monkeypatch.setenv("QKV_EPI_FAST", "1")
        else:
            monkeypatch.delenv("QKV_EPI_FAST", raising=False)
        outputs = fork.qkv_proj_heads_rope(*tensors, **kwargs)
        ns = _kernel_ns(device)
        # LoFi / bf16 DEST miss the bf16 tolerance by construction (baseline too): gate those on
        # bit-equality with the baseline instead.
        loose = case.endswith(("_lofi", "_bf16dest"))
        nocheck = bool(os.environ.get("QKV_ABLATE"))
        if nocheck:
            print(f"E2E {case} {variant} ns={ns:.0f} (ablated: {os.environ['QKV_ABLATE']})")
            continue
        if not loose:
            check_outputs(outputs, expected)
        stats = []
        for name, out, ref in zip("qkv", outputs, expected):
            got = ttnn.to_torch(out)
            stats.append(f"{name}:pcc={pcc(got, ref):.6f},rms={relative_rms(got, ref):.3e}")
            if padded:
                assert (padded_rows(out, seq) == 0).all(), f"{variant} {name}: non-zero padding"
        results[variant] = [ttnn.to_torch(o) for o in outputs]
        print(f"E2E {case} {variant} ns={ns:.0f} {' '.join(stats)}")
    if len(results) == 2 and not os.environ.get("QKV_ABLATE"):
        same = all(torch.equal(a, b) for a, b in zip(*results.values()))
        print(f"E2E {case} bitwise_equal={same}")
        assert same or not case.endswith(("_lofi", "_bf16dest")), f"{case}: fast differs from baseline"
