# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""matmul_kloop bake-off on a forked copy of qkv_proj_heads_rope (op_fork/).

Env:
  MMK_VARIANTS  comma list of VARIANTS keys, run in that order (default "base").
  MMK_CASES     comma list of CASES keys (default "8x512").
  MMK_ISO=1     isolated matmul: every DM payload, the stores and the epilogue math stubbed (ablation
                hooks), so the kernel time is the compute K-loop; correctness is not checked (garbage data).
Prints MMK <case> <variant> ns=<device kernel ns> + per-output pcc / rel rms. Correctness is the only assert.
"""

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
    "mmk_fork", _FORK / "__init__.py", submodule_search_locations=[str(_FORK)]
)
fork = importlib.util.module_from_spec(_spec)
sys.modules["mmk_fork"] = fork
_spec.loader.exec_module(fork)
fork_pd = sys.modules["mmk_fork.qkv_proj_heads_rope_program_descriptor"]

_KEY = "DEVICE KERNEL DURATION [ns]"
ISO_DEFINES = "ABL_NO_STORE,ABL_NO_XREAD,ABL_NO_WREAD,ABL_NO_MCAST,ABL_NO_TABLE,ABL_NO_BIAS,ABL_NO_EPI"


def _cfg(fidelity=ttnn.MathFidelity.HiFi3, fp32=True, full_sync=False):
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=fidelity, fp32_dest_acc_en=fp32, math_approx_mode=False, dst_full_sync_en=full_sync
    )


# name -> (k_block, CB budget bytes or None, extra compute defines, diagnostic compute config or None)
VARIANTS = {
    "base": (4, None, [], None),
    "kb6": (6, 1200 * 1024, [], None),
    "kb8": (8, 1200 * 1024, [], None),
    "kb12": (12, 1400 * 1024, [], None),
    "base_big": (4, 1200 * 1024, [], None),
    # Diagnostics only (precision contract broken on purpose, never an option): fidelity sweep.
    "diag_lofi": (4, None, [], _cfg(ttnn.MathFidelity.LoFi)),
    "diag_hifi2": (4, None, [], _cfg(ttnn.MathFidelity.HiFi2)),
    "diag_hifi4": (4, None, [], _cfg(ttnn.MathFidelity.HiFi4)),
}
for _kb in (4, 6, 8, 12):
    _sfx = "" if _kb == 4 else f"_kb{_kb}"
    _budget = None if _kb == 4 else VARIANTS[f"kb{_kb}"][1]
    for _name, _defs in {
        "raw": ["MMK_RAW"],
        "raw_nopack": ["MMK_RAW", "MMK_DIAG_NOPACK"],
        "raw_ct1": ["MMK_RAW", "MMK_DIAG_CT1"],
        "fused": ["MMK_RAW", "MMK_FUSED"],
        "fused2": ["MMK_RAW", "MMK_FUSED", "MMK_FUSED_NO_END_OP"],
        "lean": ["MMK_RAW", "MMK_FUSED", "MMK_FUSED_NO_END_OP", "MMK_FUSED_LEAN"],
    }.items():
        if _name in ("raw_nopack", "raw_ct1") and _kb != 4:
            continue
        VARIANTS[_name + _sfx] = (_kb, _budget, _defs, None)
    VARIANTS.setdefault(f"kb{_kb}", (_kb, _budget, [], None))

for _name, _defs in (
    ("fused", ["MMK_RAW", "MMK_FUSED"]),
    ("fused2", ["MMK_RAW", "MMK_FUSED", "MMK_FUSED_NO_END_OP"]),
    ("lean", ["MMK_RAW", "MMK_FUSED", "MMK_FUSED_NO_END_OP", "MMK_FUSED_LEAN"]),
    ("base", []),
):
    VARIANTS[f"diag_lofi_{_name}"] = (4, None, _defs, _cfg(ttnn.MathFidelity.LoFi))
    VARIANTS[f"diag_hifi4_{_name}"] = (4, None, _defs, _cfg(ttnn.MathFidelity.HiFi4))

# id -> (batch, seq, forced heads_per_core or 0, compute config or None = default, padded-inf inputs)
CASES = {
    "8x512": (8, 512, 0, None, False),
    "8x512_xdram": (8, 512, 0, None, False),
    "4x512": (4, 512, 0, None, False),
    "2x512": (2, 512, 0, None, False),
    "7x511": (7, 511, 0, None, True),
    "1x32": (1, 32, 0, None, False),
    "4096x1_xdram": (4096, 1, 0, None, False),
    "32x128": (32, 128, 0, None, False),
    "2x512_hifi4": (2, 512, 0, _cfg(ttnn.MathFidelity.HiFi4), False),
    "2x512_bf16dest": (2, 512, 0, _cfg(fp32=False), False),
    "2x512_lofi": (2, 512, 0, _cfg(ttnn.MathFidelity.LoFi), False),
    "2x512_hifi2": (2, 512, 0, _cfg(ttnn.MathFidelity.HiFi2), False),
    "4x256_h1": (4, 256, 1, None, False),
    "4x256_h3": (4, 256, 3, None, False),
    "2x512_fullsync": (2, 512, 0, _cfg(full_sync=True), False),
}
SELECTED = os.environ.get("MMK_CASES", "8x512").split(",")
ORDER = os.environ.get("MMK_VARIANTS", "base").split(",")
ISO = os.environ.get("MMK_ISO") == "1"


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
def test_mmk(device, monkeypatch, case):
    import torch

    batch, seq, heads, case_cfg, padded = CASES[case]
    if heads:
        monkeypatch.setattr(fork_pd, "HEADS_PER_CORE_CANDIDATES", (heads,))
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
        x_mem = ttnn.DRAM_MEMORY_CONFIG if case.endswith("_xdram") else l1
        tensors = device_inputs(device, x, w, b, cos, sin, x_mem)
    base_blk = None
    first_out = None
    for variant in ORDER:
        kb, budget, defines, diag_cfg = VARIANTS[variant]
        cfg = diag_cfg or case_cfg
        monkeypatch.setattr(fork_pd, "K_BLOCK", kb)
        monkeypatch.setattr(fork_pd, "CB_BUDGET_BYTES", budget or 900 * 1024)
        if base_blk is not None:
            # Same core assignment as the first variant: only the K-loop may change.
            monkeypatch.setattr(fork_pd, "HEADS_PER_CORE_CANDIDATES", (base_blk.heads_per_core,))
        all_defines = list(defines) + (ISO_DEFINES.split(",") if ISO else [])
        if all_defines:
            monkeypatch.setenv("QKV_ABLATE", ",".join(all_defines))
        else:
            monkeypatch.delenv("QKV_ABLATE", raising=False)
        blk = fork_pd.select_blocking(device, batch, seq, cfg or fork.default_compute_kernel_config())
        base_blk = base_blk or blk
        rows_max = -(-blk.tensor_row_tiles // blk.grid_m)
        desc = (
            f"hpc={blk.heads_per_core} grid={blk.grid_m}x{blk.grid_n} block_m={blk.block_m} kb={blk.k_block} "
            f"sb={blk.sb_h}x{blk.sb_w} wdepth={blk.weight_depth} blocks={-(-rows_max // blk.block_m)} rows_max={rows_max}"
        )
        kwargs = {} if cfg is None else {"compute_kernel_config": cfg}
        outputs = fork.qkv_proj_heads_rope(*tensors, **kwargs)
        ns = _kernel_ns(device)
        if ISO or diag_cfg is not None:
            print(f"MMK {case} {variant} ns={ns:.0f} iso={int(ISO)} {desc}")
            continue
        stats = []
        for name, out, ref in zip("qkv", outputs, expected):
            got = ttnn.to_torch(out)
            stats.append(f"{name}:pcc={pcc(got, ref):.6f},rms={relative_rms(got, ref):.3e}")
            if padded:
                assert (padded_rows(out, seq) == 0).all(), f"{variant} {name}: non-zero padding"
        got_all = [ttnn.to_torch(o) for o in outputs]
        same = (
            ""
            if first_out is None
            else f" bitwise_equal_to_{ORDER[0]}={all(torch.equal(a, b) for a, b in zip(got_all, first_out))}"
        )
        first_out = first_out if first_out is not None else got_all
        print(f"MMK {case} {variant} ns={ns:.0f} {desc} {' '.join(stats)}{same}")
        # LoFi / bf16 DEST miss the bf16 tolerance by construction (baseline too): gated on bit-equality.
        loose = case.endswith(("_lofi", "_hifi2", "_bf16dest"))
        if loose:
            assert not same or same.endswith("True"), f"{case} {variant}: differs from {ORDER[0]}"
        elif "DIAG" not in ",".join(defines):
            buffer_type = ttnn.BufferType.DRAM if case.endswith("_xdram") else ttnn.BufferType.L1
            check_outputs(outputs, expected, buffer_type)
