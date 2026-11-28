# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""tile_face_skip bake-off. Correctness = candidate logical output bit-identical to the op's (baseline) output,
and baseline bit-identical to torch .to() for non-bf8b destinations. Perf = DEVICE KERNEL DURATION per launch,
read in-process from the device profiler; median of launches 2..N. Perf is reported, never asserted.

Env: TFS_CASES (comma ids, default the focus set), TFS_VARIANTS (comma, default all), TFS_LAUNCHES (12),
     TFS_REVERSE=1 (reverse variant order), TFS_PERF=0 (correctness only), TFS_REPORT (append table to file).
Run:  scripts/run_safe_pytest.sh ttnn/ttnn/operations/g6_typecast/perf_experiments/tile_face_skip/test_tfs.py
"""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
os.environ.pop("TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES", None)

import statistics
import sys
from collections import defaultdict
from pathlib import Path

import pytest
import importlib as _importlib

torch = _importlib.import_module("torch")  # experiment bench, not ttnn package code
import ttnn

sys.path.insert(0, str(Path(__file__).parent))
from tfs_bench import VARIANTS, build  # noqa: E402

BF16, F32, I32, U32, U16, U8, BF8B = (
    ttnn.bfloat16,
    ttnn.float32,
    ttnn.int32,
    ttnn.uint32,
    ttnn.uint16,
    ttnn.uint8,
    ttnn.bfloat8_b,
)
DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG

CASES = {
    # production cells (feature_spec ids)
    "Q2185": ((16, 4096, 12), BF16, F32, DRAM),
    "Q2186": ((8, 4096, 12), BF16, F32, DRAM),
    "Q2215": ((8, 1024, 12), BF16, F32, DRAM),
    "Q2188": ((1, 3844, 12), BF16, F32, DRAM),
    "Q2219": ((1, 4096, 12), BF16, F32, DRAM),
    "Q2190": ((16, 1024, 12), BF16, F32, DRAM),
    "Q2198": ((1, 31, 12), BF16, F32, DRAM),
    "Q2191": ((16, 1, 12), BF16, F32, DRAM),
    "Q2204": ((1, 256, 12), BF16, F32, DRAM),
    "Q2227": ((1, 1, 12), BF16, F32, L1),
    "F320": ((1, 1, 196, 196), F32, BF16, DRAM),
    "F324": ((1, 1, 207, 207), F32, BF16, DRAM),
    "Q2200": ((1, 1, 33, 256), BF16, BF8B, DRAM),
    "Q2221": ((1, 1, 100, 256), BF16, BF8B, DRAM),  # h_tail 4: bottom faces dead
    # synthetic correctness / domain cells
    "S_bf8b_w12": ((4, 1000, 12), BF16, BF8B, DRAM),
    "S_bf8b_w40h33": ((2, 33, 40), BF16, BF8B, DRAM),
    "S_i32_bf16": ((2, 300, 12), I32, BF16, DRAM),
    "S_bf16_i32": ((2, 300, 20), BF16, I32, DRAM),
    "S_u16_i32": ((2, 49, 44), U16, I32, DRAM),
    "S_u32_i32": ((3, 65, 12), U32, I32, DRAM),
    "S_bf16_u8": ((2, 300, 12), BF16, U8, DRAM),
    "S_f32_bf16_l1": ((1, 2, 100, 140), F32, BF16, L1),
    "S_bf16_f32_r1": ((1000,), BF16, F32, DRAM),
    "S_aligned": ((1, 1, 512, 512), F32, BF16, DRAM),
    "S_w48": ((4, 512, 48), BF16, F32, DRAM),
    "P39": ((1, 128, 256, 256), BF16, BF8B, L1),
    "S_big_w20": ((8, 4096, 20), BF16, F32, DRAM),
}
FOCUS = "Q2185,Q2186,Q2215,Q2188,Q2219,Q2190,Q2198,Q2191,Q2204"


def _torch_input(shape, src):
    torch.manual_seed(0)
    n = 1
    for d in shape:
        n *= d
    if src == BF16:
        x = torch.randn(n).to(torch.bfloat16) * 50
    elif src == F32:
        x = torch.randn(n, dtype=torch.float32) * 50
    elif src == I32:
        x = torch.randint(-(2**31), 2**31, (n,), dtype=torch.int64).to(torch.int32)
    elif src == U32:
        x = torch.randint(0, 2**31, (n,), dtype=torch.int64).to(torch.int32)
    elif src == U16:
        x = torch.randint(0, 2**16, (n,), dtype=torch.int64).to(torch.int32)
    return x.reshape(shape)


def _torch_expected(x, src, dst):
    if dst == F32:
        return x.float()
    if dst == BF16:
        return x.to(torch.float32).to(torch.bfloat16)
    if dst == I32:
        return x.to(torch.int32)
    return None  # bf8b / uint8: candidate is gated against the op's own output only


_RAW_OFFSET = 0
_FREQ = None


def _launch_spans(device):
    """Per-program kernel span (ns), in launch order, since the previous call."""
    global _RAW_OFFSET, _FREQ
    ttnn.ReadDeviceProfiler(device)
    root = Path(
        os.environ.get("TT_METAL_PROFILER_DIR", str(Path(os.environ.get("TT_METAL_HOME", ".")) / "generated/profiler"))
    )
    path = root / ".logs/profile_log_device.csv"
    if not path.exists():
        return []
    if path.stat().st_size < _RAW_OFFSET:
        _RAW_OFFSET = 0
    with path.open() as f:
        f.seek(_RAW_OFFSET)
        lines = f.readlines()
        _RAW_OFFSET = f.tell()
    progs = defaultdict(lambda: [None, None])
    order = []
    for line in lines:
        if "CHIP_FREQ[MHz]:" in line:
            _FREQ = float(line.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
        r = [v.strip() for v in line.split(",")]
        if len(r) < 12 or not r[10].endswith("-KERNEL") or r[11] not in ("ZONE_START", "ZONE_END"):
            continue
        key = (r[0], r[7])
        if key not in progs:
            order.append(key)
        t = int(r[5])
        p = progs[key]
        if r[11] == "ZONE_START":
            p[0] = t if p[0] is None else min(p[0], t)
        else:
            p[1] = t if p[1] is None else max(p[1], t)
    return [(progs[k][1] - progs[k][0]) * 1000 / _FREQ for k in order]


def _run_case(device, cid, variants, launches, perf):
    shape, src, dst, mc = CASES[cid]
    x = _torch_input(shape, src)
    tin = ttnn.from_torch(x, dtype=src, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    outs = {}
    descs = {}
    for v in variants:
        out = ttnn.allocate_tensor_on_device(tin.shape, dst, ttnn.TILE_LAYOUT, device, mc)
        descs[v] = (build(tin, out, v), out)
    for v in variants:
        desc, out = descs[v]
        res = ttnn.to_torch(ttnn.generic_op([tin, out], desc))
        outs[v] = res
    base = outs.get("baseline")
    if base is not None:
        exp = _torch_expected(x, src, dst)
        if exp is not None and dst not in (U8,):
            assert torch.equal(base.to(exp.dtype).reshape(exp.shape), exp), f"{cid}: baseline != torch"
        for v in variants:
            assert torch.equal(
                outs[v], base
            ), f"{cid}: {v} != baseline (max diff {(outs[v].float()-base.float()).abs().max()})"
    result = {}
    if perf:
        ttnn.synchronize_device(device)
        _launch_spans(device)
        for v in variants:
            desc, out = descs[v]
            for _ in range(launches):
                ttnn.generic_op([tin, out], desc)
            ttnn.synchronize_device(device)
            spans = _launch_spans(device)
            assert len(spans) == launches, f"{cid} {v}: {len(spans)} programs"
            result[v] = statistics.median(spans[1:]) if launches > 1 else spans[0]
    return result


def test_tfs(device):
    cases = os.environ.get("TFS_CASES", FOCUS).split(",")
    variants = os.environ.get("TFS_VARIANTS", ",".join(VARIANTS)).split(",")
    if os.environ.get("TFS_REVERSE") == "1":
        variants = variants[::-1]
    launches = int(os.environ.get("TFS_LAUNCHES", "12"))
    perf = os.environ.get("TFS_PERF", "1") != "0"
    lines = [f"order={variants} launches={launches}"]
    for cid in cases:
        r = _run_case(device, cid, variants, launches, perf)
        b = r.get("baseline")
        cells = []
        for v in sorted(r, key=list(VARIANTS).index):
            rel = f" ({b / r[v]:.2f}x)" if b and v != "baseline" else ""
            cells.append(f"{v}={r[v]:.0f}{rel}")
        lines.append(f"TFS {cid:<16} " + ("  ".join(cells) if cells else "correct"))
    text = "\n".join(lines)
    print("\n" + text)
    rep = os.environ.get("TFS_REPORT")
    if rep:
        with open(rep, "a") as f:
            f.write(text + "\n")
