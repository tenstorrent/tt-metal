# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""int32 -> bfloat16 compute-variant bake-off (perf experiment int32_bf16_fused_sfpu).

Same program as the op's TILE path (reader/writer copied into kernels/, same split, CBs, compute config:
fp32_dest_acc_en=True, UnpackToDestFp32 on the input CB, math_approx_mode=False); only the compute kernel
changes. Bit-exactness vs torch (int32 -> fp32 -> bf16, both RNE) is the only pass/fail; per variant the device
kernel duration (median of launches 2..N, in-process device profiler) is printed as BENCH_RESULT lines.

Env: BENCH_SHAPES="1,1,32,128;1,1,1024,1024"  BENCH_VARIANTS="baseline,fused_plain,twopass,packer_round,direct_llk,stock"
     BENCH_LAUNCHES=12  BENCH_ORDER=fwd|rev
"""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

import statistics
from pathlib import Path

import pytest
import importlib as _importlib

torch = _importlib.import_module("torch")  # experiment bench, not ttnn package code
import ttnn

from ttnn.operations.g6_typecast import g6_typecast_program_descriptor as opd

HERE = Path(__file__).resolve().parent
KDIR = HERE / "kernels"
_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

VARIANT_IDS = {
    "fused_plain": 1,
    "twopass": 2,
    "packer_round": 3,
    "direct_llk": 4,
    "veltkamp_plain": 5,
    "twopass_veltkamp": 6,
    "twopass_veltkamp_x2": 7,
    "veltkamp_fused_x2": 8,
    "twopass_veltkamp_macro": 9,
}


def _descriptor(inp, out, variant):
    device = inp.device()
    shape = tuple(inp.shape)
    in_page = inp.buffer_page_size()
    out_page = out.buffer_page_size()
    tiles = opd._tensor_tiles(shape)
    grid = device.compute_with_storage_grid_size()
    (_, all_cores, g1, g2, t1, t2) = ttnn.split_work_to_cores(grid, tiles, row_wise=True)
    block_tiles, depth = opd.plan_blocks(t1, in_page, out_page, 0)
    cb_pages = depth * block_tiles
    cbs = [
        ttnn.CBDescriptor(
            total_size=cb_pages * in_page,
            core_ranges=all_cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=inp.dtype, page_size=in_page)],
        ),
        ttnn.CBDescriptor(
            total_size=cb_pages * out_page,
            core_ranges=all_cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=1, data_format=out.dtype, page_size=out_page)],
        ),
    ]
    rrt, wrt, crt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    start = 0
    for group, count in ((g1, t1), (g2, t2)):
        if count == 0 or group.num_cores() == 0:
            continue
        for c in ttnn.corerange_to_cores(group, row_wise=True):
            rrt[c.x][c.y] = [start, count]
            wrt[c.x][c.y] = [start, count]
            crt[c.x][c.y] = [count]
            start += count
    assert start == tiles
    w = shape[-1]
    rct = [0, block_tiles, in_page, 0, opd._div_up(w, 32), w % 32, 0, 2, 0, 0, 0, 64, 1, 1, depth]
    rct += ttnn.TensorAccessorArgs(inp).get_compile_time_args()
    wct = [1, block_tiles, out_page, 0, 3, 0, 0, 0, 64, 1] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
    reader = ttnn.KernelDescriptor(
        kernel_source=str(KDIR / "g6_typecast_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=rct,
        runtime_args=rrt,
        common_runtime_args=[inp.buffer_address()],
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KDIR / "g6_typecast_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=wct,
        runtime_args=wrt,
        common_runtime_args=[out.buffer_address()],
        config=ttnn.WriterConfigDescriptor(),
    )
    if variant == "baseline":
        src, defines = KDIR / "baseline_compute.cpp", []
    else:
        src, defines = KDIR / "variant_compute.cpp", [("VARIANT", str(VARIANT_IDS[variant]))]
    compute = ttnn.KernelDescriptor(
        kernel_source=str(src),
        core_ranges=all_cores,
        compile_time_args=[0, 1, block_tiles],
        runtime_args=crt,
        defines=defines,
        config=opd._compute_config(inp.dtype, out.dtype),
    )
    return ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)


def _run(inp, variant):
    if variant == "stock":
        return ttnn.typecast(inp, ttnn.bfloat16)
    out = ttnn.allocate_tensor_on_device(inp.shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, inp.device(), inp.memory_config())
    return ttnn.generic_op([inp, out], _descriptor(inp, out, variant))


def _read_ns(device):
    ttnn.ReadDeviceProfiler(device)
    vals = []
    for programs in (ttnn.get_latest_programs_perf_data() or {}).values():
        for p in programs:
            e = (getattr(p, "program_analyses_results", None) or {}).get(_DURATION_KEY)
            if e is not None:
                vals.append(float(e.duration))
    return vals


def _input(shape, seed=1234):
    torch.manual_seed(seed)
    n = 1
    for d in shape:
        n *= d
    x = torch.randint(-(2**31), 2**31 - 1, (n,), dtype=torch.int64).to(torch.int32)
    # Witnesses + edges + values around 2^24 / 2^25 / 2^31 and ties for both roundings.
    special = [
        33685505,
        -33685505,
        2**24 + 2**16 + 1,
        -(2**24 + 2**16 + 1),
        -(2**31),
        2**31 - 1,
        -(2**31) + 1,
        0,
        1,
        -1,
        257,
        -257,
        2**24,
        2**24 + 1,
        2**24 + 2,
        2**24 + 3,
        2**24 - 1,
        2**25 + 2,
        2**25 + 6,
        2**31 - 64,
        2**31 - 65,
        2**31 - 128,
        2**30 + 2**22,
        2**30 + 2**22 + 1,
        2**30 + 3 * 2**22,
        -(2**30 + 3 * 2**22),
        65535,
        65536,
        65537,
        16777217,
        33554433,
        2**31 - 2**23,
    ]
    k = min(len(special), n)
    x[:k] = torch.tensor(special[:k], dtype=torch.int32)
    # A slab of small / mid-range values (exact in fp32, rounding only at bf16).
    m = n // 4
    x[k : k + m] = torch.randint(-(2**26), 2**26, (m,), dtype=torch.int32)[: max(0, min(m, n - k))]
    return x.reshape(shape)


def _shapes():
    raw = os.environ.get("BENCH_SHAPES", "1,1,32,128")
    return [tuple(int(v) for v in s.split(",")) for s in raw.split(";")]


def _variants():
    v = os.environ.get("BENCH_VARIANTS", "baseline,fused_plain,twopass,packer_round,direct_llk,stock").split(",")
    if os.environ.get("BENCH_ORDER", "fwd") == "rev":
        v = v[::-1]
    return v


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0)
    yield dev
    ttnn.close_device(dev)


def test_bench(device):
    launches = int(os.environ.get("BENCH_LAUNCHES", "12"))
    failures = []
    for shape in _shapes():
        x = _input(shape, int(os.environ.get("BENCH_SEED", "1234")))
        exp = x.to(torch.float32).to(torch.bfloat16)
        assert torch.equal(exp.view(torch.int16), x.to(torch.bfloat16).view(torch.int16))
        t = ttnn.from_torch(
            x, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        for variant in _variants():
            _read_ns(device)  # flush
            samples = []
            mism = None
            for i in range(launches):
                y = _run(t, variant)
                ttnn.synchronize_device(device)
                ns = _read_ns(device)
                if i == 0:
                    got = ttnn.to_torch(y).to(torch.bfloat16).reshape(shape)
                    bad = (got.view(torch.int16) != exp.view(torch.int16)).reshape(-1)
                    mism = int(bad.sum())
                    if mism and os.environ.get("BENCH_DEBUG"):
                        g2 = bad.reshape(shape)[0, 0, :32, :32].to(torch.int32)
                        for rr in range(32):
                            print("ROW", rr, "".join("X" if v else "." for v in g2[rr].tolist()))
                        xs = x.reshape(shape)[0, 0, :4, :4].to(torch.float64)
                        gs = got.reshape(shape)[0, 0, :4, :4].to(torch.float64)
                        for rr in range(4):
                            print(
                                "VALS",
                                rr,
                                [f"{gs[rr, c].item() / xs[rr, c].item():.4g}" if xs[rr, c] else "0" for c in range(4)],
                            )
                    if mism:
                        idx = bad.nonzero().reshape(-1)[:4]
                        print(
                            f"BENCH_MISMATCH {variant} {shape} n={mism} e.g. "
                            + ", ".join(
                                f"{int(x.reshape(-1)[j])}->{float(got.reshape(-1)[j])} "
                                f"(want {float(exp.reshape(-1)[j])})"
                                for j in idx
                            )
                        )
                ttnn.deallocate(y)
                if i >= 1 and ns:
                    samples.append(max(ns))
            med = statistics.median(samples) if samples else float("nan")
            print(
                f"BENCH_RESULT shape={','.join(map(str, shape))} variant={variant} med_ns={med:.0f} "
                f"min_ns={min(samples) if samples else 0:.0f} n={len(samples)} mismatches={mism}"
            )
            if mism and variant not in ("direct_llk", "stock", "packer_round"):
                failures.append((variant, shape, mism))
    assert not failures, f"inexact variants: {failures}"
