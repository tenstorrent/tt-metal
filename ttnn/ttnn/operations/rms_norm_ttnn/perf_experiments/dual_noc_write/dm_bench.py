# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# ---------------------------------------------------------------------------
# dual_noc_write -- the isolated DM ROOFLINE INSTRUMENT for rms_norm_ttnn.
# ---------------------------------------------------------------------------
# A pure DRAM->L1->DRAM tile copy with NO Tensix compute kernel, reproducing the
# focus shape's per-core transfer pattern EXACTLY:
#
#   input (1,1,8192,2304) bf16 TILE DRAM-interleaved  ->  Rt=256 tile-rows,
#   WT=72 tiles/row, 2048 B pages, split over the full 11x10 = 110-core grid by
#   the very same `ttnn.split_work_to_cores(full_grid, Rt, row_wise=True)` the op
#   uses, so 36 cores own 3 tile-rows and 74 own 2.
#
# Everything except WHO ISSUES WHAT is held constant across variants: same
# tensors, same page size, same transaction count, same barrier structure, same
# CB depth (2 row-blocks per lane), same core set, same work split.  A measured
# delta is therefore attributable to the RISC-V / NoC assignment alone.
#
# The op's own writer takes a FULL `noc_async_write_barrier()` per row-block;
# the `wflush` variants replace it with `noc_async_writes_flushed()` (the CB slot
# only needs the data to have LEFT L1) plus one final barrier.  That is a
# second, independent write-side lever this instrument can price.

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")

import statistics
from pathlib import Path

import ttnn

HERE = Path(__file__).resolve().parent
KERNEL = str(HERE / "kernels" / "dm_lane.cpp")
_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

TILE = 32
TILE_BYTES = TILE * TILE * 2  # bf16

NOC0 = ttnn.NOC.NOC_0
NOC1 = ttnn.NOC.NOC_1
NC = ttnn.DataMovementProcessor.RISCV_1  # NCRISC -- the op's reader
BR = ttnn.DataMovementProcessor.RISCV_0  # BRISC  -- the op's writer

# ---------------------------------------------------------------------------
# A CONFIG describes, per RISC-V, which NoC it uses and which tile-column ranges
# of every tile-row it READS (producing into a CB) and WRITES (consuming a CB).
# `frac` is the fraction of WT given to the FIRST listed owner.
#
# lanes are named 'a' (cb 0) and 'b' (cb 1); lane 'a' = columns [0, split),
# lane 'b' = columns [split, WT).  A config names, for each lane, the reader
# RISC and the writer RISC.  Every CB therefore has exactly one producer RISC
# and one consumer RISC.
# ---------------------------------------------------------------------------
# name -> (lanes, noc_of_NC, noc_of_BR, split_frac)
#   lanes = tuple of (reader_risc, writer_risc) for lane a [, lane b]
CONFIGS = {
    # ---- the shipped structure: NCRISC/NoC0 reads all, BRISC/NoC1 writes all
    "base": ((("NC", "BR"),), NOC0, NOC1, 1.0),
    # ---- same split of work, NoCs swapped (reads on NoC1, writes on NoC0)
    "swap_noc": ((("NC", "BR"),), NOC1, NOC0, 1.0),
    # ---- reads AND writes both on NoC0 / both on NoC1
    "both_noc0": ((("NC", "BR"),), NOC0, NOC0, 1.0),
    "both_noc1": ((("NC", "BR"),), NOC1, NOC1, 1.0),
    # ---- roles swapped: BRISC reads, NCRISC writes (same NoC per RISC as base)
    "role_swap": ((("BR", "NC"),), NOC0, NOC1, 1.0),
    # ---- THE CANDIDATE FAMILY -------------------------------------------
    # w_split: NCRISC still reads EVERYTHING (compute needs all 72 tiles), but
    # the WRITE is split -- BRISC writes lane a, NCRISC writes lane b.
    "w_split_50": ((("NC", "BR"), ("NC", "NC")), NOC0, NOC1, 0.5),
    "w_split_25": ((("NC", "BR"), ("NC", "NC")), NOC0, NOC1, 0.75),  # NC writes 25%
    "w_split_75": ((("NC", "BR"), ("NC", "NC")), NOC0, NOC1, 0.25),  # NC writes 75%
    # the ANALYTICALLY balanced shift: move just enough of the write to NoC0 to
    # equalise the two NoCs' predicted solo times (see the report).
    "w_split_10": ((("NC", "BR"), ("NC", "NC")), NOC0, NOC1, 0.90),  # NC writes 10%
    "w_split_20": ((("NC", "BR"), ("NC", "NC")), NOC0, NOC1, 0.80),  # NC writes 20%
    # r_split: reads split, all writes on BRISC (the classic split_reader).
    "r_split_50": ((("NC", "BR"), ("BR", "BR")), NOC0, NOC1, 0.5),
    # split_rw: FULL symmetric split -- each RISC reads half and writes half on
    # its own NoC.  This is the balanced-issue upper bound.
    "split_rw_50": ((("NC", "NC"), ("BR", "BR")), NOC0, NOC1, 0.5),
    # split_rw with the lanes crossed: each RISC reads one half and writes the
    # OTHER half, so a CB always crosses RISCs (matches how the real op would
    # have to wire it, with compute in between).
    "split_rw_cross": ((("NC", "BR"), ("BR", "NC")), NOC0, NOC1, 0.5),
}


def _read_kernel_ns(device):
    ttnn.ReadDeviceProfiler(device)
    per_chip = ttnn.get_latest_programs_perf_data()
    total, found = 0.0, False
    for programs in (per_chip or {}).values():
        for program in programs:
            results = getattr(program, "program_analyses_results", None) or {}
            entry = results.get(_DURATION_KEY)
            if entry is None:
                continue
            total += float(entry.duration)
            found = True
    return total if found else None


def _cores_in(crs):
    out = []
    for cr in crs.ranges():
        for y in range(cr.start.y, cr.end.y + 1):
            for x in range(cr.start.x, cr.end.x + 1):
                out.append(ttnn.CoreCoord(x, y))
    return out


def _full_grid(device):
    g = device.compute_with_storage_grid_size()
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))])


def build_descriptor(
    x,
    out,
    *,
    config,
    ablate_read=False,
    ablate_write=False,
    write_flush=False,
    depth=2,
    nbanks=0,
    alt_tail=0,
    dyn=False,
):
    device = x.device()
    shape = list(x.shape)
    Rt = (shape[-2] + TILE - 1) // TILE
    WT = (shape[-1] + TILE - 1) // TILE

    lanes_spec, noc_nc, noc_br, frac = CONFIGS[config]
    n_lanes = len(lanes_spec)
    if nbanks and (n_lanes != 1 or WT % nbanks):
        raise ValueError("bank-run coalescing needs a single lane and NBANKS | WT")
    if n_lanes == 1:
        lane_ranges = [(0, WT)]
    else:
        split = max(1, min(WT - 1, int(round(WT * frac))))
        lane_ranges = [(0, split), (split, WT - split)]

    # --- work split: byte-identical to the op's rows scheme -----------------
    num_cores, all_cores, g1, g2, rpc1, rpc2 = ttnn.split_work_to_cores(_full_grid(device), Rt, True)
    assignment = []  # (core, row_start, num_rows)
    cursor = 0
    for group, rpc in ((_cores_in(g1), rpc1), (_cores_in(g2), rpc2)):
        for core in group:
            assignment.append((core, cursor, rpc))
            cursor += rpc
    assert cursor == Rt

    # --- CBs: one per lane, `depth` row-blocks deep -------------------------
    cbs = []
    for i, (_, cnt) in enumerate(lane_ranges):
        cbs.append(
            ttnn.CBDescriptor(
                total_size=depth * cnt * TILE_BYTES,
                core_ranges=all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=TILE_BYTES)
                ],
            )
        )

    # --- per-RISC lane assignment ------------------------------------------
    risc_lanes = {"NC": {"r": [], "w": []}, "BR": {"r": [], "w": []}}
    for i, (rd, wr) in enumerate(lanes_spec):
        start, cnt = lane_ranges[i]
        risc_lanes[rd]["r"].append((start, cnt, i))
        risc_lanes[wr]["w"].append((start, cnt, i))

    in_ta = ttnn.TensorAccessorArgs(x).get_compile_time_args()
    out_ta = ttnn.TensorAccessorArgs(out).get_compile_time_args()

    kernels = []
    for tag, proc, noc in (("NC", NC, noc_nc), ("BR", BR, noc_br)):
        rl = risc_lanes[tag]["r"]
        wl = risc_lanes[tag]["w"]
        if not rl and not wl:
            continue
        ct = [WT, TILE_BYTES, len(rl), len(wl), int(ablate_read), int(ablate_write), int(write_flush)]
        for slot in (rl + [(0, 0, 0)] * 2)[:2]:
            ct += [slot[0], slot[1], slot[2]]
        for slot in (wl + [(0, 0, 0)] * 2)[:2]:
            ct += [slot[0], slot[1], slot[2]]
        alt = alt_tail if (tag == "BR" and wl) else 0
        alt_noc = 0 if noc == NOC1 else 1
        ct += [nbanks, alt, alt_noc]
        ct += list(in_ta) + list(out_ta)
        rt = ttnn.RuntimeArgs()
        for core, row_start, num_rows in assignment:
            rt[core.x][core.y] = [x.buffer_address(), out.buffer_address(), row_start, num_rows]
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=KERNEL,
                core_ranges=all_cores,
                compile_time_args=ct,
                runtime_args=rt,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=proc,
                    noc=noc,
                    noc_mode=(ttnn.NOC_MODE.DM_DYNAMIC_NOC if dyn else ttnn.NOC_MODE.DM_DEDICATED_NOC),
                ),
            )
        )

    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)


def make_tensors(device, shape):
    import torch

    torch.manual_seed(0)
    t = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    x = ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    return t, x, out


def measure(
    device,
    shape,
    config,
    *,
    ablate_read=False,
    ablate_write=False,
    write_flush=False,
    depth=2,
    nbanks=0,
    alt_tail=0,
    dyn=False,
    trials=3,
):
    import torch

    t, x, out = make_tensors(device, shape)
    desc = build_descriptor(
        x,
        out,
        config=config,
        ablate_read=ablate_read,
        ablate_write=ablate_write,
        write_flush=write_flush,
        depth=depth,
        nbanks=nbanks,
        alt_tail=alt_tail,
        dyn=dyn,
    )

    def run():
        return ttnn.generic_op([x, out], desc)

    r = run()
    ttnn.synchronize_device(device)
    exact = None
    if not ablate_read and not ablate_write:
        got = ttnn.to_torch(r)
        exact = bool(torch.equal(got.to(torch.float32), t.to(torch.float32)))
        del got
    _read_kernel_ns(device)
    samples = []
    for _ in range(trials):
        run()
        ttnn.synchronize_device(device)
        v = _read_kernel_ns(device)
        if v is not None:
            samples.append(v)
    ns = statistics.median(samples) if samples else float("nan")
    ttnn.deallocate(x)
    ttnn.deallocate(out)
    return ns, exact, min(samples) if samples else float("nan")
