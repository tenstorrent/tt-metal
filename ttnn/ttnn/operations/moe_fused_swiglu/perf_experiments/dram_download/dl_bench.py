# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""DRAM DOWNLOAD FLOOR — how long moe_fused_swiglu's DRAM traffic takes with NOTHING in its way.

The mirror image of `scatter_matmul`. That bench removes all DRAM by construction and measures the
compute + collective; this one removes all compute and collectives by construction and measures the
DOWNLOAD. Between them the op's two rooflines are bracketed by measurement rather than by assumption.

WHAT IT ISSUES — exactly the op's reads for ONE M-block, per core, at the op's own slices:

    W_gate   reader (NOC_0)   kr requests of hn*576 B     (K-row of my column's hidden slice)
    W_up     writer (NOC_1)   kr requests of hn*576 B
    W_down   writer (NOC_1)   HID_T requests of ec*576 B  (all hidden rows of my emb-output slice)
    x        reader (NOC_0)   32*rows sub-page slices (bf16 RM) or kr*rows whole tiles (bfp8)

Same `TensorAccessor`, same placement the caller chooses (interleaved or the PERF-12 ND shard), same
grid split (`hidden` across columns, `emb`-K across rows, `emb`-output across all cores). Then it
STOPS: no tilize, no multicast, no reduce, no matmul, no output write, no consumer handshake. Every
read is in flight and ONE barrier per RISC closes them.

READ THE NUMBER AS A CEILING, NOT A PREDICTION. Issuing everything before one barrier is the most
favourable schedule that exists; the op's real per-chunk barriers can only be slower. So this is the
floor the op is measured against, and the gap between them is the schedule's cost — which is the
quantity every overlap argument in this changelog has been about.

WHAT IS DELIBERATELY NOT HERE: the x row-multicast and the h all-gather. Those are L1->L1 collective
traffic, not DRAM download; `comm_skeleton` already prices the mcast primitive, and mixing them in
would stop this from being a clean DRAM number. `MODE` selects which DRAM streams participate so the
weight half and the activation half can be attributed separately.
"""

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

TILE = 32
BFP4_TILE = 576
BFP8_TILE = 1088
HIDDEN = 2048

CB_WG, CB_WU, CB_WD, CB_X = 0, 1, 2, 3

_FORMATS = {"bf16_rm": (ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT), "bfp8_tile": (ttnn.bfloat8_b, ttnn.TILE_LAYOUT)}


def _split(total, groups):
    """VERBATIM the descriptor's `_split` — the slice plan must match the op's, not approximate it."""
    base, rem = total // groups, total % groups
    sizes = [base + (1 if i < rem else 0) for i in range(groups)]
    starts, acc = [], 0
    for s in sizes:
        starts.append(acc)
        acc += s
    return sizes, starts


def clamp_grid(device, hgroups, kgroups):
    grid = device.compute_with_storage_grid_size()
    return min(hgroups, int(grid.x)), min(kgroups, int(grid.y))


def plan(device, emb, hgroups, kgroups, m_tiles):
    """The op's own geometry: what each core reads."""
    hid_t, emb_t = HIDDEN // TILE, emb // TILE
    hn_pad = (hid_t + hgroups - 1) // hgroups
    kr_sizes, kr_starts = _split(emb_t, kgroups)  # emb K split across grid ROWS
    hn_sizes, hn_starts = _split(hid_t, hgroups)  # hidden split across grid COLUMNS (ragged last)
    ec_sizes, ec_starts = _split(emb_t, hgroups * kgroups)  # emb OUTPUT split across ALL cores
    return {
        "hid_t": hid_t,
        "emb_t": emb_t,
        "hn_pad": hn_pad,
        "kr": (kr_sizes, kr_starts),
        "hn": (hn_sizes, hn_starts),
        "ec": (ec_sizes, ec_starts),
        "m_tiles": m_tiles,
        "hgroups": hgroups,
        "kgroups": kgroups,
    }


def bytes_moved(p, emb, mode, input_format):
    """Bytes this program actually requests — the denominator for GB/s."""
    hn_sizes, _ = p["hn"]
    kr_sizes, _ = p["kr"]
    ec_sizes, _ = p["ec"]
    wg = sum(kr_sizes[y] * hn_sizes[x] for x in range(p["hgroups"]) for y in range(p["kgroups"])) * BFP4_TILE
    wd = sum(ec_sizes) * p["hid_t"] * BFP4_TILE
    total = 0
    if mode in ("weights", "all"):
        total += wg * 2 + wd  # W_gate + W_up + W_down
    if mode in ("x", "all"):
        # x is read ONCE: each of the m_tiles tile-rows is staged by exactly one injector core, and an
        # injector reads only its own grid row's K-slice of that tile-row.
        per_row = sum(kr_sizes[y] for y in range(p["kgroups"])) / p["kgroups"]
        elem = 2 if input_format == "bf16_rm" else BFP8_TILE / (TILE * TILE)
        total += int(p["m_tiles"] * TILE * per_row * TILE * elem)
    return total


def _injectors(p):
    """Which core stages which x tile-rows — column c stages tile-row c, the op's baseline map."""
    out = {}
    for r in range(p["m_tiles"]):
        col = r % p["hgroups"]
        out.setdefault((col, 0), []).append(r)
    # Contiguous runs only: give each column its first row and a count (the kernel walks r0..r0+n).
    return {core: (rows[0], len(rows)) for core, rows in out.items()}


def build(device, x_t, wg_t, wu_t, wd_t, p, *, mode, input_format):
    hgroups, kgroups = p["hgroups"], p["kgroups"]
    hn_sizes, hn_starts = p["hn"]
    kr_sizes, kr_starts = p["kr"]
    ec_sizes, ec_starts = p["ec"]
    hid_t, emb_t, hn_pad = p["hid_t"], p["emb_t"], p["hn_pad"]

    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(hgroups - 1, kgroups - 1))])
    read_wg = read_wu = read_wd = 1 if mode in ("weights", "all") else 0
    read_x = 1 if mode in ("x", "all") else 0

    x_elem = 2 if input_format == "bf16_rm" else 0
    x_page = x_t.buffer_page_size()
    kr_max, ec_max = max(kr_sizes), max(ec_sizes)

    def cb(index, pages, page_bytes, dtype):
        return ttnn.CBDescriptor(
            total_size=max(pages, 1) * page_bytes,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_bytes)],
        )

    # x landing size is the SLICE the kernel writes, NOT the accessor's page: a bf16 ROW_MAJOR page is
    # a whole `emb` stick (14 336 B) but only this core's kr-tile slice of it (1 792 B) is read. Sizing
    # from the page asked for 3.67 MB and blew L1. Pages must also match the kernel's own stride:
    # bf16 writes 32 slices per tile-row, bfp8 writes kr whole tiles.
    inj_rows = max([n for (_, n) in _injectors(p).values()] or [0]) if read_x else 0
    if input_format == "bf16_rm":
        x_page_bytes, x_pages = kr_max * TILE * 2, inj_rows * TILE
    else:
        x_page_bytes, x_pages = x_page, inj_rows * kr_max

    # Landing space only — these CBs are reserved and written, never pushed to a consumer.
    cbs = [
        cb(CB_WG, kr_max * hn_pad, BFP4_TILE, ttnn.bfloat4_b),
        cb(CB_WU, kr_max * hn_pad, BFP4_TILE, ttnn.bfloat4_b),
        cb(CB_WD, hid_t * ec_max, BFP4_TILE, ttnn.bfloat4_b),
        cb(CB_X, x_pages, x_page_bytes, ttnn.bfloat16),
    ]

    r_ct = [read_wg, read_x, hid_t, BFP4_TILE, x_elem, x_page, emb_t]
    r_ct.extend(ttnn.TensorAccessorArgs(wg_t).get_compile_time_args())
    r_ct.extend(ttnn.TensorAccessorArgs(x_t).get_compile_time_args())
    w_ct = [read_wu, read_wd, hid_t, emb_t, BFP4_TILE]
    w_ct.extend(ttnn.TensorAccessorArgs(wu_t).get_compile_time_args())
    w_ct.extend(ttnn.TensorAccessorArgs(wd_t).get_compile_time_args())

    inj = _injectors(p)
    r_rt, w_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for x in range(hgroups):
        for y in range(kgroups):
            flat = y * hgroups + x
            x_row0, x_rows = inj.get((x, 0), (0, 0)) if y == 0 else (0, 0)
            r_rt[x][y] = [
                wg_t.buffer_address(),
                x_t.buffer_address(),
                kr_starts[y],
                kr_sizes[y],
                hn_starts[x],
                hn_sizes[x],
                x_rows,
                x_row0,
            ]
            w_rt[x][y] = [
                wu_t.buffer_address(),
                wd_t.buffer_address(),
                kr_starts[y],
                kr_sizes[y],
                hn_starts[x],
                hn_sizes[x],
                ec_starts[flat],
                ec_sizes[flat],
            ]

    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "dl_reader.cpp"),
            core_ranges=cores,
            compile_time_args=r_ct,
            runtime_args=r_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "dl_writer.cpp"),
            core_ranges=cores,
            compile_time_args=w_ct,
            runtime_args=w_rt,
            config=ttnn.WriterConfigDescriptor(),
        ),
    ]
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)


def read_kernel_ns(device):
    ttnn.ReadDeviceProfiler(device)
    per_chip = ttnn.get_latest_programs_perf_data()
    total, found = 0.0, False
    for programs in (per_chip or {}).values():
        for program in programs:
            entry = (getattr(program, "program_analyses_results", None) or {}).get(_DURATION_KEY)
            if entry is not None:
                total += float(entry.duration)
                found = True
    return total if found else None


def make_tensors(device, emb, capacity, input_format, wplace, m_tiles):
    import torch

    from ttnn.operations.moe_fused_swiglu.moe_fused_swiglu_program_descriptor import weight_memory_configs

    torch.manual_seed(0)
    dt, lay = _FORMATS[input_format]
    x_t = ttnn.from_torch(
        torch.randn((1, 1, capacity, emb), dtype=torch.bfloat16),
        dtype=dt,
        layout=lay,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    if wplace == "nd_shard":
        gu_mc, dn_mc = weight_memory_configs(device, emb, HIDDEN)
    else:
        gu_mc = dn_mc = ttnn.DRAM_MEMORY_CONFIG
    w = [
        ttnn.from_torch(
            torch.randn(s, dtype=torch.bfloat16),
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=mc,
        )
        for s, mc in (((emb, HIDDEN), gu_mc), ((emb, HIDDEN), gu_mc), ((HIDDEN, emb), dn_mc))
    ]
    return x_t, w


def measure(
    device,
    *,
    emb=7168,
    capacity=5120,
    count=256,
    input_format="bf16_rm",
    wplace="nd_shard",
    mode="all",
    hgroups=11,
    kgroups=8,
):
    hgroups, kgroups = clamp_grid(device, hgroups, kgroups)
    m_tiles = -(-count // TILE)
    p = plan(device, emb, hgroups, kgroups, m_tiles)
    x_t, w = make_tensors(device, emb, capacity, input_format, wplace, m_tiles)
    program = build(device, x_t, w[0], w[1], w[2], p, mode=mode, input_format=input_format)
    dummy = ttnn.allocate_tensor_on_device(
        ttnn.Shape([TILE, TILE]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    io = [x_t, w[0], w[1], w[2], dummy]

    ttnn.synchronize_device(device)
    read_kernel_ns(device)  # drain
    ttnn.generic_op(io, program)  # warm (JIT + first-touch)
    read_kernel_ns(device)
    samples = []
    for _ in range(5):
        ttnn.generic_op(io, program)
        ns = read_kernel_ns(device)
        if ns:
            samples.append(ns)
    samples.sort()
    nbytes = bytes_moved(p, emb, mode, input_format)
    med = samples[len(samples) // 2]
    return {
        "mode": mode,
        "input_format": input_format,
        "wplace": wplace,
        "count": count,
        "cores": hgroups * kgroups,
        "ns_median": med,
        "ns_min": samples[0],
        "ns_max": samples[-1],
        "bytes": nbytes,
        "gbps": nbytes / (med * 1e-9) / 1e9,
        "pct_peak": 100.0 * nbytes / (512e9 * med * 1e-9),
    }
