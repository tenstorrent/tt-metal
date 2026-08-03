# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""NoC-SPLIT sweep for the DRAM download floor.

`dl_bench` split the weight streams by MATRIX (reader = W_gate, writer = W_up + W_down), which is a
32/68 byte split — so its 340 GB/s was the rate of a lopsided schedule, not a ceiling. This module
puts one symmetric streamer (`kernels/dl_stream.cpp`) on BOTH data-movement RISCs and assigns each a
per-matrix ROW RANGE, so `noc0_frac` moves bytes continuously from NOC_1 to NOC_0.

    noc0_frac = 1.0   all bytes on the reader   -> NOC_0 solo rate r0
    noc0_frac = 0.0   all bytes on the writer   -> NOC_1 solo rate r1
    0 < f < 1         both stream concurrently

With solo rates r0 and r1 the optimal split is f* = r0 / (r0 + r1) and the predicted floor is
bytes / (r0 + r1) — the point where both NoCs finish together. Whether the two actually ADD is the
open question this sweep answers: they share the DRAM controllers, so the sum is an upper bound.
"""

from pathlib import Path

import ttnn

from ttnn.operations.moe_fused_swiglu.perf_experiments.dram_download import dl_bench as base

KERNEL_DIR = Path(__file__).parent / "kernels"
TILE, BFP4_TILE, HIDDEN = 32, 576, 2048
CB_WG, CB_WU, CB_WD, CB_X = 0, 1, 2, 3


def _rows_for(frac, total):
    """Rows on NOC_0; the remainder goes to NOC_1. Rounded so f=1 and f=0 are exact."""
    n = int(round(frac * total))
    return max(0, min(total, n))


def build(device, x_t, wg_t, wu_t, wd_t, p, *, noc0_frac, input_format, interleave=1, with_x=True):
    hgroups, kgroups = p["hgroups"], p["kgroups"]
    hn_sizes, hn_starts = p["hn"]
    kr_sizes, kr_starts = p["kr"]
    ec_sizes, ec_starts = p["ec"]
    hid_t, emb_t, hn_pad = p["hid_t"], p["emb_t"], p["hn_pad"]
    kr_max, ec_max = max(kr_sizes), max(ec_sizes)

    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(hgroups - 1, kgroups - 1))])
    x_elem = 2 if input_format == "bf16_rm" else 0
    x_page = x_t.buffer_page_size()

    inj = base._injectors(p)
    inj_rows = max([n for (_, n) in inj.values()] or [0]) if with_x else 0
    if input_format == "bf16_rm":
        x_page_bytes, x_pages = kr_max * TILE * 2, inj_rows * TILE
    else:
        x_page_bytes, x_pages = x_page, inj_rows * kr_max

    def cb(index, pages, page_bytes, dtype):
        return ttnn.CBDescriptor(
            total_size=max(pages, 1) * page_bytes,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_bytes)],
        )

    # ONE CB SET PER RISC. A CB has a single write pointer and one producer; both RISCs reserving the
    # same CB hung the device. Each set is sized for the WHOLE slice so f=0 and f=1 both fit.
    cbs = []
    for base_idx in (0, 4):
        cbs += [
            cb(base_idx + 0, kr_max * hn_pad, BFP4_TILE, ttnn.bfloat4_b),
            cb(base_idx + 1, kr_max * hn_pad, BFP4_TILE, ttnn.bfloat4_b),
            cb(base_idx + 2, hid_t * ec_max, BFP4_TILE, ttnn.bfloat4_b),
            cb(base_idx + 3, x_pages, x_page_bytes, ttnn.bfloat16),
        ]

    def ct_for(base_idx):
        out = [
            hid_t,
            emb_t,
            BFP4_TILE,
            x_elem,
            x_page,
            interleave,
            base_idx + 0,
            base_idx + 1,
            base_idx + 2,
            base_idx + 3,
        ]
        for t in (wg_t, wu_t, wd_t, x_t):
            out.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        return out

    rt0, rt1 = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for x in range(hgroups):
        for y in range(kgroups):
            flat = y * hgroups + x
            kr, hn, ec = kr_sizes[y], hn_sizes[x], ec_sizes[flat]
            g0 = _rows_for(noc0_frac, kr)  # W_gate rows on NOC_0
            u0 = _rows_for(noc0_frac, kr)  # W_up rows on NOC_0
            d0 = _rows_for(noc0_frac, hid_t)  # W_down rows on NOC_0
            # x rides NOC_0 whenever NOC_0 has any weight work, else NOC_1 — it is 2 % of the bytes and
            # keeping it on one RISC avoids a second variable in the sweep.
            xr, xr0 = inj.get((x, 0), (0, 0)) if (y == 0 and with_x) else (0, 0)
            common = [
                wg_t.buffer_address(),
                wu_t.buffer_address(),
                wd_t.buffer_address(),
                x_t.buffer_address(),
                kr_starts[y],
                hn_starts[x],
                hn,
                ec_starts[flat],
                ec,
            ]
            rt0[x][y] = common + [0, g0, 0, u0, 0, d0, kr, xr, xr0]
            rt1[x][y] = common + [g0, kr - g0, u0, kr - u0, d0, hid_t - d0, kr, 0, 0]

    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "dl_stream.cpp"),
            core_ranges=cores,
            compile_time_args=ct_for(0),
            runtime_args=rt0,
            config=ttnn.ReaderConfigDescriptor(),  # NCRISC -> NOC_0
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "dl_stream.cpp"),
            core_ranges=cores,
            compile_time_args=ct_for(4),
            runtime_args=rt1,
            config=ttnn.WriterConfigDescriptor(),  # BRISC -> NOC_1
        ),
    ]
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)


def measure(
    device,
    *,
    emb=7168,
    capacity=5120,
    count=256,
    input_format="bf16_rm",
    wplace="nd_shard",
    noc0_frac=0.5,
    interleave=1,
    with_x=True,
    hgroups=11,
    kgroups=8,
    reps=5,
    tensors=None,
):
    hgroups, kgroups = base.clamp_grid(device, hgroups, kgroups)
    m_tiles = -(-count // TILE)
    p = base.plan(device, emb, hgroups, kgroups, m_tiles)
    if tensors is None:
        tensors = base.make_tensors(device, emb, capacity, input_format, wplace, m_tiles)
    x_t, w = tensors
    program = build(
        device,
        x_t,
        w[0],
        w[1],
        w[2],
        p,
        noc0_frac=noc0_frac,
        input_format=input_format,
        interleave=interleave,
        with_x=with_x,
    )
    dummy = ttnn.allocate_tensor_on_device(
        ttnn.Shape([TILE, TILE]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    io = [x_t, w[0], w[1], w[2], dummy]

    ttnn.synchronize_device(device)
    base.read_kernel_ns(device)
    ttnn.generic_op(io, program)  # warm
    base.read_kernel_ns(device)
    samples = []
    for _ in range(reps):
        ttnn.generic_op(io, program)
        ns = base.read_kernel_ns(device)
        if ns:
            samples.append(ns)
    samples.sort()
    med = samples[len(samples) // 2]

    # Bytes actually requested, counted from the same row split the kernels were given.
    hn_sizes, _ = p["hn"]
    kr_sizes, _ = p["kr"]
    ec_sizes, _ = p["ec"]
    b0 = b1 = 0
    for x in range(hgroups):
        for y in range(kgroups):
            flat = y * hgroups + x
            kr, hn, ec = kr_sizes[y], hn_sizes[x], ec_sizes[flat]
            g0, d0 = _rows_for(noc0_frac, kr), _rows_for(noc0_frac, p["hid_t"])
            b0 += (2 * g0 * hn + d0 * ec) * BFP4_TILE
            b1 += (2 * (kr - g0) * hn + (p["hid_t"] - d0) * ec) * BFP4_TILE
    xb = base.bytes_moved(p, emb, "x", input_format) if with_x else 0
    b0 += xb
    total = b0 + b1
    return {
        "noc0_frac": noc0_frac,
        "bytes_noc0": b0,
        "bytes_noc1": b1,
        "bytes": total,
        "ns_median": med,
        "ns_min": samples[0],
        "ns_max": samples[-1],
        "gbps": total / (med * 1e-9) / 1e9,
        "pct_peak": 100.0 * total / (512e9 * med * 1e-9),
        "gbps_noc0": b0 / (med * 1e-9) / 1e9,
        "gbps_noc1": b1 / (med * 1e-9) / 1e9,
        "interleave": interleave,
        "cores": hgroups * kgroups,
    }
