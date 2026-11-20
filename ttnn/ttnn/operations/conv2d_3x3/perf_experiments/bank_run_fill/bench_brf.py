# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""bank_run_fill bench: the conv2d_3x3 ROW_MAJOR DRAM-interleaved slab fill in isolation (one generic_op).

Per core (110 cores, the op's M split and blocking for the case), per M-block, the three kh bands are filled into an L1
slab, M-block after M-block, with nothing else running (no gather, no compute, no weights, no store). Variants: see
kernels/brf_reader.cpp. verify=True also writes every band to a DRAM output (page per (core, block, kh)) for a
bit-exact check against the input.
"""

from pathlib import Path

import ttnn

KDIR = Path(__file__).parent / "kernels"

VARIANTS = {
    "base": 0,
    "bank": 1,
    "bank_noc": 2,
    "bank_risc": 3,
    "bank_trisc": 4,
    "noc_only": 5,
    "risc_only": 6,
    "trisc_only": 7,
    "trisc_tilize_only": 7,  # mode 1: pass 1 only (cost split)
    "bank_trisc_tilize_only": 4,
}

# (H incl. halo rows, W, C_in, core_m tiles, bm) - the brief's blocking for each focus case; padding (0, 0, 1, 1).
CASES = {
    41: (1026, 2048, 32, 596, 50),
    42: (514, 1024, 64, 149, 22),
    35: (1026, 2048, 128, 596, 34),
    47: (514, 1024, 128, 149, 9),
}
import os

PAD_TOP = 0
# Band start alignment knob for the verify sweep (the op case is 1: every band starts at 7 mod 8).
PAD_LEFT = int(os.environ.get("BRF_PAD_LEFT", "1"))
GRID_X, GRID_Y = 11, 10


def geometry(idx):
    H, W, C, core_m, bm = CASES[idx]
    h_out = H - 2
    num_out_px = h_out * W
    num_in_px = H * W
    total_m = num_out_px // 32
    cores = []
    for c in range(GRID_X * GRID_Y):
        start = c * core_m
        actual = max(0, min(core_m, total_m - start))
        cores.append((start, actual))
    band_cap = ((bm * 32 + 2 + 255) // 256) * 256
    max_blocks = (core_m + bm - 1) // bm
    return dict(
        H=H,
        W=W,
        C=C,
        pb=C * 2,
        bm=bm,
        core_m=core_m,
        num_in_px=num_in_px,
        num_out_px=num_out_px,
        cores=cores,
        band_cap=band_cap,
        max_blocks=max_blocks,
    )


def bands(g, start, actual, m):
    """Mirror of brf::Geom::bands."""
    done = m * g["bm"]
    tiles = min(actual - done, g["bm"])
    o_first = (start + done) * 32
    o_end = min((start + done + tiles) * 32, g["num_out_px"])
    out = []
    for kh in range(3):
        shift = (kh - PAD_TOP) * g["W"] - PAD_LEFT
        lo = max(o_first + shift, 0)
        hi = min(o_end + shift + 2, g["num_in_px"])
        out.append((lo, max(hi, lo)))
    return out


def chunk_params(pb, chunk_bytes=None, nstg=4):
    """CH pixels per chunk (a multiple of 256) so a chunk is ~chunk_bytes, and the staging depth."""
    chunk_bytes = chunk_bytes or 65536
    ch = max(256, (chunk_bytes // pb) // 256 * 256)
    return ch, nstg


def create_descriptor(g, variant, inp, out, *, ch, nstg, verify, mode=0, verify_cores=None):
    v = VARIANTS[variant]
    trisc = v in (4, 7)
    pb, band_cap = g["pb"], g["band_cap"]
    cores = [ttnn.CoreCoord(c % GRID_X, c // GRID_X) for c in range(GRID_X * GRID_Y)]
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(GRID_X - 1, GRID_Y - 1))])
    geom = [g["W"], g["num_in_px"], g["num_out_px"], g["bm"], PAD_TOP, PAD_LEFT, 0]

    def cb(index, size):
        return ttnn.CBDescriptor(
            total_size=size,
            core_ranges=crs,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.bfloat16, page_size=2048)],
        )

    cbs = [cb(0, 3 * band_cap * pb), cb(1, nstg * ch * pb)]
    if trisc:
        cbs.append(cb(2, ch * pb))

    reader_ct = [v] + geom + [pb, band_cap, ch, nstg] + list(ttnn.TensorAccessorArgs(inp).get_compile_time_args())
    writer_ct = [v] + geom + [pb, band_cap, int(verify)] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
    compute_ct = [v] + geom + [pb, band_cap, ch, nstg, mode]
    rrt, wrt, crt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for c, core in enumerate(cores):
        start, actual = g["cores"][c]
        rrt[core.x][core.y] = [inp.buffer_address(), start, actual]
        base = c * g["max_blocks"] * 3 if (verify_cores is None or c < verify_cores) else 0xFFFFFFFF
        wrt[core.x][core.y] = [out.buffer_address(), start, actual, base]
        crt[core.x][core.y] = [start, actual]
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KDIR / "brf_reader.cpp"),
            core_ranges=crs,
            compile_time_args=reader_ct,
            runtime_args=rrt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KDIR / "brf_writer.cpp"),
            core_ranges=crs,
            compile_time_args=writer_ct,
            runtime_args=wrt,
            config=ttnn.WriterConfigDescriptor(),
        ),
    ]
    if trisc:
        cfg = ttnn.ComputeConfigDescriptor()
        cfg.math_fidelity = ttnn.MathFidelity.HiFi4
        cfg.fp32_dest_acc_en = True
        cfg.math_approx_mode = False
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=str(KDIR / "brf_compute.cpp"),
                core_ranges=crs,
                compile_time_args=compute_ct,
                runtime_args=crt,
                config=cfg,
            )
        )
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)


def run(g, variant, inp, out, **kw):
    return ttnn.generic_op([inp, out], create_descriptor(g, variant, inp, out, **kw))
