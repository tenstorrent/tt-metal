# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""write_inflight ROOFLINE GATE — the write floor of the L1 -> DRAM crossover.

    scripts/run_safe_pytest.sh --run-all \\
        ttnn/ttnn/operations/tilize/perf_experiments/write_inflight/test_probe_writefloor.py

Writer kernel ONLY: no reader, no compute, no CB handshake. Each core issues
whole TILE pages of garbage from a fixed L1 window into an interleaved DRAM
buffer. **Correctness is deliberately NOT checked here** — this probe exists to
answer one question before any candidate is built:

    Is the tilize writer's measured ~140 GB/s (16 MB in ~119.4 us on the padded
    widening cast) AT the achievable write bandwidth, or is there headroom?

If it is at the ceiling, the whole `write_inflight` idea is a measured NULL and
no amount of issue-ahead can buy anything.

The page stream reproduces the op's writer exactly (tilize_writer.cpp):
runs of WT_CHUNK consecutive tile pages, the next run WT pages later.

Reference plan — [1,1,1024,2048] bf16 -> [1,1,2048,2048] fp32 padded widening
cast, from tilize_program_descriptor.derive_blocking():
    NT_H=64, WT=64, cb_depth=2, NT_BLK=1, PIPELINE_BLOCKS_PER_CORE=4
    -> n_chunks=4, WT_CHUNK=16, 256 blocks over 64 cores = 4 blocks/core
    -> 64 pages/core x 4096 B = 256 KB/core, 16 MB total.

Nothing is asserted about speed; the numbers are printed.

MEASURED — Wormhole n150 (bgd-lab-16), 16 MB / 64 cores / 4096 B fp32 tile pages,
one fresh-cache run per arm. The GATE VERDICT is at the bottom.

  A  in-flight window (pages per write barrier), plain writes
       ppb   1: 130,584    ppb   2: 126,877    ppb   4: 129,189
       ppb   8: 123,643    ppb  16: 124,196 <- the op's cadence (B7)
       ppb  32: 123,026    ppb  64: 124,989
     FLAT across a 64x change in the window. Even a barrier PER PAGE is only
     6% off the best. The write is bandwidth-bound, not issue-bound.

  B  issue mode @ ppb 16
       plain    : 127,898
       trid ahd1: 121,338    trid ahd3: 128,033    trid ahd7: 122,626   (flat)
       VC spread: 313,223  <- 2.45x REGRESSION (rotating unicast VCs per page)

  C  page stream:  op-strided 124,645  vs  one contiguous run 123,771   (flat)

  D  core count @ 16 MB:  8: 117,854   16: 113,810   32: 123,899   64: 122,521

  E  page size @ equal page count: fp32 4096 B 16 MB 126,984 (132 GB/s);
       bf16 2048 B 8 MB 58,894 (142 GB/s)  -> the same ceiling either way.

  F  both NoCs from BRISC: 174,044  <- 1.42x REGRESSION

  G  TRANSACTION SIZE @ constant 16 MB (row-major dest, so the page really is
     that big):  2048: 115,824   4096: 128,818   8192: 137,503
                16384: 141,408  32768: 144,938
     MONOTONICALLY WORSE with bigger transactions. A merged write would lose
     even if it were expressible (it is not — see the domain note in the
     `test_g_transaction_size` comment).

  H  every page from one fixed L1 source: 127,953 vs cycling 125,909  (flat)

  I  sub-page split of the op's own 4096 B page (bank stream held fixed)
       split 1: 129,387   split 2: 120,934   split 4: 119,448   split 8: 125,321
  J  same on a bf16 2048 B page: 1: 59,771   2: 59,923   4: 61,449   (flat)

GATE VERDICT — the achievable floor to move 16 MB of tile pages out of 64 cores
is 116,000-125,000 ns (~135-145 GB/s), and NO issue-side strategy moves it. The
padded widening cast's whole wall is 141,800-148,900 ns for that same 16 MB plus
a 4 MB read, i.e. it is already within ~15% of a floor that assumes the read,
the compute and the launch ramp are all free. There is no write-in-flight
headroom to buy; see `test_attrib_subpage.py` for the one apparent exception
(axis I) being attributed away from the writer.
"""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

import pathlib

import pytest
import ttnn
from loguru import logger

KERNEL_DIR = pathlib.Path(__file__).parent / "experiment_kernels"
_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

M_PLAIN, M_TRID, M_VC, M_DUALNOC = 0, 1, 2, 3
_MODE_NAME = {M_PLAIN: "plain", M_TRID: "trid", M_VC: "vc", M_DUALNOC: "dualnoc"}

TILE = 32
WINDOW_BYTES = 128 * 1024  # L1 scratch the writes are sourced from (2 x 64 KB slot)

# --- the reference (focus) plan -------------------------------------------
F_WT = 64  # tile columns of the output
F_WT_CHUNK = 16  # pages per block  (= run_len)
F_BLOCKS_PER_CORE = 4
F_NT_H = 64
F_CORES = 64


def _read_kernel_ns(device):
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


def _dst(device, dtype, nt, wt):
    """Interleaved DRAM TILE tensor: page == one 32x32 tile of `dtype`."""
    spec = ttnn.TensorSpec(
        ttnn.Shape([1, 1, nt * TILE, wt * TILE]), dtype, ttnn.TILE_LAYOUT, buffer_type=ttnn.BufferType.DRAM
    )
    return ttnn.allocate_tensor_on_device(spec, device)


def _dst_rm(device, page_bytes, n_pages):
    """Interleaved DRAM ROW_MAJOR fp32 tensor: page == one row of `page_bytes`.

    The ONLY way to get a legal interleaved DRAM page larger than a tile page, so
    the transaction-size axis can be swept at constant total traffic.
    """
    spec = ttnn.TensorSpec(
        ttnn.Shape([1, 1, n_pages, page_bytes // 4]),
        ttnn.float32,
        ttnn.ROW_MAJOR_LAYOUT,
        buffer_type=ttnn.BufferType.DRAM,
    )
    return ttnn.allocate_tensor_on_device(spec, device)


def measure(
    device,
    label,
    *,
    dtype=ttnn.float32,
    wt=F_WT,
    nt=F_NT_H,
    run_len=F_WT_CHUNK,
    blocks_per_core=F_BLOCKS_PER_CORE,
    num_cores=F_CORES,
    ppb=None,
    mode=M_PLAIN,
    ahead=1,
    contiguous=False,
    fixed_src=False,
    rm_page_bytes=None,
    split=1,
):
    """One arm. `ppb` defaults to the op's cadence (one barrier per block)."""
    page_bytes = TILE * TILE * (4 if dtype == ttnn.float32 else 2)
    pages_per_core = blocks_per_core * run_len
    if ppb is None:
        ppb = run_len
    total_blocks = num_cores * blocks_per_core

    if rm_page_bytes is not None:
        # Transaction-size axis: constant 16 MB, one contiguous run per core.
        page_bytes = rm_page_bytes
        total_pages = (16 * 2**20) // page_bytes
        pages_per_core = total_pages // num_cores
        run_len, contiguous = pages_per_core, True
        ppb = max(1, (64 * 1024) // page_bytes)  # constant 64 KB per barrier
        dst = _dst_rm(device, page_bytes, total_pages)
    else:
        assert total_blocks * run_len <= nt * wt, "page stream must fit the destination"
        dst = _dst(device, dtype, nt, wt)
    grid = device.compute_with_storage_grid_size()
    assert num_cores <= grid.x * grid.y, f"grid is {grid.x}x{grid.y}"
    cores = [ttnn.CoreCoord(c % grid.x, c // grid.x) for c in range(num_cores)]
    core_set = ttnn.CoreRangeSet({ttnn.CoreRange(c, c) for c in cores})

    window = max(WINDOW_BYTES, page_bytes)
    cb = ttnn.CBDescriptor(
        total_size=window,
        core_ranges=core_set,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.float32, page_size=min(page_bytes, window))
        ],
    )
    run_stride = 0 if contiguous else wt
    ct = [page_bytes, ppb, window, mode, ahead, run_len, run_stride, int(fixed_src), split]
    ct.extend(ttnn.TensorAccessorArgs(dst).get_compile_time_args())

    rt = ttnn.RuntimeArgs()
    for i, core in enumerate(cores):
        if contiguous:
            start_page = i * pages_per_core
        else:
            # The op's W-chunk-major block index: b = wc * NT_H + row.
            b0 = i * blocks_per_core
            wc, row = b0 // nt, b0 % nt
            start_page = row * wt + wc * run_len
        rt[core.x][core.y] = [dst.buffer_address(), start_page, pages_per_core]

    desc = ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "probe_write.cpp"),
                core_ranges=core_set,
                compile_time_args=ct,
                runtime_args=rt,
                config=ttnn.WriterConfigDescriptor(),
            )
        ],
        semaphores=[],
        cbs=[cb],
    )
    # generic_op wants >= 2 io tensors; a tiny unused stand-in serves as "input".
    src = ttnn.allocate_tensor_on_device(
        ttnn.TensorSpec(ttnn.Shape([1, 1, 32, 32]), dtype, ttnn.TILE_LAYOUT, buffer_type=ttnn.BufferType.DRAM), device
    )
    ttnn.generic_op([src, dst], desc)
    ttnn.synchronize_device(device)
    _read_kernel_ns(device)  # flush the warm-up window
    ttnn.generic_op([src, dst], desc)
    ttnn.synchronize_device(device)
    ns = _read_kernel_ns(device)

    total_bytes = num_cores * pages_per_core * page_bytes
    logger.info(
        f"WRITEFLOOR {label}: ns={ns} mode={_MODE_NAME[mode]} ahead={ahead} fixed_src={int(fixed_src)} split={split} "
        f"page={page_bytes}B ppb={ppb} "
        f"cores={num_cores} pages/core={pages_per_core} MB={total_bytes / 2**20:.1f} "
        f"GB/s={total_bytes / ns:.1f} ns/page={ns / pages_per_core:.1f}"
    )
    assert ns is not None
    return ns


# ── A. in-flight window at the focus plan's exact traffic ──────────────────
# ppb=16 IS what the op does today (master.md B7: one barrier per block).
@pytest.mark.parametrize("ppb", [1, 2, 4, 8, 16, 32, 64], ids=lambda v: f"ppb{v}")
def test_a_inflight_window(device, ppb):
    measure(device, f"A/ppb{ppb}", ppb=ppb)


# ── B. issue mode at the op's barrier cadence ──────────────────────────────
@pytest.mark.parametrize(
    "mode,ahead",
    [(M_PLAIN, 0), (M_TRID, 1), (M_TRID, 3), (M_TRID, 7), (M_VC, 0)],
    ids=["plain", "trid1", "trid3", "trid7", "vc"],
)
def test_b_issue_mode(device, mode, ahead):
    measure(device, f"B/{_MODE_NAME[mode]}{ahead}", mode=mode, ahead=ahead)


# ── C. page-stream shape: the op's strided runs vs one contiguous run ──────
@pytest.mark.parametrize("contiguous", [False, True], ids=["opstride", "contig"])
def test_c_stream_shape(device, contiguous):
    measure(device, f"C/{'contig' if contiguous else 'opstride'}", contiguous=contiguous)


# ── D. core count on the same 16 MB of write traffic ───────────────────────
@pytest.mark.parametrize("num_cores", [8, 16, 32, 64], ids=lambda v: f"cores{v}")
def test_d_core_count(device, num_cores):
    measure(device, f"D/cores{num_cores}", num_cores=num_cores, blocks_per_core=4 * (64 // num_cores))


# ── E. page size (bf16 tile page = 2048 B) at the same page COUNT ──────────
@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16], ids=["fp32", "bf16"])
def test_e_page_size(device, dtype):
    measure(device, f"E/{dtype}", dtype=dtype)


# ── F. both NoCs from the writer RISC (run last: most likely to misbehave) ──
def test_f_dualnoc(device):
    measure(device, "F/dualnoc", mode=M_DUALNOC)


# ── G. TRANSACTION SIZE at constant 16 MB / 64 cores ────────────────────────
# The op cannot actually merge tile pages (interleaved page p lives in bank
# p % num_banks, so consecutive tile pages are in DIFFERENT banks and can never
# share one transaction) — this arm prices what that costs, i.e. whether a
# bigger transaction would buy anything at all if the layout allowed it.
@pytest.mark.parametrize("page_bytes", [2048, 4096, 8192, 16384, 32768], ids=lambda v: f"xact{v}")
def test_g_transaction_size(device, page_bytes):
    measure(device, f"G/xact{page_bytes}", rm_page_bytes=page_bytes)


# ── H. every page from ONE fixed L1 source (the pre-stamped pad tile) ───────
@pytest.mark.parametrize("fixed_src", [False, True], ids=["cycling", "fixed"])
def test_h_fixed_source(device, fixed_src):
    measure(device, f"H/{'fixed' if fixed_src else 'cycling'}", fixed_src=fixed_src)


# ── I. SUB-PAGE SPLIT on the op's own page stream ───────────────────────────
# Same destination bank stream as the op, only the transaction size changes:
# each 4096 B fp32 tile page issued as `split` consecutive sub-transactions.
# split=1 IS the shipped writer (master.md B5 `page_write`). This is the one
# axis G cannot answer, because G moved the bank map along with the size.
@pytest.mark.parametrize("split", [1, 2, 4, 8], ids=lambda v: f"split{v}")
def test_i_subpage_split(device, split):
    measure(device, f"I/split{split}", split=split)


# ── J. the same split on a bf16 (2048 B) tile page ─────────────────────────
@pytest.mark.parametrize("split", [1, 2, 4], ids=lambda v: f"split{v}")
def test_j_subpage_split_bf16(device, split):
    measure(device, f"J/bf16/split{split}", dtype=ttnn.bfloat16, split=split)
