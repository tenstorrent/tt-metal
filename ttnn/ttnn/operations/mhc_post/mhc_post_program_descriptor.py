# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""mhc_post — ProgramDescriptor (regimes R1 plain-2D and R2 fused-gather).

Work split (op_design.md "Split chooser"): the worker list L is the max_grid
rectangle in row-major order. R row groups x Q column members; row group g
is L[g*Q:(g+1)*Q] and owns a contiguous range of token-tile rows, member q a
contiguous range of C-column tiles (both ragged +-1). A core's block is one
token-tile row (BLOCK_ROWS) x block_cols column tiles x n streams.

R2 (stats / collapse): every core also produces, per token-tile row, one fp32
partial tile (its X'@W slice in cols 0..23, its sum of X'^2 in col 24). The Q
partials of a row are combined by a fixed 2-level tree: chunk k of the row
group = members [k*FANIN, (k+1)*FANIN); leader(t, k) = chunk member
(t mod chunk_size) folds its chunk's slots; root(t) = member (t mod Q) folds
the chunk sums in chunk order, writes stats and (collapse) runs the hc step.
"""

from __future__ import annotations

import struct
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE = 32

# ---------------------------------------------------------------------------
# Blocking knobs (single source of truth; every CB size / trip count derives from these)
# ---------------------------------------------------------------------------
BLOCK_ROWS = 1  # token-tile rows per block (reasoned decision, op_design.md "Why block_rows = 1")
IN_BLOCKS = 2  # input depth in blocks: the reader fetches block k+1 while compute mixes block k
OUT_BLOCKS = 2  # output depth in blocks: the writer drains block k while compute packs block k+1
DST_FULL_SYNC = True  # 8 fp32 DEST slots (vs 4 half-sync): room for the block's accumulators
MIX_FIXED_SLOTS = 2  # DEST slots of mix_block besides the per-unit accumulators (coefficient, operand)
MIX_FIXED_SLOTS_STATS = 3  # R2: + the running sum-of-squares accumulator
# R2 mechanism cap: the fused compute (mix + collapse + matmul + hc) at block_cols = 5 is 74 KB of kernel binaries
# + config, over the 70656 B TENSIX kernel-config buffer; 4 fits. With W only (never by mode: block_cols orders the
# stats sums, which must be the same bits in stats and collapse mode).
MAX_BLOCK_COLS_W = 4
COEF_DEPTH = 2  # token rows of expanded coefficients in flight
COEF_RAW_BUFS = 2  # reader landing buffers of raw coefficient tiles (row t+1 prefetched while row t expands)
BLOCK_L1_BUDGET = 384 * 1024  # bytes available to block-scaled CBs (ledger symbol)
W_RESIDENT_BUDGET = 256 * 1024  # bytes of resident W slice (R2 vs R2-W-streamed predicate)
W_STREAM_BLOCKS = 2  # W-streamed: column passes of W in flight
MAX_FEED_LINKS = 2  # feed: port cores reserved per direction (2 directions), whatever feed_num_links is
FEED_CHUNKS_IN_FLIGHT = 4  # feed: packets in flight per port (fabric_all_gather's 4-chunk batch, 112 KiB ring)
FEED_READ_BATCH = 4  # feed: chunks the port reader keeps in flight per read barrier (<= ring - in-flight packets)
FEED_WAVE_ROWS = 1  # feed: local rows per port wave (the ports stream row i while the workers finish row i+1)
# feed, T % 32 != 0: L1 of the landing port for the shared D tile rows (<= G - 1 a image, Cb tiles each; the largest
# golden shape, C 7168, needs 336 KiB)
FEED_LAND_BUDGET = 768 * 1024
COMBINE_FANIN = 11  # leader fan-in of the 2-level combine tree (<= 16, ledger symbol)
PARTIAL_DEPTH = 2  # per-row partial / sum-of-squares pages in flight to the writer
Y_BLOCKS = 2  # collapse: y blocks in flight to the writer
NORM_ROWS_IN_FLIGHT = 2  # norm: rstd rows expanded ahead of the norm phase (cb_rstd_col depth)
RSTD_SLOT_BYTES = 128  # norm: one row's rstd = 32 fp32 tokens
# norm: y_b of all a core's rows stays in L1 until rstd returns (the roots run after their own rows, Refinement 1);
# the largest golden shape (T2048 C7168: 64 rows x 3 cols) needs 384 KiB.
Y_HOLD_BUDGET = 512 * 1024

# Split-chooser model constants (catalog double_buffer measurements, Blackhole)
DRAM_BW = 512e9
CORE_BW = 17.9e9
# R2 (with W) per-core model: the fused pass is compute-bound per core, a core's time ~ rows * (ROW + cols * UNIT).
# Fitted on BH p300c (110 cores, bf16, HiFi2, T1280 C5120): R=1 709 us, R=2 490 us (stats).
ROW_NS_W = 4100
UNIT_NS_W = 6800

# CB indices (any distinct values)
CB_SUBLAYER_TILES = 0  # F tiles, one per unit
CB_STREAM_TILES = 1  # X tiles, n per unit
CB_COEF_RAW = 2  # reader-private landing of the raw post / comb / P tile rows
CB_COEF_COLS = 3  # coefficients expanded to column 0 of their own page (UnpackToDestFp32)
CB_MIXED_OUT_TILES = 4  # X' tiles, n per unit
CB_MIXED_BF16 = 5  # R2: bf16 X' of the pass (matmul in0, collapse operand)
CB_W_SLICE = 6  # R2: W rows of my columns (resident or streamed per pass)
CB_SUMSQ_ACC = 7  # R2: running elementwise sum of X'^2 over the row (reload accumulator)
CB_PROJ_INTERM = 8  # R2: matmul K-accumulation spill across the row's column passes
CB_PROJ_OUT = 9  # R2: the row's X'@W partial (compute -> writer)
CB_SUMSQ_COL = 10  # R2: row-reduced sum of squares, column 0 (compute -> writer)
CB_LEADER_SLOTS = 11  # R2: chunk members' partials (remote writes land; writer pushes)
CB_CHUNK_SUM = 12  # R2: leader's folded chunk partial (compute -> writer -> root)
CB_ROOT_SLOTS = 13  # R2: chunk sums, one region per rooted row (remote writes land; writer pushes)
CB_STATS = 14  # R2: root's stats tile (compute -> writer)
CB_HC_IN = 15  # collapse: stats in SFPU lane layout (writer -> compute)
CB_HC_OUT = 16  # collapse: hc in lane layout (compute -> writer)
CB_Y_OUT = 17  # collapse: y tiles (compute -> writer)
CB_HC_TILE = 18  # collapse: writer-private landing of the hc tile in row layout
CB_Y_HOLD = 19  # norm: bf16 y_b of every row of the core, held until its rstd lands (compute -> compute)
CB_YSQ_ACC = 20  # norm: running elementwise sum of y_b^2 over the row (reload accumulator)
CB_RSTD_COL = 21  # norm: rstd of the row in column 0 (reader -> compute, UnpackToDestFp32)
CB_NORM_W = 22  # norm: norm weight tiles of my columns, resident (reader -> compute, Row broadcast)
CB_RSTD_LAND = 23  # norm: reader-private landing of the root's rstd unicasts, one 128 B slot per row
CB_RSTD_SEND = 24  # norm: root writer's 128 B rstd source (lane vector -> token order)
CB_FEED_CHUNKS = 25  # feed (port cores): chunk ring between the port reader and the port sender
CB_FEED_POLL = 26  # feed (port cores): port reader's private landing of the workers' SEM_FEED_ROWS (16 B each)
CB_FEED_YSCR = 27  # feed, T % 32 != 0 (port cores): port reader's private y tiles of a shifted chunk (2 per tile)
CB_FEED_LAND = 28  # feed, T % 32 != 0 (port cores): landing of the shared D tile rows (compact contributor blocks)

# Semaphore ids (R2)
SEM_LEADER_COUNT = 0  # partials landed in my leader slots
SEM_ROOT_COUNT = 1  # chunk sums landed in my root slots
SEM_READY_BITS = 2  # bit (t mod 32): leader(t) has freed its slots for row t
SEM_RSTD_COUNT = 3  # norm: rstd rows landed in my cb_rstd_land slots
SEM_FEED_ROWS = 4  # feed (workers): my rows whose staged y has landed (polled by the ports)

NUM_CBS = 64


def _split(total, parts):
    """Ragged +-1 split: the first `total % parts` parts take one more. Returns [(start, count)]."""
    base, extra = divmod(total, parts)
    out, start = [], 0
    for p in range(parts):
        cnt = base + (1 if p < extra else 0)
        out.append((start, cnt))
        start += cnt
    return out


def fabric_ports_reserved(device):
    """Whether the feed's port cores are carved out of the worker list: whenever the fabric is active on a
    multi-chip mesh — never by the feed kwargs, so X' / stats / hc of a feed call are the bits of the same call
    without the feed on the same mesh (op_design.md "Worker set L")."""
    num = device.get_num_devices() if hasattr(device, "get_num_devices") else 1
    return num > 1 and ttnn.get_fabric_config() != ttnn.FabricConfig.DISABLED


def port_cores(gx, gy):
    """The 2 * MAX_FEED_LINKS reserved port cores: logical row 0 (next to the Ethernet row), centred (the
    fabric_ring_exchange / distributed_rms_norm placement); port (slot k, link l) = cores[2 * l + k]."""
    num = 2 * MAX_FEED_LINKS
    if gx * gy <= num:
        raise ValueError(f"mhc_post: a {gx}x{gy} max_grid leaves no worker beside the {num} fabric port cores")
    if gx >= num:
        x0 = gx // 2 - MAX_FEED_LINKS
        return [ttnn.CoreCoord(x0 + i, 0) for i in range(num)]
    return [ttnn.CoreCoord(i % gx, i // gx) for i in range(num)]


def worker_list(device, max_grid):
    """L: the max_grid rectangle in row-major order, minus the reserved port cores when the fabric is active.
    Returns (L, ports or None)."""
    grid = device.compute_with_storage_grid_size()
    gx, gy = grid.x, grid.y
    if max_grid is not None:
        gx, gy = min(gx, max_grid.x), min(gy, max_grid.y)
    ports = port_cores(gx, gy) if fabric_ports_reserved(device) else None
    taken = {(c.x, c.y) for c in ports} if ports else set()
    return [ttnn.CoreCoord(x, y) for y in range(gy) for x in range(gx) if (x, y) not in taken], ports


def choose_split(t_tiles, c_tiles, num_workers, unit_bytes, has_w, n, C):
    """op_design.md chooser: minimise est = max(total DRAM / DRAM_BW, per-core time); on an est tie (DRAM-bound,
    e.g. plain mode where total does not depend on R) the balanced split (smaller max core bytes) wins; then the
    smaller R. Inputs are (T, C, dtypes via unit_bytes, has_W, |L|) only — never mode / norm / feed — so the
    stats partial grouping is identical across the fused modes.

    Per-core time: plain = max core bytes / CORE_BW (DRAM-fed streaming). With W the fused pass is compute-bound
    per core: rows * (ROW_NS_W + cols * UNIT_NS_W), and only splits whose column range fits one block pass
    (cols <= MAX_BLOCK_COLS_W) are candidates when any exists (a multi-pass / ragged row measured slower: T160
    C5120 R=5 127 us vs R=2 110 us)."""
    cands = []
    for R in range(1, min(t_tiles, num_workers) + 1):
        Q = min(num_workers // R, c_tiles)
        if Q < 1:
            break
        rows, cols = -(-t_tiles // R), -(-c_tiles // Q)
        total = t_tiles * c_tiles * unit_bytes + has_w * R * n * C * 128
        max_core = rows * cols * unit_bytes + has_w * n * cols * 4096
        if has_w:
            core_s = rows * (ROW_NS_W + cols * UNIT_NS_W) * 1e-9
        else:
            core_s = max_core / CORE_BW
        est = max(total / DRAM_BW, core_s)
        cands.append(((est, max_core, R), R, Q, cols))
    if has_w and any(c[3] <= MAX_BLOCK_COLS_W for c in cands):
        cands = [c for c in cands if c[3] <= MAX_BLOCK_COLS_W]
    best = min(cands, key=lambda c: c[0])
    return best[1], best[2]


def dest_fp32_slots():
    return 8 if DST_FULL_SYNC else 4


def choose_block_cols(core_c_tiles, bytes_per_block_col, mix_fixed_slots=MIX_FIXED_SLOTS, cap=None):
    """block_cols: the whole per-core column range, capped by L1 and by DEST (one accumulator per unit).
    (Balancing the passes to equal size measured worse at T=1024 C=1792: 243 vs 215 us; a ragged pass runs as
    one-unit blocks.)"""
    l1_cap = max(1, BLOCK_L1_BUDGET // max(1, bytes_per_block_col))
    dest_cap = dest_fp32_slots() - mix_fixed_slots
    return max(1, min(core_c_tiles, l1_cap, dest_cap, cap if cap is not None else dest_cap))


def _merged_ranges(cores):
    """A core list -> few rectangles (dispatch: never one range per core): contiguous x-runs per row, runs with
    the same x-span on consecutive rows merged. A row-major prefix of the rectangle gives <= 2 rectangles; with
    the port cores carved out of row 0, <= 4."""
    rows = {}
    for c in cores:
        rows.setdefault(c.y, []).append(c.x)
    runs = []  # (x0, x1, y)
    for y in sorted(rows):
        xs = sorted(rows[y])
        start = prev = xs[0]
        for x in xs[1:]:
            if x != prev + 1:
                runs.append((start, prev, y))
                start = x
            prev = x
        runs.append((start, prev, y))
    rects = []  # [x0, x1, y0, y1]
    for x0, x1, y in runs:
        for r in rects:
            if r[0] == x0 and r[1] == x1 and r[3] == y - 1:
                r[3] = y
                break
        else:
            rects.append([x0, x1, y, y])
    return ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1)) for x0, x1, y0, y1 in rects]
    )


def _cb(index, dtype, page_size, num_pages, core_ranges):
    return ttnn.CBDescriptor(
        total_size=page_size * num_pages,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_size)],
    )


def _f32_bits(v):
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def create_program_descriptor(
    f_tensor,
    x_tensor,
    post_tensor,
    comb_tensor,
    out_tensor,
    *,
    n,
    post_col,
    comb_col,
    math_fidelity,
    math_approx_mode,
    max_grid,
    mode,
    w_tensor=None,
    pre_tensor=None,
    stats_tensor=None,
    hc_tensor=None,
    y_tensor=None,
    hc=None,
    norm_tensor=None,
    norm_eps=1e-6,
    feed=None,
):
    assert mode in ("plain", "stats", "collapse"), mode
    device = x_tensor.device()
    has_w = mode in ("stats", "collapse")
    collapse = mode == "collapse"
    norm = norm_tensor is not None
    assert not norm or collapse, "norm needs collapse mode"

    # ---------------- geometry ----------------
    f_pad = list(f_tensor.padded_shape)
    t_tiles = 1
    for d in f_pad[:-1]:
        t_tiles *= d
    t_tiles //= TILE  # tensor_t_tiles: every leading dim folds into whole (padded) images
    c_tiles = f_pad[-1] // TILE  # tensor_c_tiles
    C = f_pad[-1]
    post_wt = post_tensor.padded_shape[-1] // TILE
    comb_wt = comb_tensor.padded_shape[-1] // TILE
    pre_wt = pre_tensor.padded_shape[-1] // TILE if collapse else 1

    f_page = f_tensor.buffer_page_size()
    x_page = x_tensor.buffer_page_size()
    out_page = out_tensor.buffer_page_size()
    coef_page = post_tensor.buffer_page_size()  # fp32 tile
    f32_page = 4096
    bf16_page = 2048
    n_coef = n + n * n + (n if collapse else 0)

    # ---------------- work split ----------------
    workers, ports = worker_list(device, max_grid)
    # the score sees (T, C, dtypes, has_W, |L|) only; with W the unit always counts the y tile (op_design.md)
    unit_bytes = f_page + 2 * n * x_page + (bf16_page if has_w else 0)
    R, Q = choose_split(t_tiles, c_tiles, len(workers), unit_bytes, int(has_w), n, C)
    row_ranges = _split(t_tiles, R)
    col_ranges = _split(c_tiles, Q)
    used = workers[: R * Q]
    core_ranges = _merged_ranges(used)
    max_core_c_tiles = col_ranges[0][1]

    # R2-W-streamed predicate (regime selection, exact): depends on the split only, never on the mode
    w_resident = int(n * max_core_c_tiles * f32_page <= W_RESIDENT_BUDGET)
    # block-scaled bytes per column tile: IN_BLOCKS units of F + n X in, OUT_BLOCKS units of n X' out;
    # R2 adds the bf16 X' of the pass, the y blocks (always counted with W: block_cols must not depend on
    # the mode, it orders the stats sums) and, W-streamed, the W passes in flight.
    bytes_per_block_col = IN_BLOCKS * (f_page + n * x_page) + OUT_BLOCKS * n * out_page
    mix_fixed = MIX_FIXED_SLOTS
    if has_w:
        mix_fixed = MIX_FIXED_SLOTS_STATS
        bytes_per_block_col += n * bf16_page + Y_BLOCKS * bf16_page
        if not w_resident:
            bytes_per_block_col += W_STREAM_BLOCKS * n * f32_page
    block_cols = choose_block_cols(
        max_core_c_tiles, bytes_per_block_col, mix_fixed, MAX_BLOCK_COLS_W if has_w else None
    )
    in_depth = IN_BLOCKS * block_cols  # units
    out_ring_tiles = OUT_BLOCKS * block_cols * n
    max_passes = -(-max_core_c_tiles // block_cols)
    num_chunks = -(-Q // COMBINE_FANIN)
    max_rows = row_ranges[0][1]
    max_rooted = -(-max_rows // Q)  # rows a core roots (root(t) = member (t - t0) mod Q), deferred
    core_cols_padded = max_passes * block_cols  # a core's columns incl. the filler units of a ragged pass
    hold_pages = max_rows * core_cols_padded if norm else 0
    if hold_pages * bf16_page > Y_HOLD_BUDGET:
        raise ValueError(
            f"mhc_post: norm holds {hold_pages} y tiles per core ({max_rows} rows x {core_cols_padded} cols), "
            f"over Y_HOLD_BUDGET; use a larger max_grid"
        )

    # ---------------- CBs ----------------
    n_raw = 3 if collapse else 2
    cbs = [
        _cb(CB_SUBLAYER_TILES, f_tensor.dtype, f_page, in_depth, core_ranges),
        _cb(CB_STREAM_TILES, x_tensor.dtype, x_page, n * in_depth, core_ranges),
        _cb(CB_COEF_RAW, ttnn.float32, coef_page, COEF_RAW_BUFS * n_raw, core_ranges),  # raw tiles per row
        _cb(CB_COEF_COLS, ttnn.float32, coef_page, n_coef * COEF_DEPTH, core_ranges),
        _cb(CB_MIXED_OUT_TILES, out_tensor.dtype, out_page, out_ring_tiles, core_ranges),
    ]
    w_pages = 0
    if has_w:
        w_pages = max_passes * n * block_cols if w_resident else W_STREAM_BLOCKS * n * block_cols
        cbs += [
            _cb(CB_MIXED_BF16, ttnn.bfloat16, bf16_page, n * block_cols, core_ranges),
            _cb(CB_W_SLICE, ttnn.float32, f32_page, w_pages, core_ranges),
            _cb(CB_SUMSQ_ACC, ttnn.float32, f32_page, 1, core_ranges),
            _cb(CB_PROJ_INTERM, ttnn.float32, f32_page, 1, core_ranges),
            _cb(CB_PROJ_OUT, ttnn.float32, f32_page, PARTIAL_DEPTH, core_ranges),
            # one page per row (sum X'^2), two with the norm (+ sum y_b^2)
            _cb(CB_SUMSQ_COL, ttnn.float32, f32_page, PARTIAL_DEPTH * (2 if norm else 1), core_ranges),
            _cb(CB_LEADER_SLOTS, ttnn.float32, f32_page, COMBINE_FANIN, core_ranges),
            _cb(CB_CHUNK_SUM, ttnn.float32, f32_page, 1, core_ranges),
            _cb(CB_ROOT_SLOTS, ttnn.float32, f32_page, num_chunks * max_rooted, core_ranges),
            _cb(CB_STATS, ttnn.float32, f32_page, 1, core_ranges),
        ]
    if collapse:
        cbs += [
            _cb(CB_HC_IN, ttnn.float32, f32_page, 1, core_ranges),
            _cb(CB_HC_OUT, ttnn.float32, f32_page, 1, core_ranges),
            _cb(CB_Y_OUT, ttnn.bfloat16, bf16_page, Y_BLOCKS * block_cols, core_ranges),
            _cb(CB_HC_TILE, ttnn.float32, f32_page, 1, core_ranges),
        ]
    if norm:
        cbs += [
            _cb(CB_Y_HOLD, ttnn.bfloat16, bf16_page, hold_pages, core_ranges),
            _cb(CB_YSQ_ACC, ttnn.float32, f32_page, 1, core_ranges),
            _cb(CB_RSTD_COL, ttnn.float32, f32_page, NORM_ROWS_IN_FLIGHT, core_ranges),
            _cb(CB_NORM_W, ttnn.float32, f32_page, core_cols_padded, core_ranges),
            _cb(CB_RSTD_LAND, ttnn.float32, RSTD_SLOT_BYTES, max_rows, core_ranges),
            _cb(CB_RSTD_SEND, ttnn.float32, RSTD_SLOT_BYTES, 1, core_ranges),
        ]

    # ---------------- per-core runtime args ----------------
    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    for g in range(R):
        t0, tn = row_ranges[g]
        for q in range(Q):
            c0, cn = col_ranges[q]
            core = used[g * Q + q]
            reader_rt[core.x][core.y] = [t0, tn, c0, cn]
            writer_rt[core.x][core.y] = [t0, tn, c0, cn, g, q]
            compute_rt[core.x][core.y] = [t0, tn, cn, q]

    # coordinates of every used core (row-major L order), packed x | y << 16 (common args of the writer)
    coord_table = []
    if has_w:
        for core in used:
            v = device.worker_core_from_logical_core(core)
            coord_table.append(v.x | (v.y << 16))

    # ---------------- kernels ----------------
    reader_ct = [
        CB_SUBLAYER_TILES,
        CB_STREAM_TILES,
        CB_COEF_RAW,
        CB_COEF_COLS,
        n,
        BLOCK_ROWS,
        block_cols,
        c_tiles,
        post_wt,
        comb_wt,
        int(has_w),
        int(collapse),
        w_resident,
        CB_W_SLICE,
        pre_wt,
        int(norm),
        CB_NORM_W,
        CB_RSTD_COL,
        CB_RSTD_LAND,
        SEM_RSTD_COUNT,
    ]
    pre_for_args = pre_tensor if collapse else post_tensor
    w_for_args = w_tensor if has_w else post_tensor
    reader_ct.extend(ttnn.TensorAccessorArgs(f_tensor).get_compile_time_args())
    reader_ct.extend(ttnn.TensorAccessorArgs(x_tensor).get_compile_time_args())
    reader_ct.extend(ttnn.TensorAccessorArgs(post_tensor).get_compile_time_args())
    reader_ct.extend(ttnn.TensorAccessorArgs(comb_tensor).get_compile_time_args())
    reader_ct.extend(ttnn.TensorAccessorArgs(pre_for_args).get_compile_time_args())
    reader_ct.extend(ttnn.TensorAccessorArgs(w_for_args).get_compile_time_args())
    nw_for_args = norm_tensor if norm else post_tensor
    reader_ct.extend(ttnn.TensorAccessorArgs(nw_for_args).get_compile_time_args())
    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "mhc_post_reader.cpp"),
        core_ranges=core_ranges,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        common_runtime_args=[
            f_tensor.buffer_address(),
            x_tensor.buffer_address(),
            post_tensor.buffer_address(),
            comb_tensor.buffer_address(),
            post_col,
            comb_col,
            pre_for_args.buffer_address(),
            w_for_args.buffer_address(),
            nw_for_args.buffer_address(),
        ],
        config=ttnn.ReaderConfigDescriptor(),
    )

    stats_for_args = stats_tensor if has_w else out_tensor
    hc_for_args = hc_tensor if collapse else out_tensor
    y_for_args = y_tensor if collapse else out_tensor
    writer_ct = [
        CB_MIXED_OUT_TILES,
        n,
        BLOCK_ROWS,
        block_cols,
        c_tiles,
        out_ring_tiles,
        int(has_w),
        int(collapse),
        Q,
        COMBINE_FANIN,
        CB_PROJ_OUT,
        CB_SUMSQ_COL,
        CB_LEADER_SLOTS,
        CB_CHUNK_SUM,
        CB_ROOT_SLOTS,
        CB_STATS,
        CB_HC_IN,
        CB_HC_OUT,
        CB_Y_OUT,
        CB_HC_TILE,
        SEM_LEADER_COUNT,
        SEM_ROOT_COUNT,
        SEM_READY_BITS,
        n * (n + 2),
        int(norm),
        CB_RSTD_LAND,
        CB_RSTD_SEND,
        SEM_RSTD_COUNT,
        int(feed is not None),
        SEM_FEED_ROWS,
    ]
    writer_ct.extend(ttnn.TensorAccessorArgs(out_tensor).get_compile_time_args())
    writer_ct.extend(ttnn.TensorAccessorArgs(stats_for_args).get_compile_time_args())
    writer_ct.extend(ttnn.TensorAccessorArgs(hc_for_args).get_compile_time_args())
    writer_ct.extend(ttnn.TensorAccessorArgs(y_for_args).get_compile_time_args())
    d_for_args = feed["d_tensor"] if feed is not None else y_for_args
    writer_ct.extend(ttnn.TensorAccessorArgs(d_for_args).get_compile_time_args())
    feed_common = []
    if feed is not None:
        if not collapse:
            raise ValueError("mhc_post: the feed needs collapse mode")
        ht_img = f_pad[-2] // TILE
        # the block the workers write straight into D: mine when my rows start on a D tile boundary, else none (G)
        own_block = feed["me"] if f_tensor.shape[-2] % TILE == 0 else feed["G"]
        feed_common = [d_for_args.buffer_address(), own_block, feed["G"], c_tiles // feed["G"], ht_img]
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "mhc_post_writer.cpp"),
        core_ranges=core_ranges,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        common_runtime_args=[
            out_tensor.buffer_address(),
            stats_for_args.buffer_address(),
            hc_for_args.buffer_address(),
            y_for_args.buffer_address(),
        ]
        + feed_common
        + coord_table,
        config=ttnn.WriterConfigDescriptor(),
    )

    unpack_modes = [ttnn.UnpackToDestMode.Default] * NUM_CBS
    # Every fp32 CB the compute reads only by copy / unpack-to-dest goes to DEST unpacked (never tf32 srcA/srcB).
    # CB_W_SLICE is NOT tagged: it is the matmul's in1 (FPU; W's 9-bit mantissa is exact in srcA).
    unpack_modes[CB_COEF_COLS] = ttnn.UnpackToDestMode.UnpackToDestFp32
    if f_tensor.dtype == ttnn.float32:
        unpack_modes[CB_SUBLAYER_TILES] = ttnn.UnpackToDestMode.UnpackToDestFp32
    if x_tensor.dtype == ttnn.float32:
        unpack_modes[CB_STREAM_TILES] = ttnn.UnpackToDestMode.UnpackToDestFp32
    if has_w:
        for cb in (CB_SUMSQ_ACC, CB_PROJ_INTERM, CB_LEADER_SLOTS, CB_ROOT_SLOTS):
            unpack_modes[cb] = ttnn.UnpackToDestMode.UnpackToDestFp32
    if collapse:
        unpack_modes[CB_HC_IN] = ttnn.UnpackToDestMode.UnpackToDestFp32
    if norm:
        for cb in (CB_YSQ_ACC, CB_RSTD_COL, CB_NORM_W):
            unpack_modes[cb] = ttnn.UnpackToDestMode.UnpackToDestFp32
    compute_cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=math_fidelity,
        fp32_dest_acc_en=True,
        math_approx_mode=math_approx_mode,
        dst_full_sync_en=DST_FULL_SYNC,
    )
    compute_cfg.unpack_to_dest_mode = unpack_modes
    compute_ct = [
        CB_SUBLAYER_TILES,
        CB_STREAM_TILES,
        CB_COEF_COLS,
        CB_MIXED_OUT_TILES,
        n,
        BLOCK_ROWS,
        block_cols,
        mix_fixed,
        int(has_w),
        int(collapse),
        w_resident,
        w_pages,
        Q,
        COMBINE_FANIN,
        CB_MIXED_BF16,
        CB_W_SLICE,
        CB_SUMSQ_ACC,
        CB_PROJ_INTERM,
        CB_PROJ_OUT,
        CB_SUMSQ_COL,
        CB_LEADER_SLOTS,
        CB_CHUNK_SUM,
        CB_ROOT_SLOTS,
        CB_STATS,
        CB_HC_IN,
        CB_HC_OUT,
        CB_Y_OUT,
        int(norm),
        CB_Y_HOLD,
        CB_YSQ_ACC,
        CB_RSTD_COL,
        CB_NORM_W,
    ]
    compute_common = []
    if collapse:
        a_pre, a_post, a_res = hc["scale"]
        compute_common = [_f32_bits(a_pre), _f32_bits(a_post), _f32_bits(a_res)]
        compute_common += [_f32_bits(b) for b in hc["base"]]
        compute_common += [
            _f32_bits(1.0 / float(hc["norm_width"])),
            _f32_bits(hc["norm_eps"]),
            _f32_bits(hc["hc_eps"]),
            int(hc["sinkhorn_iters"]),
            _f32_bits(1.0 / C),  # norm: rstd = rsqrt(sum y^2 * (1/C) + norm_eps)
            _f32_bits(norm_eps),
        ]
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "mhc_post_compute.cpp"),
        core_ranges=core_ranges,
        compile_time_args=compute_ct,
        runtime_args=compute_rt,
        common_runtime_args=compute_common,
        config=compute_cfg,
    )

    semaphores = []
    if has_w:
        semaphores = [
            ttnn.SemaphoreDescriptor(id=s, core_ranges=core_ranges, initial_value=0)
            for s in (SEM_LEADER_COUNT, SEM_ROOT_COUNT, SEM_READY_BITS, SEM_RSTD_COUNT)
        ]
    if feed is not None:
        semaphores.append(ttnn.SemaphoreDescriptor(id=SEM_FEED_ROWS, core_ranges=core_ranges, initial_value=0))
    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=semaphores, cbs=cbs)
    if feed is not None:
        _add_feed_ports(
            program, device, feed, y_for_args, R, row_ranges, Q, c_tiles, f_pad[-2] // TILE, f_tensor.shape[-2], used
        )
    return program


# ---------------------------------------------------------------------------
# Feed (regime R4 feed-DRAM-staged): plan per chip + the port kernels
# ---------------------------------------------------------------------------


def _num_dram_banks(device):
    g = device.dram_grid_size()
    return g.x * g.y


def feed_plan(mesh_device, cluster_axis):
    """Per chip coordinate: its group position p, the group's coordinates and, per destination position d != p,
    the port slot of the fabric's first hop (slot_of[d]; every destination reached through one direction shares a
    port). Readies travel the reverse way (slot_of of the destination chip)."""
    rows, cols = tuple(mesh_device.shape)
    G = (rows, cols)[cluster_axis]
    node = lambda rc: mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(*rc))  # noqa: E731
    chips = {}
    for r in range(rows):
        for c in range(cols):
            rc = (r, c)
            p = rc[cluster_axis]
            group = [(r, k) for k in range(G)] if cluster_axis == 1 else [(k, c) for k in range(G)]
            dirs = {}
            for d in range(G):
                if d == p:
                    continue
                dirv = ttnn.get_eth_forwarding_direction(node(rc), node(group[d]))
                if dirv is None:
                    raise ValueError(f"mhc_post: no fabric route from {rc} to {group[d]}")
                dirs[d] = dirv
            # one port slot per distinct first-hop direction of THIS chip: a logical 1 x G group may sit on a 2-D
            # physical layout (a QuietBox 2's 2 x 2), so the directions are not E / W everywhere
            distinct = sorted(set(dirs.values()))
            if len(distinct) > 2:
                raise ValueError(f"mhc_post: the feed group leaves {rc} in {len(distinct)} directions (max 2 ports)")
            slot_of = {d: distinct.index(v) for d, v in dirs.items()}
            chips[rc] = dict(p=p, group=group, slot_of=slot_of, node=node(rc))
    return G, chips


def feed_shared_rows(T, G):
    """D tile rows (of one image, G * T rows) that hold rows of more than one chip: none when T % 32 == 0."""
    htd = -(-G * T // TILE)
    return [r for r in range(htd) if (TILE * r) // T != (min(TILE * r + TILE, G * T) - 1) // T]


def _add_feed_ports(program, device, feed, y_tensor, R, row_ranges, Q, c_tiles, ht_img, T, workers):
    """The 2 * links port cores of this chip: reader (RISCV_1) stages chunks from y_output, sender (RISCV_0) sends
    them over the fabric. Port (slot k, link l) sends to the destinations whose first hop leaves through slot k.
    T % 32 != 0 (Refinement 2b): the ports of slot 0 also place my own block in my D (local writes), and port
    (0, 0) is the landing port that merges and writes the shared D tile rows."""
    plan, me_rc, links = feed["chips"], feed["coord"], feed["links"]
    me = plan[me_rc]
    G, p = feed["G"], me["p"]
    ports = feed["ports"]  # {(slot, link): logical core}
    port_cores_used = feed["port_list"]
    port_crs = _merged_ranges(port_cores_used)
    d_tensor = feed["d_tensor"]
    page = d_tensor.buffer_page_size()
    run_pages = max(1, int(ttnn.get_tt_fabric_max_payload_size_bytes()) // page)
    ring_chunks = 2 * FEED_CHUNKS_IN_FLIGHT
    assert 1 <= FEED_READ_BATCH <= ring_chunks - FEED_CHUNKS_IN_FLIGHT, "reader batch + sender in flight <= ring"
    num_banks = _num_dram_banks(device)
    max_rows = row_ranges[0][1]
    t_tiles = sum(cnt for _, cnt in row_ranges)
    rows_base, rows_extra = divmod(t_tiles, R)
    blk = c_tiles // G
    shifted = T % TILE != 0
    nsh = len(feed_shared_rows(T, G))
    nimg = t_tiles // ht_img
    land_tiles = nimg * nsh * blk
    if land_tiles * page > FEED_LAND_BUDGET:
        raise ValueError(
            f"mhc_post: the feed's shared D tile rows ({land_tiles} tiles) exceed the landing budget "
            f"({FEED_LAND_BUDGET} B)"
        )
    geom = [c_tiles, blk, ht_img, G, p, R, rows_base, rows_extra, T, -(-G * T // TILE), nimg, nsh]

    worker_xy = [device.worker_core_from_logical_core(c) for c in workers]
    worker_xy = [v.x | (v.y << 16) for v in worker_xy]
    program.cbs = list(program.cbs) + [
        _cb(CB_FEED_CHUNKS, d_tensor.dtype, page, ring_chunks * run_pages, port_crs),
        _cb(CB_FEED_POLL, ttnn.float32, 16 * len(workers), 1, port_crs),
    ]
    if shifted:
        program.cbs = list(program.cbs) + [_cb(CB_FEED_YSCR, d_tensor.dtype, page, 2 * run_pages, port_crs)]
    if land_tiles > 0:  # same CB on every port core and every chip's program: the same address everywhere
        program.cbs = list(program.cbs) + [_cb(CB_FEED_LAND, d_tensor.dtype, page, land_tiles, port_crs)]
    ready_addr, arrival_addr = feed["semaphores"]
    land = device.worker_core_from_logical_core(ports[(0, 0)])
    reader_rt, sender_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    layout = []
    for l in range(links):
        for k in (0, 1):
            core = ports[(k, l)]
            v = device.worker_core_from_logical_core(core)
            remote = [d for d, sl in me["slot_of"].items() if sl == k]
            dests = remote + ([p] if shifted and k == 0 else [])
            # one stream per (source chip, link) whose first hop toward me leaves through slot k
            expect_in = sum(
                1 for rc, ch in plan.items() if ch["group"] == me["group"] and rc != me_rc and ch["slot_of"][p] == k
            )
            reader_rt[core.x][core.y] = geom + [l, y_tensor.buffer_address(), len(dests)] + dests + worker_xy
            args = geom + [
                l,
                d_tensor.buffer_address(),
                ready_addr,
                arrival_addr,
                v.x,
                v.y,
                expect_in,
                land.x,
                land.y,
                int((k, l) == (0, 0)),
                len(dests),
                len(remote),
                int(len(remote) > 0),
            ]
            for d in dests:  # hops 0: my own block (local writes)
                dn = plan[me["group"][d]]["node"]
                args += [d, int(dn.chip_id), int(dn.mesh_id), abs(d - p)]
            for d in remote:  # ready to the port of chip d that sends to me (its slot toward me, same link)
                dn = plan[me["group"][d]]["node"]
                tc = device.worker_core_from_logical_core(ports[(plan[me["group"][d]]["slot_of"][p], l)])
                args += [int(dn.chip_id), int(dn.mesh_id), abs(d - p), tc.x, tc.y]
            if remote:
                first = plan[me["group"][remote[0]]]["node"]
                link_ids = ttnn.get_forwarding_link_indices(me["node"], first)
                if len(link_ids) <= l:
                    raise ValueError(
                        f"mhc_post: feed_num_links={links} but {me_rc} -> {me['group'][remote[0]]} has "
                        f"{len(link_ids)} usable link(s)"
                    )
                args += list(ttnn.setup_fabric_connection(me["node"], first, link_ids[l], program, core))
            sender_rt[core.x][core.y] = args
            layout.append((k, l, len(dests), expect_in))
    feed["layout"] = layout
    port_ct = [CB_FEED_CHUNKS, page, run_pages, num_banks, links, FEED_WAVE_ROWS, max_rows]
    reader_ct = (
        port_ct
        + [Q, SEM_FEED_ROWS, CB_FEED_POLL, CB_FEED_YSCR, FEED_READ_BATCH, ring_chunks]
        + list(ttnn.TensorAccessorArgs(y_tensor).get_compile_time_args())
    )
    sender_ct = (
        port_ct
        + [FEED_CHUNKS_IN_FLIGHT, ring_chunks, CB_FEED_LAND]
        + list(ttnn.TensorAccessorArgs(d_tensor).get_compile_time_args())
    )
    program.kernels = list(program.kernels) + [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "mhc_post_port_reader.cpp"),
            core_ranges=port_crs,
            compile_time_args=reader_ct,
            runtime_args=reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "mhc_post_port_sender.cpp"),
            core_ranges=port_crs,
            compile_time_args=sender_ct,
            runtime_args=sender_rt,
            config=ttnn.WriterConfigDescriptor(),
        ),
    ]
