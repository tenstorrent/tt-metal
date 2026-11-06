# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""mhc_post — ProgramDescriptor (plain-2d, stats-2d-tree and collapse-2d-tree regimes).

Work split (op_design.md → Work Distribution): the (token-tile row × column tile) plane is
block-split 2-D into R row groups × Q column groups by `choose_split`; each core's per-pass
block (`1 row × block_c c-tiles × n streams`) comes from `choose_block_c`. Both are pure
host functions and the single sources of R, Q and block_c. Every depth / budget below is a
single-source host constant feeding both the CB capacities and the kernels' CT args.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

# ---------------------------------------------------------------------------
# Single-source constants (op_design.md → Work Distribution → Constants)
# ---------------------------------------------------------------------------
N_STREAMS = 4  # n = hc_mult (DeepSeek-V4 / V4.1)
TILE = 32
FP32_TILE_BYTES = 4096

L1_CB_BUDGET = 900 * 1024
# Block rule (Refinement 3, measured on device): block_c = min(L1 cap, the coarsest block leaving
# >= MIN_PIPELINE_BLOCKS blocks in the core's pipeline, BLOCK_C_TARGET[stream tile bytes]). The target
# caps the block at the size that keeps reader / compute / writer overlapped; it differs by stream
# dtype. Measured:
#   bf16 (2 KiB tiles), collapse HiFi2 R=5/Q=22: block_c 1 / 2 / 4 = 137 / 136 / 186 us (T160),
#     403 / 396 / 483 us (T640); plain T640/C7168: 1 / 2 / 3 / 4 = 305 / 312 / 420 / 427 us.
#   fp32 (4 KiB tiles), plain: T640/C7168 block_c 2 / 3 / 4 / 8 = 617 / 457 / 479 / 517 us;
#     T4096/C1792 996 / 738 / 754 / 776 us; T640/C1792 150 / 146 / 157 (4) us; T160/C5120 (the
#     MIN_PIPELINE_BLOCKS bound gives 2) 2 / 3 = 94 / 109 us.
MIN_PIPELINE_BLOCKS = 4
BLOCK_C_TARGET = {2048: 2, 4096: 3}  # keyed by the X stream tile size (bf16 / fp32)
W_RESIDENT_BUDGET = 384 * 1024  # fused: resident W slice bound (choose_split constraint)
# Fused split objective: a core's fixed per-row cost (coefficient expansion, ss seed / finish, the
# partial ship and combine, the rotating hc tail) in byte-equivalents of the per-c-tile stream cost.
# Fit on device (collapse, T160 / T640, C5120, 7 + 6 splits): ~20 us per row vs ~11 us per
# (row, c-tile) unit of 18 KiB -> ~33 KiB. Plain mode has no combine and keeps the pure byte cost.
FUSED_ROW_OVERHEAD_BYTES = 33 * 1024
COMBINE_FANINS = (8, 16, 32)  # fused: partials per leader, smallest that fits MAX_SEMAPHORES
COMBINE_DEPTH = 2  # fused: gather slot sets in flight per consumer
COMBINE_LAG = 1  # fused: a row is combined after the mix of row + LAG (root: + 2*LAG)
MAX_SEMAPHORES = 16  # per-core semaphore ids available to one program
# Projection passes. The FPU dot product rounds every 16-term (face) partial sum to ~tf32 at any
# fidelity (measured on device: two exact products in one face already lose ~2^-11); a single
# product per face lands in fp32 DEST exactly. EXACT = 16 column-masked passes (columns j, j+16),
# each an FPU matmul with one product per face -> fp32-exact projection. Used where the caller's
# fidelity makes every product exact (HiFi3 / HiFi4); HiFi2 / LoFi (product-limited anyway, the
# DeepSeek-V4.1 setting) run the single FPU matmul.
PROJ_EXACT_PASSES = 16
PROJ_EXACT_FIDELITIES = (ttnn.MathFidelity.HiFi3, ttnn.MathFidelity.HiFi4)
IO_DEPTH = 2  # cb_f_tiles / cb_x_tiles depth, in blocks
OUT_DEPTH = 2  # cb_out_tiles depth, in blocks (design: 1; 2 measured -8..10% on DM-bound shapes)
COEF_DEPTH = 2  # cb_coef_cols depth, in rows (2: the reader expands row r+1 while compute mixes row r)
BLOCK_ROWS = 1  # rows per block (see op_design.md "Why block_rows = 1")

# CB indices (semantic names; any distinct values)
CB_COEF_SRC = 0
CB_COEF_COLS = 1
CB_F_TILES = 2
CB_X_TILES = 3
CB_OUT_TILES = 4
# fused (stats / collapse) only
CB_XPRIME_MM = 5
CB_W_SLICE = 6
CB_PROJ_BLOCK = 7
CB_PROJ_RUN = 8
CB_SS_RUN = 9
CB_SS_ROW = 10
CB_PARTIAL = 11
CB_GATHER = 12
CB_ROOT_GATHER = 13
CB_LEAD_OUT = 14
CB_COMB_ACC = 15
CB_STATS_ROW = 16
CB_PROJ_MASKS = 17  # exact projection: PROJ_EXACT_PASSES bf16 0/1 column masks (reader-built)
CB_XPRIME_MASKED = 18  # exact projection: one masked X' block
# collapse only
CB_Y_TILES = 19  # y = sum_i P_i X'_i, one block (bf16), pushed per tile
CB_HC_LANES = 20  # root: stats row permuted into the hc lane layout (writer -> compute)
CB_HC_OUT = 21  # root: hc program result lanes (compute -> writer)
CB_HC_ROW = 22  # root: writer-private landing tile for the inverse permutation
HC_LANE_TILES = 2  # cb_hc_lanes / cb_hc_out capacity (mhc_post_hc_sfpu.hpp LANE_CB_PAGES)
# collapse: iterations between a row's stats combine and its hc program on the compute side.
# 1 decouples the compute from the writer's lane permute (which sits behind the writer's cross-core
# sends); 0 runs it right after the combine. Measured equal at T160 / T640 (C5120, HiFi2).
HC_LAG = 1


def _tile_bytes(dtype):
    return ttnn.tile_size(dtype)


def _n_src(mode, coef_source):
    """Coefficient source pages per row: packed → 1, separate → 2 (+1 separate P in collapse)."""
    if coef_source == "packed":
        return 1
    return 3 if mode == "collapse" else 2


def _n_coef_cols(mode, n=N_STREAMS):
    return n + n * n + (n if mode == "collapse" else 0)


# ---------------------------------------------------------------------------
# choose_split / choose_block_c (host, pure)
# ---------------------------------------------------------------------------


def _bank_spread(Rt, Ct, R, Q, banks, n=N_STREAMS):
    """Distinct DRAM banks the cores' first stream pages land on (interleaved page p -> bank p % banks).
    Every core walks its range in the same order, so cores whose starts share a bank keep sharing
    banks block after block."""
    starts = {((g_r * Rt // R) * n * Ct + g_c * Ct // Q) % banks for g_r in range(R) for g_c in range(Q)}
    return min(len(starts), banks)


def choose_split(Rt, Ct, *, f_tile, x_tile, n_src, G, fused=False, n=N_STREAMS, dram_banks=1):
    """(R, Q) minimizing the busiest core's cost. Tie-break: more distinct DRAM start banks
    (`_bank_spread`), then smaller R·Q, then smaller Q.

    Cost = busiest core's streamed bytes (F, X, X' and its coefficient source tiles), plus, in the
    fused modes, FUSED_ROW_OVERHEAD_BYTES per row the core owns. The fused W slice is not in the cost:
    it streams under the first row's mix (reader issues it per block) and is only a residency
    constraint. Among equal-cost splits, bank spread first: plain T160/C5120 (Ct = 160 = 0 mod 8)
    Q=20 puts every core's start on bank 0 -> 84.9 us; Q=21 / Q=22 spread them -> 74.0 / 73.4 us at
    the same busiest-core work. Then the fewest cores: each extra core re-reads the shared
    coefficient pages (plain fp32 T1/C7168 separate coefficients: Q=75 48.7 us, Q=110 53.2 us).

    Fused modes keep the core's W slice resident. The preferred bound is W_RESIDENT_BUDGET; when
    no split of a small grid (max_grid) meets it, the bound relaxes to "the whole fused inventory
    fits L1_CB_BUDGET at block_c = 1" — same function for stats and collapse (mode-invariant), and a
    no-op wherever the preferred bound is feasible."""
    unit_bytes = f_tile + 2 * n * x_tile
    row_bytes = n_src * FP32_TILE_BYTES + (FUSED_ROW_OVERHEAD_BYTES if fused else 0)

    def w_fits_budget(ct_max, Q):
        return n * ct_max * FP32_TILE_BYTES <= W_RESIDENT_BUDGET

    def w_fits_l1(ct_max, Q):
        return fused_fixed_bytes(ct_max, Q) + fused_per_c_bytes(f_tile, x_tile) <= L1_CB_BUDGET

    for feasible in (w_fits_budget, w_fits_l1) if fused else (lambda ct_max, Q: True,):
        best = None
        for R in range(1, min(Rt, G) + 1):
            rt_max = -(-Rt // R)
            for Q in range(1, min(Ct, G // R) + 1):
                ct_max = -(-Ct // Q)
                if not feasible(ct_max, Q):
                    continue
                if fused and combine_fanin(Q) is None:
                    continue
                cost = rt_max * (ct_max * unit_bytes + row_bytes)
                key = (cost, -_bank_spread(Rt, Ct, R, Q, dram_banks), R * Q, Q)
                if best is None or key < best[0]:
                    best = (key, R, Q)
        if best is not None:
            return best[1], best[2]
    raise RuntimeError(f"mhc_post: no (R, Q) split of {G} cores keeps the {Ct}-column W slice resident in L1")


def plain_fixed_bytes(n_src, n_coef_cols):
    return n_src * FP32_TILE_BYTES + n_coef_cols * COEF_DEPTH * FP32_TILE_BYTES


def plain_per_c_bytes(f_tile, x_tile, n=N_STREAMS):
    return IO_DEPTH * f_tile + IO_DEPTH * n * x_tile + OUT_DEPTH * n * x_tile


def _n_semaphores(Q, fanin):
    return 2 * COMBINE_DEPTH + 1 + -(-Q // fanin)


def combine_fanin(Q):
    """Partials per leader: the smallest COMBINE_FANINS entry whose tree fits MAX_SEMAPHORES (None if
    none does). Smaller fan-in = more leaders = more rotating roots for the hc tail (measured
    R=5/Q=22 collapse: fan-in 8 / 16 / 32 = 136 / 140 / 170 us at T160)."""
    for fanin in COMBINE_FANINS:
        if _n_semaphores(Q, fanin) <= MAX_SEMAPHORES:
            return fanin
    return None


def combine_geometry(Q):
    """(slots, num_groups): leader gather slots and the number of leader groups (tree width)."""
    fanin = combine_fanin(Q) or COMBINE_FANINS[-1]
    return min(fanin, Q), -(-Q // fanin)


def fused_fixed_bytes(core_c_max, Q, n=N_STREAMS):
    """Fused-mode constant CBs, sized against the COLLAPSE inventory for both fused modes so
    (R, Q, block_c) — and therefore the stats summation order — is mode-invariant."""
    slots, ng = combine_geometry(Q)
    n_src_max = _n_src("collapse", "separate")
    tiles = (
        n_src_max  # cb_coef_src
        + _n_coef_cols("collapse") * COEF_DEPTH  # cb_coef_cols
        + n * core_c_max  # cb_w_slice (resident)
        + 1  # cb_proj_block
        + 2  # cb_proj_run
        + 2  # cb_ss_run
        + 1  # cb_ss_row
        + 1  # cb_partial
        + slots * COMBINE_DEPTH  # cb_gather
        + (ng * COMBINE_DEPTH if ng > 1 else 0)  # cb_root_gather
        + 1  # cb_lead_out
        + 2  # cb_comb_acc
        + 1  # cb_stats_row
        + 3
        + 2
        + 1  # collapse: hc lanes, hc out, hc row
    )
    return tiles * FP32_TILE_BYTES


def fused_per_c_bytes(f_tile, x_tile, n=N_STREAMS):
    """Per c-tile of block_c: F, X (IO_DEPTH), X' out (OUT_DEPTH), cb_xprime_mm (bf16), y (collapse)."""
    bf16 = 2048
    return plain_per_c_bytes(f_tile, x_tile, n) + 2 * n * bf16 + bf16  # + cb_xprime_masked


def choose_block_c(core_c_max, core_rows_max, *, fixed_bytes, per_c_bytes, x_tile):
    block_c_cap = max(1, (L1_CB_BUDGET - fixed_bytes) // per_c_bytes)
    nb = max(-(-core_c_max // block_c_cap), min(core_c_max, -(-MIN_PIPELINE_BLOCKS // core_rows_max)))
    return max(1, min(-(-core_c_max // nb), BLOCK_C_TARGET.get(x_tile, 2)))


# ---------------------------------------------------------------------------
# Program descriptor
# ---------------------------------------------------------------------------


def _core_ranges(num_cores, gx):
    """Row-wise enumeration k → (k % gx, k // gx), as a CoreRangeSet."""
    ranges = []
    full_rows = num_cores // gx
    if full_rows:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, full_rows - 1)))
    rem = num_cores % gx
    if rem:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full_rows), ttnn.CoreCoord(rem - 1, full_rows)))
    return ttnn.CoreRangeSet(ranges)


def _cb(index, dtype, page_size, num_pages, core_ranges):
    return ttnn.CBDescriptor(
        total_size=num_pages * page_size,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_size)],
    )


def _same_buffer(a, b):
    if a is b:
        return True
    try:
        return a.buffer_address() == b.buffer_address()
    except Exception:  # noqa: BLE001
        return False


def _coef_sources(post, post_col, comb, comb_col, pre):
    """Distinct coefficient source pages per row: (n_src_read, comb_slot, pre_slot). Post reads into
    slot 0; a window on the same buffer and tile column as an earlier one shares its slot (a packed
    mixes page holding pre, post and comb is ONE read, never sliced)."""
    keys = [(post, post_col // TILE)]
    slots = []
    for t, col in ((comb, comb_col), (pre, 0)):
        if t is None:
            slots.append(0)
            continue
        for i, (kt, kc) in enumerate(keys):
            if kc == col // TILE and _same_buffer(kt, t):
                slots.append(i)
                break
        else:
            keys.append((t, col // TILE))
            slots.append(len(keys) - 1)
    return len(keys), slots[0], slots[1]


def _f32_bits(v):
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def hc_params(hc_scale, hc_base, hc_norm_width, hc_norm_eps, hc_eps, hc_sinkhorn_iters):
    """The hc program's runtime parameter block (mhc_post_hc_sfpu.hpp P_*): b_0..b_23, a_pre, a_post,
    a_res, 1/width (computed in double), norm_eps, hc_eps as fp32 bits, then the iteration count."""
    vals = [_f32_bits(b) for b in hc_base] + [_f32_bits(a) for a in hc_scale]
    vals += [_f32_bits(1.0 / float(hc_norm_width)), _f32_bits(hc_norm_eps), _f32_bits(hc_eps)]
    return vals + [int(hc_sinkhorn_iters)]


def create_program_descriptor(
    input_tensor,
    residual,
    post,
    comb,
    output_tensor,
    *,
    compute_config,
    post_col=0,
    comb_col=0,
    max_grid=None,
    stats_weight=None,
    stats_output=None,
    collapse_pre=None,
    hc_output=None,
    y_output=None,
    hc_param_block=None,
):
    n = N_STREAMS
    mode = "collapse" if collapse_pre is not None else ("stats" if stats_weight is not None else "plain")
    fused = mode != "plain"
    collapse = mode == "collapse"
    packed = _same_buffer(post, comb)
    coef_source = "packed" if packed else "separate"
    x_shape = list(residual.shape)
    T, nC = x_shape[-2], x_shape[-1]
    C = nC // n
    batch = math.prod(x_shape[:-2]) if len(x_shape) > 2 else 1
    Rt = batch * (-(-T // TILE))  # token-tile rows, per image ceil(T/32)
    Ct = C // TILE
    post_wt = -(-list(post.shape)[-1] // TILE)  # tile columns of the post tensor
    comb_wt = -(-list(comb.shape)[-1] // TILE)
    pre_wt = -(-list(collapse_pre.shape)[-1] // TILE) if collapse else 1

    # --- grid (single source G) ---
    device = residual.device()
    dev_grid = device.compute_with_storage_grid_size()
    gx, gy = dev_grid.x, dev_grid.y
    if max_grid is not None:
        gx, gy = max(1, min(max_grid.x, gx)), max(1, min(max_grid.y, gy))
    G = gx * gy
    dram_grid = device.dram_grid_size()
    dram_banks = max(1, dram_grid.x * dram_grid.y)

    f_tile = _tile_bytes(input_tensor.dtype)
    x_tile = _tile_bytes(residual.dtype)
    # Coefficient source pages actually read per row: a packed mixes tensor whose pre / post / comb
    # windows share one tile column is a single page (read once, never sliced).
    n_src_read, comb_slot, pre_slot = _coef_sources(post, post_col, comb, comb_col, collapse_pre)
    n_coef_cols = _n_coef_cols(mode)

    # Split / block sizing: the fused modes cost and size against the stats/collapse inventory
    # (mode-invariant), plain against its own.
    R, Q = choose_split(
        Rt,
        Ct,
        f_tile=f_tile,
        x_tile=x_tile,
        n_src=_n_src("stats" if fused else mode, coef_source),
        G=G,
        fused=fused,
        dram_banks=dram_banks,
    )
    core_rows_max = -(-Rt // R)
    core_c_max = -(-Ct // Q)
    if fused:
        fixed_bytes = fused_fixed_bytes(core_c_max, Q)
        per_c_bytes = fused_per_c_bytes(f_tile, x_tile)
    else:
        fixed_bytes = plain_fixed_bytes(_n_src(mode, coef_source), n_coef_cols)
        per_c_bytes = plain_per_c_bytes(f_tile, x_tile)
    block_c = choose_block_c(core_c_max, core_rows_max, fixed_bytes=fixed_bytes, per_c_bytes=per_c_bytes, x_tile=x_tile)
    num_cores = R * Q
    core_ranges = _core_ranges(num_cores, gx)

    slots, num_groups = combine_geometry(Q)
    fanin = combine_fanin(Q) or COMBINE_FANINS[-1]
    n_sems = _n_semaphores(Q, fanin)
    if fused and n_sems > MAX_SEMAPHORES:
        raise RuntimeError(f"mhc_post: Q={Q} needs {n_sems} semaphores (> {MAX_SEMAPHORES})")
    SEM_GATHER = 0  # [COMBINE_DEPTH] leader gather arrivals, per slot set
    SEM_ROOT_GATHER = COMBINE_DEPTH  # [COMBINE_DEPTH] root gather arrivals, per slot set
    SEM_LEAD_CREDIT = 2 * COMBINE_DEPTH  # grants from my group's leader
    SEM_ROOT_CREDIT = 2 * COMBINE_DEPTH + 1  # [num_groups] grants from each root

    # --- circular buffers (capacities derive from the knobs above) ---
    cbs = [
        _cb(CB_COEF_SRC, ttnn.float32, FP32_TILE_BYTES, n_src_read, core_ranges),
        _cb(CB_COEF_COLS, ttnn.float32, FP32_TILE_BYTES, n_coef_cols * COEF_DEPTH, core_ranges),
        _cb(CB_F_TILES, input_tensor.dtype, f_tile, block_c * IO_DEPTH, core_ranges),
        _cb(CB_X_TILES, residual.dtype, x_tile, n * block_c * IO_DEPTH, core_ranges),
        _cb(CB_OUT_TILES, output_tensor.dtype, x_tile, n * block_c * OUT_DEPTH, core_ranges),
    ]
    fp32_copy_cbs = []  # Float32 CBs read only by CopyTile -> UnpackToDestFp32
    proj_passes = PROJ_EXACT_PASSES if compute_config.math_fidelity in PROJ_EXACT_FIDELITIES else 1
    if fused:
        f32 = lambda idx, pages: _cb(idx, ttnn.float32, FP32_TILE_BYTES, pages, core_ranges)  # noqa: E731
        cbs += [
            _cb(CB_XPRIME_MM, ttnn.bfloat16, ttnn.tile_size(ttnn.bfloat16), n * block_c, core_ranges),
            f32(CB_W_SLICE, n * core_c_max),
            f32(CB_PROJ_BLOCK, 1),
            f32(CB_PROJ_RUN, 2),
            f32(CB_SS_RUN, 2),
            f32(CB_SS_ROW, 1),
            f32(CB_PARTIAL, 2),
            f32(CB_GATHER, slots * COMBINE_DEPTH),
            f32(CB_LEAD_OUT, 1),
            f32(CB_COMB_ACC, 2),
            f32(CB_STATS_ROW, 1),
        ]
        if num_groups > 1:
            cbs.append(f32(CB_ROOT_GATHER, num_groups * COMBINE_DEPTH))
        if proj_passes > 1:
            bf16_tile = ttnn.tile_size(ttnn.bfloat16)
            cbs.append(_cb(CB_PROJ_MASKS, ttnn.bfloat16, bf16_tile, proj_passes, core_ranges))
            cbs.append(_cb(CB_XPRIME_MASKED, ttnn.bfloat16, bf16_tile, n * block_c, core_ranges))
        fp32_copy_cbs = [CB_PROJ_BLOCK, CB_PROJ_RUN, CB_SS_RUN, CB_GATHER, CB_ROOT_GATHER, CB_COMB_ACC]
        if collapse:
            cbs += [
                _cb(CB_Y_TILES, y_output.dtype, ttnn.tile_size(y_output.dtype), block_c, core_ranges),
                f32(CB_HC_LANES, HC_LANE_TILES),
                f32(CB_HC_OUT, HC_LANE_TILES),
                f32(CB_HC_ROW, 1),
            ]
            fp32_copy_cbs.append(CB_HC_LANES)

    fused_ct = [
        int(fused),
        CB_XPRIME_MM,
        CB_W_SLICE,
        CB_PROJ_BLOCK,
        CB_PROJ_RUN,
        CB_SS_RUN,
        CB_SS_ROW,
        CB_PARTIAL,
        CB_GATHER,
        CB_ROOT_GATHER,
        CB_LEAD_OUT,
        CB_COMB_ACC,
        CB_STATS_ROW,
        Q,
        fanin,
        COMBINE_DEPTH,
        COMBINE_LAG,
        SEM_GATHER,
        SEM_ROOT_GATHER,
        SEM_LEAD_CREDIT,
        SEM_ROOT_CREDIT,
        core_c_max,
        proj_passes,
        CB_PROJ_MASKS,
        CB_XPRIME_MASKED,
        int(collapse),
        CB_Y_TILES,
        CB_HC_LANES,
        CB_HC_OUT,
        CB_HC_ROW,
    ]
    w_tensor = stats_weight if fused else residual  # placeholder accessor when not fused
    s_tensor = stats_output if fused else output_tensor
    p_tensor = collapse_pre if collapse else post  # placeholder accessors when not collapse
    y_tensor = y_output if collapse else output_tensor
    h_tensor = hc_output if collapse else output_tensor

    # --- compile-time args ---
    reader_ct = [CB_COEF_SRC, CB_COEF_COLS, CB_F_TILES, CB_X_TILES, n, block_c, Ct, n_coef_cols, post_wt, comb_wt]
    reader_ct += [n_src_read, comb_slot, pre_slot, pre_wt] + fused_ct
    reader_ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader_ct.extend(ttnn.TensorAccessorArgs(residual).get_compile_time_args())
    reader_ct.extend(ttnn.TensorAccessorArgs(post).get_compile_time_args())
    reader_ct.extend(ttnn.TensorAccessorArgs(comb).get_compile_time_args())
    reader_ct.extend(ttnn.TensorAccessorArgs(w_tensor).get_compile_time_args())
    reader_ct.extend(ttnn.TensorAccessorArgs(p_tensor).get_compile_time_args())

    writer_ct = [CB_OUT_TILES, n, block_c, Ct] + fused_ct
    writer_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    writer_ct.extend(ttnn.TensorAccessorArgs(s_tensor).get_compile_time_args())
    writer_ct.extend(ttnn.TensorAccessorArgs(y_tensor).get_compile_time_args())
    writer_ct.extend(ttnn.TensorAccessorArgs(h_tensor).get_compile_time_args())

    compute_ct = [CB_COEF_COLS, CB_F_TILES, CB_X_TILES, CB_OUT_TILES, n, block_c, n_coef_cols] + fused_ct + [HC_LAG]

    # --- physical coordinates of every core of the program (combine transport) ---
    def phys(k):
        c = device.worker_core_from_logical_core(ttnn.CoreCoord(k % gx, k // gx))
        return [c.x, c.y]

    # --- runtime args (balanced contiguous 2-D split) ---
    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    w_addr = stats_weight.buffer_address() if fused else 0
    s_addr = stats_output.buffer_address() if fused else 0
    p_addr = collapse_pre.buffer_address() if collapse else 0
    y_addr = y_output.buffer_address() if collapse else 0
    h_addr = hc_output.buffer_address() if collapse else 0
    hc_rt = list(hc_param_block) if collapse else []
    for k in range(num_cores):
        x, y = k % gx, k // gx
        g_r, g_c = k // Q, k % Q
        row_start, row_end = (g_r * Rt) // R, ((g_r + 1) * Rt) // R
        c_start, c_end = (g_c * Ct) // Q, ((g_c + 1) * Ct) // Q
        num_rows, num_c = row_end - row_start, c_end - c_start
        reader_args = [
            input_tensor.buffer_address(),
            residual.buffer_address(),
            post.buffer_address(),
            comb.buffer_address(),
            row_start,
            num_rows,
            c_start,
            num_c,
            post_col,
            comb_col,
            g_r,
            g_c,
            w_addr,
            p_addr,
        ]
        writer_args = [output_tensor.buffer_address(), row_start, num_rows, c_start, num_c, g_r, g_c, s_addr]
        writer_args += [y_addr, h_addr]
        if fused:
            base = g_r * Q
            grp = g_c // fanin
            g0 = grp * fanin
            members = [v for q in range(g0, min(Q, g0 + fanin)) for v in phys(base + q)]
            leaders = [v for g in range(num_groups) for v in phys(base + g * fanin)]
            reader_args += members + leaders  # grants: to my group (leader) / to every leader (root)
            writer_args += leaders  # my leader = leaders[grp]; roots = leaders[r % num_groups]
        reader_rt[x][y] = reader_args
        writer_rt[x][y] = writer_args
        compute_rt[x][y] = [num_rows, num_c, g_c] + hc_rt

    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "mhc_post_reader.cpp"),
        core_ranges=core_ranges,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "mhc_post_writer.cpp"),
        core_ranges=core_ranges,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )

    # fp32 operands enter DEST exactly: every Float32 CB read by CopyTile / UnaryBcast is tagged
    # UnpackToDestFp32 — F / X when fp32, the coefficient columns, and the fused fp32 running /
    # gather tiles. unary_bcast keys its unpack-to-dest branch on the CB's unpack *dst* format,
    # which is TF32 unless the CB is tagged (measured: an untagged cb_coef_cols truncates every
    # coefficient by ~2^-11). bf16 CBs (exact through srcA), cb_xprime_mm and cb_w_slice (matmul
    # operands, FPU) stay Default.
    unpack_modes = [ttnn.UnpackToDestMode.Default] * 64
    for cb_idx, dtype in (
        (CB_F_TILES, input_tensor.dtype),
        (CB_X_TILES, residual.dtype),
        (CB_COEF_COLS, ttnn.float32),
    ):
        if dtype == ttnn.float32:
            unpack_modes[cb_idx] = ttnn.UnpackToDestMode.UnpackToDestFp32
    for cb_idx in fp32_copy_cbs:
        unpack_modes[cb_idx] = ttnn.UnpackToDestMode.UnpackToDestFp32
    kernel_cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=compute_config.math_fidelity,
        fp32_dest_acc_en=True,
        math_approx_mode=compute_config.math_approx_mode,
    )
    kernel_cfg.unpack_to_dest_mode = unpack_modes

    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "mhc_post_compute.cpp"),
        core_ranges=core_ranges,
        compile_time_args=compute_ct,
        runtime_args=compute_rt,
        config=kernel_cfg,
    )

    semaphores = []
    if fused:
        semaphores = [ttnn.SemaphoreDescriptor(id=i, core_ranges=core_ranges, initial_value=0) for i in range(n_sems)]
    return ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=semaphores, cbs=cbs)
