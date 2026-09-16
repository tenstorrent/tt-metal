# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ProgramDescriptor for groupnorm_sc_N_1_HW_C.

Realises op_design.md's Blocking Model:

* one **rectangle of P_n cores per image**, cut 2-D along ``hw`` (Pr) and ``ct`` (Pc) with the
  design's ceil/floor-balanced ragged split (``rows [i*Ht // Pr, (i+1)*Ht // Pr)``, same for cols);
  ``N >= num_cores`` degenerates to one core per image (P_n = 1, images looped);
* per core, ``chunk_rows x cols_per_group`` tile blocks (the last row chunk / column group of a core
  may be ragged: nominal CB quanta, narrowed work); ``resident_2d`` keeps the whole per-core block in
  an aliased ring across both passes, ``streaming_2d`` re-reads it;
* per-(image, group) statistics in lane form (Kg = ceil(G/32) tiles per statistic), built by
  a membership-matrix matmul, combined at an image root through a NoC gather + multicast.

Every block knob is defined ONCE here and passed to the kernels as CT/RT args.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

# ---------------------------------------------------------------------------
# Knobs (single source of truth — see op_design.md → Parameters / Buffer-depth knobs)
# ---------------------------------------------------------------------------
CHUNK_TILES_TARGET = 32  # tiles per (chunk_rows x cols_per_group) block; chunk_rows derives from it
X_DEPTH = 2  # streaming x ring depth, in chunks
X_RM_DEPTH = 2  # RM stick ring depth, in tile-rows
MEMBERSHIP_DEPTH = 1  # membership (E) blocks buffered
OUT_BLOCK_TILES_TARGET = 8  # output tiles per writer barrier (examples/double_buffer: 4-8 in flight saturates)
OUT_DEPTH_FACTOR = 2  # cb_out depth = OUT_DEPTH_FACTOR * out_block tiles (writer double-buffering)
MIN_TILES_PER_CORE = 1  # grid-synchronisation lamp: fewer, fatter cores per image when raised
# Tie-break among the core splits that use the most cores, per input layout:
#   "wide" = largest Pc (narrowest per-core Ct_core) — TILE input: the pass-2 affine build (one matmul + chains per
#            channel tile), membership generation and the column-group fixed costs all scale with Ct_core;
#   "tall" = largest Pr (narrowest per-core Ht_core) — ROW_MAJOR input: the stick reader issues one NoC read per
#            stick (32 * Ht_core per core per pass), which dominates the RM path.
# Measured on the 11x10 BH grid (Refinement 1, (1,1,1024,640) G=32 bf16): TILE wide 18.4 us vs tall 22.8 us;
# RM tall 28.7 us vs wide 34.8 us (resident).
SPLIT_TIE_BREAK = {"TILE": "wide", "ROW_MAJOR": "tall"}
# After the split, drop the cores that do not shorten the per-core critical extent: Pr -> ceil(Ht / Ht_core_max),
# Pc -> ceil(Ct / Ct_core_max). Every remaining core still owns <= the same max block, so the critical path is
# unchanged while the gather/multicast has fewer participants (measured ~1 us per 30 participants). Enabled for
# ROW_MAJOR input, whose per-stick reader IS the critical path (RM (1,1,1024,640): 22x5=110 cores 28.9 us vs
# 16x5=80 cores 28.0 us). Kept off for TILE: the DRAM-bound large shapes gain 3-4% from every extra core and
# op_requirements.md pins device_num_cores >= 108 on the flagship shapes ((1,1,1024,640) TILE would drop to 80
# cores at ~17.6 us vs 18.3 us — a finding for the perf rounds, not taken here).
SHRINK_TO_CRITICAL_EXTENT = {"TILE": False, "ROW_MAJOR": True}
L1_BUDGET_BYTES_DEFAULT = 1_000_000  # min(1 MB, worker L1 - 200 KiB) on WH/BH is the 1 MB term
MATH_FIDELITY = ttnn.MathFidelity.HiFi4
FP32_DEST_ACC_EN = True
DST_FULL_SYNC_EN = True
MATH_APPROX_MODE = False
# Design lamp (op_design.md -> cb_xsq "Float16_b pages for bf16/bf8b inputs"): x^2 pages at the INPUT's 16-bit width
# even under fp32 DEST (one extra bf16 rounding of x^2 before the column sum; halves the bytes through the packer /
# unpacker of the square + colsum pair). Off = the ledger rule (page follows DEST width). Under a 16-bit DEST the
# pages are Float16_b regardless. Measured in Refinement 3 (see op_requirements.md -> Outcome).
XSQ_16BIT_FOR_16BIT_INPUT = False

TILE = 32

# Bytes per element for the RM-stick arithmetic (host `element_size()` raises for block formats).
# bfloat8_b never takes the RM path (feature_spec INVALID), so its entry only keeps the expressions finite.
_ELEM_SIZE = {ttnn.bfloat16: 2, ttnn.float32: 4, ttnn.bfloat8_b: 1}


def _statistic_page_dtype(cfg):
    """Page format of every compute-produced intermediate (numeric-formats-metal §4: it follows the DEST width, not
    the input dtype). fp32 DEST -> Float32 pages; 16-bit DEST -> Float16_b (the value is already rounded to bf16 in
    DEST, so a Float32 page would only double the packer / unpacker bytes). Exceptions, Float32 whatever the DEST:
      * cb_partial / cb_gather / cb_totals_src / cb_totals_recv — the cross-core record format (writer face-row
        shuffle + multicast landing; both ends must change together);
      * cb_agg_interm — matmul_block re-targets the packer from interm to out only under fp32 DEST
        (matmul_block_helpers.inl), so with a 16-bit DEST the K-spill region must carry cb_partial's format."""
    return ttnn.float32 if bool(cfg.fp32_dest_acc_en) else ttnn.bfloat16


def _xsq_page_dtype(cfg, input_dtype):
    if not bool(cfg.fp32_dest_acc_en):
        return ttnn.bfloat16
    if XSQ_16BIT_FOR_16BIT_INPUT and input_dtype != ttnn.float32:
        return ttnn.bfloat16
    return ttnn.float32


# Regime-pin contract (acceptance test): module-level overrides of the two host knobs.
_l1_budget_bytes_override = None
_max_cores_override = None


def set_l1_budget_bytes_override(value):
    """Force the resident predicate (0 → nothing resident → streaming_2d). None clears."""
    global _l1_budget_bytes_override
    _l1_budget_bytes_override = value


def set_max_cores_override(value):
    """Cap the number of cores the op may use (N >= cap → single_core_per_image). None clears."""
    global _max_cores_override
    _max_cores_override = value


def default_compute_kernel_config():
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=MATH_FIDELITY,
        math_approx_mode=MATH_APPROX_MODE,
        fp32_dest_acc_en=FP32_DEST_ACC_EN,
        dst_full_sync_en=DST_FULL_SYNC_EN,
    )


def _dest_limit(cfg) -> int:
    """Mirror dest_helpers.hpp: DEST tile capacity from sync mode x accumulation width."""
    fp32 = bool(cfg.fp32_dest_acc_en)
    full = bool(cfg.dst_full_sync_en)
    if full:
        return 8 if fp32 else 16
    return 4 if fp32 else 8


# ---------------------------------------------------------------------------
# CB indices (semantic names; the index is just the slot)
# ---------------------------------------------------------------------------
CB_X_PASS1 = 0
CB_X_PASS2 = 1
CB_X_RM = 2
CB_XSQ = 3
CB_SCALER = 4
CB_COLSUM = 5
CB_MEMBERSHIP = 6
CB_AGG_INTERM = 7
CB_PARTIAL = 8
CB_GATHER = 9
CB_TOTALS_SRC = 10
CB_TOTALS_RECV = 11
CB_STATS_G_FULL = 12
CB_GAMMA_ROW = 13
CB_BETA_ROW = 14
CB_STATS_T = 15
CB_BETA_FULL = 16  # transient beta_T broadcast to all rows (beta path)
CB_A_FULL = 17
CB_B_FULL = 18
CB_OUT = 19
CB_STATS_ROW = 20

SEM_GATHER = 0
SEM_MCAST_READY = 1
SEM_MCAST_CONSUMED = 2

ROLE_IDLE = 0  # inside an image rectangle but p >= P_used: mcast handshake only
ROLE_MEMBER = 1
ROLE_ROOT = 2


# ---------------------------------------------------------------------------
# Small integer helpers
# ---------------------------------------------------------------------------
def _divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def _largest_divisor_leq(n, cap):
    cap = max(1, cap)
    best = 1
    for d in _divisors(n):
        if d <= cap:
            best = d
    return best


def _f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


# ---------------------------------------------------------------------------
# Work distribution
# ---------------------------------------------------------------------------
@dataclass
class ImageGroup:
    images: list  # image indices owned by this rectangle (len > 1 only when P_n == 1)
    cores: list  # (x, y) logical coords, row-major inside the rectangle; len == rect area
    rect_w: int
    rect_h: int
    pr: int  # cores cutting hw
    pc: int  # cores cutting ct
    p_used: int  # pr * pc  (<= rect area; the remainder are handshake-only idle cores)

    @property
    def root(self):
        return ttnn.CoreCoord(*self.cores[0])

    @property
    def core_range_set(self):
        x0, y0 = self.cores[0]
        return ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x0 + self.rect_w - 1, y0 + self.rect_h - 1))]
        )


def _split_2d(Ht, Ct, p_target, tie_break="wide"):
    """op_design.md -> In-image split: Pr <= min(P_n, Ht), Pc = min(P_n // Pr, Ct), maximising the used
    core count Pr*Pc; ties resolved by `tie_break` ("wide" -> largest Pc, "tall" -> largest Pr; see
    SPLIT_TIE_BREAK). No divisor constraint — per-core extents are ceil/floor balanced (see _axis_range)."""
    best = (1, 1)
    for pr in range(1, min(p_target, Ht) + 1):
        pc = min(p_target // pr, Ct)
        used, best_used = pr * pc, best[0] * best[1]
        tie_wins = (pc > best[1]) if tie_break == "wide" else (pr > best[0])
        if used > best_used or (used == best_used and tie_wins):
            best = (pr, pc)
    return best


def _axis_range(extent, parts, i):
    """Balanced split of `extent` units over `parts`: part i owns [i*extent // parts, (i+1)*extent // parts)."""
    begin = (i * extent) // parts
    end = ((i + 1) * extent) // parts
    return begin, end - begin


def _balanced_block(extent_max, target):
    """Block size <= target that covers `extent_max` with the fewest blocks and the least nominal padding:
    ceil(extent_max / ceil(extent_max / target)). Every smaller per-core extent then needs <= the same
    number of blocks, so CBs sized on `extent_max` hold every core's nominal block count."""
    target = max(1, min(target, extent_max))
    num_blocks = math.ceil(extent_max / target)
    return math.ceil(extent_max / num_blocks)


def _tight_rect(p_used, w, h):
    """Smallest (rw, rh) with rw <= w, rh <= h, rw*rh >= p_used; exact fit preferred."""
    for rh in range(1, h + 1):
        if p_used % rh == 0 and p_used // rh <= w:
            return p_used // rh, rh
    best = None
    for rh in range(1, h + 1):
        rw = math.ceil(p_used / rh)
        if rw <= w and (best is None or rw * rh < best[0] * best[1]):
            best = (rw, rh)
    assert best is not None
    return best


def _usable_grid(device):
    grid = device.compute_with_storage_grid_size()
    Gx, Gy = grid.x, grid.y
    cap = _max_cores_override if _max_cores_override is not None else Gx * Gy
    cap = max(1, min(cap, Gx * Gy))
    if cap < Gx * Gy:
        Gx2 = min(Gx, cap)
        Gy2 = max(1, cap // Gx2)
        return Gx2, Gy2
    return Gx, Gy


def _shrink_to_critical_extent(Ht, Ct, pr, pc):
    """Fewest cores per axis that keep the same ceil-balanced max extent (see SHRINK_TO_CRITICAL_EXTENT)."""
    pr = math.ceil(Ht / math.ceil(Ht / pr))
    pc = math.ceil(Ct / math.ceil(Ct / pc))
    return pr, pc


def _assign_images(N, Gx, Gy, Ht, Ct, tie_break="wide", shrink=False):
    """op_design.md → Work Distribution → Image rectangles / In-image split."""
    num_cores = Gx * Gy
    groups = []
    if N >= num_cores:
        # single_core_per_image: core c owns images c, c + num_cores, ...
        for c in range(num_cores):
            x, y = c % Gx, c // Gx
            groups.append(ImageGroup(list(range(c, N, num_cores)), [(x, y)], 1, 1, 1, 1, 1))
        return groups

    if N <= Gy:
        rows_per_image = Gy // N
        rects = [(0, n * rows_per_image, Gx, rows_per_image) for n in range(N)]
    else:
        k = math.ceil(N / Gy)  # images per grid row
        w = Gx // k
        rects = [((n % k) * w, n // k, w, 1) for n in range(N)]

    for n, (x0, y0, w, h) in enumerate(rects):
        p_target = min(w * h, max(1, (Ht * Ct) // MIN_TILES_PER_CORE))
        pr, pc = _split_2d(Ht, Ct, p_target, tie_break)
        if shrink:
            pr, pc = _shrink_to_critical_extent(Ht, Ct, pr, pc)
        p_used = pr * pc
        rw, rh = _tight_rect(p_used, w, h)
        cores = [(x0 + i, y0 + j) for j in range(rh) for i in range(rw)]
        groups.append(ImageGroup([n], cores, rw, rh, pr, pc, p_used))
    return groups


# ---------------------------------------------------------------------------
# Program descriptor
# ---------------------------------------------------------------------------
def _cb(index, core_ranges, num_pages, page_size, dtype):
    return ttnn.CBDescriptor(
        total_size=num_pages * page_size,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_size)],
    )


def create_program_descriptor(input_tensor, output_tensor, *, num_groups, gamma, beta, eps, compute_kernel_config=None):
    cfg = compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()
    device = input_tensor.device()

    N, _, HW, C = (int(v) for v in input_tensor.shape)
    G = int(num_groups)
    Cg = C // G
    Ht = math.ceil(HW / TILE)
    Ct = math.ceil(C / TILE)
    Kg = math.ceil(G / TILE)
    # hw_non_aligned: rows >= HW of the image's last tile-row are padding. The statistics must not depend on
    # what those rows hold (TILE input: whatever the producer left there; RM input: the reader zero-fills the
    # stick slots), so the reader emits a [full, partial] REDUCE_COL scaler pair and compute applies the partial
    # scaler to the row chunk that holds the image's last tile-row (reduce_helpers: ReducePartialScaler).
    hw_tail = HW % TILE
    scaler_tiles = 2 if hw_tail else 1

    is_rm = input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT
    has_gamma = gamma is not None
    has_beta = beta is not None
    affine = gamma if has_gamma else beta
    dest_limit = _dest_limit(cfg)

    if Kg > dest_limit:
        raise NotImplementedError(
            f"groupnorm_sc_N_1_HW_C: num_groups={G} needs Kg={Kg} lane-form tiles per statistic, "
            f"above the single-subblock DEST cap {dest_limit} (refinement candidate)"
        )

    # ---- image rectangles + in-image split -------------------------------------------
    Gx, Gy = _usable_grid(device)
    layout_key = "ROW_MAJOR" if is_rm else "TILE"
    groups = _assign_images(N, Gx, Gy, Ht, Ct, SPLIT_TIE_BREAK[layout_key], SHRINK_TO_CRITICAL_EXTENT[layout_key])
    pr, pc = groups[0].pr, groups[0].pc
    assert all(g.pr == pr and g.pc == pc for g in groups), "uniform split across image rectangles"
    # Ragged split: per-core extents are ceil/floor balanced (RT args); CBs size to the largest.
    Ht_core_max = math.ceil(Ht / pr)
    Ct_core_max = math.ceil(Ct / pc)
    p_max = max(g.p_used for g in groups)
    gather_tiles_per_stat = math.ceil(p_max / TILE)

    # ---- block knobs (derived once, on the max per-core extents) --------------------------
    cols = _balanced_block(Ct_core_max, dest_limit)  # cols_per_group (<= DEST cap of the REDUCE_COL block)
    chunk_rows = _balanced_block(Ht_core_max, max(1, CHUNK_TILES_TARGET // cols))
    num_col_groups_max = math.ceil(Ct_core_max / cols)
    num_row_chunks_max = math.ceil(Ht_core_max / chunk_rows)
    chunk = chunk_rows * cols
    # writer store block: tiles per NoC barrier, independent of cols (cols can be 1 on wide grids)
    out_block = _largest_divisor_leq(chunk, OUT_BLOCK_TILES_TARGET)
    # nominal pages of the largest per-core block (ragged last chunk / group pushes its full quantum)
    blk_max = num_col_groups_max * num_row_chunks_max * chunk
    in1_num_subblocks = math.ceil(Kg / dest_limit)
    out_subblock_w = min(Kg, dest_limit)

    # ---- byte sizes ------------------------------------------------------------------------
    x_tile_bytes = ttnn.tile_size(input_tensor.dtype)
    y_tile_bytes = output_tensor.buffer_page_size()
    f32_tile_bytes = ttnn.tile_size(
        ttnn.float32
    )  # cross-core record pages + matmul K-spill (see _statistic_page_dtype)
    stat_dtype = _statistic_page_dtype(cfg)
    stat_tile_bytes = ttnn.tile_size(stat_dtype)
    stat_elem_size = _ELEM_SIZE[stat_dtype]  # membership 0/1 element width the reader writes
    xsq_dtype = _xsq_page_dtype(cfg, input_tensor.dtype)
    xsq_tile_bytes = ttnn.tile_size(xsq_dtype)
    scaler_tile_bytes = ttnn.tile_size(ttnn.bfloat16)
    x_elem_size = _ELEM_SIZE[input_tensor.dtype]
    x_page_bytes = input_tensor.buffer_page_size()  # tile (TILE input) or stick (RM input)
    g_dtype = affine.dtype if affine is not None else ttnn.bfloat16
    g_tile_bytes = ttnn.tile_size(g_dtype)
    g_elem_size = _ELEM_SIZE[g_dtype]
    g_is_tile = int(affine is not None and affine.layout == ttnn.TILE_LAYOUT)
    g_is_bfp = int(g_dtype == ttnn.bfloat8_b)  # block format: the reader fetches the whole tile page
    g_page_bytes = affine.buffer_page_size() if affine is not None else g_tile_bytes

    # ---- core ranges -------------------------------------------------------------------------
    all_cores = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for g in groups for (x, y) in g.cores]
    )

    # ---- CBs (op_design.md → Circular Buffers; l1_ledger.md) ---------------------------------
    cbs = []
    fixed_bytes = 0

    def add(index, num_pages, page_size, dtype):
        nonlocal fixed_bytes
        fixed_bytes += num_pages * page_size
        cbs.append(_cb(index, all_cores, num_pages, page_size, dtype))

    if is_rm:
        add(CB_X_RM, X_RM_DEPTH * cols, x_tile_bytes, input_tensor.dtype)
    add(CB_XSQ, chunk, xsq_tile_bytes, xsq_dtype)
    add(CB_SCALER, scaler_tiles, scaler_tile_bytes, ttnn.bfloat16)
    add(CB_COLSUM, 2 * cols, stat_tile_bytes, stat_dtype)
    add(CB_MEMBERSHIP, MEMBERSHIP_DEPTH * cols * Kg, stat_tile_bytes, stat_dtype)
    add(CB_AGG_INTERM, 2 * Kg, f32_tile_bytes, ttnn.float32)  # must match cb_partial (16-bit DEST: no interm->out swap)
    add(CB_PARTIAL, 2 * Kg, f32_tile_bytes, ttnn.float32)  # cross-core record format (Float32 face rows)
    add(CB_GATHER, 2 * Kg * gather_tiles_per_stat, f32_tile_bytes, ttnn.float32)
    add(CB_TOTALS_SRC, 2 * Kg, f32_tile_bytes, ttnn.float32)
    add(CB_TOTALS_RECV, 2 * Kg, f32_tile_bytes, ttnn.float32)
    add(CB_STATS_ROW, 2 * Kg, stat_tile_bytes, stat_dtype)
    add(CB_STATS_G_FULL, 2 * Kg, stat_tile_bytes, stat_dtype)
    if has_gamma:
        add(CB_GAMMA_ROW, cols, g_tile_bytes, g_dtype)
    if has_beta:
        add(CB_BETA_ROW, cols, g_tile_bytes, g_dtype)
        add(CB_BETA_FULL, 1, stat_tile_bytes, stat_dtype)
    add(CB_STATS_T, 2, stat_tile_bytes, stat_dtype)
    add(CB_A_FULL, cols, stat_tile_bytes, stat_dtype)
    add(CB_B_FULL, cols, stat_tile_bytes, stat_dtype)
    add(CB_OUT, OUT_DEPTH_FACTOR * out_block, y_tile_bytes, output_tensor.dtype)

    # ---- regime: resident_2d vs streaming_2d (host-side, exact) -----------------------------
    l1_budget = _l1_budget_bytes_override if _l1_budget_bytes_override is not None else L1_BUDGET_BYTES_DEFAULT
    resident = (fixed_bytes + blk_max * x_tile_bytes) <= l1_budget
    if resident:
        # one L1 region, two credit counters (pass 1 / pass 2)
        cbs.append(
            ttnn.CBDescriptor(
                total_size=blk_max * x_tile_bytes,
                core_ranges=all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_X_PASS1, data_format=input_tensor.dtype, page_size=x_tile_bytes
                    ),
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_X_PASS2, data_format=input_tensor.dtype, page_size=x_tile_bytes
                    ),
                ],
            )
        )
    else:
        # streaming: separate rings so the reader may prefetch pass 2 while compute drains pass 1
        cbs.append(_cb(CB_X_PASS1, all_cores, X_DEPTH * chunk, x_tile_bytes, input_tensor.dtype))
        cbs.append(_cb(CB_X_PASS2, all_cores, X_DEPTH * chunk, x_tile_bytes, input_tensor.dtype))

    # ---- semaphores + multicast families (one per image rectangle) ---------------------------
    semaphores = [
        ttnn.SemaphoreDescriptor(id=SEM_GATHER, core_ranges=all_cores, initial_value=0),
        ttnn.SemaphoreDescriptor(id=SEM_MCAST_READY, core_ranges=all_cores, initial_value=0),
        ttnn.SemaphoreDescriptor(id=SEM_MCAST_CONSUMED, core_ranges=all_cores, initial_value=0),
    ]
    mcast_cfg = ttnn.McastConfig(handshake=False, sem_ids=[SEM_MCAST_READY, SEM_MCAST_CONSUMED])
    helpers = [ttnn.Mcast2D(device, g.core_range_set, g.root, mcast_cfg) for g in groups]
    mcast_ct = list(helpers[0].compile_time_args())
    assert all(list(h.compile_time_args()) == mcast_ct for h in helpers), "uniform mcast CT args"

    # ---- kernels -------------------------------------------------------------------------------
    inv_n_bits = _f32_bits(1.0 / float(Cg * HW))
    eps_bits = _f32_bits(float(eps))

    reader_ct = [
        CB_X_PASS1,
        CB_X_PASS2,
        CB_X_RM,
        CB_SCALER,
        CB_MEMBERSHIP,
        CB_GAMMA_ROW,
        CB_BETA_ROW,
        int(is_rm),
        int(resident),
        int(has_gamma),
        int(has_beta),
        chunk_rows,
        cols,
        Kg,
        x_tile_bytes,
        x_elem_size,
        x_page_bytes,
        g_elem_size,
        g_is_tile,
        g_is_bfp,
        g_page_bytes,
        C,
        G,
        HW,
        Ht,
        Ct,
        stat_elem_size,
    ]  # 27 scalars, then the accessors (TA_BASE = 27 in the reader)
    assert len(reader_ct) == 27
    reader_ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader_ct.extend(
        ttnn.TensorAccessorArgs(gamma).get_compile_time_args()
        if has_gamma
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )
    reader_ct.extend(
        ttnn.TensorAccessorArgs(beta).get_compile_time_args()
        if has_beta
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )

    writer_ct = [
        CB_PARTIAL,
        CB_GATHER,
        CB_TOTALS_SRC,
        CB_TOTALS_RECV,
        CB_OUT,
        Kg,
        cols,
        chunk_rows,
        Ht,
        Ct,
        gather_tiles_per_stat,
        SEM_GATHER,
        out_block,
    ]  # 13 scalars → McastArgs CT base = 13
    writer_mc_ct_base = len(writer_ct)
    writer_ct.extend(mcast_ct)
    writer_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    writer_rt_scalars = 14  # McastArgs RT base (MC_RT in the writer)
    assert writer_mc_ct_base == 13

    compute_ct = [
        CB_X_PASS1,
        CB_X_PASS2,
        CB_X_RM,
        CB_XSQ,
        CB_SCALER,
        CB_COLSUM,
        CB_MEMBERSHIP,
        CB_AGG_INTERM,
        CB_PARTIAL,
        CB_GATHER,
        CB_TOTALS_SRC,
        CB_TOTALS_RECV,
        CB_STATS_G_FULL,
        CB_GAMMA_ROW,
        CB_BETA_ROW,
        CB_STATS_T,
        CB_BETA_FULL,
        CB_A_FULL,
        CB_B_FULL,
        CB_OUT,
        CB_STATS_ROW,
        int(is_rm),
        int(resident),
        int(has_gamma),
        int(has_beta),
        chunk_rows,
        cols,
        Kg,
        gather_tiles_per_stat,
        in1_num_subblocks,
        out_subblock_w,
        out_block,
        hw_tail,
    ]

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    gamma_addr = gamma.buffer_address() if has_gamma else 0
    beta_addr = beta.buffer_address() if has_beta else 0

    for group, helper in zip(groups, helpers):
        root_virtual = device.worker_core_from_logical_core(group.root)
        image_begin = group.images[0]
        image_count = len(group.images)
        image_stride = Gx * Gy if image_count > 1 else 1
        for p, (x, y) in enumerate(group.cores):
            core = ttnn.CoreCoord(x, y)
            if p >= group.p_used:
                role = ROLE_IDLE
                row_begin, Ht_core, col_begin, Ct_core = 0, 0, 0, 0  # idle: no block at all
            else:
                role = ROLE_ROOT if p == 0 else ROLE_MEMBER
                i, j = p // group.pc, p % group.pc
                row_begin, Ht_core = _axis_range(Ht, group.pr, i)  # rows [i*Ht // Pr, (i+1)*Ht // Pr)
                col_begin, Ct_core = _axis_range(Ct, group.pc, j)  # cols [j*Ct // Pc, (j+1)*Ct // Pc)
            active = int(role != ROLE_IDLE)
            reader_rt[x][y] = [
                input_tensor.buffer_address(),
                gamma_addr,
                beta_addr,
                image_begin,
                image_count,
                image_stride,
                row_begin,
                col_begin,
                active,
                Ht_core,
                Ct_core,
            ]
            writer_rt[x][y] = [
                output_tensor.buffer_address(),
                image_begin,
                image_count,
                image_stride,
                row_begin,
                col_begin,
                role,
                p,
                group.p_used,
                root_virtual.x,
                root_virtual.y,
                len(group.cores),  # num_participants: every rectangle core increments the gather semaphore
                Ht_core,
                Ct_core,
            ]
            assert len(writer_rt[x][y]) == writer_rt_scalars
            writer_rt[x][y] = list(writer_rt[x][y]) + list(helper.runtime_args(core))
            owns_last_row = int(active and row_begin + Ht_core == Ht)  # holds the image's ragged last tile-row
            compute_rt[x][y] = [image_count, role, inv_n_bits, eps_bits, Ht_core, Ct_core, owns_last_row]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "groupnorm_sc_N_1_HW_C_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "groupnorm_sc_N_1_HW_C_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "groupnorm_sc_N_1_HW_C_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=compute_ct,
        runtime_args=compute_rt,
        config=cfg,
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=semaphores,
        cbs=cbs,
    )
