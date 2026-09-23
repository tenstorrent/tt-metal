# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ProgramDescriptor for groupnorm_sc_N_1_HW_C.

Realizes op_design.md: a 2-D `(hw_splits x c_splits)` tile split of every
image across the grid, per-channel column sums on each core, a membership
matmul that aggregates them by group, and an ack-free all-gather of the
per-core group partials (multicast + monotone counter semaphores), two
rounds per image (mean, then centered variance). Two regimes share every
kernel: `resident_2d` (the whole per-core assignment stays in L1 across all
three passes) and `streaming_2d` (input re-read once per pass; two passes when
`two_pass`: both statistics from one read of each chunk, then apply).

Combine: every core unicasts its partial rows into the
root's `cb_gather`; the root's compute sums them with the SAME matmul_block
instantiation as the group aggregation — `(1 x GT) @ (GT x Ng)` against
`cb_inv_rows` (row 0 = 1/n) so the result is the mean / variance directly —
and the root's writer multicasts the Ng stat tiles (mcast_pipe). Every core
forms `rstd = rsqrt(var + eps)` on the Ng group tiles (SFPU work scales with
Ng, not K), expands it and multiplies by gamma. The writer builds the
membership matrix twice — M (K x Ng tiles, the reduce-side mask) and its transpose Mᵀ (the apply-side mask) — so the
aggregation, the root combine and the expansions are all the same
non-transposed `(1 x K') @ (K' x N)` matmul: ONE matmul_block body in the
compute binary, which has to fit the kernel-config ring together with the
reader and writer in the --dev (watcher) build.

Every block knob (K = block_c_tiles, Q = chunk_hw_tiles, D = STREAM_DEPTH,
Ng, GT, the split) is derived ONCE here and handed to the kernels as CT/RT
args; nothing downstream restates a literal.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

import ttnn

from . import config

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE = 32
FP32_TILE_BYTES = 4096
BF16_TILE_BYTES = 2048

# ---- CB indices (semantic names; the index is just the slot) --------------
CB_INPUT_TILES = 0
CB_INPUT_STICKS = 1
CB_SCALER = 2
CB_MEMBERSHIP = 3
CB_GAMMA_ROWS = 4
CB_BETA_ROWS = 5
CB_INV_ROWS = 6  # GT fp32 tiles, row 0 = 1/(HW*Cg): the combine matmul's in0 (root only)
CB_COLSUM_ROWS = 7  # also the cross-chunk raw-sum accumulator (pop-before-pack per output)
CB_PARTIAL_ROWS = 8
CB_GATHER = 9
CB_GROUP_MEAN = 10  # writer (broadcast landing) -> compute
CB_GROUP_VAR = 11  # writer (broadcast landing) -> compute: group variance (rstd is formed per channel in pass 3)
CB_MEAN_ROWS = 12
CB_SCALE_ROWS = 13
CB_SHIFT_FULL = 14
CB_FP32_SCRATCH = 15
CB_OUTPUT_TILES = 16
CB_OUTPUT_STICKS = 17
CB_STATS_BCAST = 18  # root compute -> root writer: reduced group stats, the multicast source
CB_MEMBERSHIP_T = 19  # transposed membership (Ng x K tiles): in1 of the expansion matmuls
CB_GROUP_RSTD = 20  # compute -> compute: rsqrt(var + eps) on the Ng group tiles (pass-3 expansion in0)
CB_MASKED_MEAN = 21  # writer -> compute (hw_mask programs): group-mean tiles with rows outside the image zeroed
CB_ZERO_ROW = 22  # writer-owned 128 B of zeros: the partial row a core sends for an image its shard misses
CB_INPUT_SHARD = 23  # RM input shard (stick pages), placed on the shard buffer — the reader's staging source
CB_OUTPUT_SHARD = 24  # RM output shard (stick pages) — the writer's valid-stick destination (== input when in_place)
ZERO_ROW_BYTES = 128  # 2 x 64 B fp32 face rows (row 0 of faces 0/1 of one slot tile)
# `two_pass` programs only: the constant 1/32 row tile — the in0 of the shift matmul (1 x 1) @ (1 x K) that
# turns tile-row 0 of the first chunk into per-channel column means. (The second cross-chunk accumulator U
# lives BEHIND S in CB_COLSUM_ROWS, sized 2K: the accumulate reduce reloads its K partials from the front and
# packs the new ones at the back, so alternating S / U calls keep the ring in [S, U] order.)
CB_INV32_ROW = 25

SEM_ROUND0 = 0  # root's monotone row counter, round 0 (mean)
SEM_ROUND1 = 1  # root's monotone row counter, round 1 (variance)
SEM_MCAST_READY = 2  # mcast_pipe data-ready flag (root -> all)
SEM_MCAST_CONSUMED = 3  # mcast_pipe consumer-ready counter (all -> root)

# The combine root: active core 0 (logical (0,0)); it gathers every core's partial
# rows, reduces them and multicasts the Ng stat tiles back (op_design.md "Gather +
# multicast": one sender per round instead of a flat all-to-all).
ROOT_CORE = (0, 0)


def _div_up(a, b):
    return (a + b - 1) // b


def _f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


# ---------------------------------------------------------------------------
# Per-core L1 footprint (closed form)
# ---------------------------------------------------------------------------
def fixed_footprint(
    K, Ng, GT, Q, D, is_rm, has_gamma, has_beta, TB, TA, sharded=False, hw_mask=False, rm_direct=False, two_pass=False
):
    """Bytes of the CBs the program ALLOCATES (excludes shard-placed CBs: a TILE shard's input
    and output CBs sit on the tensor buffers; an RM shard's stick CBs likewise, and with the
    direct view the tilize / untilize stick CBs are the shards themselves). `hw_mask` programs
    (some tile-row partly outside its image) add the writer's masked group-mean CB; `two_pass`
    streaming programs add the second row accumulator and the 1/32 row tile."""
    T4 = FP32_TILE_BYTES
    total = (
        T4 * K * Ng  # cb_membership (reduce-side mask: in1 of the aggregation matmul)
        + T4 * K * Ng  # cb_membership_t (apply-side mask: in1 of the expansion matmuls)
        + BF16_TILE_BYTES  # cb_scaler
        + TA * K * (int(has_gamma) + int(has_beta))  # cb_gamma_rows, cb_beta_rows
        + T4 * K  # cb_colsum_rows
        + T4 * Ng  # cb_partial_rows
        + T4 * Ng * GT  # cb_gather (single landing region, root-reduce protocol)
        + T4 * GT  # cb_inv_rows (combine matmul in0)
        + T4 * 2 * Ng  # cb_group_mean, cb_group_var
        + T4 * Ng  # cb_group_rstd
        + T4 * Ng  # cb_stats_bcast
        + T4 * K  # cb_mean_rows
        + T4 * K  # cb_scale_rows
        + T4 * K  # cb_shift_full
        + T4 * Q * K  # cb_fp32_scratch
    )
    if is_rm or not sharded:
        total += TB * D * Q * K  # cb_output_tiles (a TILE shard's output CB is the output shard itself)
    if is_rm and not rm_direct:
        total += 2 * TB * D * K  # cb_input_sticks + cb_output_sticks (direct view: placed on the shards)
    if sharded:
        total += ZERO_ROW_BYTES  # cb_zero_row
    if hw_mask:
        total += 2 * Ng * T4  # cb_masked_mean (head + tail masked group-mean tiles)
    if two_pass:
        total += T4 * K + T4  # cb_colsum_rows grows to 2K (S and U accumulators) + cb_inv32_row (shift matmul in0)
    return total


def _has_partial_rows(N, HW, hw_stride, shard_rows, ny):
    """True iff some shard row holds a tile-row whose sticks are only partly inside one image (the
    kernels' image_work_sharded: a stick range that starts or ends mid tile-row). Image n owns
    sticks [n*hw_stride, n*hw_stride + HW) — hw_stride = HW for ROW_MAJOR, HWt*32 for TILE (each
    image padded to whole tile-rows). Drives the `hw_mask` compile-time knob — the masked-mean
    pass-2 path is compiled ONLY into programs that need it (not into shards with 32-multiple
    heights and N = 1)."""
    total = N * hw_stride
    for y in range(ny):
        s0 = y * shard_rows
        sticks = max(0, min(shard_rows, total - s0))
        for n in range(N):
            lo, hi = max(n * hw_stride, s0), min(n * hw_stride + HW, s0 + sticks)
            if lo < hi and ((lo - s0) % TILE != 0 or (hi - s0) % TILE != 0):
                return True
    return False


def _chunk_hw_tiles(K):
    return max(1, config.CHUNK_TILES_TARGET // K)


# ---------------------------------------------------------------------------
# Work split (op_design.md "Work split and regimes")
# ---------------------------------------------------------------------------
def _split_candidates(HWt, Ct, num_cores, Ng, D, is_rm, has_gamma, has_beta, TB, TA, hw_mask=False):
    """Every legal (hw_splits, c_splits) with uniform K = Ct / c_splits <= MAX_CORE_C_TILES
    whose fixed footprint fits the budget (worst-case GT). Yields
    (hw_splits, c_splits, K, Q, max_tiles_per_core, resident).

    The chunk Q is halved (down to 1 tile-row) until the whole per-core assignment
    is L1-resident, if any Q achieves it; otherwise Q keeps its default (a streaming
    assignment wants the coarsest chunk).
    """
    budget = config.L1_CB_BUDGET_BYTES
    GT_worst = _div_up(num_cores, TILE)
    for c_splits in range(1, min(Ct, num_cores) + 1):
        if Ct % c_splits != 0:
            continue
        K = Ct // c_splits
        if K > config.MAX_CORE_C_TILES:
            continue
        hw_splits = min(HWt, num_cores // c_splits)
        if hw_splits < 1:
            continue
        max_tiles = _div_up(HWt, hw_splits) * K
        Q0 = _chunk_hw_tiles(K)

        def fits(Q):
            fixed = fixed_footprint(K, Ng, GT_worst, Q, D, is_rm, has_gamma, has_beta, TB, TA, hw_mask=hw_mask)
            return fixed <= budget, max_tiles * TB <= budget - fixed

        fixed_ok, resident = fits(Q0)
        Q = Q0
        if not resident:
            q = Q0
            while q > 1:
                q //= 2
                ok, res = fits(q)
                if ok and res:
                    fixed_ok, resident, Q = ok, res, q
                    break
        if not fixed_ok:
            continue
        yield hw_splits, c_splits, K, Q, max_tiles, resident


def choose_split(HWt, Ct, num_cores, Ng, D, is_rm, has_gamma, has_beta, TB, TA, hw_mask=False):
    """Return (hw_splits, c_splits, Q) — op_design.md "Split search" with three rules
    layered on the min-max-tiles objective:

      1. TILE only: an L1-resident split (input crosses DRAM once) beats any
         non-resident one (input re-read per pass), whatever the tile count; the chunk
         Q shrinks if that is what makes an assignment fit. ROW_MAJOR skips this rule:
         there the stick-slice width K dominates, so it keeps the widest K and takes
         residency only when it comes free.
      2. config.SPLIT_COST_TOLERANCE_{TILE,RM}: among the remaining candidates, those
         within (1 + tol) x the minimum max-tiles-per-core are equivalent on the tile
         objective. ROW_MAJOR then widens the per-core column block K (stick slices
         are K*64 B) up to SPLIT_RM_STICK_K_TARGET and, at or above it, takes the
         fewest tiles then the narrowest K; TILE takes the fewest tiles then the
         narrowest K. Then more hw_splits, then fewer cores.
      3. config.FORCE_C_SPLITS (test knob) pins c_splits.

    K must be UNIFORM across cores (tilize/untilize take the block width as a
    compile-time template argument), so c_splits is restricted to divisors of Ct.
    """
    cands = list(_split_candidates(HWt, Ct, num_cores, Ng, D, is_rm, has_gamma, has_beta, TB, TA, hw_mask))
    if config.FORCE_C_SPLITS is not None:
        cands = [c for c in cands if c[1] == config.FORCE_C_SPLITS]
    if not cands:
        raise NotImplementedError(
            "groupnorm_sc_N_1_HW_C: no column split with uniform K <= MAX_CORE_C_TILES fits the "
            f"L1 budget for Ct={Ct} on {num_cores} cores (FORCE_C_SPLITS={config.FORCE_C_SPLITS})"
        )
    if not is_rm and any(c[5] for c in cands):
        cands = [c for c in cands if c[5]]
    tol = config.SPLIT_COST_TOLERANCE_RM if is_rm else config.SPLIT_COST_TOLERANCE_TILE
    min_tiles = min(c[4] for c in cands)
    limit = min_tiles * (1.0 + max(0.0, float(tol)))
    cands = [c for c in cands if c[4] <= limit]

    def key(c):
        hw_splits, c_splits, K, _Q, max_tiles, _res = c
        tail = (-hw_splits, hw_splits * c_splits)
        if is_rm:
            # Widen K up to the stick target; at/above it, fewest tiles then narrower K.
            wide = K >= config.SPLIT_RM_STICK_K_TARGET
            return (0 if wide else 1, max_tiles if wide else -K, K if wide else max_tiles) + tail
        # TILE: fewest tiles, then the narrower K (measured faster).
        return (max_tiles, K) + tail

    best = min(cands, key=key)
    return best[0], best[1], best[3]


def _balanced(total, splits, idx):
    base, rem = divmod(total, splits)
    start = idx * base + min(idx, rem)
    count = base + (1 if idx < rem else 0)
    return start, count


def _cb(index, core_ranges, num_pages, page_size, dtype):
    return ttnn.CBDescriptor(
        total_size=num_pages * page_size,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_size)],
    )


def _shard_cb(index, tensor, alias_index=None, alias_dtype=None, alias_page_size=None, page_size=None):
    """Zero-copy CB placed on a sharded tensor's L1 buffer (memory-layouts §3.4). With `alias_index`
    a second buffer index shares the SAME region (the `in_place` output: pass 3 consumes each input
    tile last, exactly once, and packs the result over it). `page_size` re-pages the region (the RM
    direct view hands the shard to the tilize / untilize as tile-sized pages: one page = one
    32-row x 32-column block of the row-major view)."""
    cb = ttnn.cb_descriptor_from_sharded_tensor(index, tensor)
    fds = list(cb.format_descriptors)  # def_rw on a std::vector hands back a copy: reassign, never append in place.
    if page_size is not None:
        assert cb.total_size % page_size == 0, (cb.total_size, page_size)
        fds = [ttnn.CBFormatDescriptor(buffer_index=index, data_format=tensor.dtype, page_size=page_size)]
    if alias_index is not None:
        fds.append(
            ttnn.CBFormatDescriptor(buffer_index=alias_index, data_format=alias_dtype, page_size=alias_page_size)
        )
    cb.format_descriptors = fds
    return cb


# ---------------------------------------------------------------------------
# Block-sharded geometry (op_design.md regime `block_sharded_resident`)
# ---------------------------------------------------------------------------
def _shard_geometry(input_tensor):
    """Read (hw_splits, c_splits, per-core extents) off the shard spec: the shard IS the per-core block.

    Returns dict(nx, ny, shard_rows, shard_w, K, Hs, stick_bytes). Only ROW_MAJOR-oriented
    rectangles anchored at (0,0) are accepted (what `ttnn.CoreGrid` realizes);
    anything else is a NotImplementedError, not a silent fallback to the interleaved path.
    """
    mc = input_tensor.memory_config()
    ss = mc.shard_spec
    if ss is None or mc.buffer_type != ttnn.BufferType.L1:
        raise NotImplementedError("groupnorm_sc_N_1_HW_C: BLOCK_SHARDED input must be an L1 shard with a shard spec")
    # ROW_MAJOR: core (x, y) holds row-block y / column-block x (HW down the grid rows, C across the
    # columns). COL_MAJOR (the transposed SDXL layout): core (x, y) holds row-block x / column-block y,
    # i.e. HW runs across the grid columns and C down the rows. Only the per-core (c0, s0) mapping and
    # the split counts change; the kernels see per-core stick/channel ranges either way.
    col_major = ss.orientation == ttnn.ShardOrientation.COL_MAJOR
    if not col_major and ss.orientation != ttnn.ShardOrientation.ROW_MAJOR:
        raise NotImplementedError("groupnorm_sc_N_1_HW_C: unsupported shard orientation")
    bbox = ss.grid.bounding_box()
    if ss.grid.num_cores() != bbox.grid_size().x * bbox.grid_size().y or (bbox.start.x, bbox.start.y) != (0, 0):
        raise NotImplementedError("groupnorm_sc_N_1_HW_C: the shard grid must be one rectangle anchored at (0,0)")
    shard_rows, shard_w = int(ss.shape[0]), int(ss.shape[1])
    nx, ny = bbox.grid_size().x, bbox.grid_size().y
    K = _div_up(shard_w, TILE)
    if K > config.MAX_CORE_C_TILES:
        raise NotImplementedError(
            f"groupnorm_sc_N_1_HW_C: shard width {shard_w} needs K={K} channel tiles > MAX_CORE_C_TILES="
            f"{config.MAX_CORE_C_TILES}"
        )
    stick_bytes = int(input_tensor.buffer_aligned_page_size())  # RM: the shard-width stick; TILE: a tile
    return dict(
        nx=nx,
        ny=ny,
        col_major=col_major,
        shard_rows=shard_rows,
        shard_w=shard_w,
        K=K,
        Hs=_div_up(shard_rows, TILE),
        stick_bytes=stick_bytes,
    )


def _rm_direct_view(geo, elem, N, HW):
    """The RM shard as an in-place row-major block of width lcm(shard_w, 32) elements.

    A ROW_MAJOR L1 shard [shard_rows, shard_w] whose stick page is exactly shard_w*elem bytes is one
    contiguous row-major byte array, so it is ALSO the block [shard_rows/m, m*shard_w] for any m —
    and with m*shard_w = lcm(shard_w, 32) = 32*K' that block is K' whole tiles wide: the tilize can
    read it zero-copy and the untilize can pack straight into the output shard (no pad lanes, no
    per-stick staging). Lane j of the view is channel c0 + (j % shard_w) — periodic — which only the
    membership build and the gamma/beta rows have to know. The view is exact iff every image
    boundary and the shard boundary fall on whole view tile-rows (32*m sticks), so no view row is
    ever partly outside its image (the staged path's zero pad sticks do not exist here).

    Returns (m, K') or None when the shard must take the staged path.
    """
    W = geo["shard_w"]
    lcm = W * TILE // math.gcd(W, TILE)
    m, Kd = lcm // W, lcm // TILE
    exact = (
        geo["stick_bytes"] == W * elem  # no alignment padding between sticks: the view is contiguous
        and Kd <= config.MAX_CORE_C_TILES
        and geo["shard_rows"] % (TILE * m) == 0  # shard boundaries on whole view tile-rows
        and HW % (TILE * m) == 0  # image boundaries likewise (RM images are HW sticks apart)
        and (W * elem) % 4 == 0  # the affine-row fill copies whole words at period boundaries
    )
    return (m, Kd) if exact else None


# ---------------------------------------------------------------------------
# Descriptor
# ---------------------------------------------------------------------------
def default_compute_kernel_config():
    """The default compute configuration: HiFi4, fp32 DEST accumulation, exact SFPU. The op's stat
    path (column sums, membership / combine / expansion matmuls, centered squares) accumulates in
    fp32 DEST into fp32 CBs; `fp32_dest_acc_en` is therefore mandatory (the op file refuses False)."""
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
        dst_full_sync_en=False,
    )


def create_program_descriptor(
    input_tensor,
    output_tensor,
    num_groups,
    *,
    gamma=None,
    beta=None,
    eps=1e-5,
    compute_kernel_config=None,
    activation=None,
):
    device = input_tensor.device()
    if compute_kernel_config is None:
        compute_kernel_config = default_compute_kernel_config()
    N, _, HW, C = [int(v) for v in input_tensor.shape]
    G = int(num_groups)
    Cg = C // G
    HWt = _div_up(HW, TILE)
    Ct = _div_up(C, TILE)
    Ng = _div_up(G, TILE)
    if Ng > config.MAX_GROUP_TILES:
        raise NotImplementedError(
            f"groupnorm_sc_N_1_HW_C: num_groups={G} needs {Ng} group-slot tiles > MAX_GROUP_TILES="
            f"{config.MAX_GROUP_TILES} (regime sparse_membership is deferred)"
        )

    is_rm = input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT
    has_gamma = gamma is not None
    has_beta = beta is not None
    weight = gamma if has_gamma else beta
    affine_dtype = weight.dtype if weight is not None else ttnn.bfloat16
    # Weight layout / format: TILE weights are lane-gathered by the reader from row 0 of their tile pages; bf8b weights (TILE-only, a block format) are decoded lane by lane into a bf16
    # rows CB (exact: bf8b carries a 7-bit mantissa), every other dtype's rows CB is the weight's own.
    affine_tile = weight is not None and weight.layout == ttnn.TILE_LAYOUT
    affine_bf8 = affine_dtype == ttnn.bfloat8_b
    rows_dtype = ttnn.bfloat16 if affine_bf8 else affine_dtype
    affine_page_bytes = int(weight.buffer_aligned_page_size()) if weight is not None else 0

    TB = ttnn.tile_size(input_tensor.dtype)
    TA = ttnn.tile_size(rows_dtype)
    T4 = FP32_TILE_BYTES
    D = config.STREAM_DEPTH

    grid = device.compute_with_storage_grid_size()
    num_cores = grid.x * grid.y
    sharded = input_tensor.memory_config().memory_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED
    in_place = output_tensor.buffer_address() == input_tensor.buffer_address()
    elem = TB // 1024
    # Non-tile-aligned HW / C: images are HW sticks apart in ROW_MAJOR and HWt*32 in TILE (each image padded to whole tile-rows); the kernels intersect their block with
    # [n*hw_stride, n*hw_stride + HW) so a ragged last tile-row carries `row_hi` valid sticks.
    hw_stride = HW if is_rm else HWt * TILE
    row_hi_last = HW - TILE * (HWt - 1)  # valid sticks of an image's last tile-row (32 when aligned)
    # Aligned page of each tensor (RM: the stick — shard-width or full-row; TILE: a tile): the
    # kernels' TensorAccessor page size / RM shard stick pitch.
    input_page_bytes = int(input_tensor.buffer_aligned_page_size())
    output_page_bytes = int(output_tensor.buffer_aligned_page_size())

    if sharded:
        # Regime `block_sharded_resident`: the shard IS the per-core block, so the split is read off
        # the shard spec (no search), the assignment is resident by construction, and the grid
        # rectangle is the shard grid. The chunk Q shrinks only if the fixed CBs (+ the RM tiled
        # copy of the shard) do not fit next to the shard.
        geo = _shard_geometry(input_tensor)
        # An RM shard whose row-major view of width lcm(shard_w, 32) is exact is consumed
        # in place as that K'-tile-wide block (m sticks per view row); the kernels see the VIEW's
        # geometry (K', Hs' = shard_rows/(32 m), stick counts / m) and a periodic channel map
        # (c_period = shard_w). Otherwise m = 1 and every value below is the shard's own.
        # The direct view is tried first and the staged view (m = 1) is the fallback: the fixed CBs
        # scale with K, so a wide view (K' = 15 for a 120-channel shard) can miss the budget where
        # the staged K = 4 fits; a view that does not fit is skipped, never an error.
        view = _rm_direct_view(geo, elem, N, HW) if is_rm else None
        candidates = ([(True,) + view] if view is not None else []) + [(False, 1, geo["K"])]
        hw_splits, c_splits = (geo["nx"], geo["ny"]) if geo["col_major"] else (geo["ny"], geo["nx"])
        rect_x, rect_y = geo["nx"], geo["ny"]
        num_active = rect_x * rect_y
        GT = _div_up(num_active, TILE)
        # The shards live in the same L1 (allocated top-down): the CBs must end below the lowest one.
        cb_ceiling = min(input_tensor.buffer_address(), output_tensor.buffer_address()) - int(
            ttnn.get_allocator_base_address(device, ttnn.BufferType.L1)
        )
        budget = min(config.L1_CB_BUDGET_BYTES, cb_ceiling - config.L1_CB_SAFETY_MARGIN_BYTES)
        for rm_direct, m, K in candidates:
            Hmax = geo["Hs"] // m
            Q = _chunk_hw_tiles(K)
            # Geometry the kernels intersect their block with, in view rows (m = 1: sticks).
            hw_geom, hw_stride_geom, shard_rows_geom = HW // m, hw_stride // m, geo["shard_rows"] // m
            # Pass-2 row masks are needed only when a tile-row is partly outside its image (RM shard
            # heights that are not multiples of 32, N > 1 shards straddling images, HW % 32 != 0 on a
            # TILE shard). The direct view is exact only without them (its rows have no zero pad sticks).
            hw_mask = _has_partial_rows(N, hw_geom, hw_stride_geom, shard_rows_geom, hw_splits)
            assert not (rm_direct and hw_mask), "RM direct view requires whole view tile-rows per image"
            tiled_copy = Hmax * K * TB if is_rm else 0  # RM shards tilize once into a resident tiled CB
            while True:
                fixed = fixed_footprint(
                    K,
                    Ng,
                    GT,
                    Q,
                    D,
                    is_rm,
                    has_gamma,
                    has_beta,
                    TB,
                    TA,
                    sharded=True,
                    hw_mask=hw_mask,
                    rm_direct=rm_direct,
                )
                if fixed + tiled_copy <= budget or Q == 1:
                    break
                Q //= 2
            if fixed + tiled_copy <= budget:
                break
        else:
            raise NotImplementedError(
                f"groupnorm_sc_N_1_HW_C: sharded block K={K}, Hs={Hmax} does not fit the L1 CB budget "
                f"({fixed + tiled_copy} B > {budget} B below the shard)"
            )
        resident = True
        input_cb_pages = Hmax * K  # RM: the tiled copy; TILE: the shard itself (placed, not allocated)
        # Channel period of a block lane: shard_w on the direct view ONLY (lane j of the lcm-wide view
        # is channel c0 + j % shard_w). On the staged path (m = 1) the block is K tiles wide with lanes
        # [c_valid, K*32) as zero pad, and shard_w may be SMALLER than K*32 (a 176- or 240-wide shard):
        # a period of shard_w there wraps the pad lanes onto the shard's first channels and hands them
        # membership rows — stale L1 (a tile-aligned c_valid < shard_w on the last column shard is
        # never zeroed) would then land in one group's stats. Identity map (>= K*32) everywhere but
        # the direct view.
        c_period = geo["shard_w"] if rm_direct else K * TILE
        assert rm_direct or c_period >= K * TILE
        two_pass = False  # the shard is resident: one read, no streaming passes
    else:
        # Interleaved: the image's last tile-row is ragged when HW % 32 != 0 -> masked-mean pass 2
        # on the core(s) that own it (every other tile-row is fully inside the image).
        hw_mask = HW % TILE != 0
        hw_splits, c_splits, Q = choose_split(HWt, Ct, num_cores, Ng, D, is_rm, has_gamma, has_beta, TB, TA, hw_mask)
        K = Ct // c_splits
        num_active = hw_splits * c_splits
        GT = _div_up(num_active, TILE)
        Hmax = _div_up(HWt, hw_splits)
        rm_direct, m = False, 1
        hw_geom, hw_stride_geom = HW, hw_stride
        c_period = K * TILE  # no periodicity: lane j is channel c0 + j

        fixed = fixed_footprint(K, Ng, GT, Q, D, is_rm, has_gamma, has_beta, TB, TA, hw_mask=hw_mask)
        resident = (not config.FORCE_STREAMING) and (Hmax * K * TB <= config.L1_CB_BUDGET_BYTES - fixed)
        input_cb_pages = Hmax * K if resident else D * Q * K
        # A streaming program computes both statistics from ONE read per chunk (pass A) and applies in
        # pass B — 3 tensor volumes of DRAM traffic instead of 4. The per-chunk shift / the per-core
        # stick count n assume every tile-row is fully inside its image, so a ragged last tile-row
        # (hw_mask) keeps the three-pass schedule with its masked-mean segments.
        two_pass = bool(config.STREAMING_TWO_PASS) and not resident and not hw_mask
        fixed = fixed_footprint(K, Ng, GT, Q, D, is_rm, has_gamma, has_beta, TB, TA, hw_mask=hw_mask, two_pass=two_pass)
        rect_x = min(num_active, grid.x)
        rect_y = _div_up(num_active, rect_x)

    # ---- grid rectangle (row-wise enumeration of active cores) -----------
    rect_num_cores = rect_x * rect_y
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(rect_x - 1, rect_y - 1))])
    root = ttnn.CoreCoord(*ROOT_CORE)
    root_v = device.worker_core_from_logical_core(root)
    # One sender (the root) per round over the whole rectangle; every core in the
    # rectangle — active or idle — acknowledges readiness, so the dense default
    # ack count applies. The writer runs on NOC_1.
    mcast = ttnn.Mcast2D(
        device,
        core_grid,
        root,
        ttnn.McastConfig(
            noc=ttnn.NOC.NOC_1,
            handshake=True,
            rotating_sender=False,
            sem_ids=[SEM_MCAST_READY, SEM_MCAST_CONSUMED],
        ),
    )

    # ---- circular buffers -------------------------------------------------
    cbs = []
    if sharded and not is_rm:
        # TILE shard: the input CB is the shard (zero-copy); the output CB is the output shard, or a
        # second buffer index on the SAME region when in_place.
        if in_place:
            cbs.append(_shard_cb(CB_INPUT_TILES, input_tensor, CB_OUTPUT_TILES, output_tensor.dtype, TB))
        else:
            cbs.append(_shard_cb(CB_INPUT_TILES, input_tensor))
            cbs.append(_shard_cb(CB_OUTPUT_TILES, output_tensor))
    else:
        cbs.append(_cb(CB_INPUT_TILES, core_grid, input_cb_pages, TB, input_tensor.dtype))
        cbs.append(_cb(CB_OUTPUT_TILES, core_grid, D * Q * K, TB, output_tensor.dtype))
    if sharded and is_rm and rm_direct:
        # RM shard, direct view: the shard IS the tilize's row-major input, re-paged as
        # tile-sized pages (one page = one 32 x 32 block of the view); the output shard IS the
        # untilize's destination the same way (the same region when in_place: pass 1 tilizes image
        # n's rows out of it before pass 3 untilizes over them). The reader pushes credits only, the
        # writer stores nothing.
        if in_place:
            cbs.append(
                _shard_cb(CB_INPUT_STICKS, input_tensor, CB_OUTPUT_STICKS, output_tensor.dtype, TB, page_size=TB)
            )
        else:
            cbs.append(_shard_cb(CB_INPUT_STICKS, input_tensor, page_size=TB))
            cbs.append(_shard_cb(CB_OUTPUT_STICKS, output_tensor, page_size=TB))
    elif sharded and is_rm:
        # RM shard, staged: stick pages. The reader stages sticks from CB_INPUT_SHARD into
        # cb_input_sticks (tilize stride), the writer copies the valid sticks of each untilized
        # tile-row into CB_OUTPUT_SHARD (the same region when in_place).
        if in_place:
            cbs.append(_shard_cb(CB_INPUT_SHARD, input_tensor, CB_OUTPUT_SHARD, output_tensor.dtype, output_page_bytes))
        else:
            cbs.append(_shard_cb(CB_INPUT_SHARD, input_tensor))
            cbs.append(_shard_cb(CB_OUTPUT_SHARD, output_tensor))
    if hw_mask:
        # writer -> compute: masked copies of the landed group-mean tiles (head + tail segments)
        cbs.append(_cb(CB_MASKED_MEAN, core_grid, 2 * Ng, T4, ttnn.float32))
    if sharded:
        cbs.append(_cb(CB_ZERO_ROW, core_grid, 1, ZERO_ROW_BYTES, ttnn.float32))
    if two_pass:
        # the writer-filled 1/32 row tile (in0 of compute's shift matmul)
        cbs.append(_cb(CB_INV32_ROW, core_grid, 1, T4, ttnn.float32))
    cbs += [
        _cb(CB_SCALER, core_grid, 1, BF16_TILE_BYTES, ttnn.bfloat16),
        _cb(CB_MEMBERSHIP, core_grid, K * Ng, T4, ttnn.float32),
        _cb(CB_MEMBERSHIP_T, core_grid, K * Ng, T4, ttnn.float32),
        # two_pass: S and U accumulate in this one ring ([S, U] order, 2K pages)
        _cb(CB_COLSUM_ROWS, core_grid, K * (2 if two_pass else 1), T4, ttnn.float32),
        _cb(CB_PARTIAL_ROWS, core_grid, Ng, T4, ttnn.float32),
        _cb(CB_GATHER, core_grid, Ng * GT, T4, ttnn.float32),
        _cb(CB_INV_ROWS, core_grid, GT, T4, ttnn.float32),
        _cb(CB_GROUP_MEAN, core_grid, Ng, T4, ttnn.float32),
        _cb(CB_GROUP_VAR, core_grid, Ng, T4, ttnn.float32),
        _cb(CB_GROUP_RSTD, core_grid, Ng, T4, ttnn.float32),
        _cb(CB_STATS_BCAST, core_grid, Ng, T4, ttnn.float32),
        _cb(CB_MEAN_ROWS, core_grid, K, T4, ttnn.float32),
        _cb(CB_SCALE_ROWS, core_grid, K, T4, ttnn.float32),
        _cb(CB_SHIFT_FULL, core_grid, K, T4, ttnn.float32),
        _cb(CB_FP32_SCRATCH, core_grid, Q * K, T4, ttnn.float32),
    ]
    if is_rm and not rm_direct:
        cbs.append(_cb(CB_INPUT_STICKS, core_grid, D * K, TB, input_tensor.dtype))
        cbs.append(_cb(CB_OUTPUT_STICKS, core_grid, D * K, TB, output_tensor.dtype))
    if has_gamma:
        cbs.append(_cb(CB_GAMMA_ROWS, core_grid, K, TA, rows_dtype))
    if has_beta:
        cbs.append(_cb(CB_BETA_ROWS, core_grid, K, TA, rows_dtype))

    semaphores = [
        ttnn.SemaphoreDescriptor(id=SEM_ROUND0, core_ranges=core_grid, initial_value=0),
        ttnn.SemaphoreDescriptor(id=SEM_ROUND1, core_ranges=core_grid, initial_value=0),
        ttnn.SemaphoreDescriptor(id=SEM_MCAST_READY, core_ranges=core_grid, initial_value=0),
        ttnn.SemaphoreDescriptor(id=SEM_MCAST_CONSUMED, core_ranges=core_grid, initial_value=0),
    ]

    # ---- compile-time args ------------------------------------------------
    knobs = [int(is_rm), int(resident), K, Q, int(has_gamma), int(has_beta), int(sharded), int(hw_mask), int(rm_direct)]
    knobs += [int(two_pass)]  # streaming programs read the input twice (pass A statistics, pass B apply)

    reader_ct = knobs + [CB_INPUT_TILES, CB_INPUT_STICKS, CB_SCALER, CB_GAMMA_ROWS, CB_BETA_ROWS, CB_INPUT_SHARD]
    reader_ct += [int(affine_tile), int(affine_bf8)]  # reader-only weight knobs (accessor_ct_base = 18)
    reader_ct += ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args()
    # ONE accessor-args block for both weights: the reader builds gamma's and beta's row tiles with the
    # same accessor type (their own base addresses), so fill_affine_rows is instantiated once — the
    # program must fit the kernel-config ring in the --dev build (two instantiations overflowed it on
    # the K = 2 staged RM hw_mask shards). Requires equal placement CT args (both DRAM interleaved, as
    # every producer makes them); anything else is refused, never silently mis-addressed.
    affine_args = (
        ttnn.TensorAccessorArgs(weight).get_compile_time_args()
        if weight is not None
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )
    if has_gamma and has_beta and list(ttnn.TensorAccessorArgs(beta).get_compile_time_args()) != list(affine_args):
        raise NotImplementedError(
            "groupnorm_sc_N_1_HW_C: gamma and beta must share their memory placement (buffer type / layout)"
        )
    reader_ct += affine_args

    writer_ct = knobs + [
        CB_MEMBERSHIP,
        CB_PARTIAL_ROWS,
        CB_GATHER,
        CB_STATS_BCAST,
        CB_GROUP_MEAN,
        CB_GROUP_VAR,
        CB_OUTPUT_TILES,
        CB_OUTPUT_STICKS,
        SEM_ROUND0,
        SEM_ROUND1,
        CB_INV_ROWS,
        CB_MEMBERSHIP_T,
        CB_MASKED_MEAN,
        CB_ZERO_ROW,
        CB_OUTPUT_SHARD,
        CB_INV32_ROW,
    ]
    writer_ct += list(mcast.compile_time_args())  # McastArgs<26, 11> in the kernel
    writer_ct += ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args()

    compute_ct = knobs + [
        CB_INPUT_TILES,
        CB_INPUT_STICKS,
        CB_SCALER,
        CB_MEMBERSHIP,
        CB_GAMMA_ROWS,
        CB_BETA_ROWS,
        CB_COLSUM_ROWS,
        CB_PARTIAL_ROWS,
        CB_GATHER,
        CB_GROUP_MEAN,
        CB_GROUP_VAR,
        CB_MEAN_ROWS,
        CB_SCALE_ROWS,
        CB_SHIFT_FULL,
        CB_FP32_SCRATCH,
        CB_OUTPUT_TILES,
        CB_OUTPUT_STICKS,
        CB_STATS_BCAST,
        CB_INV_ROWS,
        CB_MEMBERSHIP_T,
        CB_GROUP_RSTD,
        CB_MASKED_MEAN,
        CB_INV32_ROW,
    ]
    compute_ct += [int(activation == "silu")]  # index 33: SiLU fused into the pass-3 apply chain

    # ---- runtime args -----------------------------------------------------
    inv_bits = _f32_bits(1.0 / float(HW * Cg))  # 1/n for the combine matmul's in0 row
    eps_bits = _f32_bits(float(eps))
    input_addr = input_tensor.buffer_address()
    output_addr = output_tensor.buffer_address()
    gamma_addr = gamma.buffer_address() if has_gamma else 0
    beta_addr = beta.buffer_address() if has_beta else 0

    # Grid-wide constants travel as COMMON runtime args (one copy per kernel, not
    # one per core): the kernel-config ring holds every core's arg table, and at
    # 110 cores the per-core copies of these constants alone cost ~11 KB of it.
    # HW / hw_stride travel in the kernels' block-row units (view rows on the RM direct view, sticks
    # otherwise — identical unless m > 1); 1/n above uses the true HW. c_period is the channel period
    # of a block lane (shard_w on the direct view; >= K*32, i.e. the identity, everywhere else).
    reader_common = [input_addr, gamma_addr, beta_addr, N, HWt, Ct, hw_geom, Hmax, input_page_bytes, hw_stride_geom]
    reader_common += [c_period, affine_page_bytes]
    writer_common = [output_addr, N, HWt, Ct, hw_geom, C, Cg, Ng, GT, num_active, root_v.x, root_v.y, inv_bits]
    writer_common += [output_page_bytes, hw_stride_geom, c_period]
    compute_common = [N, Ng, GT, eps_bits, Hmax, hw_geom, hw_stride_geom]

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()

    # Per-core geometry. Interleaved: tile-row range [r0, r0 + H_core) of every image and channel
    # tiles [t0, t0 + K); `row_hi` = valid sticks of the block's last tile-row (HW % 32 for the core
    # that owns the image's ragged last row, else 32). Sharded: sticks [s0, s0 + sticks_valid) of
    # the flattened (N*hw_stride) rows and channels [c0, c0 + c_valid) — the kernels intersect the
    # stick range with each image (kernels/groupnorm_sc_N_1_HW_C_geometry.hpp). (c0, c_valid) is
    # the valid-lane pair both placements use for the membership rows and the gamma/beta slices (a
    # 40-channel shard is K = 2 tiles with 40 valid lanes; the last column block of a C % 32 != 0
    # tensor, sharded or interleaved, is clipped at C).
    active_idx = 0
    for y in range(rect_y):
        for x in range(rect_x):
            k = y * rect_x + x
            if sharded:
                hw_blk, c_blk = (x, y) if geo["col_major"] else (y, x)
                c0 = c_blk * geo["shard_w"]
                s0 = hw_blk * shard_rows_geom  # view rows (m = 1: sticks)
                c_valid = max(0, min(geo["shard_w"], C - c0))  # valid channels inside one period
                sticks_valid = max(0, min(shard_rows_geom, N * hw_stride_geom - s0))
                active = c_valid > 0 and sticks_valid > 0
                r0, H_core, t0 = 0, Hmax, c0 // TILE
                row_hi = TILE  # unused: sharded rows come from the stick intersection
            else:
                active = k < num_active
                if active:
                    hw_idx, c_idx = divmod(k, c_splits)
                    r0, H_core = _balanced(HWt, hw_splits, hw_idx)
                    t0 = c_idx * K
                else:
                    r0, H_core, t0 = 0, 0, 0
                c0 = t0 * TILE
                c_valid = max(0, min(K * TILE, C - c0)) if active else 0
                row_hi = row_hi_last if (active and r0 + H_core == HWt) else TILE
                s0, sticks_valid = 0, 0
            core_idx = active_idx if active else 0
            active_idx += int(active)
            is_root = (x, y) == ROOT_CORE
            reader_rt[x][y] = [int(active), r0, H_core, t0, c0, c_valid, s0, sticks_valid, row_hi]
            # mcast per-core args follow the 11 op args: McastArgs<CT, 11> in the writer
            writer_rt[x][y] = [
                int(active),
                r0,
                H_core,
                t0,
                core_idx,
                int(is_root),
                c0,
                c_valid,
                s0,
                sticks_valid,
                row_hi,
            ] + list(mcast.runtime_args(ttnn.CoreCoord(x, y)))
            # two_pass: -n/2 as fp32 bits, n = the core's sticks per image (every tile-row is
            # full: no hw_mask) — the scalar of the per-channel variance combine v = U + 2 (s - m)(S - (n/2)(m + s)).
            n_sticks = TILE * H_core if (two_pass and active) else 0
            compute_rt[x][y] = [int(active), H_core, int(is_root), s0, sticks_valid, row_hi, _f32_bits(-0.5 * n_sticks)]
    if sharded and active_idx != num_active:
        # Every shard core must carry data (PAD/EVEN auto shards and the model shards do); the
        # root's gather waits for exactly num_active rows.
        writer_common[9] = active_idx

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "groupnorm_sc_N_1_HW_C_reader.cpp"),
        core_ranges=core_grid,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        common_runtime_args=reader_common,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "groupnorm_sc_N_1_HW_C_writer.cpp"),
        core_ranges=core_grid,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        common_runtime_args=writer_common,
        config=ttnn.WriterConfigDescriptor(),
    )
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "groupnorm_sc_N_1_HW_C_compute.cpp"),
        core_ranges=core_grid,
        compile_time_args=compute_ct,
        runtime_args=compute_rt,
        common_runtime_args=compute_common,
        # The user's ttnn.ComputeKernelConfig. Every intermediate CB is fp32 because DEST is fp32;
        # the op file refuses fp32_dest_acc_en=False.
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=compute_kernel_config.math_fidelity,
            fp32_dest_acc_en=compute_kernel_config.fp32_dest_acc_en,
            math_approx_mode=compute_kernel_config.math_approx_mode,
            dst_full_sync_en=compute_kernel_config.dst_full_sync_en,
        ),
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=semaphores,
        cbs=cbs,
    )
