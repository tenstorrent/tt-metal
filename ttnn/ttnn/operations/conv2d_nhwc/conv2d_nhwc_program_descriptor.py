# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""conv2d_nhwc — ProgramDescriptor (CBs, kernels, args).

Multi-core over the full compute-with-storage grid. Implicit im2col + blocked matmul:

    DRAM NHWC RM activation
       │ NCRISC reader — one NoC read per (output position × tap × channel slice)
       ▼
    cb_act_rm ─(TRISC tilize, as matmul_block's PreKBlockFn)─► cb_act_tiles
                                                                   │
    cb_weight_tiles ◄── BRISC writer ◄── DRAM TILE weight          │
                    └──────► matmul_block (K-accumulating) ◄───────┘
                                    │
                 no bias ───► cb_mm_out       bias ───► cb_partials
                                                            │ + cb_bias_tiles
                                                            ▼ add_bias_bcast_rows
                                                        cb_mm_out
                                    │ TRISC untilize
                                    ▼
                                 cb_out_rm ── BRISC writer ──► DRAM NHWC RM output

Work distribution (Refinement 2): the M dimension — `M_total = N*H_out*W_out`
output positions, grouped into `num_m_blocks` blocks of `Mt` tile-rows each —
is the split unit. Both tensors are DRAM-interleaved and every m_block is
computed from activation/weight reads alone (no inter-core data dependency), so
the split is embarrassingly parallel: `ttnn.split_work_to_cores` hands each core
a contiguous `[start_m_block, start_m_block + num_m_blocks_here)` range, passed
as runtime args to all three kernels. Per-core CB sizes are unchanged — they are
set by `(Kb, Mt, Nt_b)`, not by the number of m_blocks.
"""

from math import gcd
from pathlib import Path

from loguru import logger

import ttnn


KERNEL_DIR = Path(__file__).parent / "kernels"

TILE = 32

# Byte alignment every NoC source offset must respect. The sanitizer's real
# rule is that the L1 destination and the NoC source must share the same
# alignment residue; every L1 write pointer in the reader sits on a multiple of
# `stick_bytes` (itself a multiple of 64), so keeping the DRAM-side offsets on a
# 64-byte grid keeps both residues at 0 on Wormhole (DRAM_ALIGNMENT 32) and
# Blackhole (DRAM_ALIGNMENT 64) alike.
NOC_OFFSET_ALIGNMENT = 64

# CB indices — 0-7 input, 8-15 special, 16-23 output, 24-31 intermediate.
CB_ACT_RM = 0  # row-major im2col sticks (reader -> tilize)
CB_WEIGHT_TILES = 1  # prepared weight tiles (writer -> matmul in1)
CB_BIAS_TILES = 2  # prepared bias tiles (writer -> bias helper)
CB_ZERO_SCRATCH = 3  # reader-private zero page for out-of-bounds taps
CB_OUT_RM = 16  # untilized output sticks (untilize -> writer)
CB_ACT_TILES = 24  # tilized activation block (tilize -> matmul in0)
CB_PARTIALS = 25  # matmul K-accumulation spill/reload (and bias input)
CB_MM_OUT = 26  # matmul/bias result block (-> untilize)

# L1 budget for all CBs combined. 1.5 MB physical; leave headroom for
# firmware, stacks and kernel binaries.
L1_CB_BUDGET = 1_000_000


def _dest_limit(*, fp32_dest_acc_en, dst_full_sync_en):
    """DEST tile capacity (mirrors dest_helpers.hpp get_dest_limit()).

    half-sync fp16 = 8, half-sync fp32 = 4, full-sync doubles both.
    `out_subblock_h * out_subblock_w` must not exceed this.
    """
    return (4 if fp32_dest_acc_en else 8) * (2 if dst_full_sync_en else 1)


def _div_up(a, b):
    return (a + b - 1) // b


def _round_up_to_tile(n):
    return _div_up(int(n), TILE) * TILE


# Activation element size in bytes. Activations are never block-float (they are
# ROW_MAJOR by op contract), so the tile-size / 1024 identity is exact.
def elem_size_of(dtype):
    return ttnn.tile_size(dtype) // (TILE * TILE)


# ---------------------------------------------------------------------------
# Grouped / depthwise column blocking  (Refinement 4)
# ---------------------------------------------------------------------------


class GroupBlocking:
    """How the C_out columns are partitioned into group-aligned column blocks.

    Grouped convolution makes the im2col gather depend on the *output* column:
    output channel `co` only sees input channels
    `[(co // cols_per_group) * chans_per_group, ... + chans_per_group)`. The
    matmul, however, reduces a whole K-window against a whole N-window, so the
    channel window has to be constant across every column of an N-block.

    The fix is to bundle `G_blk` consecutive groups into one *column block*
    whose column count is a multiple of the tile side, and give the matmul that
    block's union of channels as its K-window:

        G_blk      = 32 / gcd(cols_per_group, 32)        (capped at `groups`)
        cols_cb    = G_blk * cols_per_group              (multiple of 32)
        chans_cb   = G_blk * chans_per_group             (the K-window)

    Within a column block the weight is block-diagonal — column `co` carries
    zeros on every channel outside its own group — so the extra MACs multiply
    by zero. `G_blk` is the *smallest* bundle that tile-aligns the columns, so
    that waste is the minimum the tile granularity allows:

      * dense (groups == 1)            -> G_blk = 1, one column block, this
                                          whole mechanism is the identity.
      * grouped, cols_per_group >= 32  -> G_blk = 1, zero waste.
      * depthwise (cols_per_group = 1) -> G_blk = 32: one 32-column block reads
                                          32 channels instead of 1. K per block
                                          is kH*kW*32 regardless of C_in, i.e.
                                          the cost stays O(C_out) rather than
                                          the O(C_out^2) a fully dense
                                          expansion would cost.

    Column blocks are laid out contiguously in column space, which is what lets
    the writer keep indexing weights/bias/output by the flat N-block index: the
    reader is the only kernel that needs to know about `cblock` at all.

    Two fallbacks to a single column block (`G_blk = groups`, i.e. a plain
    dense expansion over all C_in channels — always correct, just wider):
      1. the bundled columns would not fit inside `roundup32(C_out)` tiles
         (possible when cols_per_group is neither a divisor nor a multiple of
         32 and `groups` is odd), which would desynchronize the flat N-block
         indexing from the prepared weight's real tile count;
      2. `chans_cb * elem_size` is not a multiple of `NOC_OFFSET_ALIGNMENT`, so
         the per-cblock channel base offset would break the NoC alignment
         residue in the reader.
    """

    __slots__ = ("G_blk", "cols_cb", "chans_cb", "num_cblocks", "fallback")

    def __init__(self, G_blk, cols_cb, chans_cb, num_cblocks, fallback):
        self.G_blk = G_blk
        self.cols_cb = cols_cb
        self.chans_cb = chans_cb
        self.num_cblocks = num_cblocks
        self.fallback = fallback


def group_blocking(*, C_in, C_out, groups, elem_size):
    """Pick the column-block bundle size. See `GroupBlocking`."""
    if groups == 1:
        return GroupBlocking(1, C_out, C_in, 1, fallback=False)

    cols_per_group = C_out // groups
    chans_per_group = C_in // groups

    G_blk = min(groups, TILE // gcd(cols_per_group, TILE))
    num_cblocks = _div_up(groups, G_blk)
    cols_cb = G_blk * cols_per_group
    chans_cb = G_blk * chans_per_group

    fits_in_padded_cout = num_cblocks * cols_cb <= _round_up_to_tile(C_out)
    noc_aligned = num_cblocks == 1 or (chans_cb * elem_size) % NOC_OFFSET_ALIGNMENT == 0
    if fits_in_padded_cout and noc_aligned:
        return GroupBlocking(G_blk, cols_cb, chans_cb, num_cblocks, fallback=False)

    # Dense expansion over the full channel axis: one column block, base
    # offset 0, block-diagonal weight. Correct for any `groups`.
    return GroupBlocking(groups, C_out, C_in, 1, fallback=True)


# ---------------------------------------------------------------------------
# Compute-kernel configuration
# ---------------------------------------------------------------------------


class ResolvedComputeConfig:
    """The four knobs the descriptor actually needs, after defaulting and
    clamping. Kept as a tiny value object so the blocking search, the CB
    format derivation and the ComputeConfigDescriptor all read the same thing."""

    __slots__ = ("math_fidelity", "fp32_dest_acc_en", "math_approx_mode", "dst_full_sync_en")

    def __init__(self, math_fidelity, fp32_dest_acc_en, math_approx_mode, dst_full_sync_en):
        self.math_fidelity = math_fidelity
        self.fp32_dest_acc_en = fp32_dest_acc_en
        self.math_approx_mode = math_approx_mode
        self.dst_full_sync_en = dst_full_sync_en


def default_math_fidelity(dtype, weight_dtype):
    """Fidelity the op picks when the caller supplies no compute_kernel_config.

    Both operands fp32 -> HiFi3. Any narrower operand -> HiFi2: HiFi4's extra
    passes buy nothing once a 7-bit-mantissa operand is in the product.

    Refinement 5b — why fp32 is HiFi3 and not HiFi4
    -----------------------------------------------
    Phase 0 read issue #38306 as "HiFi4 + fp32_dest_acc_en + a *narrow* operand
    corrupts the K-accumulator" and so kept HiFi4 for the all-fp32 branch. That
    reading was inverted. The upstream matmul applies the workaround in exactly
    the opposite direction (`matmul_device_operation.cpp`):

        // Due to hardware bug (#38306), HiFi4 + fp32_dest_acc_en can sometime
        // produce incorrect results on Wormhole.
        math_fidelity = are_inputs_32F ? (is_wormhole ? HiFi3 : HiFi4) : ...;

    i.e. fp32 inputs are the *dangerous* case, because fp32 inputs are what
    drive `fp32_dest_acc_en=True` in the first place.

    Measured here: with HiFi4 + fp32 DEST, a `matmul_block` that accumulates
    onto a NON-ZERO fp32 DEST can return the result low by exactly `16 * 2**E`,
    where `2**E <= |DEST_before| < 2**(E+1)` — a high bit lost out of the FPU's
    fixed-point accumulate headroom during HiFi4's multi-pass fp32 reduction.
    The error is always negative and always an exact power of two (observed
    -16, -32, -64), and it lands on a single output element per run.

    HiFi3 removes it outright and costs almost nothing: over a 40-seed sweep on
    three shapes the relative RMS moves 0.00152 -> 0.00163, ~200x inside the
    0.02 fp32 band, while the outlier rate goes from 9/360 to 0/360. HiFi2
    (0.0075) and LoFi (0.031) are far worse, so HiFi3 is the right rung.
    """
    both_fp32 = dtype == ttnn.float32 and weight_dtype == ttnn.float32
    return ttnn.MathFidelity.HiFi3 if both_fp32 else ttnn.MathFidelity.HiFi2


def resolve_compute_kernel_config(dtype, weight_dtype, compute_kernel_config=None):
    """Fill in the op's defaults, then apply the WH-B0 HiFi4 guard.

    The guard survives user control as a *clamp*: an explicit
    `HiFi4 + fp32_dest_acc_en` request is demoted to HiFi3 (explicitly listed as
    safe by matmul_block_helpers.hpp) and logged, rather than silently honoured
    into a corrupt accumulator.

    Refinement 5b widened this clamp. It used to fire only for a *narrow*
    operand; the operand-width qualifier was wrong. #38306 is a property of
    `HiFi4 + fp32_dest_acc_en` alone — all-fp32 operands are if anything the
    most exposed case, since they are what turn `fp32_dest_acc_en` on. See
    `default_math_fidelity` for the measurement. `fp32_dest_acc_en=False` is
    still not clamped: with a 16-bit DEST there is no fp32 accumulate to
    corrupt, and over-clamping would cost fidelity for nothing.
    """
    cfg = compute_kernel_config
    math_fidelity = getattr(cfg, "math_fidelity", None) if cfg is not None else None
    if math_fidelity is None:
        math_fidelity = default_math_fidelity(dtype, weight_dtype)
    fp32_dest_acc_en = bool(getattr(cfg, "fp32_dest_acc_en", True)) if cfg is not None else True
    math_approx_mode = bool(getattr(cfg, "math_approx_mode", False)) if cfg is not None else False
    dst_full_sync_en = bool(getattr(cfg, "dst_full_sync_en", False)) if cfg is not None else False

    if fp32_dest_acc_en and math_fidelity == ttnn.MathFidelity.HiFi4:
        logger.warning(
            "conv2d_nhwc: clamping math_fidelity HiFi4 -> HiFi3 for "
            f"dtype={dtype} weight_dtype={weight_dtype}: HiFi4 + fp32_dest_acc_en "
            "corrupts the K-accumulator on Wormhole B0 (issue #38306)."
        )
        math_fidelity = ttnn.MathFidelity.HiFi3

    return ResolvedComputeConfig(math_fidelity, fp32_dest_acc_en, math_approx_mode, dst_full_sync_en)


def packer_l1_acc_enabled(*, num_k_blocks, Kb, fuse_bias, fp32_dest_acc_en):
    """Whether the matmul K-accumulates in the packer's L1 accumulator.

    Refinement 5. matmul_block's default software K-accumulation spills each
    partial output block to `cb_partials` and reloads it into DEST through
    `copy_block_matmul_partials` — i.e. through SrcA, which holds ~11 mantissa
    bits on Wormhole. That is a *biased* (truncating) rounding of the running
    sum applied once per K-block, so the relative error grows LINEARLY in
    `num_k_blocks` rather than staying flat: measured rel_rms fits
    `0.0013 + 1.9e-4 * Kt` to three digits over Kt = 1, 9, 25, 49, 121, which
    is how a k=11 conv (121 K-blocks) blew past the fp32 0.02 band while k=3
    sat comfortably at 0.003.

    With `packer_l1_acc` the packer adds each block's result straight into the
    fp32 L1 region instead, so no partial sum ever round-trips through SrcA and
    only the per-product SrcA/SrcB floor is left.

    Gated on `fp32_dest_acc_en` so the L1 accumulator is always Float32 (the
    reference matmul factory falls back to Float16_b for the fp16-DEST case;
    accumulating a deep reduction in bf16 would trade one precision problem for
    a worse one, and every default config this op ships has fp32 DEST on).

    The block-count thresholds match the reference matmul factory: with a
    downstream bias the last K-block stays in the intermediate buffer, so L1
    accumulation pays off from 2 blocks; without one the last block spills and
    reloads regardless, so it only pays off from 3.

    No longer restricted to `Kb == 1` (Refinement 5b)
    -------------------------------------------------
    Refinement 5 shipped a `Kb != 1` refusal because L1 accumulation at an
    in-DEST K-depth >= 2 corrupted a *single output element* per run (one
    element reading -48.90625 against a +15.12401 reference, 1.73 sigma).
    That gate was aimed at the wrong variable. The corruption is issue #38306
    — `MathFidelity::HiFi4` + `fp32_dest_acc_en` on Wormhole B0 — and has
    nothing to do with the packer:

      * The error is always an exact negative power of two (`-16 * 2**E`, with
        `2**E <= |DEST_before| < 2**(E+1)`), which is a bit lost out of the
        FPU's fixed-point accumulate headroom, not a bad accumulate.
      * A DEVICE_PRINT trace of the packer thread shows every accumulate
        satisfying `partials[j] == partials[j-1] + dest[j]` exactly, including
        the failing step — the corruption is already in DEST before the pack.
      * It is not the multi-tile pack: it reproduces bit-identically at
        `Mt=1, Nt_b=1, out_subblock_w=1`, where the pack is a single tile.
      * It does not need zero data: a padding-free `k=1` pure matmul fires too.

    `Kb >= 2` was only the *exposure amplifier*. On this path `enable_reload`
    is false, so DEST starts at zero each K-block and the only accumulate onto
    a non-zero DEST is the second `matmul_tiles` of a `Kb >= 2` block — whose
    two operands are halves of the same tap's dot product, i.e. the
    equal-magnitude/opposite-sign cancellation regime that trips #38306. At
    `Kb == 1` DEST_before is always zero, so the hazard cannot arise, which is
    why the old gate looked like it worked.

    `default_math_fidelity` now returns HiFi3 for all-fp32 operands (and
    `resolve_compute_kernel_config` clamps any HiFi4 + fp32 DEST request),
    which removes the cause. Verified over a 40-seed sweep on three shapes and
    every legal `Kb` (1, 2, 4, 5): 9/360 runs carried an outlier at HiFi4,
    0/360 at HiFi3, with relative RMS flat at 0.00163 independent of `Kb`.
    """
    if not fp32_dest_acc_en:
        return False
    return num_k_blocks > 1 if fuse_bias else num_k_blocks > 2


def _partials_pages(*, Mt, Nt_b, out_subblock_w, packer_l1_acc_en):
    """Page count for cb_partials.

    Under packer L1 accumulation the size must be EXACTLY one output block:
    the helper pushes and then drains a whole block per K-block, so the CB's
    write pointer only wraps back onto the same L1 region — which is what makes
    the next block's pack accumulate onto the previous one — when the FIFO
    holds exactly `Mt * Nt_b` pages. (`matmul_multicore_reuse_mcast_*` sizes
    interm0 at exactly `out_block_tiles` for the same reason.)

    The software spill/reload path has no such constraint; it keeps one extra
    sub-block of slack so the last K-block's per-sub-block reload-then-repack
    never has to wait on its own pop.
    """
    return Mt * Nt_b if packer_l1_acc_en else Mt * Nt_b + out_subblock_w


def _divisors_desc(n):
    return [d for d in range(n, 0, -1) if n % d == 0]


def _cb_bytes(*, Kb, Mt, Nt_b, out_subblock_w, elem_size, act_tile, w_tile, interm_tile, fuse_bias):
    """Total L1 footprint of the CB set for one candidate blocking.

    `act_tile` / `w_tile` / `interm_tile` are independent: the activation, the
    prepared weight and the matmul K-accumulator each carry their own dtype
    (bf8b weights are 1088 B/tile against a 2048 B bf16 or 4096 B fp32
    activation tile, and the accumulator is fp32 whenever fp32_dest_acc_en is).
    """
    stick_bytes = Kb * TILE * elem_size
    total = 0
    total += 64 * stick_bytes  # cb_act_rm  (two 32-stick tile-rows)
    total += 1 * stick_bytes  # cb_zero_scratch
    total += Mt * Kb * act_tile  # cb_act_tiles
    total += 2 * Kb * Nt_b * w_tile  # cb_weight_tiles (double-buffered K-block)
    total += (Nt_b * w_tile) if fuse_bias else 0  # cb_bias_tiles
    total += (Mt * Nt_b + out_subblock_w) * interm_tile  # cb_partials
    total += Mt * Nt_b * act_tile  # cb_mm_out
    total += 2 * Nt_b * act_tile  # cb_out_rm
    return total


def _mt_cap(Mt_total, num_cores_available):
    """Largest `Mt` that still leaves at least `num_cores_available` M-blocks.

    `Mt` is simultaneously the per-core matmul block height and the *inverse* of
    the parallelism available to the grid split: `num_m_blocks =
    ceil(Mt_total / Mt)` is the number of work units. Phase 0's search always
    picked the largest feasible `Mt` (8) because a single core has no
    parallelism to trade against; under the grid split that choice can starve
    63 of 64 cores.

    Capping at `ceil(Mt_total / num_cores)` keeps `num_m_blocks >= num_cores`
    whenever there is enough M to go around, and degrades to `Mt = 1` (one
    tile-row per unit, the finest split this kernel supports) for small M.
    The cap only *shrinks* CBs, so it can never make the search infeasible.
    """
    return max(1, _div_up(Mt_total, max(1, num_cores_available)))


def _pick_blocking(*, Ct, Nt, Mt_total, Mt_cap, elem_size, act_tile, w_tile, interm_tile, dest_limit, fuse_bias):
    """Deterministic host-side blocking search.

    `Kb` divides `Ct`, `Nt_b` divides `Nt`, `out_subblock_h` is fixed at 1
    (keeps SubblockMajor pack order == tile-row-major so plain `untilize`
    can consume cb_mm_out, and keeps the DEST budget satisfiable).
    Objective: maximize `Mt * Nt_b` under `L1_CB_BUDGET`, tie-break on the
    larger `Nt_b`. Prefer the largest feasible `Kb` (fewest K-blocks).
    `Mt` is additionally capped by the grid split — see `_mt_cap`.
    """
    mt_candidates = sorted({min(c, Mt_total, Mt_cap) for c in (8, 4, 2, 1)}, reverse=True)

    for Kb in _divisors_desc(Ct):
        best = None
        for Mt in mt_candidates:
            for Nt_b in _divisors_desc(Nt):
                osw = max(d for d in _divisors_desc(Nt_b) if d <= dest_limit)
                size = _cb_bytes(
                    Kb=Kb,
                    Mt=Mt,
                    Nt_b=Nt_b,
                    out_subblock_w=osw,
                    elem_size=elem_size,
                    act_tile=act_tile,
                    w_tile=w_tile,
                    interm_tile=interm_tile,
                    fuse_bias=fuse_bias,
                )
                if size > L1_CB_BUDGET:
                    continue
                key = (Mt * Nt_b, Nt_b)
                if best is None or key > best[0]:
                    best = (key, Mt, Nt_b, osw)
        if best is not None:
            _, Mt, Nt_b, osw = best
            return Kb, Mt, Nt_b, osw

    raise RuntimeError(
        f"conv2d_nhwc: no feasible blocking under {L1_CB_BUDGET} B " f"(Ct={Ct}, Nt={Nt}, Mt_total={Mt_total})"
    )


def _grid_assignment(device, num_m_blocks):
    """Split the M-blocks across the full Tensix grid.

    Returns `(all_cores, [(core, start_m_block, num_m_blocks_here), ...])`.
    `split_work_to_cores` already clips to `min(num_cores, num_m_blocks)`, so a
    shape with fewer M-blocks than cores simply lights up fewer cores, and the
    assignment list is exactly the set of cores with work.
    """
    grid_size = device.compute_with_storage_grid_size()
    (
        _num_cores,
        all_cores,
        core_group_1,
        core_group_2,
        units_g1,
        units_g2,
    ) = ttnn.split_work_to_cores(grid_size, num_m_blocks, row_wise=True)

    assignment = []
    start = 0
    for group, per_core in ((core_group_1, units_g1), (core_group_2, units_g2)):
        if per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            assignment.append((core, start, per_core))
            start += per_core
    assert start == num_m_blocks, f"conv2d_nhwc: work split lost units ({start} != {num_m_blocks})"
    return all_cores, assignment


def _cb(index, dtype, page_size, num_pages, core_grid):
    return ttnn.CBDescriptor(
        total_size=page_size * num_pages,
        core_ranges=core_grid,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_size)],
    )


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    weight_tensor: ttnn.Tensor,
    bias_tensor,
    output_tensor: ttnn.Tensor,
    *,
    kernel_size: int,
    padding: int,
    stride: int,
    groups: int,
    dilation: int,
    compute_kernel_config=None,
) -> ttnn.ProgramDescriptor:
    # ---------------- 1. metadata + derived geometry ----------------
    N, H, W, C_in = (int(d) for d in input_tensor.shape)
    C_out = int(weight_tensor.shape[-1])

    dtype = input_tensor.dtype
    weight_dtype = weight_tensor.dtype
    elem_size = input_tensor.element_size()
    act_tile = ttnn.tile_size(dtype)
    w_tile = ttnn.tile_size(weight_dtype)

    ckc = resolve_compute_kernel_config(dtype, weight_dtype, compute_kernel_config)

    # cb_partials is the matmul's K-accumulation spill/reload region. Its format
    # is governed by the DEST accumulator, NOT by the activation dtype: with
    # fp32_dest_acc_en a bf16 partials CB would round the running sum to 7
    # mantissa bits at every K-block boundary and throw away the fp32 DEST.
    # cb_act_tiles follows the activation (it is the tilize target and the
    # matmul's in0) and cb_mm_out follows the output (untilize's source, and the
    # output tensor is allocated with the activation dtype), so both stay `dtype`.
    interm_dtype = ttnn.float32 if ckc.fp32_dest_acc_en else dtype
    interm_tile = ttnn.tile_size(interm_dtype)
    dest_limit = _dest_limit(
        fp32_dest_acc_en=ckc.fp32_dest_acc_en,
        dst_full_sync_en=ckc.dst_full_sync_en,
    )

    eff_k = dilation * (kernel_size - 1) + 1
    H_out = (H + 2 * padding - eff_k) // stride + 1
    W_out = (W + 2 * padding - eff_k) // stride + 1
    M_total = N * H_out * W_out
    Mt_total = _div_up(M_total, TILE)
    num_taps = kernel_size * kernel_size
    # Refinement 4 — grouped / depthwise. `gb.chans_cb` is the per-column-block
    # channel window (== C_in for dense), `gb.num_cblocks` is how many such
    # blocks tile the C_out axis. See `GroupBlocking`.
    gb = group_blocking(C_in=C_in, C_out=C_out, groups=groups, elem_size=elem_size)

    # Refinement 3 — channel alignment. Both tile counts round UP:
    #   Ct (K dim): the reader gathers Ct*32 channels per tap and zero-fills
    #     the bytes past the real channel count; prepare_conv2d_weights
    #     zero-pads the matching weight rows. `//` here used to make Ct == 0
    #     for C_in < 32, which is what raised "no feasible blocking" before
    #     that refinement.
    #   Nt (N dim): the matmul produces Nt*32 output columns; the writer
    #     truncates the last N-block's scatter to `out_row_bytes`.
    Ct = _div_up(gb.chans_cb, TILE)
    Nt = _div_up(C_out, TILE)
    # Column blocks partition the padded column-tile axis exactly — guaranteed
    # by group_blocking's `fits_in_padded_cout` check, asserted here because
    # every downstream index (weight page, bias page, output byte offset)
    # depends on n_block enumerating the *flat* column-tile blocks.
    assert Nt % gb.num_cblocks == 0, f"conv2d_nhwc: column blocking mismatch (Nt={Nt}, cblocks={gb.num_cblocks})"
    Nt_cb = Nt // gb.num_cblocks
    fuse_bias = bias_tensor is not None

    device = input_tensor.device()
    grid_size = device.compute_with_storage_grid_size()
    num_cores_available = int(grid_size.x) * int(grid_size.y)

    # Nt_b must divide the *column block*, not the whole N axis: an N-block that
    # straddled a column-block boundary would need two different K-windows.
    Kb, Mt, Nt_b, out_subblock_w = _pick_blocking(
        Ct=Ct,
        Nt=Nt_cb,
        Mt_total=Mt_total,
        Mt_cap=_mt_cap(Mt_total, num_cores_available),
        elem_size=elem_size,
        act_tile=act_tile,
        w_tile=w_tile,
        interm_tile=interm_tile,
        dest_limit=dest_limit,
        fuse_bias=fuse_bias,
    )
    sub_per_tap = Ct // Kb
    num_k_blocks = num_taps * sub_per_tap
    num_m_blocks = _div_up(Mt_total, Mt)
    # N-blocks enumerate the flat column-tile axis; the first
    # `n_sub_per_cblock` of them belong to column block 0, and so on.
    n_sub_per_cblock = Nt_cb // Nt_b
    num_n_blocks = Nt // Nt_b
    in1_num_subblocks = Nt_b // out_subblock_w

    # Refinement 5 — how the matmul K-accumulates. See `packer_l1_acc_enabled`.
    packer_l1_acc_en = packer_l1_acc_enabled(
        num_k_blocks=num_k_blocks,
        Kb=Kb,
        fuse_bias=fuse_bias,
        fp32_dest_acc_en=ckc.fp32_dest_acc_en,
    )

    stick_bytes = Kb * TILE * elem_size  # one im2col row restricted to the K-block
    out_slice_bytes = Nt_b * TILE * elem_size  # per-(m,n) chunk of an output stick

    # Channel-alignment masks. `c_in_bytes` is how many bytes an activation
    # stick actually holds (the Ct*32-channel gather window of a column block
    # may run past it, and gets zero-filled); `out_row_bytes` is how many bytes
    # of an output stick are real (the rest of the Nt*32-column matmul result
    # is dropped). `chans_cb_bytes` is the reader's per-column-block channel
    # base stride.
    c_in_bytes = C_in * elem_size
    chans_cb_bytes = gb.chans_cb * elem_size
    out_row_bytes = C_out * elem_size
    # True iff any (cblock, channel-slice) gather window overhangs the real
    # channel axis. False for the dense tile-aligned case, where every masking
    # branch in the reader compiles out.
    has_chan_tail = (Ct * TILE != gb.chans_cb) or (C_in != gb.num_cblocks * gb.chans_cb)

    # ---------------- 2. core grid + M-block work split ----------------
    core_grid, assignment = _grid_assignment(device, num_m_blocks)

    # ---------------- 3. circular buffers ----------------
    cbs = [
        _cb(CB_ACT_RM, dtype, stick_bytes, 64, core_grid),
        _cb(CB_WEIGHT_TILES, weight_dtype, w_tile, 2 * Kb * Nt_b, core_grid),
        _cb(CB_ZERO_SCRATCH, dtype, stick_bytes, 1, core_grid),
        _cb(CB_OUT_RM, dtype, act_tile, 2 * Nt_b, core_grid),
        _cb(CB_ACT_TILES, dtype, act_tile, Mt * Kb, core_grid),
        _cb(
            CB_PARTIALS,
            interm_dtype,
            interm_tile,
            _partials_pages(Mt=Mt, Nt_b=Nt_b, out_subblock_w=out_subblock_w, packer_l1_acc_en=packer_l1_acc_en),
            core_grid,
        ),
        _cb(CB_MM_OUT, dtype, act_tile, Mt * Nt_b, core_grid),
    ]
    if fuse_bias:
        cbs.append(_cb(CB_BIAS_TILES, weight_dtype, w_tile, Nt_b, core_grid))

    # ---------------- 4. kernels ----------------
    # Every kernel binary is identical across the grid; the per-core M-block
    # range (start_m_block, num_m_blocks_here) rides in as runtime args.
    act_addr = input_tensor.buffer_address()
    weight_addr = weight_tensor.buffer_address()
    bias_addr = bias_tensor.buffer_address() if fuse_bias else 0
    out_addr = output_tensor.buffer_address()

    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    compute_rt_args = ttnn.RuntimeArgs()
    for core, start_m_block, num_m_blocks_here in assignment:
        reader_rt_args[core.x][core.y] = [act_addr, start_m_block, num_m_blocks_here]
        writer_rt_args[core.x][core.y] = [weight_addr, bias_addr, out_addr, start_m_block, num_m_blocks_here]
        compute_rt_args[core.x][core.y] = [num_m_blocks_here]

    # Reader (NCRISC): implicit im2col gather.
    reader_ct_args = [
        H,
        W,
        H_out,
        W_out,
        M_total,
        Mt,
        num_n_blocks,
        num_k_blocks,
        sub_per_tap,
        kernel_size,
        padding,
        stride,
        dilation,
        stick_bytes,
        c_in_bytes,
        chans_cb_bytes,
        n_sub_per_cblock,
        1 if has_chan_tail else 0,
    ]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "conv2d_nhwc_reader.cpp"),
        core_ranges=core_grid,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # Writer (BRISC): weight + bias feed, output scatter.
    writer_ct_args = [
        M_total,
        Mt,
        num_n_blocks,
        num_k_blocks,
        Kb,
        Nt,
        Nt_b,
        1 if fuse_bias else 0,
        out_slice_bytes,
        out_row_bytes,
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(weight_tensor).get_compile_time_args())
    writer_ct_args.extend(
        ttnn.TensorAccessorArgs(bias_tensor).get_compile_time_args()
        if fuse_bias
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "conv2d_nhwc_writer.cpp"),
        core_ranges=core_grid,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # Compute (TRISC): tilize (hook) -> matmul_block -> bias -> untilize.
    compute_ct_args = [
        Mt,
        num_n_blocks,
        num_k_blocks,
        Kb,
        Nt_b,
        out_subblock_w,
        in1_num_subblocks,
        1 if fuse_bias else 0,
        1 if packer_l1_acc_en else 0,
    ]
    # No CB in this pipeline qualifies for UnpackToDestMode::UnpackToDestFp32:
    # the tag is exclusive to CBs whose every consumer reloads them with
    # copy_tile, and each intermediate here is read by an FPU op —
    # cb_act_tiles is matmul's in0, cb_mm_out is untilize's unpack source, and
    # cb_partials feeds add_tiles_bcast_rows on the fused-bias path (it is
    # copy_tile-only just on the no-bias path, so tagging it would make the two
    # bias modes numerically divergent for no measurable gain). Left Default.
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "conv2d_nhwc_compute.cpp"),
        core_ranges=core_grid,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt_args,
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ckc.math_fidelity,
            fp32_dest_acc_en=ckc.fp32_dest_acc_en,
            math_approx_mode=ckc.math_approx_mode,
            dst_full_sync_en=ckc.dst_full_sync_en,
        ),
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=cbs,
    )
