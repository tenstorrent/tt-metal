// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// groupnorm_sc_N_1_HW_C compute — regime `cluster_parallel_two_pass`.
//
// One work unit = (batch n, channel cluster k). A cluster is a whole number of
// groups AND a whole number of 32-channel tiles, so a unit is self-contained and
// needs no cross-core traffic. Per unit:
//
//   pass 1  accumulate_moments_block   sum(x) and sum(x^2) per channel-tile,
//                                      L1-accumulated across HW blocks
//   ------  group_stats                per group j: mask -> REDUCE_SCALAR -> mean,
//                                      var = E[x^2] - mean^2, scale = rsqrt(var+eps)
//   ------  expand_group_block         splat (mean, scale) back onto channel-tiles
//                                      through the same group masks
//   pass 2  apply_block                (x - mean) * scale * gamma + beta
//
// The HW axis is walked in blocks of BLOCK_HW_TILES — the live block knob shared
// with the reader and writer; nothing here is hardcoded to 1.
//
// DEVIATIONS from op_design.md (all recorded in the commit message):
//  (a) accumulate_moments_block uses pure L1Accumulation, not the design's
//      DestAccumulation::PerRow + L1Accumulation: the chain static_asserts that
//      L1 and DEST accumulation cannot be combined. DEST-resident accumulation
//      stays a perf lamp.
//  (b) expand_group_block is a per-tile loop over the group's span (span <= 2 for
//      every supported geometry, and it is NOT a block knob) because
//      L1Accumulation::AddToExisting pins the pack index to `base`, so a
//      tiles(span) walk cannot address span distinct outputs.
//  (c) the design's single cb_group_stats is split into cb_moment_scalar (reduce
//      outputs) + cb_mean_scalar / cb_scale_scalar (chain outputs): an in-place
//      pack over an already-pushed CB is not addressable.
//  (d) gamma is NOT folded into the scale tiles; it is applied as an extra
//      DestReuseBinary in the apply chain. Folding in place would need a
//      packer->unpacker barrier that the CB set cannot express, and the fused
//      form costs no extra pass.

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/common.h"

#include "../dependencies/ttnn/cpp/ttnn/kernel_lib/eltwise/core/chain.hpp"
#include "../dependencies/ttnn/cpp/ttnn/kernel_lib/eltwise/generators/fill.hpp"
#include "../dependencies/ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"
#include "../dependencies/ttnn/cpp/ttnn/kernel_lib/eltwise/unary/scalar.hpp"
#include "../dependencies/ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/basic.hpp"
#include "../dependencies/ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "../dependencies/ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

using namespace compute_kernel_lib;

namespace {

constexpr uint32_t cb_input_tiles = 0;
constexpr uint32_t cb_output_tiles = 1;
constexpr uint32_t cb_mask_tiles = 2;
constexpr uint32_t cb_scaler_ones = 3;
constexpr uint32_t cb_moment_sum = 4;
constexpr uint32_t cb_moment_sq = 5;
constexpr uint32_t cb_masked_moment = 6;
constexpr uint32_t cb_moment_scalar = 7;
constexpr uint32_t cb_mean_tiles = 8;
constexpr uint32_t cb_scale_tiles = 9;
constexpr uint32_t cb_gamma_tiles = 10;
constexpr uint32_t cb_beta_tiles = 11;
constexpr uint32_t cb_rm_sticks = 12;
constexpr uint32_t cb_mean_scalar = 13;
constexpr uint32_t cb_scale_scalar = 14;
constexpr uint32_t cb_row_mask = 16;
constexpr uint32_t cb_partial_moments = 17;
constexpr uint32_t cb_moment_total_out = 18;
constexpr uint32_t cb_moment_total_in = 19;

// --- geometry (shared geom_ct prefix) ---------------------------------------
constexpr uint32_t tensor_hw_tiles = get_compile_time_arg_val(0);
constexpr uint32_t cluster_c_tiles = get_compile_time_arg_val(2);
constexpr uint32_t groups_per_cluster = get_compile_time_arg_val(3);
constexpr uint32_t num_clusters = get_compile_time_arg_val(4);
constexpr uint32_t BLOCK_HW_TILES = get_compile_time_arg_val(5);
constexpr uint32_t num_hw_blocks = get_compile_time_arg_val(6);
constexpr uint32_t Cg = get_compile_time_arg_val(7);
constexpr uint32_t num_mask_tiles = get_compile_time_arg_val(9);
constexpr uint32_t max_span = get_compile_time_arg_val(10);
// Refinement 1 — ragged tails. hw_tail == 0 means HW is tile-aligned and the
// whole row-masking path below is compiled out.
constexpr uint32_t hw_tail = get_compile_time_arg_val(11);
// (12) c_tail — reader-side only; the channel padding lanes are already
// excluded by every group mask, since num_groups * Cg == C.

// Refinement 3 — regime selector. REGIME_HW_SPLIT (1) splits the spatial axis
// across `hw_split_factor` cores per work unit and combines the per-core
// (Sigma-x, Sigma-x^2) partials cross-core; everything downstream of the combine
// is the unchanged Phase-0 schedule, just fed from cb_moment_total_in.
constexpr uint32_t regime_id = get_compile_time_arg_val(13);
constexpr uint32_t hw_split_factor = get_compile_time_arg_val(14);
constexpr uint32_t resident_input = get_compile_time_arg_val(15);
constexpr uint32_t inv_n_g_bits = get_compile_time_arg_val(16);
constexpr uint32_t eps_bits = get_compile_time_arg_val(17);
constexpr uint32_t input_is_rm = get_compile_time_arg_val(18);

constexpr bool HW_SPLIT = (regime_id == 1);
// Remote partial pairs landed on the combiner: (S-1) slots of [sum tiles | sq tiles].
constexpr uint32_t n_partial_tiles = HW_SPLIT ? (hw_split_factor - 1) * 2 * cluster_c_tiles : 0;
constexpr uint32_t n_total_tiles = 2 * cluster_c_tiles;

constexpr uint32_t TILE_HW = 32;

// ---------------------------------------------------------------------------
// Block operation 1 — accumulate_moments_block
// ---------------------------------------------------------------------------
// One HW block of `rows_this` tiles for channel-tile `t` is folded into
// moment tile `t`. ACC selects seed-vs-accumulate for the first HW block.
template <L1Accumulation ACC>
FORCE_INLINE void accumulate_moments_span(uint32_t t, uint32_t base, uint32_t count) {
    // sum(x)
    eltwise_chain(
        IterationShape::tiles(count),
        CopyTile<
            input(
                cb_input_tiles,
                WaitPolicy::None,
                PopPolicy::None,
                OperandKind::Block,
                DataFormatReconfig::Enabled,
                TileOffset::Set),
            Dst::D0>{base},
        PackTile<
            output(
                cb_moment_sum,
                ReservePolicy::None,
                PushPolicy::None,
                DataFormatReconfig::Enabled,
                PackRelu::Disabled,
                ACC,
                DestAccumulation::Disabled,
                TileOffset::Set),
            Dst::D0>{t});

    // sum(x^2)
    eltwise_chain(
        IterationShape::tiles(count),
        BinaryFpu<
            BinaryFpuOp::Mul,
            input(
                cb_input_tiles,
                WaitPolicy::None,
                PopPolicy::None,
                OperandKind::Block,
                DataFormatReconfig::Enabled,
                TileOffset::Set),
            input(
                cb_input_tiles,
                BroadcastDim::None,
                WaitPolicy::None,
                PopPolicy::None,
                OperandKind::Block,
                TileOffset::Set),
            Dst::D0>{base, base},
        PackTile<
            output(
                cb_moment_sq,
                ReservePolicy::None,
                PushPolicy::None,
                DataFormatReconfig::Enabled,
                PackRelu::Disabled,
                ACC,
                DestAccumulation::Disabled,
                TileOffset::Set),
            Dst::D0>{t});
}

// Refinement 1 (`hw_non_aligned`): the SINGLE trailing HW tile of the tensor.
// Its rows >= HW % 32 are padding, so both moments are pre-multiplied by the
// row mask (1.0 on real rows, 0.0 on padding) before being folded into the
// same L1 accumulator. `n_g = Cg * HW` already counts only the real rows.
template <L1Accumulation ACC>
FORCE_INLINE void accumulate_moments_tail_tile(uint32_t t, uint32_t base) {
    // sum(x * rowmask)
    eltwise_chain(
        IterationShape::one_tile(),
        BinaryFpu<
            BinaryFpuOp::Mul,
            input(
                cb_input_tiles,
                WaitPolicy::None,
                PopPolicy::None,
                OperandKind::Block,
                DataFormatReconfig::Enabled,
                TileOffset::Set),
            input(cb_row_mask, BroadcastDim::None, WaitPolicy::None, PopPolicy::None, OperandKind::Scalar),
            Dst::D0>{base},
        PackTile<
            output(
                cb_moment_sum,
                ReservePolicy::None,
                PushPolicy::None,
                DataFormatReconfig::Enabled,
                PackRelu::Disabled,
                ACC,
                DestAccumulation::Disabled,
                TileOffset::Set),
            Dst::D0>{t});

    // sum(x^2 * rowmask)  — the mask is 0/1, so masking after the square is
    // identical to masking before it.
    eltwise_chain(
        IterationShape::one_tile(),
        BinaryFpu<
            BinaryFpuOp::Mul,
            input(
                cb_input_tiles,
                WaitPolicy::None,
                PopPolicy::None,
                OperandKind::Block,
                DataFormatReconfig::Enabled,
                TileOffset::Set),
            input(
                cb_input_tiles,
                BroadcastDim::None,
                WaitPolicy::None,
                PopPolicy::None,
                OperandKind::Block,
                TileOffset::Set),
            Dst::D0>{base, base},
        DestReuseBinary<
            input(cb_row_mask, WaitPolicy::None, PopPolicy::None, OperandKind::Scalar),
            BinaryFpuOp::Mul,
            DestReuseType::DEST_TO_SRCB,
            Dst::D0>{},
        PackTile<
            output(
                cb_moment_sq,
                ReservePolicy::None,
                PushPolicy::None,
                DataFormatReconfig::Enabled,
                PackRelu::Disabled,
                ACC,
                DestAccumulation::Disabled,
                TileOffset::Set),
            Dst::D0>{t});
}

// Dispatcher: fold one HW block of channel-tile `t` into moment tile `t`.
// `has_tail_tile` is true only for the last tile of the last block when HW is
// not tile-aligned; everything else takes the unchanged bulk path.
template <L1Accumulation ACC>
FORCE_INLINE void accumulate_moments_block(uint32_t t, uint32_t rows_this, bool has_tail_tile, uint32_t block_off) {
    // `block_off` is 0 on the streaming path (the block is always at the CB
    // front). Under the residency fast path nothing is popped during the moment
    // pass, so block `b` sits at `b * BLOCK_HW_TILES * cluster_c_tiles`.
    const uint32_t base = block_off + t * rows_this;
    if constexpr (hw_tail == 0) {
        (void)has_tail_tile;
        accumulate_moments_span<ACC>(t, base, rows_this);
    } else {
        if (!has_tail_tile) {
            accumulate_moments_span<ACC>(t, base, rows_this);
        } else if (rows_this > 1) {
            accumulate_moments_span<ACC>(t, base, rows_this - 1);
            accumulate_moments_tail_tile<L1Accumulation::AddToExisting>(t, base + rows_this - 1);
        } else {
            accumulate_moments_tail_tile<ACC>(t, base);
        }
    }
}

// ---------------------------------------------------------------------------
// Block operation 1b — combine_partial_moments (Refinement 3, combiner core only)
// ---------------------------------------------------------------------------
// Fold this core's own moments plus the (S-1) remote partial pairs that landed
// in cb_partial_moments into cb_moment_total_out, laid out as
// [Sigma-x tiles | Sigma-x^2 tiles]. Both moments are plain sums over disjoint
// HW slices, so the combine is a plain add — no rescaling, and
// `inv_n_g = 1/(Cg * HW)` still divides by the FULL spatial extent.
//
// The seed pass copies this core's own moment tile into the total. The fold pass
// is an explicit FPU add reading the total back in place, NOT a pack-side L1
// accumulation: measured on device, `L1Accumulation::AddToExisting` into
// cb_moment_total_out produced garbage bit patterns here (own partial 16.0 and
// remote partial 16.0 both verified correct going in, total came back 5.23 /
// -3.2e-38 / 9.4e16), while the seed-only path (S=1) was exact. The in-place add
// keeps the fold in the FPU where both operands carry an explicit tile index.
template <uint32_t cb_src>
FORCE_INLINE void seed_total_tile(uint32_t src_idx, uint32_t dst_idx) {
    eltwise_chain(
        IterationShape::one_tile(),
        CopyTile<
            input(
                cb_src,
                WaitPolicy::None,
                PopPolicy::None,
                OperandKind::Block,
                DataFormatReconfig::Enabled,
                TileOffset::Set),
            Dst::D0>{src_idx},
        PackTile<
            output(
                cb_moment_total_out,
                ReservePolicy::None,
                PushPolicy::None,
                DataFormatReconfig::Enabled,
                PackRelu::Disabled,
                L1Accumulation::Disabled,
                DestAccumulation::Disabled,
                TileOffset::Set),
            Dst::D0>{dst_idx});
}

// total[dst_idx] += cb_partial_moments[src_idx], in place in cb_moment_total_out.
// Legal per tile: the unpack of the total tile completes before the pack
// overwrites it, and cb_moment_total_out has no concurrent consumer here (the
// writer only waits on it after cb_push_back below).
FORCE_INLINE void fold_partial_tile(uint32_t src_idx, uint32_t dst_idx) {
    eltwise_chain(
        IterationShape::one_tile(),
        BinaryFpu<
            BinaryFpuOp::Add,
            input(
                cb_moment_total_out,
                WaitPolicy::None,
                PopPolicy::None,
                OperandKind::Block,
                DataFormatReconfig::Enabled,
                TileOffset::Set),
            input(
                cb_partial_moments,
                BroadcastDim::None,
                WaitPolicy::None,
                PopPolicy::None,
                OperandKind::Block,
                TileOffset::Set),
            Dst::D0>{dst_idx, src_idx},
        PackTile<
            output(
                cb_moment_total_out,
                ReservePolicy::None,
                PushPolicy::None,
                DataFormatReconfig::Enabled,
                PackRelu::Disabled,
                L1Accumulation::Disabled,
                DestAccumulation::Disabled,
                TileOffset::Set),
            Dst::D0>{dst_idx});
}

FORCE_INLINE void combine_partial_moments() {
    cb_reserve_back(cb_moment_total_out, n_total_tiles);
    for (uint32_t t = 0; t < cluster_c_tiles; ++t) {
        seed_total_tile<cb_moment_sum>(t, t);
        seed_total_tile<cb_moment_sq>(t, cluster_c_tiles + t);
    }
    for (uint32_t i = 0; i + 1 < hw_split_factor; ++i) {
        const uint32_t slot = i * n_total_tiles;
        for (uint32_t t = 0; t < cluster_c_tiles; ++t) {
            fold_partial_tile(slot + t, t);
            fold_partial_tile(slot + cluster_c_tiles + t, cluster_c_tiles + t);
        }
    }
    cb_push_back(cb_moment_total_out, n_total_tiles);
}

// ---------------------------------------------------------------------------
// Block operation 2 — masked group reduction of one moment CB
// ---------------------------------------------------------------------------
template <uint32_t cb_moment>
FORCE_INLINE void reduce_group_moment(uint32_t t0, uint32_t span, uint32_t mask_base) {
    // Pre-mask: REDUCE_SCALAR cannot take a partial scaler, so the channel lanes
    // outside the group are zeroed before the reduce and the scaler stays 1.0.
    eltwise_chain(
        IterationShape::tiles(span),
        BinaryFpu<
            BinaryFpuOp::Mul,
            input(
                cb_moment,
                WaitPolicy::None,
                PopPolicy::None,
                OperandKind::Block,
                DataFormatReconfig::Enabled,
                TileOffset::Set),
            input(
                cb_mask_tiles,
                BroadcastDim::None,
                WaitPolicy::None,
                PopPolicy::None,
                OperandKind::Block,
                TileOffset::Set),
            Dst::D0>{t0, mask_base},
        PackTile<output(cb_masked_moment), Dst::D0>{});

    // Pad the block out to `max_span` tiles with zeros. A circular buffer never
    // wraps mid-block, so every push into cb_masked_moment must be the SAME size
    // as its capacity (`max_span`); a group with `span < max_span` would
    // otherwise leave the write pointer off a block boundary and the next
    // group's multi-tile push would run off the end of the buffer. The zero
    // tiles contribute nothing to the SUM reduce below.
    if (span < max_span) {
        eltwise_chain(
            IterationShape::tiles(max_span - span),
            FillScalar<Dst::D0>{0.0f},
            PackTile<output(cb_masked_moment), Dst::D0>{});
    }

    reduce<
        PoolType::SUM,
        ReduceDim::REDUCE_SCALAR,
        cb_masked_moment,
        cb_scaler_ones,
        cb_moment_scalar,
        ReduceInputPolicy::BulkWaitBulkPop>(
        ReduceInputBlockShape::of(1, max_span, 1),
        ReduceInputMemoryLayout::contiguous(),
        NoAccumulation{},
        NoOp{},
        ReducePartialScaler::none());
}

// ---------------------------------------------------------------------------
// Block operation 3 — group_stats
// ---------------------------------------------------------------------------
// Consumes the two scalar moment tiles and emits mean and rsqrt(var+eps) from a
// single DEST window (two PackTile elements to two distinct output CBs).
FORCE_INLINE void group_stats() {
    eltwise_chain(
        IterationShape::one_tile(),
        CopyTile<
            input(
                cb_moment_scalar,
                WaitPolicy::None,
                PopPolicy::None,
                OperandKind::Scalar,
                DataFormatReconfig::Enabled,
                TileOffset::Set),
            Dst::D0>{0},
        MulUnary<Dst::D0>{inv_n_g_bits},  // mean = sum / (Cg * HW)
        CopyTile<
            input(
                cb_moment_scalar,
                WaitPolicy::Upfront,
                PopPolicy::AtEnd,
                OperandKind::Scalar,
                DataFormatReconfig::Disabled,
                TileOffset::Set),
            Dst::D1>{1},
        MulUnary<Dst::D1>{inv_n_g_bits},               // E[x^2]
        MulBinary<Dst::D0, Dst::D0, Dst::D2>{},        // mean^2
        SubBinary<Dst::D1, Dst::D2, Dst::D1>{},        // var
        AddUnary<Dst::D1>{eps_bits},                   // var + eps
        Rsqrt<Approx::Exact, Legacy::Off, Dst::D1>{},  // scale
        PackTile<output(cb_mean_scalar), Dst::D0>{},
        PackTile<output(cb_scale_scalar), Dst::D1>{});
}

// ---------------------------------------------------------------------------
// Block operation 4 — expand_group_block
// ---------------------------------------------------------------------------
// Splat one group statistic back onto its channel lanes via the group mask,
// accumulating into the (already zeroed) per-channel-tile stat block.
template <uint32_t cb_stat_scalar, uint32_t cb_stat_tiles>
FORCE_INLINE void expand_group_block(uint32_t t0, uint32_t span, uint32_t mask_base) {
    for (uint32_t k = 0; k < span; ++k) {
        eltwise_chain(
            IterationShape::one_tile(),
            BinaryFpu<
                BinaryFpuOp::Mul,
                input(
                    cb_mask_tiles,
                    WaitPolicy::None,
                    PopPolicy::None,
                    OperandKind::Scalar,
                    DataFormatReconfig::Enabled,
                    TileOffset::Set),
                input(
                    cb_stat_scalar,
                    BroadcastDim::Scalar,
                    WaitPolicy::None,
                    PopPolicy::None,
                    OperandKind::Scalar,
                    TileOffset::Unset),
                Dst::D0>{mask_base + k},
            PackTile<
                output(
                    cb_stat_tiles,
                    ReservePolicy::None,
                    PushPolicy::None,
                    DataFormatReconfig::Enabled,
                    PackRelu::Disabled,
                    L1Accumulation::AddToExisting,
                    DestAccumulation::Disabled,
                    TileOffset::Set),
                Dst::D0>{t0 + k});
    }
}

template <uint32_t cb_stat_tiles>
FORCE_INLINE void zero_stat_tiles() {
    eltwise_chain(
        IterationShape::tiles(cluster_c_tiles),
        FillScalar<Dst::D0>{0.0f},
        PackTile<output(cb_stat_tiles, ReservePolicy::None, PushPolicy::None), Dst::D0>{});
}

// ---------------------------------------------------------------------------
// Block operation 5 — apply_block
// ---------------------------------------------------------------------------
// (x - mean) * scale * gamma + beta over one HW block, walked as a
// (cluster_c_tiles x rows_this) grid so the per-channel-tile statistics and the
// affine tiles are Col-indexed operands.
FORCE_INLINE void apply_block(uint32_t rows_this) {
    eltwise_chain(
        IterationShape::grid(cluster_c_tiles, rows_this),
        BinaryFpu<
            BinaryFpuOp::Sub,
            input(cb_input_tiles, WaitPolicy::Upfront, PopPolicy::AtEnd, OperandKind::Block),
            input(
                cb_mean_tiles,
                BroadcastDim::None,
                WaitPolicy::None,
                PopPolicy::None,
                OperandKind::Col,
                TileOffset::Unset),
            Dst::D0>{},
        DestReuseBinary<
            input(cb_scale_tiles, WaitPolicy::None, PopPolicy::None, OperandKind::Col),
            BinaryFpuOp::Mul,
            DestReuseType::DEST_TO_SRCB,
            Dst::D0>{},
        DestReuseBinary<
            input(cb_gamma_tiles, WaitPolicy::None, PopPolicy::None, OperandKind::Col),
            BinaryFpuOp::Mul,
            DestReuseType::DEST_TO_SRCB,
            Dst::D0>{},
        DestReuseBinary<
            input(cb_beta_tiles, WaitPolicy::None, PopPolicy::None, OperandKind::Col),
            BinaryFpuOp::Add,
            DestReuseType::DEST_TO_SRCB,
            Dst::D0>{},
        PackTile<output(cb_output_tiles), Dst::D0>{});
}

// The host snaps BLOCK_HW_TILES to a divisor of the PER-CORE HW extent, so every
// block on every core is exactly BLOCK_HW_TILES tall (the uniform-block invariant
// the CBs depend on). Kept as a named accessor so the block knob stays the single
// source of truth for the loop extent.
FORCE_INLINE uint32_t rows_in_block(uint32_t) { return BLOCK_HW_TILES; }

// ROW_MAJOR input: turn the reader's 32-channel sub-sticks into `rows_this`
// contiguous tiles per channel-tile, preserving the channel-major CB contract.
FORCE_INLINE void tilize_block(uint32_t rows_this) {
    if constexpr (input_is_rm == 1) {
        for (uint32_t t = 0; t < cluster_c_tiles; ++t) {
            compute_kernel_lib::tilize<1, cb_rm_sticks, cb_input_tiles>(rows_this, TILE_HW * rows_this);
        }
    } else {
        (void)rows_this;
    }
}

}  // namespace

void kernel_main() {
    compute_kernel_hw_startup(cb_input_tiles, cb_mask_tiles, cb_output_tiles);

    const uint32_t units = get_arg_val<uint32_t>(1);
    const bool is_combiner = get_arg_val<uint32_t>(2) != 0;
    // Only the core holding the LAST HW tile of the tensor sees the ragged tail.
    // In REGIME_CLUSTER_PARALLEL that is every core (each owns the whole axis).
    const bool owns_hw_tail = get_arg_val<uint32_t>(3) != 0;

    // Kernel-lifetime constants produced once by the reader; never popped.
    cb_wait_front(cb_mask_tiles, num_mask_tiles);
    cb_wait_front(cb_scaler_ones, 1);
    if constexpr (hw_tail != 0) {
        cb_wait_front(cb_row_mask, 1);
    }

    const uint32_t unit_start = get_arg_val<uint32_t>(0);
    // Refinement 4 lever 2 — gamma/beta are cluster-invariant, so the reader
    // ingests them once per distinct cluster. Mirror the reader's predicate
    // exactly (same `unit % num_clusters`) so wait/pop stay matched.
    uint32_t prev_k = 0xFFFFFFFFu;

    for (uint32_t u = 0; u < units; ++u) {
        const uint32_t unit = unit_start + u;
        const uint32_t k = unit % num_clusters;
        if (k != prev_k) {
            cb_wait_front(cb_gamma_tiles, cluster_c_tiles);
            cb_wait_front(cb_beta_tiles, cluster_c_tiles);
            prev_k = k;
        }

        // ---------------- pass 1: moments ---------------------------------
        cb_reserve_back(cb_moment_sum, cluster_c_tiles);
        cb_reserve_back(cb_moment_sq, cluster_c_tiles);

        for (uint32_t b = 0; b < num_hw_blocks; ++b) {
            const uint32_t rows_this = rows_in_block(b);
            // Under residency the tilize (ROW_MAJOR path) also runs once: the
            // tiles it produces stay in cb_input_tiles for the apply pass.
            tilize_block(rows_this);
            const uint32_t block_tiles = rows_this * cluster_c_tiles;
            // Residency: nothing is popped in this pass, so wait cumulatively
            // and index block `b` at its offset from the (unmoving) CB front.
            const uint32_t block_off = (resident_input != 0) ? b * block_tiles : 0;
            cb_wait_front(cb_input_tiles, block_off + block_tiles);
            const bool has_tail_tile = (hw_tail != 0) && owns_hw_tail && (b == num_hw_blocks - 1);
            for (uint32_t t = 0; t < cluster_c_tiles; ++t) {
                if (b == 0) {
                    accumulate_moments_block<L1Accumulation::Enabled>(t, rows_this, has_tail_tile, block_off);
                } else {
                    accumulate_moments_block<L1Accumulation::AddToExisting>(t, rows_this, has_tail_tile, block_off);
                }
            }
            if constexpr (resident_input == 0) {
                cb_pop_front(cb_input_tiles, block_tiles);
            }
        }

        // packer -> unpacker barrier on the L1-accumulated moment tiles.
        // In REGIME_HW_SPLIT this push ALSO publishes the partials to the local
        // writer: on a non-combiner core the writer is the sole consumer (it
        // NoC-writes them to the combiner and pops); on the combiner this kernel
        // is the sole consumer. One producer, one consumer, per core.
        cb_push_back(cb_moment_sum, cluster_c_tiles);
        cb_push_back(cb_moment_sq, cluster_c_tiles);

        if constexpr (HW_SPLIT) {
            if (is_combiner) {
                cb_wait_front(cb_moment_sum, cluster_c_tiles);
                cb_wait_front(cb_moment_sq, cluster_c_tiles);
                cb_wait_front(cb_partial_moments, n_partial_tiles);
                combine_partial_moments();
                cb_pop_front(cb_partial_moments, n_partial_tiles);
                cb_pop_front(cb_moment_sum, cluster_c_tiles);
                cb_pop_front(cb_moment_sq, cluster_c_tiles);
            }
            // Every core (the combiner included, via the loopback multicast)
            // takes the totals from the writer-published landing CB.
            cb_wait_front(cb_moment_total_in, n_total_tiles);
        } else {
            cb_wait_front(cb_moment_sum, cluster_c_tiles);
            cb_wait_front(cb_moment_sq, cluster_c_tiles);
        }

        // ---------------- group statistics + expansion ---------------------
        cb_reserve_back(cb_mean_tiles, cluster_c_tiles);
        cb_reserve_back(cb_scale_tiles, cluster_c_tiles);
        zero_stat_tiles<cb_mean_tiles>();
        zero_stat_tiles<cb_scale_tiles>();

        uint32_t mask_base = 0;
        for (uint32_t j = 0; j < groups_per_cluster; ++j) {
            const uint32_t lo = j * Cg;
            const uint32_t t0 = lo / TILE_HW;
            const uint32_t span = ((lo + Cg - 1) / TILE_HW) - t0 + 1;

            if constexpr (HW_SPLIT) {
                // Same reduction, fed from the combined totals: [sum | sq].
                reduce_group_moment<cb_moment_total_in>(t0, span, mask_base);
                reduce_group_moment<cb_moment_total_in>(cluster_c_tiles + t0, span, mask_base);
            } else {
                reduce_group_moment<cb_moment_sum>(t0, span, mask_base);
                reduce_group_moment<cb_moment_sq>(t0, span, mask_base);
            }
            group_stats();

            cb_wait_front(cb_mean_scalar, 1);
            cb_wait_front(cb_scale_scalar, 1);
            expand_group_block<cb_mean_scalar, cb_mean_tiles>(t0, span, mask_base);
            expand_group_block<cb_scale_scalar, cb_scale_tiles>(t0, span, mask_base);
            cb_pop_front(cb_mean_scalar, 1);
            cb_pop_front(cb_scale_scalar, 1);

            mask_base += span;
        }

        // packer -> unpacker barrier on the expanded statistic tiles
        cb_push_back(cb_mean_tiles, cluster_c_tiles);
        cb_push_back(cb_scale_tiles, cluster_c_tiles);
        cb_wait_front(cb_mean_tiles, cluster_c_tiles);
        cb_wait_front(cb_scale_tiles, cluster_c_tiles);

        // ---------------- pass 2: apply ------------------------------------
        for (uint32_t b = 0; b < num_hw_blocks; ++b) {
            const uint32_t rows_this = rows_in_block(b);
            // Residency: the tiles are already in cb_input_tiles (never popped
            // by the moment pass), so neither the reader nor the tilize runs a
            // second time. apply_block's own Upfront/AtEnd lifecycle then walks
            // the CB block by block exactly as on the streaming path.
            if constexpr (resident_input == 0) {
                tilize_block(rows_this);
            }
            apply_block(rows_this);
        }

        if constexpr (HW_SPLIT) {
            cb_pop_front(cb_moment_total_in, n_total_tiles);
        } else {
            cb_pop_front(cb_moment_sum, cluster_c_tiles);
            cb_pop_front(cb_moment_sq, cluster_c_tiles);
        }
        cb_pop_front(cb_mean_tiles, cluster_c_tiles);
        cb_pop_front(cb_scale_tiles, cluster_c_tiles);
        // Lever 2: retire the affine tiles only when the NEXT unit uses a
        // different cluster (or there is no next unit) — that is exactly when
        // the reader will push a fresh pair.
        const uint32_t next_k = (u + 1 < units) ? ((unit + 1) % num_clusters) : 0xFFFFFFFFu;
        if (next_k != k) {
            cb_pop_front(cb_gamma_tiles, cluster_c_tiles);
            cb_pop_front(cb_beta_tiles, cluster_c_tiles);
            prev_k = 0xFFFFFFFFu;
        }
    }
}
