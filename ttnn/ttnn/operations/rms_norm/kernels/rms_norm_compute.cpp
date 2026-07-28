// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// rms_norm compute — op_design.md §8.
//
// Per core, per row-block of HT_BLOCK tile-rows:
//
//   pass A (all NW chunks):  [tilize] -> square -> reduce<SUM,REDUCE_ROW,
//                            AccumulateViaAdd> into cb_partials, finalizing the
//                            last chunk with reduce_mean(n_reduced = W)
//   phase 4 (once):          AddUnary(eps) -> Rsqrt  => cb_rms_recip (1/rms)
//   pass B (all NW chunks):  [tilize] -> mul<Col>(x, 1/rms) -> mul<Row>(., gamma)
//                            -> [untilize]
//
// Under the cross-core W-split (W_SPLIT, §4.2) this core owns only a SLICE of W,
// so pass A stops one step earlier: no chunk finalizes, cb_partials keeps the
// RAW elementwise x^2 accumulator, and a copy publishes it for the writer's
// gather leg. The combine then folds the gathered accumulators with the SAME
// reduce — the local chunk-accumulate, done across cores instead of across
// chunks — and n_reduced stays the grand total W. Its shape is the CW1 x CW2
// topology knob (Refinement 3): CW2 == 1 is one flat fold on the root over all
// CW tiles; CW2 > 1 stages it, so each row LEADER folds CW1 tiles raw (again
// never finalizing) and the root finalizes over CW2 row-sums. Phase 4 onwards is
// byte identical on every core; only the producer of cb_rms_sum changes (the
// reader's multicast receive instead of phase 3).
//
// Every loop trip count and every helper block shape is a function of the block
// knobs (HT_BLOCK / WT_CHUNK / NW / CW) — never of a whole-op dimension.
//
// All compute goes through ttnn/cpp/ttnn/kernel_lib helpers. Phases 2, 5 and 6
// drop from the `square`/`mul` convenience wrappers to `eltwise_chain` directly
// in the resident regimes, because only the chain surface exposes the
// per-operand TileOffset the resident-block fast path needs (the convenience
// wrappers do not forward it). That is a helper *overload* choice, not a raw-LLK
// substitution: the same BinaryFpu + PackTile elements the wrappers emit.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_scalar.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

namespace {
constexpr uint32_t cb_input_tiles = 0;
constexpr uint32_t cb_gamma = 1;
constexpr uint32_t cb_scaler = 2;
constexpr uint32_t cb_input_rm = 3;
constexpr uint32_t cb_gamma_rm = 4;
constexpr uint32_t cb_ones = 5;
constexpr uint32_t cb_group_partials = 6;
constexpr uint32_t cb_rms_mean = 7;
constexpr uint32_t cb_partial_out = 8;
constexpr uint32_t cb_group_partials2 = 9;
constexpr uint32_t cb_output_tiles = 16;
constexpr uint32_t cb_output_rm = 17;
constexpr uint32_t cb_x_squared = 24;
constexpr uint32_t cb_partials = 25;
constexpr uint32_t cb_rms_sum = 26;
constexpr uint32_t cb_rms_recip = 27;
constexpr uint32_t cb_scaled = 28;
}  // namespace

namespace ckl = compute_kernel_lib;

void kernel_main() {
    // ---- regime flags (§5.2) ----
    constexpr bool IS_RM = get_compile_time_arg_val(0) != 0;
    constexpr bool HAS_GAMMA = get_compile_time_arg_val(1) != 0;
    constexpr bool IS_RM_GAMMA = get_compile_time_arg_val(2) != 0;
    constexpr bool X_RESIDENT = get_compile_time_arg_val(3) != 0;
    constexpr bool GAMMA_RESIDENT = get_compile_time_arg_val(4) != 0;
    constexpr bool HAS_PARTIAL_W = get_compile_time_arg_val(5) != 0;
    // ---- block knobs (§1.2) ----
    constexpr uint32_t WT = get_compile_time_arg_val(6);
    constexpr uint32_t WT_CHUNK = get_compile_time_arg_val(7);
    constexpr uint32_t WT_LAST = get_compile_time_arg_val(8);
    constexpr uint32_t NW = get_compile_time_arg_val(9);
    constexpr uint32_t HT_BLOCK = get_compile_time_arg_val(10);
    // W-chunks the reader coalesces per push on the resident TILE path (see
    // rms_norm_program_descriptor._x_read_chunks). The cumulative wait below has
    // to be quantized to this, since that is the granularity data becomes
    // visible at. Always 1 on the RM path (compute's own tilize is the producer).
    constexpr uint32_t X_READ_CHUNKS = IS_RM ? 1u : get_compile_time_arg_val(11);
    // ---- geometry ----
    constexpr uint32_t W_VALID_LAST = get_compile_time_arg_val(12);
    constexpr uint32_t N_REDUCED = get_compile_time_arg_val(13);  // true element count == W
    // ---- cross-core W-split (§4.2) ----
    constexpr bool W_SPLIT = get_compile_time_arg_val(14) != 0;
    constexpr uint32_t CW = get_compile_time_arg_val(15);   // cores per combine group
    constexpr uint32_t CW1 = get_compile_time_arg_val(16);  // stage-1 fan-in (row -> leader)
    constexpr uint32_t CW2 = get_compile_time_arg_val(17);  // stage-2 fan-in (leaders -> root)
    constexpr bool TWO_STAGE = CW2 > 1;
    static_assert(CW1 * CW2 == CW, "combine stages must tile CW");

    static_assert(WT_LAST == WT_CHUNK, "compute assumes uniform chunk widths");
    static_assert(NW * WT_CHUNK == WT, "chunking must tile Wt exactly");
    static_assert(!(NW > 1 && HT_BLOCK > 1), "R7: NW > 1 requires HT_BLOCK == 1");
    static_assert(X_READ_CHUNKS >= 1 && NW % X_READ_CHUNKS == 0, "read batch must tile NW");

    const uint32_t num_tile_rows = get_arg_val<uint32_t>(0);
    const uint32_t eps_bits = get_arg_val<uint32_t>(1);
    const uint32_t is_root = get_arg_val<uint32_t>(2);
    const uint32_t is_last_w_core = get_arg_val<uint32_t>(3);
    const uint32_t is_leader = get_arg_val<uint32_t>(4);

    // Filler core (inside a group's multicast rectangle, owns no data).
    if (num_tile_rows == 0) {
        return;
    }

    compute_kernel_hw_startup(cb_input_tiles, cb_scaler, cb_output_tiles);

    // ---- phase 0a: RM gamma tilized once and held resident -----------------
    if constexpr (HAS_GAMMA && GAMMA_RESIDENT && IS_RM_GAMMA) {
        for (uint32_t wc = 0; wc < NW; ++wc) {
            ckl::tilize<WT_CHUNK, cb_gamma_rm, cb_gamma>(/*num_blocks=*/1, /*total_input_pages=*/1);
        }
    }
    if constexpr (HAS_GAMMA && GAMMA_RESIDENT) {
        // R8: waited once, held for the whole kernel, never popped.
        cb_wait_front(cb_gamma, WT);
    }

    // Non-tile-aligned W: the 0/1 mask tile the reader filled zeroes the padded
    // lanes of the LAST reduce-dim tile. n_reduced stays the true count (W).
    // Under a W-split only the core whose slice ENDS on the tensor's last
    // W-tile owns that tile, so only it applies the mask.
    const auto partial = (HAS_PARTIAL_W && (!W_SPLIT || is_last_w_core != 0))
                             ? ckl::ReducePartialScaler::partial_mask(W_VALID_LAST, 0)
                             : ckl::ReducePartialScaler::none();

    constexpr uint32_t cb_scale_out = HAS_GAMMA ? cb_scaled : cb_output_tiles;
    // A REDUCE_ROW result is column-shaped, so it broadcasts back across
    // columns via BroadcastDim::Col (eltwise_chain.hpp:526-528).
    constexpr auto rms_kind = (HT_BLOCK > 1) ? ckl::OperandKind::Col : ckl::OperandKind::Scalar;
    constexpr auto gamma_kind = (HT_BLOCK > 1) ? ckl::OperandKind::Row : ckl::OperandKind::Block;
    constexpr auto x_life = X_RESIDENT ? ckl::InputLifecycle::CallerManaged : ckl::InputLifecycle::Bulk;

    const uint32_t num_row_blocks = (num_tile_rows + HT_BLOCK - 1) / HT_BLOCK;
    for (uint32_t hb = 0; hb < num_row_blocks; ++hb) {
        uint32_t ht = num_tile_rows - hb * HT_BLOCK;
        if (ht > HT_BLOCK) {
            ht = HT_BLOCK;
        }
        const auto blk = ckl::EltwiseShape::grid(ht, WT_CHUNK);

        // ================= pass A: mean(x^2) over the whole W ==============
        for (uint32_t wc = 0; wc < NW; ++wc) {
            if constexpr (IS_RM) {
                // Resident regime: this fills the strip in place, chunk by chunk,
                // so pass B needs no re-tilize. Streaming: one block at a time.
                ckl::tilize<WT_CHUNK, cb_input_rm, cb_input_tiles>(ht, ht * 32u);
            }
            const uint32_t x_base = wc * WT_CHUNK;
            if constexpr (X_RESIDENT) {
                // R8: CallerManaged — the chain neither waits nor pops; we do
                // both. Waiting CUMULATIVELY (rather than for the whole strip
                // upfront) is what lets the producer stay a batch ahead of
                // compute. Rounded UP to the producer's push granularity:
                // X_READ_CHUNKS == NW collapses to one wait for the full strip.
                // NW > 1 => HT_BLOCK == 1 (R7), so the strip is one flat Wt
                // strip and chunk wc occupies [wc*WT_CHUNK, +WT_CHUNK).
                const uint32_t batches_ready = (wc / X_READ_CHUNKS) + 1u;
                cb_wait_front(cb_input_tiles, batches_ready * X_READ_CHUNKS * ht * WT_CHUNK);
            }

            // ---- phase 2: x^2 ----
            if constexpr (X_RESIDENT) {
                ckl::eltwise_chain(
                    blk,
                    ckl::BinaryFpu<
                        cb_input_tiles,
                        cb_input_tiles,
                        ckl::BinaryFpuOp::Mul,
                        ckl::BroadcastDim::None,
                        ckl::InputLifecycle::CallerManaged,
                        ckl::InputLifecycle::CallerManaged,
                        ckl::BinaryDataFormatReconfig::Input,
                        ckl::Dst::D0,
                        ckl::OperandKind::Block,
                        ckl::OperandKind::Block,
                        ckl::TileOffset::Set,
                        ckl::TileOffset::Set>{x_base, x_base},
                    ckl::PackTile<cb_x_squared, ckl::OutputLifecycle::Streaming>{});
            } else {
                ckl::square<
                    cb_input_tiles,
                    cb_x_squared,
                    ckl::InputLifecycle::Bulk,
                    ckl::OutputLifecycle::Streaming,
                    ckl::BinaryDataFormatReconfig::Input,
                    ckl::PackTileReconfig::Output,
                    ckl::OperandKind::Block>(blk);
            }

            // ---- phase 3: chunked SUM -> mean on the finalizing chunk ----
            //
            // Under a W-split NO chunk finalizes: this core owns only a SLICE of
            // W, so both the within-tile fold and the 1/N are premature. Every
            // chunk therefore uses Accumulate::at (never at_last), which leaves
            // cb_partials holding the RAW elementwise-accumulated x^2 tile — the
            // exact object the cross-core combine needs. Shipping the *reduced*
            // tile instead would be wrong: AccumulateViaAdd's finalize writes the
            // row sum into column 0 and leaves the surviving x^2 lanes in columns
            // 1..31, so a second REDUCE_ROW over such tiles double-counts them
            // (measured: mean(x^2) of an all-ones W=64 came out 8.75, not 1.0).
            const auto rshape = ckl::ReduceInputBlockShape::of(ht, WT_CHUNK, 1);
            const bool finalize_here = !W_SPLIT && (wc + 1 == NW);
            if constexpr (NW == 1) {
                if constexpr (W_SPLIT) {
                    // Single chunk, so the accumulator is written ONCE and never
                    // reloaded: pack it straight into the writer's gather CB and
                    // skip the republishing copy below. cb_partial_out still has
                    // exactly one producer (this) and one consumer (the writer).
                    ckl::reduce<
                        ckernel::PoolType::SUM,
                        ckernel::ReduceDim::REDUCE_ROW,
                        cb_x_squared,
                        cb_scaler,
                        cb_partial_out,
                        ckl::ReduceInputPolicy::BulkWaitBulkPop,
                        ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                        ckl::ReduceAlgorithm::AccumulateViaAdd>(
                        rshape,
                        ckl::ReduceInputMemoryLayout::contiguous(),
                        ckl::Accumulate::at(cb_partial_out, wc),
                        ckl::NoOp{},
                        partial);
                } else {
                    ckl::reduce_mean<
                        ckernel::ReduceDim::REDUCE_ROW,
                        cb_x_squared,
                        cb_scaler,
                        cb_rms_sum,
                        ckl::ReduceInputPolicy::BulkWaitBulkPop,
                        ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                        ckl::ReduceAlgorithm::AccumulateViaAdd>(
                        rshape, N_REDUCED, ckl::ReduceInputMemoryLayout::contiguous(), ckl::NoAccumulation{}, partial);
                }
            } else if (finalize_here) {
                ckl::reduce_mean<
                    ckernel::ReduceDim::REDUCE_ROW,
                    cb_x_squared,
                    cb_scaler,
                    cb_rms_sum,
                    ckl::ReduceInputPolicy::BulkWaitBulkPop,
                    ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                    ckl::ReduceAlgorithm::AccumulateViaAdd>(
                    rshape,
                    N_REDUCED,
                    ckl::ReduceInputMemoryLayout::contiguous(),
                    ckl::Accumulate::at_last(cb_partials, wc),
                    partial);
            } else {
                // Non-finalizing chunk. The partial-W mask rides the chunk that
                // owns the tensor's last W-tile, which under a W-split is this
                // core's last chunk (and only on the last-W core).
                const bool last_chunk = (wc + 1 == NW);
                ckl::reduce<
                    ckernel::PoolType::SUM,
                    ckernel::ReduceDim::REDUCE_ROW,
                    cb_x_squared,
                    cb_scaler,
                    cb_partials,
                    ckl::ReduceInputPolicy::BulkWaitBulkPop,
                    ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                    ckl::ReduceAlgorithm::AccumulateViaAdd>(
                    rshape,
                    ckl::ReduceInputMemoryLayout::contiguous(),
                    ckl::Accumulate::at(cb_partials, wc),
                    ckl::NoOp{},
                    last_chunk ? partial : ckl::ReducePartialScaler::none());
            }
        }

        // Hand the raw accumulator to the writer's gather leg. With NW > 1
        // cb_partials is a compute->compute read-modify-write across the chunk
        // loop, so it cannot ALSO be the writer's CB (single producer / single
        // consumer) and one copy publishes the settled tile per tile-row. With
        // NW == 1 there is no read-modify-write, so the reduce above already
        // packed into cb_partial_out and this whole pass is gone — worth having
        // as its own case because a compute pass costs ~320 ns of fixed
        // overhead (examples/compute_block_size) and this one sits on the
        // combine's serial path, ahead of the gather.
        if constexpr (W_SPLIT && NW > 1) {
            ckl::copy<cb_partials, cb_partial_out>(ckl::EltwiseShape::tiles(ht));
        }

        // ========== phase 3b: cross-core combine (W-split only) ============
        // The combine folds the raw slice-accumulators the writers gathered into
        // ONE mean(x^2) per tile-row. That fold is EXACTLY the local chunk
        // accumulate, done across cores instead of across chunks: AccumulateViaAdd
        // elementwise-adds the gathered tiles into DEST, folds the result within
        // the tile ONCE, and applies 1/n_reduced with n_reduced = W, the GRAND
        // total (§4.2 "Finalize"). Gathered tiles are laid out h-major
        // (tile h*fan_in + slot), so of(ht, fan_in) reads them contiguously.
        //
        // With CW2 == 1 there is one fold, on the root, over all CW tiles.
        // With CW2 > 1 it is staged: every LEADER first folds its row's CW1
        // tiles WITHOUT finalizing — Accumulate::at (never at_last) keeps the raw
        // elementwise accumulator, the same object a worker's chunk loop
        // produces, so the second fold cannot double-count the surviving x^2
        // lanes — and republishes it through cb_partial_out for its own writer.
        // The root then finalizes over just the CW2 row-sums.
        if constexpr (W_SPLIT) {
            if constexpr (TWO_STAGE) {
                if (is_leader) {
                    // One accumulate call, never reloaded -> pack the row sum
                    // straight back into cb_partial_out for this core's own
                    // writer to ship on to the root (same CB the slice partial
                    // rode: one producer, one consumer, two sequential pushes).
                    ckl::reduce<
                        ckernel::PoolType::SUM,
                        ckernel::ReduceDim::REDUCE_ROW,
                        cb_group_partials,
                        cb_ones,
                        cb_partial_out,
                        ckl::ReduceInputPolicy::BulkWaitBulkPop,
                        ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                        ckl::ReduceAlgorithm::AccumulateViaAdd>(
                        ckl::ReduceInputBlockShape::of(ht, CW1, 1),
                        ckl::ReduceInputMemoryLayout::contiguous(),
                        ckl::Accumulate::at(cb_partial_out, 0),
                        ckl::NoOp{},
                        ckl::ReducePartialScaler::none());
                    if (ht < HT_BLOCK) {
                        cb_pop_front(cb_group_partials, (HT_BLOCK - ht) * CW1);
                    }
                }
                if (is_root) {
                    ckl::reduce_mean<
                        ckernel::ReduceDim::REDUCE_ROW,
                        cb_group_partials2,
                        cb_ones,
                        cb_rms_mean,
                        ckl::ReduceInputPolicy::BulkWaitBulkPop,
                        ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                        ckl::ReduceAlgorithm::AccumulateViaAdd>(
                        ckl::ReduceInputBlockShape::of(ht, CW2, 1),
                        N_REDUCED,
                        ckl::ReduceInputMemoryLayout::contiguous(),
                        ckl::NoAccumulation{},
                        ckl::ReducePartialScaler::none());
                    if (ht < HT_BLOCK) {
                        cb_pop_front(cb_group_partials2, (HT_BLOCK - ht) * CW2);
                    }
                }
            } else if (is_root) {
                ckl::reduce_mean<
                    ckernel::ReduceDim::REDUCE_ROW,
                    cb_group_partials,
                    cb_ones,
                    cb_rms_mean,
                    ckl::ReduceInputPolicy::BulkWaitBulkPop,
                    ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                    ckl::ReduceAlgorithm::AccumulateViaAdd>(
                    ckl::ReduceInputBlockShape::of(ht, CW1, 1),
                    N_REDUCED,
                    ckl::ReduceInputMemoryLayout::contiguous(),
                    ckl::NoAccumulation{},
                    ckl::ReducePartialScaler::none());
                // The reader publishes a fixed HT_BLOCK*CW1 block so the gather
                // slots stay at a constant L1 offset; drop the unused tail.
                if (ht < HT_BLOCK) {
                    cb_pop_front(cb_group_partials, (HT_BLOCK - ht) * CW1);
                }
            }
        }

        // ================= phase 4: 1/sqrt(mean + eps) =====================
        // One dst-sync window for both SFPU ops; the FPU consumer in phase 5
        // reads it back from L1 (DEST reuse measures slower for an FPU consumer).
        // Under a W-split cb_rms_sum is produced by the READER (the root's
        // broadcast), not by phase 3 — every core then finalizes identically.
        ckl::eltwise_chain(
            ckl::EltwiseShape::tiles(ht),
            ckl::CopyTile<cb_rms_sum, ckl::Dst::D0, ckl::InputLifecycle::Streaming>{},
            ckl::AddUnary<ckl::Dst::D0>{eps_bits},
            ckl::Rsqrt<>{},
            ckl::PackTile<cb_rms_recip, ckl::OutputLifecycle::Streaming>{});
        if constexpr (W_SPLIT) {
            // Same fixed-block contract as the gather: the reader publishes
            // HT_BLOCK pages so the multicast lands at a constant L1 offset.
            if (ht < HT_BLOCK) {
                cb_pop_front(cb_rms_sum, HT_BLOCK - ht);
            }
        }

        // ================= pass B: scale (and gamma), then write ===========
        for (uint32_t wc = 0; wc < NW; ++wc) {
            if constexpr (IS_RM && !X_RESIDENT) {
                ckl::tilize<WT_CHUNK, cb_input_rm, cb_input_tiles>(ht, ht * 32u);
            }
            const uint32_t x_base = wc * WT_CHUNK;

            // ---- phase 5: x * (1/rms), broadcast across columns ----
            if constexpr (X_RESIDENT) {
                ckl::eltwise_chain(
                    blk,
                    ckl::BinaryFpu<
                        cb_input_tiles,
                        cb_rms_recip,
                        ckl::BinaryFpuOp::Mul,
                        ckl::BroadcastDim::Col,
                        ckl::InputLifecycle::CallerManaged,
                        ckl::InputLifecycle::HeldBulk,
                        ckl::BinaryDataFormatReconfig::Input,
                        ckl::Dst::D0,
                        ckl::OperandKind::Block,
                        rms_kind,
                        ckl::TileOffset::Set,
                        ckl::TileOffset::Unset>{x_base, 0},
                    ckl::PackTile<cb_scale_out, ckl::OutputLifecycle::Streaming>{});
            } else {
                ckl::mul<
                    cb_input_tiles,
                    cb_rms_recip,
                    cb_scale_out,
                    ckl::BroadcastDim::Col,
                    x_life,
                    ckl::InputLifecycle::HeldBulk,
                    ckl::OutputLifecycle::Streaming,
                    ckl::BinaryDataFormatReconfig::Input,
                    ckl::PackTileReconfig::Output,
                    ckl::OperandKind::Block,
                    rms_kind>(blk);
            }

            // ---- phase 6: * gamma, broadcast down the rows ----
            if constexpr (HAS_GAMMA) {
                if constexpr (IS_RM_GAMMA && !GAMMA_RESIDENT) {
                    ckl::tilize<WT_CHUNK, cb_gamma_rm, cb_gamma>(/*num_blocks=*/1, /*total_input_pages=*/1);
                }
                if constexpr (GAMMA_RESIDENT) {
                    ckl::eltwise_chain(
                        blk,
                        ckl::BinaryFpu<
                            cb_scaled,
                            cb_gamma,
                            ckl::BinaryFpuOp::Mul,
                            ckl::BroadcastDim::Row,
                            ckl::InputLifecycle::Bulk,
                            ckl::InputLifecycle::CallerManaged,
                            ckl::BinaryDataFormatReconfig::Input,
                            ckl::Dst::D0,
                            ckl::OperandKind::Block,
                            gamma_kind,
                            ckl::TileOffset::Unset,
                            ckl::TileOffset::Set>{0, x_base},
                        ckl::PackTile<cb_output_tiles, ckl::OutputLifecycle::Streaming>{});
                } else {
                    ckl::mul<
                        cb_scaled,
                        cb_gamma,
                        cb_output_tiles,
                        ckl::BroadcastDim::Row,
                        ckl::InputLifecycle::Bulk,
                        ckl::InputLifecycle::Bulk,
                        ckl::OutputLifecycle::Streaming,
                        ckl::BinaryDataFormatReconfig::Input,
                        ckl::PackTileReconfig::Output,
                        ckl::OperandKind::Block,
                        gamma_kind>(blk);
                }
            }

            // ---- phase 7: back to row-major sticks ----
            if constexpr (IS_RM) {
                ckl::untilize<WT_CHUNK, cb_output_tiles, cb_output_rm>(ht);
            }
        }

        // ================= phase 8: release the held CBs ===================
        // R2: cb_rms_recip is HeldBulk across all NW chunks of pass B.
        cb_pop_front(cb_rms_recip, ht);
        if constexpr (X_RESIDENT) {
            cb_pop_front(cb_input_tiles, ht * WT);
        }
    }
}
