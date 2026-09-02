// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Implementation file for reduce_helpers_compute.hpp
// Do not include directly - include reduce_helpers_compute.hpp instead

#include "api/compute/add_int_sfpu.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/debug/assert.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/dfb_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_common.hpp"

namespace compute_kernel_lib {

namespace detail {

// SFPU MAX fold
template <DataFormat format>
ALWI void sfpu_reduce_max_fold_init() {
    if constexpr (format == DataFormat::Int32) {
        binary_max_int32_tile_init();
    } else {
        binary_max_tile_init();
    }
}

template <DataFormat format>
ALWI void sfpu_reduce_max_fold_tile(uint32_t a, uint32_t b, uint32_t out) {
    if constexpr (format == DataFormat::Int32) {
        binary_max_int32_tile(a, b, out);
    } else {
        binary_max_tile(a, b, out);
    }
}

// SFPU MIN fold
template <DataFormat format>
ALWI void sfpu_reduce_min_fold_init() {
    if constexpr (format == DataFormat::Int32) {
        binary_min_int32_tile_init();
    } else {
        binary_min_tile_init();
    }
}

template <DataFormat format>
ALWI void sfpu_reduce_min_fold_tile(uint32_t a, uint32_t b, uint32_t out) {
    if constexpr (format == DataFormat::Int32) {
        binary_min_int32_tile(a, b, out);
    } else {
        binary_min_tile(a, b, out);
    }
}

// SFPU cross-tile add. Int32 uses add_int_tile; Float32 uses add_binary_tile for
// accurate fp32 accumulation. add_binary_tile is unavailable on Quasar, so guard
// it with ARCH_QUASAR to avoid template lookup failures.
template <DataFormat format>
ALWI void sfpu_reduce_sum_fold_init() {
    if constexpr (format == DataFormat::Int32) {
        add_int_tile_init();
    } else {
#ifndef ARCH_QUASAR
        add_binary_tile_init();
#else
        static_assert(format == DataFormat::Int32, "Accurate fp32 SFPU reduce is not supported on Quasar");
#endif
    }
}

template <DataFormat format>
ALWI void sfpu_reduce_sum_fold_tile(uint32_t a, uint32_t b, uint32_t out) {
    if constexpr (format == DataFormat::Int32) {
        add_int_tile<format>(a, b, out);
    } else {
#ifndef ARCH_QUASAR
        add_binary_tile(a, b, out);
#else
        static_assert(format == DataFormat::Int32, "Accurate fp32 SFPU reduce is not supported on Quasar");
#endif
    }
}

// Pool-type dispatched cross-tile fold init (MAX -> binary_max, MIN -> binary_min, SUM -> add).
// Used by compute_kernel_lib::reduce() for the Int32 SFPU path and for accurate fp32 reduces.
template <PoolType pool_type, DataFormat format>
ALWI void sfpu_reduce_fold_init() {
    if constexpr (pool_type == PoolType::SUM) {
        sfpu_reduce_sum_fold_init<format>();
#ifndef ARCH_QUASAR  // Quasar's ckernel::PoolType has no MIN (and no SFPU reduce path)
    } else if constexpr (pool_type == PoolType::MIN) {
        sfpu_reduce_min_fold_init<format>();
#endif
    } else {
        sfpu_reduce_max_fold_init<format>();
    }
}

// Copy one input tile into DST and fold into the running accumulator (first tile seeds dst_idx
// directly). Fold op is selected by pool_type: MAX -> running max, MIN -> running min, SUM -> running sum.
template <PoolType pool_type, DataFormat format>
ALWI void sfpu_copy_and_fold(
    uint32_t input_cb_id, uint32_t tile_idx, uint32_t dst_idx, uint32_t work_dst, bool is_first_tile) {
    if (is_first_tile) {
        copy_tile(input_cb_id, tile_idx, dst_idx);
    } else {
        copy_tile(input_cb_id, tile_idx, work_dst);
        if constexpr (pool_type == PoolType::SUM) {
            sfpu_reduce_sum_fold_tile<format>(dst_idx, work_dst, dst_idx);
#ifndef ARCH_QUASAR  // Quasar's ckernel::PoolType has no MIN (and no SFPU reduce path)
        } else if constexpr (pool_type == PoolType::MIN) {
            sfpu_reduce_min_fold_tile<format>(dst_idx, work_dst, dst_idx);
#endif
        } else {
            sfpu_reduce_max_fold_tile<format>(dst_idx, work_dst, dst_idx);
        }
    }
}

// Matches sfpu_copy_and_fold_max is_first_tile: copy on axis 0 unless Accumulate already reloaded DST.
template <typename AccumulateT>
ALWI bool sfpu_is_first_tile(uint32_t axis_index, const AccumulateT& accumulate) {
    if constexpr (is_accumulate_v<AccumulateT>) {
        return axis_index == 0 && accumulate.is_first();
    }
    return axis_index == 0;
}

// Post-reduce scalar multiply. mul_unary_tile is fp32-only, so Int32 is bracketed with typecasts
// (truncates toward zero on the way back); all other formats use plain mul_unary_tile.
template <DataFormat reduce_format>
ALWI void reduce_post_mul_tile(uint32_t dst, uint32_t scaler_bits) {
    if constexpr (reduce_format == DataFormat::Int32) {
        typecast_tile_init<(uint32_t)DataFormat::Int32, (uint32_t)DataFormat::Float32>();
        typecast_tile<(uint32_t)DataFormat::Int32, (uint32_t)DataFormat::Float32>(dst);
        binop_with_scalar_tile_init();
        mul_unary_tile(dst, scaler_bits);
        typecast_tile_init<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Int32>();
        typecast_tile<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Int32>(dst);
    } else {
        binop_with_scalar_tile_init();
        mul_unary_tile(dst, scaler_bits);
    }
}

// Does `dfb_id` unpack straight into DEST (bypassing SrcA/SrcB)? True only for a 32-bit CB tagged
// UnpackToDestFp32, where the JIT keeps unpack_dst_format == unpack_src_format (Default downgrades to Tf32, bf16
// is never 32-bit). FoldViaAdd reads the accumulator via SrcA/SrcB, so it is invalid for such a CB. UNPACK/MATH
// only (PACK cannot see unpack_dst_format); mirrors tilize_helpers' has_unpack_to_dest_fp32.
ALWI bool dfb_unpacks_to_dest(uint32_t dfb_id) {
#if defined(UCK_CHLKC_PACK)
    (void)dfb_id;
    return false;
#else
    const uint32_t src = unpack_src_format[dfb_id];
    const bool src_is_32bit = (src == (uint32_t)DataFormat::Float32) || (src == (uint32_t)DataFormat::Int32);
    return src_is_32bit && (src == unpack_dst_format[dfb_id]);
#endif
}

// -----------------------------------------------------------------------------
// AccumulateViaAdd datapath (ReduceAlgorithm::AccumulateViaAdd).
//
// Each output tile is produced independently: sum its reduce-dim tiles into a DST slot with pairwise
// add_tiles(acc_to_dest) (parity resolved at the seed — copy_tile one tile when the count is odd, add
// the first pair when even, no phantom zero tile), finalize within the tile on the SFPU (sfpu_reduce
// SUM, which reads DST in place), and for AVG multiply by the compile-time 1/reduce_factor once. One DST
// register per active output tile; grouped COL input keeps a host-planned set of those slots live together.
//
// Restrictions (enforced by reduce()): float SUM or AVG. AVG's reduce_factor is caller-owned, so standalone,
// partial, cross-chunk, sharded, and uneven means all use the same reduce<AVG> entry point.
// All policies except WaitAndPopPerTile + COL are supported. WaitUpfrontNoPop / NoWaitNoPop index a resident
// block; ROW/SCALAR can stream either one tile or a host-planned chunk, while COL requires grouped Bulk or
// Chunked input. should_pop policies (Bulk / WaitAndPop / Chunked)
// pop the input and pack per output; no-pop policies (WaitUpfront / NoWait) leave the input resident and
// bulk-reserve the outputs upfront, packing output o -> its OWN page o. The one-time SFPU-macro load
// (sfpu_reduce_init) is hoisted OUT of the per-output loop; only the light MOP inits (add_tiles/copy) run per
// output.
//
// PARTIAL (non-tile-aligned) reduce dims — ROW/COL only, signalled by partial_scaler.use_partial: the last tile
// is folded in with a DEST-ACCUMULATING masked broadcast-mul (0/1 mask at
// scaler_dfb_id[partial_tile_idx]; row-0 mask for ROW, col-0 for COL) via fold_partial_last(), so the padding
// contributes 0. The bulk stays pure add_tiles (fidelity-flat, 2 tiles/op); only the one partial tile is a
// (fidelity-affected) mul. The bcast shorthands overwrite DEST (clear_fp32_dst_acc=true), so the accumulating
// variant is the LLK directly with acc_to_dest=1 at init and clear_fp32_dst_acc=false at the op.
// Partial is supported for SUM/AVG standalone (any should_pop / no-pop policy), under ROW streaming, and folded
// into cross-call Accumulate (ROW/COL). SCALAR partial is rejected (a 2-D corner mask a single row/col tile
// can't encode).
//
// CROSS-CALL ACCUMULATE (AccumulateT == Accumulate) — the accumulator CB holds the RAW partial-sum tile per
// output (NOT a reduced tile). On the first call we sum the new input (+ its masked partial, if any). Later
// resident-input calls use the selected AccumulateReloadMode; streamed/grouped calls safely copy-seed DEST
// from the accumulator before folding the arriving tiles.
// The within-tile finalize (sfpu_reduce [+ 1/reduce_factor for AVG] + post_reduce_op) runs only when
// accumulate.is_last(); non-last chunks pack the raw partial sum back to the output CB (which the caller points
// at the accumulator).
template <
    PoolType reduce_type,
    ReduceDim reduce_dim,
    uint32_t input_dfb_id,
    uint32_t scaler_dfb_id,
    uint32_t output_dfb_id,
    ReduceInputPolicy input_policy,
    ReduceDataFormatReconfigMode reconfig_mode,
    typename AccumulateT,
    typename PostReduceOp,
    ReduceWithinTile within_tile = ReduceWithinTile::Collapse,
    uint32_t reduce_factor = 1>
ALWI void reduce_accumulate_via_add(
    ReduceInputBlockShape shape,
    ReduceInputMemoryLayout input_memory_layout,
    ReducePartialScaler partial_scaler,
    AccumulateT accumulate,
    PostReduceOp post_reduce_op,
    ReduceInputChunk input_chunk) {
    const uint32_t Ht = shape.rows, Wt = shape.cols, NC = shape.batches;
    // row_pitch = tile distance between consecutive rows of the resident block (>= Wt). row_stride > Wt lets
    // the reduce run over the first Wt columns of a WIDER resident tensor — the padding tiles [Wt, row_pitch)
    // are simply never indexed. 0 => contiguous (row_pitch = Wt). Honored for ROW/COL under BulkWaitBulkPop;
    // SCALAR / streaming / cross-call accumulate require contiguous (asserted in reduce()).
    const uint32_t row_pitch = (input_memory_layout.row_stride > 0u) ? input_memory_layout.row_stride : Wt;
    const uint32_t in_tiles = Ht * row_pitch * NC;

    // DST accumulation format drives the SFPU finalize (fp32 DST when fp32_dest_acc_en is on).
    constexpr DataFormat dst_fmt = DST_ACCUM_MODE ? DataFormat::Float32 : DataFormat::Float16_b;

    constexpr bool is_row = (reduce_dim == ReduceDim::REDUCE_ROW);
    constexpr bool is_col = (reduce_dim == ReduceDim::REDUCE_COL);
    constexpr auto MASK_BCAST = is_col ? ckernel::BroadcastType::COL : ckernel::BroadcastType::ROW;
    // WaitAndPopPerTile streams one contiguous output at a time (ROW/SCALAR). ChunkedWaitChunkedPop does the
    // same for ROW/SCALAR, while COL streams a row-major group of output columns and keeps one running DEST
    // slot per column. A later cross-call Accumulate copy-seeds those slots from the accumulator CB first.
    constexpr bool chunked = input_policy == ReduceInputPolicy::ChunkedWaitChunkedPop;
    constexpr bool grouped_col = is_col && (input_policy == ReduceInputPolicy::BulkWaitBulkPop || chunked);
    constexpr bool streaming = input_policy == ReduceInputPolicy::WaitAndPopPerTile || (chunked && !is_col);
    constexpr bool streamed_input = streaming || grouped_col;
    constexpr bool has_accum = is_accumulate_v<AccumulateT>;  // cross-call CB accumulator (raw partial sum)

    // CB-policy predicates (match the standard path). should_pop_p: the output is popped per output tile
    // (Bulk / WaitAndPop) vs bulk-reserved upfront + bulk-pushed at the end (WaitUpfrontNoPop / NoWaitNoPop).
    // helper_waits_block: the whole resident block is waited once (Bulk / WaitUpfront) — NoWaitNoPop trusts the
    // caller to have it resident, WaitAndPop streams per pair. helper_pops_block: only BulkWaitBulkPop pops it.
    constexpr bool should_pop_p =
        (input_policy == ReduceInputPolicy::WaitAndPopPerTile || input_policy == ReduceInputPolicy::BulkWaitBulkPop ||
         input_policy == ReduceInputPolicy::ChunkedWaitChunkedPop);
    constexpr bool no_wait_p = (input_policy == ReduceInputPolicy::NoWaitNoPop);
    constexpr bool bulk_per_output = input_policy == ReduceInputPolicy::BulkWaitBulkPop && !is_col;
    constexpr bool helper_waits_block = (!streamed_input && !no_wait_p && !bulk_per_output);
    constexpr bool helper_pops_block = (!streamed_input && should_pop_p && !bulk_per_output);

    // tiles that collapse into one output, and their stride in the row-major (batch-major) input block.
    const uint32_t cnt = is_row ? Wt : (is_col ? Ht : (Ht * Wt));
    const uint32_t stride = is_col ? row_pitch : 1u;  // COL steps down a column by the row pitch
    const uint32_t n_out = is_row ? (Ht * NC) : (is_col ? (Wt * NC) : NC);

    // The pairwise accumulation and within-tile finalize produce a SUM. AVG applies the caller-owned
    // compile-time 1/reduce_factor below; no divisor is inferred from this call's tile geometry.
    const bool has_partial = partial_scaler.use_partial;
    const uint32_t mask_idx = partial_scaler.partial_tile_idx;
    const uint32_t full_cnt = has_partial ? (cnt - 1u) : cnt;  // tiles summed via pure add_tiles

    DataflowBuffer input_dfb(input_dfb_id), scaler_dfb(scaler_dfb_id), output_dfb(output_dfb_id);
    DataflowBuffer accum_dfb([&]() -> uint32_t {
        if constexpr (has_accum) {
            return accumulate.config.cb_accumulator;
        } else {
            return 0;
        }
    }());

    // Accumulate: reload the running accumulator on every chunk except the first; finalize only on the last.
    // (do_finalize is always true when there is no cross-call accumulation.)
    bool do_finalize = true;
    if constexpr (has_accum) {
        do_finalize = accumulate.is_last();
    }

    // Per-reduce()-call setup — LIGHT only. The heavy hw_configure (unpack/math/pack HW setup + pack_dest_init)
    // is the once-per-kernel boot (compute_kernel_hw_startup, same as every reduce) and must NEVER run per
    // reduce() call, so it is not done here. Per call we do only the light format reconfig (gated by
    // reconfig_mode, to adapt SrcA/SrcB/packer formats when this reduce chains after a different-format op —
    // the AccumulateViaAdd analogue of the standard path's reconfig_data_format) plus the light SFPU-macro
    // (re)load; the per-output add_tiles_init / copy_tile_init below re-arm the MOP. This mirrors how ReduceTile
    // relies on boot hw_configure + light reduce_init.
    constexpr bool reconfig_in =
        (reconfig_mode == ReduceDataFormatReconfigMode::INPUT ||
         reconfig_mode == ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT);
    constexpr bool reconfig_out =
        (reconfig_mode == ReduceDataFormatReconfigMode::OUTPUT ||
         reconfig_mode == ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT);
    if constexpr (reconfig_in) {
        reconfig_data_format(input_dfb_id, input_dfb_id);  // both add operands = the input CB
    }
    if constexpr (reconfig_out) {
        pack_reconfig_data_format(output_dfb_id);
    }
    // Light: (re)load the SFPU reduce macro (persists across the adds). Skipped under ReduceWithinTile::Skip —
    // there is no sfpu_reduce to arm. The later AVG scale initializes its scalar op when needed, but a caller's
    // SFPU post_reduce_op should still follow the normal contract and run its own <op>_tile_init.
    if constexpr (within_tile == ReduceWithinTile::Collapse) {
        sfpu_reduce_init<PoolType::SUM, dst_fmt>();
    }
    // Basic validity the reduce() dispatch skips on this path (its compile-time restrictions are asserted
    // there). Capacity self-asserts in each wait_front/reserve_back, except NoWaitNoPop which does neither.
    ASSERT(input_dfb_id != output_dfb_id && Ht > 0 && Wt > 0 && NC > 0);
    if constexpr (chunked) {
        ASSERT(input_chunk.reduce_axis_tiles > 0 && input_chunk.output_tiles > 0);
        if constexpr (is_col) {
            // A COL chunk must expose at least a pair of rows so AccumulateViaAdd can make progress while
            // retaining one running DEST slot for each output column in the group.
            ASSERT(input_chunk.reduce_axis_tiles >= 2);
        } else {
            ASSERT(input_chunk.output_tiles == 1);
            ASSERT((input_chunk.reduce_axis_tiles & 1u) == 0);
        }
    }
#ifndef ARCH_QUASAR  // is_valid_dfb_tile_page_size is WH/BH only
    UNPACK(ASSERT(is_valid_dfb_tile_page_size(input_dfb_id, (DataFormat)unpack_src_format[input_dfb_id])));
    PACK(ASSERT(is_valid_dfb_tile_page_size(output_dfb_id, (DataFormat)pack_dst_format[output_dfb_id])));
#endif
    if constexpr (no_wait_p) {  // no wait/reserve to self-assert capacity: caller must have the block resident
        ASSERT(get_dfb_num_pages(input_dfb_id) >= in_tiles);
    }

    // The auxiliary CB is never popped here. A partial consumes its mask/scaler representation; a
    // CopySeedZeroPair zero follows that representation, so mask and zero can coexist in the same CB.
    // With AccumulateViaAdd's mask-only partial layout this is [mask@0, zero@1]; without a partial it is
    // [zero@0].
    const uint32_t zero_idx = has_partial ? partial_scaler.scaler_tile_count() : 0u;
    uint32_t required_aux_tiles = has_partial ? mask_idx + 1u : 0u;
    if constexpr (has_accum) {
        // AccumulateReloadMode contracts for indexed input. Streamed/grouped paths normally use their safe
        // copy-seed fold, but honor CopySeedZeroPair for an odd tile/row once DST has already been seeded.
        // acc_cb (running RAW partial sum) is maybe_unused: only ASSERTs read it.
        [[maybe_unused]] const uint32_t acc_cb = accumulate.config.cb_accumulator;
        ASSERT(input_dfb_id != acc_cb);
        if constexpr (!streamed_input) {
            // FoldViaAdd reads acc_cb via SrcA/SrcB — invalid for an UnpackToDestFp32 CB (see
            // dfb_unpacks_to_dest).
            UNPACK(ASSERT(accumulate.reload != AccumulateReloadMode::FoldViaAdd || !dfb_unpacks_to_dest(acc_cb)));
#ifdef ARCH_QUASAR
            ASSERT(accumulate.reload != AccumulateReloadMode::CopySeedSfpuAdd);  // needs add_binary_tile (WH/BH only)
#endif
        }
        if (accumulate.reload == AccumulateReloadMode::CopySeedZeroPair) {
            const uint32_t zero_tile_count = zero_idx + 1u;
            required_aux_tiles = required_aux_tiles < zero_tile_count ? zero_tile_count : required_aux_tiles;
        }
    }
    if (required_aux_tiles > 0) {
        ASSERT(input_dfb_id != scaler_dfb_id && output_dfb_id != scaler_dfb_id);
        scaler_dfb.wait_front(required_aux_tiles);
    }
    if constexpr (helper_waits_block) {
        input_dfb.wait_front(in_tiles);  // Bulk / WaitUpfront: whole resident block, indexed per output
    }
    if constexpr (!should_pop_p) {
        output_dfb.reserve_back(n_out);  // no-pop: reserve every output page upfront (pack o -> page o below)
    }

    // Fold the masked partial LAST reduce-dim tile into DST, ACCUMULATING (acc_to_dest=1 at init,
    // clear_fp32_dst_acc=false at the op — the bcast shorthands would overwrite). Shared by the standalone,
    // cross-call-accumulate, and streaming paths so the partial fold lives in one place. `last_idx` is the
    // input-CB index of that tile (absolute into the resident block, or front-relative 0 for streaming).
    // Referenced from a runtime `if (has_partial)` in every instantiation, so it is never truly unused.
    [[maybe_unused]] auto fold_partial_last = [&](uint32_t last_idx, uint32_t dst_idx = 0) {
        MATH((llk_math_eltwise_binary_init<ckernel::EltwiseBinaryType::ELWMUL, MASK_BCAST, MATH_FIDELITY>(
            input_dfb_id, scaler_dfb_id, 1)));
        UNPACK((llk_unpack_AB_init<MASK_BCAST>(input_dfb_id, scaler_dfb_id)));
        UNPACK((llk_unpack_AB<MASK_BCAST>(input_dfb_id, scaler_dfb_id, last_idx, mask_idx)));
        MATH((llk_math_eltwise_binary<ckernel::EltwiseBinaryType::ELWMUL, MASK_BCAST, DST_ACCUM_MODE, MATH_FIDELITY>(
            dst_idx, false)));
    };

    // Finalize a raw cross-tile sum only on the last cross-call accumulation step. Keeping this indexed by
    // DST lets grouped COL chunks finalize every output column without duplicating the policy-independent tail.
    auto finalize_output = [&](uint32_t dst_idx) {
        if (!do_finalize) {
            return;
        }
        if constexpr (within_tile == ReduceWithinTile::Collapse) {
            if constexpr (is_row) {
                sfpu_reduce<PoolType::SUM, dst_fmt, ReduceDim::REDUCE_ROW>(dst_idx, 1, 1);
            } else if constexpr (is_col) {
                sfpu_reduce<PoolType::SUM, dst_fmt, ReduceDim::REDUCE_COL>(dst_idx, 1, 1);
            } else {
                sfpu_reduce<PoolType::SUM, dst_fmt, ReduceDim::REDUCE_ROW>(dst_idx, 1, 1);
                sfpu_reduce<PoolType::SUM, dst_fmt, ReduceDim::REDUCE_COL>(dst_idx, 1, 1);
            }
        }
        if constexpr (reduce_factor != 1) {
            constexpr uint32_t inv_bits = __builtin_bit_cast(uint32_t, 1.0f / static_cast<float>(reduce_factor));
            if constexpr (within_tile == ReduceWithinTile::Skip) {
                binop_with_scalar_tile_init();
            }
            mul_unary_tile(dst_idx, inv_bits);
        }
        post_reduce_op(dst_idx);
    };

    if constexpr (grouped_col) {
        // The COL reader emits N, W-group, H, W-within-group order. Keep the complete output group in DEST,
        // synchronize one row chunk at a time, and fold each column independently. Unlike WaitAndPopPerTile,
        // the chunk metadata tells us both dimensions of the resident row-major block.
        const uint32_t default_output_group = Wt < DEST_AUTO_LIMIT ? Wt : DEST_AUTO_LIMIT;
        const uint32_t output_group = input_chunk.output_tiles > 0 ? input_chunk.output_tiles : default_output_group;
        const uint32_t axis_chunk = chunked ? input_chunk.reduce_axis_tiles : Ht;
        ASSERT(output_group > 0 && output_group <= DEST_AUTO_LIMIT && axis_chunk > 0);

        for (uint32_t nc = 0; nc < NC; ++nc) {
            for (uint32_t wt = 0; wt < Wt; wt += output_group) {
                const uint32_t current_outputs = output_group < Wt - wt ? output_group : Wt - wt;
                tile_regs_acquire();

                bool dst_seeded = false;
                if constexpr (has_accum) {
                    if (!accumulate.is_first()) {
                        const uint32_t acc_cb = accumulate.config.cb_accumulator;
                        accum_dfb.wait_front(current_outputs);
                        reconfig_data_format_srca(input_dfb_id, acc_cb);
                        copy_tile_init(acc_cb);
                        for (uint32_t out = 0; out < current_outputs; ++out) {
                            copy_tile(acc_cb, out, out);
                        }
                        accum_dfb.pop_front(current_outputs);
                        reconfig_data_format_srca(acc_cb, input_dfb_id);
                        dst_seeded = true;
                    }
                }

                for (uint32_t ht = 0; ht < Ht; ht += axis_chunk) {
                    const uint32_t current_rows = axis_chunk < Ht - ht ? axis_chunk : Ht - ht;
                    const uint32_t input_tiles = current_rows * current_outputs;
                    input_dfb.wait_front(input_tiles);

                    const uint32_t remaining_full_rows = ht < full_cnt ? full_cnt - ht : 0;
                    const uint32_t full_rows = current_rows < remaining_full_rows ? current_rows : remaining_full_rows;
                    uint32_t local_h = 0;
                    if (full_rows & 1u) {
                        if (dst_seeded) {
                            if constexpr (has_accum) {
                                if (accumulate.reload == AccumulateReloadMode::CopySeedZeroPair) {
                                    add_tiles_init(input_dfb_id, scaler_dfb_id, true);
                                    for (uint32_t out = 0; out < current_outputs; ++out) {
                                        add_tiles(input_dfb_id, scaler_dfb_id, out, zero_idx, out);
                                    }
                                } else {
                                    binary_dest_reuse_tiles_init<
                                        ckernel::EltwiseBinaryType::ELWADD,
                                        ckernel::EltwiseBinaryReuseDestType::DEST_TO_SRCB>(input_dfb_id);
                                    for (uint32_t out = 0; out < current_outputs; ++out) {
                                        binary_dest_reuse_tiles<
                                            ckernel::EltwiseBinaryType::ELWADD,
                                            ckernel::EltwiseBinaryReuseDestType::DEST_TO_SRCB>(input_dfb_id, out, out);
                                    }
                                }
                            } else {
                                binary_dest_reuse_tiles_init<
                                    ckernel::EltwiseBinaryType::ELWADD,
                                    ckernel::EltwiseBinaryReuseDestType::DEST_TO_SRCB>(input_dfb_id);
                                for (uint32_t out = 0; out < current_outputs; ++out) {
                                    binary_dest_reuse_tiles<
                                        ckernel::EltwiseBinaryType::ELWADD,
                                        ckernel::EltwiseBinaryReuseDestType::DEST_TO_SRCB>(input_dfb_id, out, out);
                                }
                            }
                        } else {
                            copy_tile_init(input_dfb_id);
                            for (uint32_t out = 0; out < current_outputs; ++out) {
                                copy_tile(input_dfb_id, out, out);
                            }
                            dst_seeded = true;
                        }
                        local_h = 1;
                    }

                    if (local_h < full_rows) {
                        add_tiles_init(input_dfb_id, input_dfb_id, true);
                        for (; local_h < full_rows; local_h += 2) {
                            const uint32_t first_row = local_h * current_outputs;
                            const uint32_t second_row = (local_h + 1) * current_outputs;
                            for (uint32_t out = 0; out < current_outputs; ++out) {
                                add_tiles(input_dfb_id, input_dfb_id, first_row + out, second_row + out, out);
                            }
                        }
                        dst_seeded = true;
                    }

                    if (has_partial && ht + current_rows == Ht) {
                        const uint32_t partial_row = (current_rows - 1) * current_outputs;
                        for (uint32_t out = 0; out < current_outputs; ++out) {
                            fold_partial_last(partial_row + out, out);
                        }
                        dst_seeded = true;
                    }
                    input_dfb.pop_front(input_tiles);
                }
                ASSERT(dst_seeded);

                for (uint32_t out = 0; out < current_outputs; ++out) {
                    finalize_output(out);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t out = 0; out < current_outputs; ++out) {
                    output_dfb.reserve_back(1);
                    pack_tile(out, output_dfb_id);
                    output_dfb.push_back(1);
                }
                tile_regs_release();
            }
        }
        return;
    }

    for (uint32_t o = 0; o < n_out; ++o) {
        if constexpr (bulk_per_output) {
            input_dfb.wait_front(cnt);
        }
        tile_regs_acquire();

        if constexpr (streaming) {
            // Stream this output's reduce-dim tiles through DST in pairs, waiting/popping as they arrive
            // (front-relative indices 0/1). Contiguous per output (row/scalar), so tiles arrive in reduce
            // order. A later cross-call accumulation first copy-seeds DST from the accumulator CB; the first
            // call starts from a fresh DST. An odd new-input count is then handled by either a unary seed
            // (first call) or one DEST-reuse add (later calls), leaving an even pairwise tail.
            bool loaded_accumulator = false;
            if constexpr (has_accum) {
                if (!accumulate.is_first()) {
                    const uint32_t acc_cb = accumulate.config.cb_accumulator;
                    accum_dfb.wait_front(1);
                    reconfig_data_format_srca(input_dfb_id, acc_cb);
                    copy_tile_init(acc_cb);
                    copy_tile(acc_cb, 0, 0);
                    reconfig_data_format_srca(acc_cb, input_dfb_id);
                    loaded_accumulator = true;
                }
            }

            uint32_t consumed = 0;
            if (full_cnt & 1u) {
                input_dfb.wait_front(1);
                if (loaded_accumulator) {
                    if constexpr (has_accum) {
                        if (accumulate.reload == AccumulateReloadMode::CopySeedZeroPair) {
                            add_tiles_init(input_dfb_id, scaler_dfb_id, true);
                            add_tiles(input_dfb_id, scaler_dfb_id, 0, zero_idx, 0);
                        } else {
                            binary_dest_reuse_tiles_init<
                                ckernel::EltwiseBinaryType::ELWADD,
                                ckernel::EltwiseBinaryReuseDestType::DEST_TO_SRCB>(input_dfb_id);
                            binary_dest_reuse_tiles<
                                ckernel::EltwiseBinaryType::ELWADD,
                                ckernel::EltwiseBinaryReuseDestType::DEST_TO_SRCB>(input_dfb_id, 0, 0);
                        }
                    }
                } else {
                    copy_tile_init(input_dfb_id);
                    copy_tile(input_dfb_id, 0, 0);
                }
                input_dfb.pop_front(1);
                consumed = 1;
            }
            add_tiles_init(input_dfb_id, input_dfb_id, true);
            if constexpr (chunked) {
                while (consumed < full_cnt) {
                    const uint32_t remaining = full_cnt - consumed;
                    const uint32_t current_chunk =
                        remaining < input_chunk.reduce_axis_tiles ? remaining : input_chunk.reduce_axis_tiles;
                    ASSERT((current_chunk & 1u) == 0);
                    input_dfb.wait_front(current_chunk);
                    for (uint32_t k = 0; k < current_chunk; k += 2) {
                        add_tiles(input_dfb_id, input_dfb_id, k, k + 1, 0);
                    }
                    input_dfb.pop_front(current_chunk);
                    consumed += current_chunk;
                }
            } else {
                for (; consumed < full_cnt; consumed += 2) {
                    input_dfb.wait_front(2);
                    add_tiles(input_dfb_id, input_dfb_id, 0, 1, 0);
                    input_dfb.pop_front(2);
                }
            }
            if constexpr (has_accum) {
                if (loaded_accumulator) {
                    accum_dfb.pop_front(1);
                }
            }
            if (has_partial) {  // ROW partial: the LAST reduce-dim tile is now at the CB front; fold it masked
                input_dfb.wait_front(1);
                fold_partial_last(0);
                input_dfb.pop_front(1);
            }
        } else {
            // Indexed access into the resident block; `start` is output o's first reduce-dim tile. row_pitch
            // is the per-row tile pitch (== Wt when contiguous), so padded rows are skipped automatically.
            uint32_t start;
            if constexpr (bulk_per_output) {
                start = 0;
            } else if constexpr (is_row) {
                start = o * row_pitch;
            } else if constexpr (is_col) {
                start = (o / Wt) * (Ht * row_pitch) + (o % Wt);
            } else {
                start = o * (Ht * row_pitch);  // scalar: row_pitch == Wt (contiguous, asserted in reduce())
            }

            if constexpr (has_accum) {
                if (accumulate.is_first()) {
                    // First chunk: no accumulator yet — sum this chunk's full_cnt tiles (aligned; accumulate
                    // rejects partial). acc_to_dest=true throughout: a freshly-acquired DST reads 0 on its
                    // first write, so the first add is the plain sum. Odd count: seed DST with a unary copy.
                    uint32_t k = 0;
                    if (full_cnt & 1u) {
                        copy_tile_init(input_dfb_id);
                        copy_tile(input_dfb_id, start, 0);
                        k = 1;
                    }
                    add_tiles_init(input_dfb_id, input_dfb_id, true);
                    for (; k < full_cnt; k += 2) {
                        add_tiles(input_dfb_id, input_dfb_id, start + k * stride, start + (k + 1) * stride, 0);
                    }
                    if (has_partial) {  // ROW/COL partial: fold the masked last tile into this chunk's sum
                        fold_partial_last(start + full_cnt * stride);
                    }
                } else {
                    // Later chunk: fold output o's running accumulator (raw partial sum, front of accum CB)
                    // with this chunk's new tiles. Strategy = accumulate.reload (see AccumulateReloadMode):
                    // FoldViaAdd reads the accumulator via SrcB (fast, Default-acc only); the CopySeed* modes
                    // reload it into DST via copy_tile (the only access a UnpackToDestFp32 acc_cb allows).
                    const uint32_t acc_cb = accumulate.config.cb_accumulator;
                    accum_dfb.wait_front(1);
                    if (accumulate.reload == AccumulateReloadMode::FoldViaAdd) {
                        // Fold the accumulator as an add_tiles SRCB operand — no dest reload. Reads acc via
                        // SrcB, so ONLY valid when acc_cb is UnpackToDestMode::Default. Parity of full_cnt
                        // decides; add_tiles_init does NOT reconfig format, so reconfig SRCB around the acc-add
                        // (acc may be fp32 while the input is bf16) and restore it after.
                        if (full_cnt & 1u) {
                            if (full_cnt == 1u) {
                                reconfig_data_format_srcb(input_dfb_id, acc_cb);
                                add_tiles_init(input_dfb_id, acc_cb, true);  // fresh DST reads 0 -> new[0] + acc
                                add_tiles(input_dfb_id, acc_cb, start, 0, 0);
                                reconfig_data_format_srcb(acc_cb, input_dfb_id);
                            } else {
                                add_tiles_init(input_dfb_id, input_dfb_id, true);                 // fresh DST reads 0
                                add_tiles(input_dfb_id, input_dfb_id, start, start + stride, 0);  // seed new pair
                                add_tiles_init(input_dfb_id, input_dfb_id, true);
                                for (uint32_t k = 2; k + 1 < full_cnt; k += 2) {
                                    add_tiles(
                                        input_dfb_id, input_dfb_id, start + k * stride, start + (k + 1) * stride, 0);
                                }
                                reconfig_data_format_srcb(input_dfb_id, acc_cb);
                                add_tiles_init(input_dfb_id, acc_cb, true);  // last new tile + accumulator
                                add_tiles(input_dfb_id, acc_cb, start + (full_cnt - 1u) * stride, 0, 0);
                                reconfig_data_format_srcb(acc_cb, input_dfb_id);
                            }
                        } else {
                            reconfig_data_format_srca(input_dfb_id, acc_cb);
                            copy_tile_init(acc_cb);
                            copy_tile(acc_cb, 0, 0);  // DST = accumulator (even count reloads as the seed)
                            reconfig_data_format_srca(acc_cb, input_dfb_id);
                            add_tiles_init(input_dfb_id, input_dfb_id, true);
                            for (uint32_t k = 0; k < full_cnt; k += 2) {
                                add_tiles(input_dfb_id, input_dfb_id, start + k * stride, start + (k + 1) * stride, 0);
                            }
                        }
                    } else if (accumulate.reload == AccumulateReloadMode::CopySeedSfpuAdd) {
                        // Sum this chunk's new tiles into DST[0] with pure pairwise add_tiles (fresh DST reads
                        // 0 -> full fp32 accumulation, no DEST-reuse TF32 truncation), reload the accumulator
                        // into DST[1] via copy_tile (U2D-safe, lossless), then SFPU-add DST[0] += DST[1] (the
                        // SFPU operates on DST in fp32). The accumulator is never an FPU SrcA/B operand. WH/BH
                        // only (add_binary_tile is not on Quasar).
                        {
                            uint32_t k = 0;
                            if (full_cnt & 1u) {
                                copy_tile_init(input_dfb_id);
                                copy_tile(input_dfb_id, start, 0);  // DST[0] = new[0] (odd seed, fresh)
                                k = 1;
                            }
                            add_tiles_init(input_dfb_id, input_dfb_id, true);
                            for (; k < full_cnt; k += 2) {
                                add_tiles(input_dfb_id, input_dfb_id, start + k * stride, start + (k + 1) * stride, 0);
                            }
                        }
                        reconfig_data_format_srca(input_dfb_id, acc_cb);
                        copy_tile_init(acc_cb);
                        copy_tile(acc_cb, 0, 1);  // DST[1] = accumulator (adjacent slot)
                        reconfig_data_format_srca(acc_cb, input_dfb_id);
#ifndef ARCH_QUASAR
                        add_binary_tile_init();
                        add_binary_tile(0, 1, 0);  // DST[0] = DST[0] + DST[1] (fp32 SFPU add)
                        if constexpr (within_tile == ReduceWithinTile::Collapse) {
                            sfpu_reduce_init<PoolType::SUM, dst_fmt>();  // restore the reduce macro to finalize with
                        }
#else
                        ASSERT(false);  // CopySeedSfpuAdd needs add_binary_tile (WH/BH only)
#endif
                    } else {
                        // CopySeed*: reload the accumulator into DST via copy_tile — the ONLY access a
                        // UnpackToDestFp32 acc_cb allows (the accumulator is never an FPU operand). copy_tile
                        // uses SrcA (or unpack-direct-to-dest when tagged), so reconfig SRCA around it; SrcB is
                        // left untouched (== input from the per-call reconfig), which the partial fold needs.
                        reconfig_data_format_srca(input_dfb_id, acc_cb);
                        copy_tile_init(acc_cb);
                        copy_tile(acc_cb, 0, 0);  // DST = accumulator
                        reconfig_data_format_srca(acc_cb, input_dfb_id);
                        if (accumulate.reload == AccumulateReloadMode::CopySeedUniform) {
                            // Add every new tile via a DEST-reuse add (new tile -> SrcA, running sum reused
                            // from DST). 1 tile/op; acc stays resident in DST, never an FPU CB operand.
                            binary_dest_reuse_tiles_init<
                                ckernel::EltwiseBinaryType::ELWADD,
                                ckernel::EltwiseBinaryReuseDestType::DEST_TO_SRCB>(input_dfb_id);
                            for (uint32_t k = 0; k < full_cnt; ++k) {
                                binary_dest_reuse_tiles<
                                    ckernel::EltwiseBinaryType::ELWADD,
                                    ckernel::EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
                                    input_dfb_id, start + k * stride, 0);
                            }
                        } else if (accumulate.reload == AccumulateReloadMode::CopySeedZeroPair) {
                            // Odd leftover pairs with the planned ZERO tile via an acc_to_dest add_tiles:
                            // DST += input[leftover] + 0, keeping the running sum in fp32 DST (no DEST-reuse
                            // truncation, no SFPU). The zero follows any partial mask in the auxiliary CB.
                            uint32_t k = 0;
                            if (full_cnt & 1u) {
                                add_tiles_init(input_dfb_id, scaler_dfb_id, true);
                                add_tiles(input_dfb_id, scaler_dfb_id, start, zero_idx, 0);
                                k = 1;
                            }
                            add_tiles_init(input_dfb_id, input_dfb_id, true);
                            for (; k < full_cnt; k += 2) {
                                add_tiles(input_dfb_id, input_dfb_id, start + k * stride, start + (k + 1) * stride, 0);
                            }
                        } else {
                            // CopySeedPairs: odd leftover first via one DEST-reuse add, then the bulk in pairs
                            // (2 tiles/op). Ending on add_tiles(input,input) leaves SrcB=input for the fold.
                            uint32_t k = 0;
                            if (full_cnt & 1u) {
                                binary_dest_reuse_tiles_init<
                                    ckernel::EltwiseBinaryType::ELWADD,
                                    ckernel::EltwiseBinaryReuseDestType::DEST_TO_SRCB>(input_dfb_id);
                                binary_dest_reuse_tiles<
                                    ckernel::EltwiseBinaryType::ELWADD,
                                    ckernel::EltwiseBinaryReuseDestType::DEST_TO_SRCB>(input_dfb_id, start, 0);
                                k = 1;
                            }
                            add_tiles_init(input_dfb_id, input_dfb_id, true);
                            for (; k < full_cnt; k += 2) {
                                add_tiles(input_dfb_id, input_dfb_id, start + k * stride, start + (k + 1) * stride, 0);
                            }
                        }
                    }
                    accum_dfb.pop_front(1);
                    if (has_partial) {  // ROW/COL partial: fold the masked last tile in after the accumulator
                        fold_partial_last(start + full_cnt * stride);
                    }
                }
            } else {
                // acc_to_dest=true throughout: a freshly-acquired DST reads 0 on its first write, so the first
                // add is the plain sum — no separate overwrite-seed init. Odd count: seed DST with a unary copy.
                uint32_t k = 0;
                if (full_cnt & 1u) {
                    copy_tile_init(input_dfb_id);
                    copy_tile(input_dfb_id, start, 0);
                    k = 1;
                }
                add_tiles_init(input_dfb_id, input_dfb_id, true);
                for (; k < full_cnt; k += 2) {
                    add_tiles(input_dfb_id, input_dfb_id, start + k * stride, start + (k + 1) * stride, 0);
                }
                // partial: fold the LAST reduce-dim tile in, masked, ACCUMULATING into DST.
                if (has_partial) {
                    fold_partial_last(start + full_cnt * stride);
                }
            }
        }

        // Non-last cross-call accumulation steps leave the raw sum in DST; the last call performs the
        // within-tile collapse, AVG normalization, and caller post-op.
        finalize_output(0);

        tile_regs_commit();
        tile_regs_wait();
        if constexpr (should_pop_p) {  // Bulk / WaitAndPop: reserve + pack + push per output tile
            output_dfb.reserve_back(1);
            pack_tile(0, output_dfb_id);
            output_dfb.push_back(1);
        } else {  // no-pop: bulk-reserved upfront; write output o to its OWN page o. (The standard no-pop body
                  // packs every output to the default page 0 — correct only for a single output; the fast path
                  // passes o explicitly so multi-output no-pop is correct.)
            pack_tile(0, output_dfb_id, o);
        }
        tile_regs_release();
        if constexpr (bulk_per_output) {
            input_dfb.pop_front(cnt);
        }
    }
    if constexpr (!should_pop_p) {
        output_dfb.push_back(n_out);  // no-pop: bulk-push all outputs at the end
    }
    if constexpr (helper_pops_block) {
        input_dfb.pop_front(in_tiles);  // only BulkWaitBulkPop pops the resident block
    }
}

}  // namespace detail

// =============================================================================
// ReduceDataFormatReconfigMode Helper Functions
// =============================================================================

constexpr bool reconfig_input(ReduceDataFormatReconfigMode mode) {
    return mode == ReduceDataFormatReconfigMode::INPUT || mode == ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT;
}

constexpr bool reconfig_output(ReduceDataFormatReconfigMode mode) {
    return mode == ReduceDataFormatReconfigMode::OUTPUT || mode == ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT;
}

// =============================================================================
// ReduceInputPolicy Helper Functions
// =============================================================================

constexpr bool waits_per_tile(ReduceInputPolicy p) { return p == ReduceInputPolicy::WaitAndPopPerTile; }
constexpr bool waits_bulk(ReduceInputPolicy p) { return p == ReduceInputPolicy::BulkWaitBulkPop; }
constexpr bool waits_upfront(ReduceInputPolicy p) { return p == ReduceInputPolicy::WaitUpfrontNoPop; }
constexpr bool no_wait(ReduceInputPolicy p) { return p == ReduceInputPolicy::NoWaitNoPop; }
constexpr bool waits_chunked(ReduceInputPolicy p) { return p == ReduceInputPolicy::ChunkedWaitChunkedPop; }
constexpr bool should_pop(ReduceInputPolicy p) {
    return p == ReduceInputPolicy::WaitAndPopPerTile || p == ReduceInputPolicy::BulkWaitBulkPop ||
           p == ReduceInputPolicy::ChunkedWaitChunkedPop;
}
constexpr bool manages_cb(ReduceInputPolicy p) {
    // Returns true if the reduce function manages CB wait/reserve/push (not preloaded)
    return p != ReduceInputPolicy::NoWaitNoPop;
}

// =============================================================================
// Helper Function Implementations
// =============================================================================

template <PoolType reduce_type, ReduceDim reduce_dim>
ALWI void reduce_init_short_with_dt(uint32_t old_dfb_id, uint32_t input_dfb_id, uint32_t scaler_dfb_id) {
    constexpr bool swap_operands = reduce_swaps_operands<reduce_type, reduce_dim, false>();
    const uint32_t srca_dfb_id = swap_operands ? scaler_dfb_id : input_dfb_id;

    // Reconfigure SRCA data format from old_dfb_id to the correct SrcA format
    UNPACK(
        (llk_unpack_reconfig_data_format_srca<DST_ACCUM_MODE, p_dim_stride_target::IGNORE>(old_dfb_id, srca_dfb_id)));
    MATH((llk_math_reconfig_data_format_srca<DST_ACCUM_MODE>(old_dfb_id, srca_dfb_id)));

    // Reconfigure unpacker for reduce operation (SRCA and SRCB)
    UNPACK((llk_unpack_AB_reduce_init<reduce_type, reduce_dim>(input_dfb_id, scaler_dfb_id)));

    // Reconfigure math for reduce operation
    MATH((llk_math_reduce_init<reduce_type, reduce_dim, DST_ACCUM_MODE, MATH_FIDELITY>(input_dfb_id, scaler_dfb_id)));

    // Skip packer reconfiguration - it remains valid from initial reduce_init call
}

template <typename AccumulateT>
ALWI constexpr uint32_t get_dst_index(const AccumulateT& accumulate) {
    if constexpr (is_accumulate_v<AccumulateT>) {
        return accumulate.config.dst_index;
    } else {
        return 0;
    }
}

template <PoolType reduce_type, ReduceDim reduce_dim, typename AccumulateT, bool is_sfpu = false>
ALWI void reload_accumulator_if_needed(
    DataflowBuffer& accum_dfb,
    uint32_t input_dfb_id,
    uint32_t scaler_dfb_id,
    const AccumulateT& accumulate,
    uint32_t num_tiles = 1) {
    if constexpr (is_accumulate_v<AccumulateT>) {
        if (!accumulate.is_first()) {  // Reload on all iterations except first
            accum_dfb.wait_front(num_tiles);
            constexpr bool swap_operands = reduce_swaps_operands<reduce_type, reduce_dim, is_sfpu>();
            const uint32_t prev_srca_cb = swap_operands ? scaler_dfb_id : input_dfb_id;

            // For MAX + REDUCE_ROW, GMPOOL's running accumulator lives at row 0 of face 0
            // (max for rows 0-15) and row 0 of face 2 (max for rows 16-31); faces 1 and 3
            // are never read. The LLK's reduce_row_perform_transpose then rotates those
            // row-0 accumulators into col 0 of face 0 and col 0 of face 2 for packing.
            // A vanilla copy_tile reload would leave the running max at col 0, but the
            // next GMPOOL iteration only reads row 0 — so it would be silently dropped.
            // Within-face-16x16-transpose on reload puts col 0 of each face back at row 0
            // of that face, restoring the exact layout GMPOOL expects. The SFPU path never runs
            // GMPOOL — its running max is a plain full-tile value — so it must not transpose.
            constexpr bool reload_within_face_transpose =
                (reduce_type == PoolType::MAX && reduce_dim == ReduceDim::REDUCE_ROW && !is_sfpu);

            reconfig_data_format_srca(prev_srca_cb, accumulate.config.cb_accumulator);
            copy_tile_to_dst_init_short(
                accumulate.config.cb_accumulator,
                /*transpose_of_faces=*/0,
                /*transpose_within_16x16_face=*/reload_within_face_transpose ? 1u : 0u);
            for (uint32_t tile = 0; tile < num_tiles; ++tile) {
                copy_tile(accumulate.config.cb_accumulator, tile, accumulate.config.dst_index + tile);
            }
            accum_dfb.pop_front(num_tiles);

            // CRITICAL: Re-init after copy_tile corrupts SRCA config
            // Use short version since packer config is still valid from initial init
            // Pass accumulator DFB as old_dfb_id to reconfigure data format from accumulator to input DFB
            if constexpr (is_sfpu) {
                // Point SrcA back at the input for the next fold; the caller does the fold init.
                reconfig_data_format_srca(accumulate.config.cb_accumulator, input_dfb_id);
                copy_tile_to_dst_init_short(input_dfb_id);
            } else {
                reduce_init_short_with_dt<reduce_type, reduce_dim>(
                    accumulate.config.cb_accumulator, input_dfb_id, scaler_dfb_id);
            }
        }
    }
}

template <ReduceInputPolicy input_policy>
ALWI void assert_input_dfb_size(
    uint32_t input_dfb_id, uint32_t tiles_per_bulk, uint32_t total_tiles, uint32_t tiles_per_chunk = 1) {
    if constexpr (waits_per_tile(input_policy)) {
        ASSERT(get_dfb_num_pages(input_dfb_id) >= 1);
    } else if constexpr (waits_bulk(input_policy)) {
        ASSERT(get_dfb_num_pages(input_dfb_id) >= tiles_per_bulk);
        ASSERT(get_dfb_num_pages(input_dfb_id) % tiles_per_bulk == 0);
    } else if constexpr (waits_chunked(input_policy)) {
        ASSERT(tiles_per_chunk > 0);
        ASSERT(get_dfb_num_pages(input_dfb_id) >= tiles_per_chunk);
    } else {  // waits_upfront or no_wait
        ASSERT(get_dfb_num_pages(input_dfb_id) >= total_tiles);
    }
}

ALWI void assert_output_dfb_size(uint32_t output_dfb_id) {
    // Outputs are reserved and pushed one tile at a time for every input policy.
    ASSERT(get_dfb_num_pages(output_dfb_id) >= 1);
}

// =============================================================================
// Main Reduce Function Implementation
// =============================================================================

template <
    PoolType reduce_type,
    ReduceDim reduce_dim,
    uint32_t input_dfb_id,
    uint32_t scaler_dfb_id,
    uint32_t output_dfb_id,
    ReduceInputPolicy input_policy,
    ReduceDataFormatReconfigMode reconfig_mode,
    ReduceFp32Mode fp32_mode,
    ReduceAlgorithm algorithm,
    ReduceWithinTile within_tile,
    uint32_t reduce_factor,
    typename AccumulateT,
    typename PostReduceOp>
ALWI void reduce(
    ReduceInputBlockShape input_block_shape,
    ReduceInputMemoryLayout input_memory_layout,
    AccumulateT accumulate,
    PostReduceOp post_reduce_op,
    ReducePartialScaler partial_scaler,
    ReduceInputChunk input_chunk) {
    // Int32 and Accurate fp32 route to the SFPU via is_sfpu_reduce_path<>(); others use FPU/GMPOOL.
    constexpr DataFormat reduce_format = static_cast<DataFormat>(unpack_src_format[input_dfb_id]);
    // =============================================================================
    // Static Assertions (compile-time validation)
    // =============================================================================
    static_assert(
        (reduce_type != PoolType::MAX && reduce_type != PoolType::SUM) || reduce_dim != ReduceDim::REDUCE_SCALAR ||
            reduce_format != DataFormat::Int32,
        "Int32 MAX/SUM REDUCE_SCALAR is not supported (host decomposes Int32 HW reduce into W-then-H)");
    static_assert(
        reduce_type != PoolType::AVG || reduce_format != DataFormat::Int32, "Int32 AVG (mean) is not supported");
    static_assert(reduce_factor != 0, "reduce_factor must not be zero");
    static_assert(
        reduce_factor == 1 || reduce_type == PoolType::AVG,
        "A non-default reduce_factor is only valid with PoolType::AVG");
#ifndef ARCH_QUASAR  // Quasar's ckernel::PoolType has no MIN, so this check is vacuous there
    static_assert(
        reduce_type != PoolType::MIN || is_sfpu_reduce_path<reduce_type, reduce_dim, reduce_format, fp32_mode>(),
        "MIN is only valid on an SFPU path (Int32 or Accurate fp32); FPU MIN arrives as PoolType::MAX via -MAX(-x)");
#endif
    static_assert(
        is_accumulation_type_v<AccumulateT>,
        "AccumulateT must be a valid accumulation type (NoAccumulation or Accumulate)");
    static_assert(is_post_reduce_op_v<PostReduceOp>, "PostReduceOp must be callable with a uint32_t argument");
    static_assert(
        !is_accumulate_v<AccumulateT> || !(reduce_type == PoolType::MAX && reduce_dim == ReduceDim::REDUCE_SCALAR),
        "Accumulate with PoolType::MAX + REDUCE_SCALAR is not supported: the pack edge mask "
        "keeps only DST(0,0), but GMPOOL needs that running max broadcast across face-0 row 4 "
        "on the reload pass, which the current copy_tile reload cannot reproduce.");
#ifdef ARCH_QUASAR
    // The MAX + REDUCE_ROW accumulator reload relies on a within-16x16-face transpose during
    // copy_tile_to_dst_init_short (see reload_accumulator_if_needed). That transpose is rejected
    // by copy_tile_to_dst_init_short on Quasar ("Transpose within face not supported on Quasar"),
    // and there is no Quasar-compatible reload that restores the layout GMPOOL expects.
    static_assert(
        !is_accumulate_v<AccumulateT> || !(reduce_type == PoolType::MAX && reduce_dim == ReduceDim::REDUCE_ROW),
        "Accumulate with PoolType::MAX + REDUCE_ROW is not supported on Quasar: the accumulator "
        "reload requires a within-16x16-face transpose, which copy_tile_to_dst_init_short asserts "
        "against on Quasar.");
#endif

    // =============================================================================
    // Algorithm selection. AccumulateViaAdd is a restricted, faster datapath for wide float SUM/AVG reduces;
    // anything it cannot express is rejected here (compile-time where possible) and must use ReduceTile.
    // =============================================================================
    constexpr bool is_sfpu = is_sfpu_reduce_path<reduce_type, reduce_dim, reduce_format, fp32_mode>();
    static_assert(
        reduce_type != PoolType::AVG || !(algorithm == ReduceAlgorithm::AccumulateViaAdd || is_sfpu) ||
            reduce_factor != 1,
        "PoolType::AVG requires reduce_factor != 1 on AccumulateViaAdd and SFPU reduce paths");
    if constexpr (algorithm == ReduceAlgorithm::AccumulateViaAdd) {
        static_assert(
            reduce_type == PoolType::SUM || reduce_type == PoolType::AVG,
            "AccumulateViaAdd computes SUM or AVG. AVG uses the caller-supplied compile-time reduce_factor; "
            "MAX/MIN are not expressible via additive accumulate, so use ReduceTile.");
        static_assert(
            reduce_format != DataFormat::Int32,
            "AccumulateViaAdd: float only (add_tiles + sfpu_reduce). Int32 must use ReduceTile.");
        // ReduceWithinTile::Skip drops only the lane collapse. AVG remains well-defined because reduce_factor
        // is caller-owned (for example, the number of contributors in a cross-core combine).
        // All policies except WaitAndPopPerTile + COL are supported. Resident policies index the source;
        // Chunked COL carries explicit reduction-axis/output-group geometry and retains that group in DEST.
        // Cross-call Accumulate keeps a raw partial sum in its accumulator CB and normalizes AVG only when the
        // last call finalizes. Indexed policies honor AccumulateReloadMode; streaming ROW/SCALAR copy-seed the
        // accumulator into DST before consuming the next input chunk.
        // A one-tile-at-a-time stream has no column-group geometry, so REDUCE_COL cannot identify which DEST
        // slot owns an arriving tile. ChunkedWaitChunkedPop carries both axis/output chunk sizes and supports
        // COL once the axis chunk is large enough for additive progress (validated below at runtime).
        static_assert(
            input_policy != ReduceInputPolicy::WaitAndPopPerTile || reduce_dim != ReduceDim::REDUCE_COL,
            "AccumulateViaAdd REDUCE_COL requires grouped input; use ChunkedWaitChunkedPop or an indexed "
            "resident-input policy.");
        // Streaming + partial is supported for ROW and for grouped Chunked COL: the masked last tile/row folds
        // in as the final streamed op. Partial ROW/COL is also supported by the indexed policies. REDUCE_SCALAR
        // can be partial in BOTH axes at once (a single row/col mask cannot express the corner), so it is
        // rejected — use ReduceTile.
        if constexpr (reduce_dim == ReduceDim::REDUCE_SCALAR) {
            ASSERT(!partial_scaler.use_partial);
        }
        // Skip + partial is a contradiction: use_partial requests a lane mask along an axis whose inputs are
        // already collapsed. Reject it rather than pretend the mask did useful work.
        if constexpr (within_tile == ReduceWithinTile::Skip) {
            ASSERT(!partial_scaler.use_partial);
        }
        // Cross-call Accumulate + partial is supported for ROW/COL (the masked last tile folds into each
        // chunk's sum via fold_partial_last). SCALAR partial is rejected above (622-ish) regardless of accumulate.
        // row_stride (a WIDER resident block, padded rows) is honored for ROW/COL indexed reduces — the
        // per-output indexing steps by the row pitch and skips the padding tiles — under BulkWaitBulkPop,
        // WaitUpfrontNoPop, NoWaitNoPop, AND cross-call Accumulate (the fold uses the same start/stride/pitch).
        // SCALAR walks a 2-D block (a single linear reduce-dim stride cannot skip per-row padding) and streaming
        // is a pure contiguous stream (no indexing) — both require a contiguous layout (row_stride 0 or == Wt).
        if (input_memory_layout.row_stride != 0) {
            ASSERT(input_memory_layout.row_stride >= input_block_shape.cols);
            if constexpr (reduce_dim == ReduceDim::REDUCE_SCALAR) {
                ASSERT(input_memory_layout.row_stride == input_block_shape.cols);
            }
            if constexpr (
                input_policy == ReduceInputPolicy::WaitAndPopPerTile ||
                input_policy == ReduceInputPolicy::ChunkedWaitChunkedPop) {
                ASSERT(input_memory_layout.row_stride == input_block_shape.cols);
            }
        }
        detail::reduce_accumulate_via_add<
            reduce_type,
            reduce_dim,
            input_dfb_id,
            scaler_dfb_id,
            output_dfb_id,
            input_policy,
            reconfig_mode,
            AccumulateT,
            PostReduceOp,
            within_tile,
            reduce_factor>(
            input_block_shape, input_memory_layout, partial_scaler, accumulate, post_reduce_op, input_chunk);
        return;
    }

    // Past this point is the ReduceTile datapath, where reduce_tile (matmul-with-ones) IS the within-tile
    // collapse — there is no separate finalize pass to elide, so Skip has no meaning here.
    // The condition MUST be predicated on algorithm: `if constexpr` discards only the statements
    // INSIDE its branch, so everything after the AccumulateViaAdd block (including this static_assert) is
    // still instantiated for an AccumulateViaAdd call. Asserting `within_tile == Collapse` bare here made
    // ReduceWithinTile::Skip fail to compile on EVERY algorithm, including the one that implements it.
    static_assert(
        algorithm == ReduceAlgorithm::AccumulateViaAdd || within_tile == ReduceWithinTile::Collapse,
        "ReduceWithinTile::Skip is AccumulateViaAdd-only: on the ReduceTile datapath the reduce_tile "
        "matmul-with-ones performs the within-tile collapse itself, so there is nothing to skip. Select "
        "ReduceAlgorithm::AccumulateViaAdd, or drop the Skip.");

    // =============================================================================
    // Runtime Assertions (parameter validation)
    // =============================================================================
    ASSERT(input_dfb_id != output_dfb_id);
    ASSERT(input_dfb_id != scaler_dfb_id);
    ASSERT(output_dfb_id != scaler_dfb_id);
#ifndef ARCH_QUASAR
    // is_valid_dfb_tile_page_size() is a debug validator only defined on WH/BH
    UNPACK(ASSERT(is_valid_dfb_tile_page_size(input_dfb_id, (DataFormat)unpack_src_format[input_dfb_id])));
    UNPACK(ASSERT(is_valid_dfb_tile_page_size(scaler_dfb_id, (DataFormat)unpack_src_format[scaler_dfb_id])));
    PACK(ASSERT(is_valid_dfb_tile_page_size(output_dfb_id, (DataFormat)pack_dst_format[output_dfb_id])));
#endif
    ASSERT(input_block_shape.rows > 0);
    ASSERT(input_block_shape.cols > 0);
    ASSERT(input_block_shape.batches > 0);
    if (input_memory_layout.row_stride != 0) {
        ASSERT(input_memory_layout.row_stride >= input_block_shape.cols);
    }
    if constexpr (waits_chunked(input_policy)) {
        ASSERT(input_chunk.reduce_axis_tiles > 0);
        ASSERT(input_chunk.output_tiles > 0);
    }

    // Compile-time flag: true when Accumulate type is passed, false otherwise
    constexpr bool enable_accumulation = is_accumulate_v<AccumulateT>;
    // Extract block shape components
    const uint32_t Ht = input_block_shape.rows;
    const uint32_t Wt = input_block_shape.cols;
    const uint32_t num_batches = input_block_shape.batches;

    DataflowBuffer input_dfb(input_dfb_id);
    DataflowBuffer scaler_dfb(scaler_dfb_id);
    DataflowBuffer output_dfb(output_dfb_id);
    DataflowBuffer accum_dfb([&]() -> uint32_t {
        if constexpr (enable_accumulation) {
            return accumulate.config.cb_accumulator;
        } else {
            return 0;
        }
    }());

    // Apply reconfig based on mode
    constexpr bool swap_operands = reduce_swaps_operands<reduce_type, reduce_dim, is_sfpu>();
    if constexpr (reconfig_input(reconfig_mode)) {
        if constexpr (swap_operands) {
            reconfig_data_format(scaler_dfb_id, input_dfb_id);
        } else {
            reconfig_data_format(input_dfb_id, scaler_dfb_id);
        }
    }
    if constexpr (reconfig_output(reconfig_mode)) {
        pack_reconfig_data_format(output_dfb_id);
    }
    // Initialization
    if constexpr (is_sfpu) {
        init_sfpu(input_dfb_id, output_dfb_id);
        copy_tile_to_dst_init_short(input_dfb_id);
    } else {
        reduce_init<reduce_type, reduce_dim>(input_dfb_id, scaler_dfb_id, output_dfb_id);
    }
    // Partial scaler: REDUCE_SCALAR can't use it (applies the scaler twice).
    // Other reduce dims may add a partial-fill tile at index >0; wait for both.
    if constexpr (reduce_dim == ReduceDim::REDUCE_SCALAR) {
        ASSERT(!partial_scaler.use_partial);
    }
    scaler_dfb.wait_front(partial_scaler.scaler_tile_count());
    if constexpr (is_sfpu) {
        PACK((llk_pack_reduce_mask_config<reduce_dim, PackMode::Default>(output_dfb_id)));
    }

    constexpr uint32_t onetile = 1;

    // Pattern dispatch based on reduce_dim
    if constexpr (reduce_dim == ReduceDim::REDUCE_SCALAR) {
        // =================================================================
        // REDUCE_SCALAR: HW reduction - all tiles -> 1 output tile per batch
        // =================================================================
        const uint32_t stride = (input_memory_layout.row_stride > 0) ? input_memory_layout.row_stride : Wt;
        const uint32_t tiles_per_bulk = Ht * stride;
        const uint32_t total_input_tiles = tiles_per_bulk * num_batches;
        UNPACK((assert_input_dfb_size<input_policy>(
            input_dfb_id, tiles_per_bulk, total_input_tiles, input_chunk.reduce_axis_tiles)));
        PACK((assert_output_dfb_size(output_dfb_id)));

        // PersistentPolicy: wait for all tiles upfront
        if constexpr (waits_upfront(input_policy)) {
            input_dfb.wait_front(total_input_tiles);
        }

        uint32_t batch_offset = 0;
        for (uint32_t nc = 0; nc < num_batches; ++nc) {
            // BulkWaitBulkPop: wait for all Ht×Wt tiles in bulk
            if constexpr (waits_bulk(input_policy)) {
                input_dfb.wait_front(tiles_per_bulk);
            }

            tile_regs_acquire();

            // Reload accumulator if needed (zero overhead when AccumulateT is NoAccumulation)
            reload_accumulator_if_needed<reduce_type, reduce_dim, AccumulateT, is_sfpu>(
                accum_dfb, input_dfb_id, scaler_dfb_id, accumulate);

            const uint32_t dst_idx = get_dst_index(accumulate);
            if constexpr (waits_chunked(input_policy)) {
                uint32_t consumed = 0;
                const uint32_t input_tiles = Ht * Wt;
                while (consumed < input_tiles) {
                    const uint32_t remaining = input_tiles - consumed;
                    const uint32_t current_chunk =
                        remaining < input_chunk.reduce_axis_tiles ? remaining : input_chunk.reduce_axis_tiles;
                    input_dfb.wait_front(current_chunk);
                    for (uint32_t tile_idx = 0; tile_idx < current_chunk; ++tile_idx) {
                        reduce_tile<reduce_type, reduce_dim>(input_dfb_id, scaler_dfb_id, tile_idx, 0, dst_idx);
                    }
                    input_dfb.pop_front(current_chunk);
                    consumed += current_chunk;
                }
            } else {
                for (uint32_t ht = 0; ht < Ht; ++ht) {
                    for (uint32_t wt = 0; wt < Wt; ++wt) {
                        if constexpr (waits_per_tile(input_policy)) {
                            // One-at-a-time: wait/pop per tile
                            input_dfb.wait_front(onetile);
                            reduce_tile<reduce_type, reduce_dim>(input_dfb_id, scaler_dfb_id, 0, 0, dst_idx);
                            input_dfb.pop_front(onetile);
                        } else if constexpr (waits_bulk(input_policy)) {
                            // BulkWaitBulkPop: use indexed access
                            uint32_t tile_idx = ht * stride + wt;
                            reduce_tile<reduce_type, reduce_dim>(input_dfb_id, scaler_dfb_id, tile_idx, 0, dst_idx);
                        } else {  // PreloadedPolicy or PersistentPolicy: indexed access
                            uint32_t tile_idx = batch_offset + ht * stride + wt;
                            reduce_tile<reduce_type, reduce_dim>(input_dfb_id, scaler_dfb_id, tile_idx, 0, dst_idx);
                        }
                    }
                }
            }

            // Call post-reduce operation on the single accumulated DST register.
            // No-op when PostReduceOp is the default NoOp.
            post_reduce_op(dst_idx);

            output_dfb.reserve_back(onetile);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(get_dst_index(accumulate), output_dfb_id);
            tile_regs_release();
            output_dfb.push_back(onetile);

            // BulkWaitBulkPop: pop all tiles after processing
            if constexpr (waits_bulk(input_policy)) {
                input_dfb.pop_front(tiles_per_bulk);
            }

            // PreloadedPolicy or PersistentPolicy: update batch offset
            if constexpr (!should_pop(input_policy)) {
                batch_offset += tiles_per_bulk;
            }
        }
    } else if constexpr (reduce_dim == ReduceDim::REDUCE_ROW) {
        // =================================================================
        // REDUCE_ROW: W reduction - each row -> 1 output tile (Ht outputs per batch)
        // =================================================================
        const uint32_t stride = (input_memory_layout.row_stride > 0) ? input_memory_layout.row_stride : Wt;
        const uint32_t total_input_tiles = Ht * stride * num_batches;
        UNPACK(
            (assert_input_dfb_size<input_policy>(input_dfb_id, Wt, total_input_tiles, input_chunk.reduce_axis_tiles)));
        PACK((assert_output_dfb_size(output_dfb_id)));

        // PersistentPolicy: wait for all tiles upfront
        if constexpr (waits_upfront(input_policy)) {
            input_dfb.wait_front(total_input_tiles);
        }

        uint32_t index_offset = 0;
        for (uint32_t nc = 0; nc < num_batches; ++nc) {
            for (uint32_t ht = 0; ht < Ht; ++ht) {
                // BulkWaitBulkPop: wait for entire row upfront
                if constexpr (waits_bulk(input_policy)) {
                    input_dfb.wait_front(Wt);
                }

                tile_regs_acquire();

                // Reload accumulator if needed (zero overhead when AccumulateT is NoAccumulation)
                reload_accumulator_if_needed<reduce_type, reduce_dim, AccumulateT, is_sfpu>(
                    accum_dfb, input_dfb_id, scaler_dfb_id, accumulate);
                if constexpr (is_sfpu) {
                    // Fold needed if the axis has >1 tile, or Accumulate reloaded a result into DST.
                    if (Wt > 1 || !detail::sfpu_is_first_tile(0, accumulate)) {
                        detail::sfpu_reduce_fold_init<reduce_type, reduce_format>();
                    }
                }

                const uint32_t dst_idx = get_dst_index(accumulate);
                if constexpr (waits_chunked(input_policy)) {
                    uint32_t wt = 0;
                    while (wt < Wt) {
                        const uint32_t remaining = Wt - wt;
                        const uint32_t current_chunk =
                            remaining < input_chunk.reduce_axis_tiles ? remaining : input_chunk.reduce_axis_tiles;
                        input_dfb.wait_front(current_chunk);
                        for (uint32_t local_wt = 0; local_wt < current_chunk; ++local_wt) {
                            const uint32_t global_wt = wt + local_wt;
                            if constexpr (is_sfpu) {
                                constexpr uint32_t sfpu_work_dst = 1;
                                const bool is_first_tile = detail::sfpu_is_first_tile(global_wt, accumulate);
                                detail::sfpu_copy_and_fold<reduce_type, reduce_format>(
                                    input_dfb_id, local_wt, dst_idx, sfpu_work_dst, is_first_tile);
                            } else {
                                const uint32_t scaler_idx =
                                    (global_wt == Wt - 1) ? partial_scaler.partial_scaler_idx() : 0;
                                reduce_tile<reduce_type, reduce_dim>(
                                    input_dfb_id, scaler_dfb_id, local_wt, scaler_idx, dst_idx);
                            }
                        }
                        input_dfb.pop_front(current_chunk);
                        wt += current_chunk;
                    }
                } else {
                    for (uint32_t wt = 0; wt < Wt; ++wt) {
                        if constexpr (is_sfpu) {
                            constexpr uint32_t sfpu_work_dst = 1;
                            const bool is_first_tile = detail::sfpu_is_first_tile(wt, accumulate);
                            if constexpr (waits_per_tile(input_policy)) {
                                input_dfb.wait_front(onetile);
                                detail::sfpu_copy_and_fold<reduce_type, reduce_format>(
                                    input_dfb_id, 0, dst_idx, sfpu_work_dst, is_first_tile);
                                input_dfb.pop_front(onetile);
                            } else if constexpr (waits_bulk(input_policy)) {
                                detail::sfpu_copy_and_fold<reduce_type, reduce_format>(
                                    input_dfb_id, wt, dst_idx, sfpu_work_dst, is_first_tile);
                            } else {
                                detail::sfpu_copy_and_fold<reduce_type, reduce_format>(
                                    input_dfb_id, wt + index_offset, dst_idx, sfpu_work_dst, is_first_tile);
                            }
                        } else {
                            // Last W-tile picks up the partial scaler when one was prepared by the reader.
                            const uint32_t scaler_idx = (wt == Wt - 1) ? partial_scaler.partial_scaler_idx() : 0;
                            if constexpr (waits_per_tile(input_policy)) {
                                // One-at-a-time: wait/pop per tile
                                input_dfb.wait_front(onetile);
                                reduce_tile<reduce_type, reduce_dim>(
                                    input_dfb_id, scaler_dfb_id, 0, scaler_idx, dst_idx);
                                input_dfb.pop_front(onetile);
                            } else if constexpr (waits_bulk(input_policy)) {
                                // BulkWaitBulkPop: use indexed access
                                reduce_tile<reduce_type, reduce_dim>(
                                    input_dfb_id, scaler_dfb_id, wt, scaler_idx, dst_idx);
                            } else {  // PreloadedPolicy or PersistentPolicy: indexed access
                                reduce_tile<reduce_type, reduce_dim>(
                                    input_dfb_id, scaler_dfb_id, wt + index_offset, scaler_idx, dst_idx);
                            }
                        }
                    }
                }

                // SFPU intra-tile finalize
                if constexpr (is_sfpu) {
#ifndef ARCH_QUASAR
                    sfpu_reduce_init<reduce_type, reduce_format>();
                    sfpu_reduce<reduce_type, reduce_format, reduce_dim>(dst_idx, /*ct_dim=*/1, /*rt_dim=*/1);
#else
                    // The SFPU reduce path (Int32, or accurate-fp32 SUM) is unported on Quasar:
                    // sfpu_reduce/_init are ARCH_QUASAR-guarded out. is_sfpu_reduce_path() is false for the
                    // FPU/GMPOOL paths Quasar does support (e.g. avg_pool SUM, MAX), so this branch is dead
                    // there; static_assert makes an actual Quasar SFPU-reduce instantiation fail loudly
                    // rather than silently drop the finalize.
                    static_assert(!is_sfpu, "SFPU reduce path is not supported on Quasar");
#endif
                }

                // Call post-reduce operation (e.g., recip_tile for softmax)
                // User's lambda can include reduce_uninit() if needed before custom ops
                post_reduce_op(dst_idx);

                output_dfb.reserve_back(onetile);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(dst_idx, output_dfb_id);
                tile_regs_release();
                output_dfb.push_back(onetile);

                // BulkWaitBulkPop: pop all tiles after processing
                if constexpr (waits_bulk(input_policy)) {
                    input_dfb.pop_front(Wt);
                }

                // PreloadedPolicy or PersistentPolicy: update index offset
                if constexpr (!should_pop(input_policy)) {
                    index_offset += stride;
                }
            }
        }
    } else {
        // =================================================================
        // REDUCE_COL: H reduction - each column -> 1 output tile (Wt outputs per batch)
        // Need chunking due to DEST register limits
        // StreamingPolicy: Tiles arrive in N C W_skip H W_chunk order (chunked by chunk_size)
        // PreloadedPolicy: Tiles in row-major order, indexed as batch_offset + ht*stride + wt
        // =================================================================

        // Auto-detect chunk size from DEST register capacity
        // Both reader (dataflow) and compute kernels compute this identically via DEST_AUTO_LIMIT
        constexpr uint32_t default_chunk_size = is_sfpu ? (DEST_AUTO_LIMIT - 1) : DEST_AUTO_LIMIT;
        const uint32_t chunk_size = waits_chunked(input_policy) ? input_chunk.output_tiles : default_chunk_size;
        const uint32_t stride = (input_memory_layout.row_stride > 0) ? input_memory_layout.row_stride : Wt;
        const uint32_t tiles_per_bulk = Ht * stride;
        const uint32_t total_input_tiles = tiles_per_bulk * num_batches;
        UNPACK((assert_input_dfb_size<input_policy>(
            input_dfb_id, Ht * chunk_size, total_input_tiles, input_chunk.reduce_axis_tiles * chunk_size)));
        PACK((assert_output_dfb_size(output_dfb_id)));

        // PersistentPolicy: wait for all tiles upfront
        if constexpr (waits_upfront(input_policy)) {
            input_dfb.wait_front(total_input_tiles);
        }

        uint32_t batch_offset = 0;
        for (uint32_t nc = 0; nc < num_batches; ++nc) {
            for (uint32_t wt = 0; wt < Wt; wt += chunk_size) {
                uint32_t chunk_end = (wt + chunk_size < Wt) ? (wt + chunk_size) : Wt;
                uint32_t current_chunk = chunk_end - wt;
                uint32_t tiles_in_chunk = Ht * current_chunk;

                // BulkWaitBulkPop: wait for entire chunk upfront
                if constexpr (waits_bulk(input_policy)) {
                    input_dfb.wait_front(tiles_in_chunk);
                }

                tile_regs_acquire();

                // Reload accumulator if needed (zero overhead when AccumulateT is NoAccumulation)
                reload_accumulator_if_needed<reduce_type, reduce_dim, AccumulateT, is_sfpu>(
                    accum_dfb, input_dfb_id, scaler_dfb_id, accumulate, current_chunk);
                if constexpr (is_sfpu) {
                    // Fold needed if the axis has >1 tile, or Accumulate reloaded a result into DST.
                    if (Ht > 1 || !detail::sfpu_is_first_tile(0, accumulate)) {
                        detail::sfpu_reduce_fold_init<reduce_type, reduce_format>();
                    }
                }

                if constexpr (waits_chunked(input_policy)) {
                    uint32_t ht_base = 0;
                    while (ht_base < Ht) {
                        const uint32_t remaining = Ht - ht_base;
                        const uint32_t current_h =
                            remaining < input_chunk.reduce_axis_tiles ? remaining : input_chunk.reduce_axis_tiles;
                        const uint32_t input_tiles = current_h * current_chunk;
                        input_dfb.wait_front(input_tiles);
                        for (uint32_t local_ht = 0; local_ht < current_h; ++local_ht) {
                            const uint32_t ht = ht_base + local_ht;
                            uint32_t dst_idx = get_dst_index(accumulate);
                            const uint32_t scaler_idx = (ht == Ht - 1) ? partial_scaler.partial_scaler_idx() : 0;
                            for (uint32_t local_wt = 0; local_wt < current_chunk; ++local_wt) {
                                const uint32_t tile_idx = local_ht * current_chunk + local_wt;
                                if constexpr (is_sfpu) {
                                    const bool is_first_tile = detail::sfpu_is_first_tile(ht, accumulate);
                                    constexpr uint32_t sfpu_work_dst = default_chunk_size;
                                    detail::sfpu_copy_and_fold<reduce_type, reduce_format>(
                                        input_dfb_id, tile_idx, dst_idx, sfpu_work_dst, is_first_tile);
                                } else {
                                    reduce_tile<reduce_type, reduce_dim>(
                                        input_dfb_id, scaler_dfb_id, tile_idx, scaler_idx, dst_idx);
                                }
                                ++dst_idx;
                            }
                        }
                        input_dfb.pop_front(input_tiles);
                        ht_base += current_h;
                    }
                } else {
                    for (uint32_t ht = 0; ht < Ht; ++ht) {
                        // Base dst_index: from accumulation config or 0 for multi-column output
                        uint32_t dst_idx = get_dst_index(accumulate);
                        // Last H-tile picks up the partial scaler when one was prepared by the reader.
                        [[maybe_unused]] const uint32_t scaler_idx =
                            (ht == Ht - 1) ? partial_scaler.partial_scaler_idx() : 0;
                        for (uint32_t i = wt; i < chunk_end; ++i) {
                            if constexpr (is_sfpu) {
                                const bool is_first_tile = detail::sfpu_is_first_tile(ht, accumulate);
                                constexpr uint32_t sfpu_work_dst = default_chunk_size;
                                if constexpr (waits_per_tile(input_policy)) {
                                    input_dfb.wait_front(onetile);
                                    detail::sfpu_copy_and_fold<reduce_type, reduce_format>(
                                        input_dfb_id, 0, dst_idx, sfpu_work_dst, is_first_tile);
                                    input_dfb.pop_front(onetile);
                                } else if constexpr (waits_bulk(input_policy)) {
                                    const uint32_t tile_idx = ht * current_chunk + (i - wt);
                                    detail::sfpu_copy_and_fold<reduce_type, reduce_format>(
                                        input_dfb_id, tile_idx, dst_idx, sfpu_work_dst, is_first_tile);
                                } else {
                                    const uint32_t tile_idx = batch_offset + ht * stride + i;
                                    detail::sfpu_copy_and_fold<reduce_type, reduce_format>(
                                        input_dfb_id, tile_idx, dst_idx, sfpu_work_dst, is_first_tile);
                                }
                            } else if constexpr (waits_per_tile(input_policy)) {
                                // One-at-a-time: wait/pop per tile
                                input_dfb.wait_front(onetile);
                                reduce_tile<reduce_type, reduce_dim>(
                                    input_dfb_id, scaler_dfb_id, 0, scaler_idx, dst_idx);
                                input_dfb.pop_front(onetile);
                            } else if constexpr (waits_bulk(input_policy)) {
                                // BulkWaitBulkPop: use indexed access
                                uint32_t tile_idx = ht * current_chunk + (i - wt);
                                reduce_tile<reduce_type, reduce_dim>(
                                    input_dfb_id, scaler_dfb_id, tile_idx, scaler_idx, dst_idx);
                            } else {  // PreloadedPolicy or PersistentPolicy: indexed access
                                uint32_t tile_idx = batch_offset + ht * stride + i;
                                reduce_tile<reduce_type, reduce_dim>(
                                    input_dfb_id, scaler_dfb_id, tile_idx, scaler_idx, dst_idx);
                            }
                            ++dst_idx;
                        }
                    }
                }

                // SFPU intra-tile finalize per output slot
                if constexpr (is_sfpu) {
#ifndef ARCH_QUASAR
                    const uint32_t sfpu_base_dst = get_dst_index(accumulate);
                    sfpu_reduce_init<reduce_type, reduce_format>();
                    for (uint32_t k = 0; k < current_chunk; ++k) {
                        sfpu_reduce<reduce_type, reduce_format, reduce_dim>(
                            sfpu_base_dst + k, /*ct_dim=*/1, /*rt_dim=*/1);
                    }
#else
                    // SFPU reduce path unported on Quasar (see the matching guard above); dead for the
                    // FPU/GMPOOL paths Quasar supports, static_assert catches a real Quasar SFPU reduce.
                    static_assert(!is_sfpu, "SFPU reduce path is not supported on Quasar");
#endif
                }

                // Post-reduce operation for each output tile in chunk
                const uint32_t base_dst = get_dst_index(accumulate);
                for (uint32_t i = 0; i < current_chunk; ++i) {
                    post_reduce_op(base_dst + i);
                }

                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < current_chunk; ++i) {
                    output_dfb.reserve_back(onetile);
                    pack_tile(base_dst + i, output_dfb_id);
                    output_dfb.push_back(onetile);
                }
                tile_regs_release();

                // BulkWaitBulkPop: pop all tiles after processing
                if constexpr (waits_bulk(input_policy)) {
                    input_dfb.pop_front(tiles_in_chunk);
                }
            }
            // Update batch_offset for indexed modes (PreloadedPolicy and PersistentPolicy)
            if constexpr (!should_pop(input_policy)) {
                batch_offset += tiles_per_bulk;
            }
        }
    }

    // Cleanup
    if constexpr (is_sfpu) {
        PACK((llk_pack_reduce_mask_clear()));
    } else {
        reduce_uninit();
    }
}

}  // namespace compute_kernel_lib
