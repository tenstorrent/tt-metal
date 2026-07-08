// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Implementation file for reduce_helpers_compute.hpp
// Do not include directly - include reduce_helpers_compute.hpp instead

#include "api/compute/add_int_sfpu.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/compute_kernel_api.h"
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

// SFPU MAX fold (also used by reduce_{h,w}_neg for -MAX(-x) MIN).
template <DataFormat format>
ALWI void sfpu_reduce_max_fold_init() {
    static_assert(format == DataFormat::Int32, "SFPU reduce MAX fold: Int32 only");
    binary_max_int32_tile_init();
}

template <DataFormat format>
ALWI void sfpu_reduce_max_fold_tile(uint32_t a, uint32_t b, uint32_t out) {
    static_assert(format == DataFormat::Int32, "SFPU reduce MAX fold: Int32 only");
    binary_max_int32_tile(a, b, out);
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
        static_assert(format == DataFormat::Int32, "Accurate fp32 SFPU mean is not supported on Quasar");
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
        static_assert(format == DataFormat::Int32, "Accurate fp32 SFPU mean is not supported on Quasar");
#endif
    }
}

// Pool-type dispatched cross-tile fold init (MAX -> binary_max, SUM -> add_int).
// Used only by compute_kernel_lib::reduce(); the _neg kernels call the MAX fold directly.
template <PoolType pool_type, DataFormat format>
ALWI void sfpu_reduce_fold_init() {
    if constexpr (pool_type == PoolType::SUM) {
        sfpu_reduce_sum_fold_init<format>();
    } else {
        sfpu_reduce_max_fold_init<format>();
    }
}

// Copy one input tile into DST and fold into the running accumulator (first tile seeds dst_idx
// directly). Fold op is selected by pool_type: MAX -> running max, SUM -> running sum.
template <PoolType pool_type, DataFormat format>
ALWI void sfpu_copy_and_fold(
    uint32_t input_cb_id, uint32_t tile_idx, uint32_t dst_idx, uint32_t work_dst, bool is_first_tile) {
    if (is_first_tile) {
        copy_tile(input_cb_id, tile_idx, dst_idx);
    } else {
        copy_tile(input_cb_id, tile_idx, work_dst);
        if constexpr (pool_type == PoolType::SUM) {
            sfpu_reduce_sum_fold_tile<format>(dst_idx, work_dst, dst_idx);
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
constexpr bool should_pop(ReduceInputPolicy p) {
    return p == ReduceInputPolicy::WaitAndPopPerTile || p == ReduceInputPolicy::BulkWaitBulkPop;
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
    UNPACK((llk_unpack_reconfig_data_format_srca<DST_ACCUM_MODE, p_dim_stride_target::IGNORE>(old_dfb_id, srca_dfb_id)));
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

template <
    PoolType reduce_type,
    ReduceDim reduce_dim,
    DataFormat reduce_format,
    typename AccumulateT,
    bool is_sfpu = false>
ALWI void reload_accumulator_if_needed(
    DataflowBuffer& accum_dfb, uint32_t input_dfb_id, uint32_t scaler_dfb_id, const AccumulateT& accumulate) {
    if constexpr (is_accumulate_v<AccumulateT>) {
        if (!accumulate.is_first()) {  // Reload on all iterations except first
            constexpr uint32_t onetile = 1;
            accum_dfb.wait_front(onetile);
            constexpr bool swap_operands = reduce_swaps_operands<reduce_type, reduce_dim, is_sfpu>();
            const uint32_t prev_srca_cb = swap_operands ? scaler_dfb_id : input_dfb_id;

            // For MAX + REDUCE_ROW, GMPOOL's running accumulator lives at row 0 of face 0
            // (max for rows 0-15) and row 0 of face 2 (max for rows 16-31); faces 1 and 3
            // are never read. The LLK's reduce_row_perform_transpose then rotates those
            // row-0 accumulators into col 0 of face 0 and col 0 of face 2 for packing.
            // A vanilla copy_tile reload would leave the running max at col 0, but the
            // next GMPOOL iteration only reads row 0 — so it would be silently dropped.
            // Within-face-16x16-transpose on reload puts col 0 of each face back at row 0
            // of that face, restoring the exact layout GMPOOL expects.
            constexpr bool reload_within_face_transpose =
                (reduce_type == PoolType::MAX && reduce_dim == ReduceDim::REDUCE_ROW);

            reconfig_data_format_srca(prev_srca_cb, accumulate.config.cb_accumulator);
            copy_tile_to_dst_init_short(
                accumulate.config.cb_accumulator,
                /*transpose_of_faces=*/0,
                /*transpose_within_16x16_face=*/reload_within_face_transpose ? 1u : 0u);
            copy_tile(accumulate.config.cb_accumulator, 0, accumulate.config.dst_index);
            accum_dfb.pop_front(onetile);

            // CRITICAL: Re-init after copy_tile corrupts SRCA config
            // Use short version since packer config is still valid from initial init
            // Pass accumulator DFB as old_dfb_id to reconfigure data format from accumulator to input DFB
            if constexpr (is_sfpu) {
                detail::sfpu_reduce_fold_init<reduce_type, reduce_format>();
            } else {
                reduce_init_short_with_dt<reduce_type, reduce_dim>(
                    accumulate.config.cb_accumulator, input_dfb_id, scaler_dfb_id);
            }
        }
    }
}

template <ReduceInputPolicy input_policy>
ALWI void assert_input_dfb_size(uint32_t input_dfb_id, uint32_t tiles_per_bulk, uint32_t total_tiles) {
    if constexpr (waits_per_tile(input_policy)) {
        ASSERT(get_dfb_num_pages(input_dfb_id) >= 1);
    } else if constexpr (waits_bulk(input_policy)) {
        ASSERT(get_dfb_num_pages(input_dfb_id) >= tiles_per_bulk);
        ASSERT(get_dfb_num_pages(input_dfb_id) % tiles_per_bulk == 0);
    } else {  // waits_upfront or no_wait
        ASSERT(get_dfb_num_pages(input_dfb_id) >= total_tiles);
    }
}

template <ReduceInputPolicy input_policy>
ALWI void assert_output_dfb_size(uint32_t output_dfb_id, uint32_t total_outputs) {
    if constexpr (should_pop(input_policy)) {
        // Per-tile reserve/push: only needs 1 page
        ASSERT(get_dfb_num_pages(output_dfb_id) >= 1);
    } else {
        // Bulk reserve upfront: needs all outputs
        ASSERT(get_dfb_num_pages(output_dfb_id) >= total_outputs);
    }
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
    typename AccumulateT,
    typename PostReduceOp>
ALWI void reduce(
    ReduceInputBlockShape input_block_shape,
    ReduceInputMemoryLayout input_memory_layout,
    AccumulateT accumulate,
    PostReduceOp post_reduce_op) {
    // Int32 MAX is routed to the SFPU path via is_sfpu_reduce_path<>(); all other formats use FPU/GMPOOL.
    constexpr DataFormat reduce_format = static_cast<DataFormat>(unpack_src_format[input_dfb_id]);
    // =============================================================================
    // Static Assertions (compile-time validation)
    // =============================================================================
    static_assert(
        (reduce_type != PoolType::MAX && reduce_type != PoolType::SUM) ||
            reduce_dim != ReduceDim::REDUCE_SCALAR || reduce_format != DataFormat::Int32,
        "Int32 MAX/SUM REDUCE_SCALAR is not supported (host decomposes Int32 HW reduce into W-then-H)");
    static_assert(
        reduce_type != PoolType::AVG || reduce_format != DataFormat::Int32,
        "Int32 AVG (mean) is not supported");
    static_assert(
        is_accumulation_type_v<AccumulateT>,
        "AccumulateT must be a valid accumulation type (NoAccumulation or Accumulate)");
    static_assert(
        is_post_reduce_op_v<PostReduceOp>,
        "PostReduceOp must be callable with a uint32_t argument");
    static_assert(
        !is_accumulate_v<AccumulateT> ||
            !(reduce_type == PoolType::MAX && reduce_dim == ReduceDim::REDUCE_SCALAR),
        "Accumulate with PoolType::MAX + REDUCE_SCALAR is not supported: the pack edge mask "
        "keeps only DST(0,0), but GMPOOL needs that running max broadcast across face-0 row 4 "
        "on the reload pass, which the current copy_tile reload cannot reproduce.");
#ifdef ARCH_QUASAR
    // The MAX + REDUCE_ROW accumulator reload relies on a within-16x16-face transpose during
    // copy_tile_to_dst_init_short (see reload_accumulator_if_needed). That transpose is rejected
    // by copy_tile_to_dst_init_short on Quasar ("Transpose within face not supported on Quasar"),
    // and there is no Quasar-compatible reload that restores the layout GMPOOL expects.
    static_assert(
        !is_accumulate_v<AccumulateT> ||
            !(reduce_type == PoolType::MAX && reduce_dim == ReduceDim::REDUCE_ROW),
        "Accumulate with PoolType::MAX + REDUCE_ROW is not supported on Quasar: the accumulator "
        "reload requires a within-16x16-face transpose, which copy_tile_to_dst_init_short asserts "
        "against on Quasar.");
#endif

    // =============================================================================
    // Runtime Assertions (parameter validation)
    // =============================================================================
    ASSERT(input_dfb_id != output_dfb_id);
    ASSERT(input_dfb_id != scaler_dfb_id);
    ASSERT(output_dfb_id != scaler_dfb_id);
    UNPACK(ASSERT(is_valid_dfb_tile_page_size(input_dfb_id, (DataFormat)unpack_src_format[input_dfb_id])));
    UNPACK(ASSERT(is_valid_dfb_tile_page_size(scaler_dfb_id, (DataFormat)unpack_src_format[scaler_dfb_id])));
    PACK(ASSERT(is_valid_dfb_tile_page_size(output_dfb_id, (DataFormat)pack_dst_format[output_dfb_id])));
    ASSERT(input_block_shape.rows > 0);
    ASSERT(input_block_shape.cols > 0);
    ASSERT(input_block_shape.batches > 0);
    if (input_memory_layout.row_stride != 0) {
        ASSERT(input_memory_layout.row_stride >= input_block_shape.cols);
    }

    // Compile-time flag: true when Accumulate type is passed, false otherwise
    constexpr bool enable_accumulation = is_accumulate_v<AccumulateT>;
    // Extract block shape components
    const uint32_t Ht = input_block_shape.rows;
    const uint32_t Wt = input_block_shape.cols;
    const uint32_t num_batches = input_block_shape.batches;

    constexpr bool is_sfpu = is_sfpu_reduce_path<reduce_type, reduce_dim, reduce_format, fp32_mode>();

    DataflowBuffer input_dfb(input_dfb_id);
    DataflowBuffer scaler_dfb(scaler_dfb_id);
    DataflowBuffer output_dfb(output_dfb_id);
    DataflowBuffer accum_dfb([&]() -> uint32_t {
        if constexpr (enable_accumulation) { return accumulate.config.cb_accumulator; }
        else { return 0; }
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
    scaler_dfb.wait_front(1);  // Wait for scaler tile
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
        const uint32_t total_output_tiles = num_batches;
        UNPACK((assert_input_dfb_size<input_policy>(input_dfb_id, tiles_per_bulk, total_input_tiles)));
        PACK((assert_output_dfb_size<input_policy>(output_dfb_id, total_output_tiles)));

        // No-pop modes: bulk reserve output upfront
        if constexpr (!should_pop(input_policy)) {
            output_dfb.reserve_back(total_output_tiles);
        }

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
            reload_accumulator_if_needed<reduce_type, reduce_dim, reduce_format, AccumulateT, is_sfpu>(
                accum_dfb, input_dfb_id, scaler_dfb_id, accumulate);

            const uint32_t dst_idx = get_dst_index(accumulate);
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
                        reduce_tile<reduce_type, reduce_dim>(
                            input_dfb_id, scaler_dfb_id, tile_idx, 0, dst_idx);
                    } else {  // PreloadedPolicy or PersistentPolicy: indexed access
                        uint32_t tile_idx = batch_offset + ht * stride + wt;
                        reduce_tile<reduce_type, reduce_dim>(
                            input_dfb_id, scaler_dfb_id, tile_idx, 0, dst_idx);
                    }
                }
            }

            // Call post-reduce operation on the single accumulated DST register.
            // No-op when PostReduceOp is the default NoOp.
            post_reduce_op(dst_idx);

            // Pop modes: reserve per-batch
            if constexpr (should_pop(input_policy)) {
                output_dfb.reserve_back(onetile);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(get_dst_index(accumulate), output_dfb_id);
            tile_regs_release();
            if constexpr (should_pop(input_policy)) {
                output_dfb.push_back(onetile);
            }

            // BulkWaitBulkPop: pop all tiles after processing
            if constexpr (waits_bulk(input_policy)) {
                input_dfb.pop_front(tiles_per_bulk);
            }

            // PreloadedPolicy or PersistentPolicy: update batch offset
            if constexpr (!should_pop(input_policy)) {
                batch_offset += tiles_per_bulk;
            }
        }

        // No-pop modes: bulk push output at end
        if constexpr (!should_pop(input_policy)) {
            output_dfb.push_back(total_output_tiles);
        }
    } else if constexpr (reduce_dim == ReduceDim::REDUCE_ROW) {
        // =================================================================
        // REDUCE_ROW: W reduction - each row -> 1 output tile (Ht outputs per batch)
        // =================================================================
        const uint32_t stride = (input_memory_layout.row_stride > 0) ? input_memory_layout.row_stride : Wt;
        const uint32_t total_output_tiles = Ht * num_batches;
        const uint32_t total_input_tiles = Ht * stride * num_batches;
        UNPACK((assert_input_dfb_size<input_policy>(input_dfb_id, Wt, total_input_tiles)));
        PACK((assert_output_dfb_size<input_policy>(output_dfb_id, total_output_tiles)));

        // No-pop modes: bulk reserve output upfront
        if constexpr (!should_pop(input_policy)) {
            output_dfb.reserve_back(total_output_tiles);
        }

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
                reload_accumulator_if_needed<reduce_type, reduce_dim, reduce_format, AccumulateT, is_sfpu>(
                    accum_dfb, input_dfb_id, scaler_dfb_id, accumulate);
                if constexpr (is_sfpu) {
                    if (Wt > 1) {
                        detail::sfpu_reduce_fold_init<reduce_type, reduce_format>();
                    }
                }

                const uint32_t dst_idx = get_dst_index(accumulate);
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
                    } else if constexpr (waits_per_tile(input_policy)) {
                        // One-at-a-time: wait/pop per tile
                        input_dfb.wait_front(onetile);
                        reduce_tile<reduce_type, reduce_dim>(input_dfb_id, scaler_dfb_id, 0, 0, dst_idx);
                        input_dfb.pop_front(onetile);
                    } else if constexpr (waits_bulk(input_policy)) {
                        // BulkWaitBulkPop: use indexed access
                        reduce_tile<reduce_type, reduce_dim>(
                            input_dfb_id, scaler_dfb_id, wt, 0, dst_idx);
                    } else {  // PreloadedPolicy or PersistentPolicy: indexed access
                        reduce_tile<reduce_type, reduce_dim>(
                            input_dfb_id, scaler_dfb_id, wt + index_offset, 0, dst_idx);
                    }
                }

                // SFPU intra-tile finalize
                if constexpr (is_sfpu) {
                    sfpu_reduce_init<reduce_type, reduce_format>();
                    sfpu_reduce<reduce_type, reduce_format, reduce_dim>(dst_idx, /*ct_dim=*/1, /*rt_dim=*/1);
                }

                // Call post-reduce operation (e.g., recip_tile for softmax)
                // User's lambda can include reduce_uninit() if needed before custom ops
                post_reduce_op(dst_idx);

                // Pop modes: reserve per-row to avoid deadlock
                if constexpr (should_pop(input_policy)) {
                    output_dfb.reserve_back(onetile);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(dst_idx, output_dfb_id);
                tile_regs_release();
                if constexpr (should_pop(input_policy)) {
                    output_dfb.push_back(onetile);
                }

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

        // No-pop modes: bulk push output at end
        if constexpr (!should_pop(input_policy)) {
            output_dfb.push_back(total_output_tiles);
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
        constexpr uint32_t chunk_size = is_sfpu ? (DEST_AUTO_LIMIT - 1) : DEST_AUTO_LIMIT;
        const uint32_t stride = (input_memory_layout.row_stride > 0) ? input_memory_layout.row_stride : Wt;
        const uint32_t tiles_per_bulk = Ht * stride;
        const uint32_t total_output_tiles = Wt * num_batches;
        const uint32_t total_input_tiles = tiles_per_bulk * num_batches;
        UNPACK((assert_input_dfb_size<input_policy>(input_dfb_id, Ht * chunk_size, total_input_tiles)));
        PACK((assert_output_dfb_size<input_policy>(output_dfb_id, total_output_tiles)));

        // No-pop modes: bulk reserve output upfront
        if constexpr (!should_pop(input_policy)) {
            output_dfb.reserve_back(total_output_tiles);
        }

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
                reload_accumulator_if_needed<reduce_type, reduce_dim, reduce_format, AccumulateT, is_sfpu>(
                    accum_dfb, input_dfb_id, scaler_dfb_id, accumulate);
                if constexpr (is_sfpu) {
                    if (Ht > 1) {
                        detail::sfpu_reduce_fold_init<reduce_type, reduce_format>();
                    }
                }

                for (uint32_t ht = 0; ht < Ht; ++ht) {
                    // Base dst_index: from accumulation config or 0 for multi-column output
                    uint32_t dst_idx = get_dst_index(accumulate);
                    for (uint32_t i = wt; i < chunk_end; ++i) {
                        if constexpr (is_sfpu) {
                            const bool is_first_tile = detail::sfpu_is_first_tile(ht, accumulate);
                            constexpr uint32_t sfpu_work_dst = chunk_size;
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
                                input_dfb_id, scaler_dfb_id, 0, 0, dst_idx);
                            input_dfb.pop_front(onetile);
                        } else if constexpr (waits_bulk(input_policy)) {
                            // BulkWaitBulkPop: use indexed access
                            uint32_t tile_idx = ht * current_chunk + (i - wt);
                            reduce_tile<reduce_type, reduce_dim>(
                                input_dfb_id, scaler_dfb_id, tile_idx, 0, dst_idx);
                        } else {  // PreloadedPolicy or PersistentPolicy: indexed access
                            uint32_t tile_idx = batch_offset + ht * stride + i;
                            reduce_tile<reduce_type, reduce_dim>(
                                input_dfb_id, scaler_dfb_id, tile_idx, 0, dst_idx);
                        }
                        ++dst_idx;
                    }
                }

                // SFPU intra-tile finalize per output slot
                if constexpr (is_sfpu) {
                    const uint32_t sfpu_base_dst = get_dst_index(accumulate);
                    sfpu_reduce_init<reduce_type, reduce_format>();
                    for (uint32_t k = 0; k < current_chunk; ++k) {
                        sfpu_reduce<reduce_type, reduce_format, reduce_dim>(
                            sfpu_base_dst + k, /*ct_dim=*/1, /*rt_dim=*/1);
                    }
                }

                // Post-reduce operation for each output tile in chunk
                const uint32_t base_dst = get_dst_index(accumulate);
                for (uint32_t i = 0; i < current_chunk; ++i) {
                    post_reduce_op(base_dst + i);
                }

                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < current_chunk; ++i) {
                    // Pop modes: reserve/push per output tile
                    if constexpr (should_pop(input_policy)) {
                        output_dfb.reserve_back(onetile);
                    }
                    pack_tile(base_dst + i, output_dfb_id);
                    if constexpr (should_pop(input_policy)) {
                        output_dfb.push_back(onetile);
                    }
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

        // No-pop modes: bulk push output at end
        if constexpr (!should_pop(input_policy)) {
            output_dfb.push_back(total_output_tiles);
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
