// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Implementation file for reduce_helpers_compute.hpp
// Do not include directly - include reduce_helpers_compute.hpp instead

#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/debug/assert.h"
#include "experimental/circular_buffer.h"
#include "tt-metalium/circular_buffer_constants.h"
#include "ttnn/cpp/ttnn/kernel_lib/cb_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_common.hpp"


namespace compute_kernel_lib {

// HiFi4 fidelity for matmul-based reduce (higher precision than kernel default)
constexpr ckernel::MathFidelity REDUCE_MATMUL_FIDELITY = ckernel::MathFidelity::HiFi4;

// Matmul wrappers that use REDUCE_MATMUL_FIDELITY instead of MATH_FIDELITY
ALWI void reduce_mm_init_short(uint32_t in0_cb_id, uint32_t in1_cb_id) {
    state_configure(in1_cb_id, in0_cb_id);
    MATH((llk_math_matmul_init<REDUCE_MATMUL_FIDELITY, MM_THROTTLE>(in0_cb_id, in1_cb_id, 0)));
    UNPACK((llk_unpack_AB_matmul_init(in0_cb_id, in1_cb_id, 0)));
}

ALWI void reduce_mm_init_short_with_dt(uint32_t in0_cb_id, uint32_t in1_cb_id, uint32_t c_in_old_srca) {
    UNPACK((llk_unpack_reconfig_data_format_srca<DST_ACCUM_MODE>(c_in_old_srca, in1_cb_id)));
    MATH((llk_math_reconfig_data_format_srca<DST_ACCUM_MODE>(c_in_old_srca, in1_cb_id)));
    reduce_mm_init_short(in0_cb_id, in1_cb_id);
}

ALWI void reduce_matmul_tiles(
    uint32_t in0_cb_id, uint32_t in1_cb_id, uint32_t in0_tile_index, uint32_t in1_tile_index, uint32_t idst) {
    UNPACK((llk_unpack_AB_matmul(in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index)));
    MATH((llk_math_matmul<REDUCE_MATMUL_FIDELITY, MM_THROTTLE>(idst)));
}

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
ALWI void reduce_init_short_with_dt(uint32_t old_cbid, uint32_t input_cb, uint32_t scaler_cb) {
    // Reconfigure SRCA data format from old_cbid to input_cb (similar to copy_tile_to_dst_init_short_with_dt)
    UNPACK((llk_unpack_reconfig_data_format_srca<DST_ACCUM_MODE>(old_cbid, input_cb)));
    MATH((llk_math_reconfig_data_format_srca<DST_ACCUM_MODE>(old_cbid, input_cb)));

    // Reconfigure unpacker for reduce operation (SRCA and SRCB)
    UNPACK((llk_unpack_AB_reduce_init<reduce_type, reduce_dim>(input_cb, scaler_cb)));

    // Reconfigure math for reduce operation
    MATH((llk_math_reduce_init<reduce_type, reduce_dim, DST_ACCUM_MODE, MATH_FIDELITY>()));

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
    typename AccumulateT,
    bool use_matmul = false>
ALWI void reload_accumulator_if_needed(
    experimental::CircularBuffer& cb_accum, uint32_t input_cb, uint32_t scaler_cb, const AccumulateT& accumulate) {
    if constexpr (is_accumulate_v<AccumulateT>) {
        if (!accumulate.is_first()) {  // Reload on all iterations except first
            constexpr uint32_t onetile = 1;
            cb_accum.wait_front(onetile);
            copy_tile_to_dst_init_short_with_dt(input_cb, accumulate.config.cb_accumulator);
            copy_tile(accumulate.config.cb_accumulator, 0, accumulate.config.dst_index);
            cb_accum.pop_front(onetile);

            // CRITICAL: Re-init after copy_tile corrupts SRCA config
            // Use short version since packer config is still valid from initial init
            // Pass accumulator CB as old_cbid to reconfigure data format from accumulator to input CB
            if constexpr (use_matmul) {
                reduce_mm_init_short_with_dt(input_cb, scaler_cb, accumulate.config.cb_accumulator);
            } else {
                reduce_init_short_with_dt<reduce_type, reduce_dim>(
                    accumulate.config.cb_accumulator, input_cb, scaler_cb);
            }
        }
    }
}

template <ReduceInputPolicy input_policy>
ALWI void assert_input_cb_size(uint32_t input_cb, uint32_t tiles_per_bulk, uint32_t total_tiles) {
    if constexpr (waits_per_tile(input_policy)) {
        ASSERT(get_cb_num_pages(input_cb) >= 1);
    } else if constexpr (waits_bulk(input_policy)) {
        ASSERT(get_cb_num_pages(input_cb) >= tiles_per_bulk);
        ASSERT(get_cb_num_pages(input_cb) % tiles_per_bulk == 0);
    } else {  // waits_upfront or no_wait
        ASSERT(get_cb_num_pages(input_cb) >= total_tiles);
    }
}

template <ReduceInputPolicy input_policy>
ALWI void assert_output_cb_size(uint32_t output_cb, uint32_t total_outputs) {
    if constexpr (should_pop(input_policy)) {
        // Per-tile reserve/push: only needs 1 page
        ASSERT(get_cb_num_pages(output_cb) >= 1);
    } else {
        // Bulk reserve upfront: needs all outputs
        ASSERT(get_cb_num_pages(output_cb) >= total_outputs);
    }
}

// =============================================================================
// Main Reduce Function Implementation
// =============================================================================

template <
    PoolType reduce_type,
    ReduceDim reduce_dim,
    ReduceInputPolicy input_policy,
    ReduceDataFormatReconfigMode reconfig_mode,
    typename AccumulateT,
    typename PostReduceOp>
ALWI void reduce(
    uint32_t input_cb,
    uint32_t scaler_cb,
    uint32_t output_cb,
    ReduceInputBlockShape input_block_shape,
    ReduceInputMemoryLayout input_memory_layout,
    AccumulateT accumulate,
    PostReduceOp post_reduce_op) {
    // =============================================================================
    // Static Assertions (compile-time validation)
    // =============================================================================
    static_assert(
        is_accumulation_type_v<AccumulateT>,
        "AccumulateT must be a valid accumulation type (NoAccumulation or Accumulate)");
    static_assert(
        is_post_reduce_op_v<PostReduceOp>,
        "PostReduceOp must be callable with a uint32_t argument");

    // =============================================================================
    // Runtime Assertions (parameter validation)
    // =============================================================================
    ASSERT(input_cb < NUM_CIRCULAR_BUFFERS);
    ASSERT(scaler_cb < NUM_CIRCULAR_BUFFERS);
    ASSERT(output_cb < NUM_CIRCULAR_BUFFERS);
    ASSERT(input_cb != output_cb);
    ASSERT(input_cb != scaler_cb);
    ASSERT(output_cb != scaler_cb);
    UNPACK(ASSERT(is_valid_cb_tile_page_size(input_cb, (DataFormat)unpack_src_format[input_cb])));
    UNPACK(ASSERT(is_valid_cb_tile_page_size(scaler_cb, (DataFormat)unpack_src_format[scaler_cb])));
    PACK(ASSERT(is_valid_cb_tile_page_size(output_cb, (DataFormat)pack_dst_format[output_cb])));
    ASSERT(input_block_shape.rows > 0);
    ASSERT(input_block_shape.cols > 0);
    ASSERT(input_block_shape.batches > 0);
    if (input_memory_layout.row_stride != 0) {
        ASSERT(input_memory_layout.row_stride >= input_block_shape.cols);
    }
    if constexpr (is_accumulate_v<AccumulateT>) {
        ASSERT(accumulate.config.cb_accumulator < NUM_CIRCULAR_BUFFERS);
    }

    // Compile-time flag: true when Accumulate type is passed, false otherwise
    constexpr bool enable_accumulation = is_accumulate_v<AccumulateT>;
    // Extract block shape components
    const uint32_t Ht = input_block_shape.rows;
    const uint32_t Wt = input_block_shape.cols;
    const uint32_t num_batches = input_block_shape.batches;

    constexpr bool use_matmul = reduce_uses_matmul<reduce_type, reduce_dim>();

    experimental::CircularBuffer cb_input(input_cb);
    experimental::CircularBuffer cb_scaler(scaler_cb);
    experimental::CircularBuffer cb_output(output_cb);
    experimental::CircularBuffer cb_accum([&]() -> uint32_t {
        if constexpr (enable_accumulation) { return accumulate.config.cb_accumulator; }
        else { return 0; }
    }());

    // Apply reconfig based on mode
    if constexpr (reconfig_input(reconfig_mode)) {
        if constexpr (use_matmul) {
            reconfig_data_format(scaler_cb, input_cb);
        } else {
            reconfig_data_format(input_cb, scaler_cb);
        }
    }
    if constexpr (reconfig_output(reconfig_mode)) {
        pack_reconfig_data_format(output_cb);
    }
    // Initialization
    if constexpr (use_matmul) {
        reduce_mm_init_short(input_cb, scaler_cb);
    } else {
        reduce_init<reduce_type, reduce_dim>(input_cb, scaler_cb, output_cb);
    }
    cb_scaler.wait_front(1);  // Wait for scaler tile

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
        UNPACK((assert_input_cb_size<input_policy>(input_cb, tiles_per_bulk, total_input_tiles)));
        PACK((assert_output_cb_size<input_policy>(output_cb, total_output_tiles)));

        // No-pop modes: bulk reserve output upfront
        if constexpr (!should_pop(input_policy)) {
            cb_output.reserve_back(total_output_tiles);
        }

        // PersistentPolicy: wait for all tiles upfront
        if constexpr (waits_upfront(input_policy)) {
            cb_input.wait_front(total_input_tiles);
        }

        uint32_t batch_offset = 0;
        for (uint32_t nc = 0; nc < num_batches; ++nc) {
            // BulkWaitBulkPop: wait for all Ht×Wt tiles in bulk
            if constexpr (waits_bulk(input_policy)) {
                cb_input.wait_front(tiles_per_bulk);
            }

            tile_regs_acquire();

            // Reload accumulator if needed (zero overhead when AccumulateT is NoAccumulation)
            reload_accumulator_if_needed<reduce_type, reduce_dim, AccumulateT>(
                cb_accum, input_cb, scaler_cb, accumulate);

            const uint32_t dst_idx = get_dst_index(accumulate);
            for (uint32_t ht = 0; ht < Ht; ++ht) {
                for (uint32_t wt = 0; wt < Wt; ++wt) {
                    if constexpr (waits_per_tile(input_policy)) {
                        // One-at-a-time: wait/pop per tile
                        cb_input.wait_front(onetile);
                        reduce_tile<reduce_type, reduce_dim>(input_cb, scaler_cb, 0, 0, dst_idx);
                        cb_input.pop_front(onetile);
                    } else if constexpr (waits_bulk(input_policy)) {
                        // BulkWaitBulkPop: use indexed access
                        uint32_t tile_idx = ht * stride + wt;
                        reduce_tile<reduce_type, reduce_dim>(
                            input_cb, scaler_cb, tile_idx, 0, dst_idx);
                    } else {  // PreloadedPolicy or PersistentPolicy: indexed access
                        uint32_t tile_idx = batch_offset + ht * stride + wt;
                        reduce_tile<reduce_type, reduce_dim>(
                            input_cb, scaler_cb, tile_idx, 0, dst_idx);
                    }
                }
            }
            // Pop modes: reserve per-batch
            if constexpr (should_pop(input_policy)) {
                cb_output.reserve_back(onetile);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(get_dst_index(accumulate), output_cb);
            tile_regs_release();
            if constexpr (should_pop(input_policy)) {
                cb_output.push_back(onetile);
            }

            // BulkWaitBulkPop: pop all tiles after processing
            if constexpr (waits_bulk(input_policy)) {
                cb_input.pop_front(tiles_per_bulk);
            }

            // PreloadedPolicy or PersistentPolicy: update batch offset
            if constexpr (!should_pop(input_policy)) {
                batch_offset += tiles_per_bulk;
            }
        }

        // No-pop modes: bulk push output at end
        if constexpr (!should_pop(input_policy)) {
            cb_output.push_back(total_output_tiles);
        }
    } else if constexpr (reduce_dim == ReduceDim::REDUCE_ROW) {
        // =================================================================
        // REDUCE_ROW: W reduction - each row -> 1 output tile (Ht outputs per batch)
        // =================================================================
        const uint32_t stride = (input_memory_layout.row_stride > 0) ? input_memory_layout.row_stride : Wt;
        const uint32_t total_output_tiles = Ht * num_batches;
        const uint32_t total_input_tiles = Ht * stride * num_batches;
        UNPACK((assert_input_cb_size<input_policy>(input_cb, Wt, total_input_tiles)));
        PACK((assert_output_cb_size<input_policy>(output_cb, total_output_tiles)));

        // No-pop modes: bulk reserve output upfront
        if constexpr (!should_pop(input_policy)) {
            cb_output.reserve_back(total_output_tiles);
        }

        // PersistentPolicy: wait for all tiles upfront
        if constexpr (waits_upfront(input_policy)) {
            cb_input.wait_front(total_input_tiles);
        }

        uint32_t index_offset = 0;
        for (uint32_t nc = 0; nc < num_batches; ++nc) {
            for (uint32_t ht = 0; ht < Ht; ++ht) {
                // BulkWaitBulkPop: wait for entire row upfront
                if constexpr (waits_bulk(input_policy)) {
                    cb_input.wait_front(Wt);
                }

                tile_regs_acquire();

                // Reload accumulator if needed (zero overhead when AccumulateT is NoAccumulation)
                reload_accumulator_if_needed<reduce_type, reduce_dim, AccumulateT, use_matmul>(
                    cb_accum, input_cb, scaler_cb, accumulate);

                const uint32_t dst_idx = get_dst_index(accumulate);
                for (uint32_t wt = 0; wt < Wt; ++wt) {
                    if constexpr (waits_per_tile(input_policy)) {
                        // One-at-a-time: wait/pop per tile
                        cb_input.wait_front(onetile);
                        if constexpr (use_matmul) {
                            reduce_matmul_tiles(input_cb, scaler_cb, 0, 0, dst_idx);
                        } else {
                            reduce_tile<reduce_type, reduce_dim>(input_cb, scaler_cb, 0, 0, dst_idx);
                        }
                        cb_input.pop_front(onetile);
                    } else if constexpr (waits_bulk(input_policy)) {
                        // BulkWaitBulkPop: use indexed access
                        if constexpr (use_matmul) {
                            reduce_matmul_tiles(input_cb, scaler_cb, wt, 0, dst_idx);
                        } else {
                            reduce_tile<reduce_type, reduce_dim>(
                                input_cb, scaler_cb, wt, 0, dst_idx);
                        }
                    } else {  // PreloadedPolicy or PersistentPolicy: indexed access
                        if constexpr (use_matmul) {
                            reduce_matmul_tiles(input_cb, scaler_cb, wt + index_offset, 0, dst_idx);
                        } else {
                            reduce_tile<reduce_type, reduce_dim>(
                                input_cb, scaler_cb, wt + index_offset, 0, dst_idx);
                        }
                    }
                }

                // Call post-reduce operation (e.g., recip_tile for softmax)
                // User's lambda can include reduce_uninit() if needed before custom ops
                post_reduce_op(dst_idx);

                // Pop modes: reserve per-row to avoid deadlock
                if constexpr (should_pop(input_policy)) {
                    cb_output.reserve_back(onetile);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(dst_idx, output_cb);
                tile_regs_release();
                if constexpr (should_pop(input_policy)) {
                    cb_output.push_back(onetile);
                }

                // BulkWaitBulkPop: pop all tiles after processing
                if constexpr (waits_bulk(input_policy)) {
                    cb_input.pop_front(Wt);
                }

                // PreloadedPolicy or PersistentPolicy: update index offset
                if constexpr (!should_pop(input_policy)) {
                    index_offset += stride;
                }
            }
        }

        // No-pop modes: bulk push output at end
        if constexpr (!should_pop(input_policy)) {
            cb_output.push_back(total_output_tiles);
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
        constexpr uint32_t chunk_size = DEST_AUTO_LIMIT;
        const uint32_t stride = (input_memory_layout.row_stride > 0) ? input_memory_layout.row_stride : Wt;
        const uint32_t tiles_per_bulk = Ht * stride;
        const uint32_t total_output_tiles = Wt * num_batches;
        const uint32_t total_input_tiles = tiles_per_bulk * num_batches;
        UNPACK((assert_input_cb_size<input_policy>(input_cb, Ht * chunk_size, total_input_tiles)));
        PACK((assert_output_cb_size<input_policy>(output_cb, total_output_tiles)));

        // No-pop modes: bulk reserve output upfront
        if constexpr (!should_pop(input_policy)) {
            cb_output.reserve_back(total_output_tiles);
        }

        // PersistentPolicy: wait for all tiles upfront
        if constexpr (waits_upfront(input_policy)) {
            cb_input.wait_front(total_input_tiles);
        }

        uint32_t batch_offset = 0;
        for (uint32_t nc = 0; nc < num_batches; ++nc) {
            for (uint32_t wt = 0; wt < Wt; wt += chunk_size) {
                uint32_t chunk_end = (wt + chunk_size < Wt) ? (wt + chunk_size) : Wt;
                uint32_t current_chunk = chunk_end - wt;
                uint32_t tiles_in_chunk = Ht * current_chunk;

                // BulkWaitBulkPop: wait for entire chunk upfront
                if constexpr (waits_bulk(input_policy)) {
                    cb_input.wait_front(tiles_in_chunk);
                }

                tile_regs_acquire();

                // Reload accumulator if needed (zero overhead when AccumulateT is NoAccumulation)
                reload_accumulator_if_needed<reduce_type, reduce_dim, AccumulateT>(
                    cb_accum, input_cb, scaler_cb, accumulate);

                for (uint32_t ht = 0; ht < Ht; ++ht) {
                    // Base dst_index: from accumulation config or 0 for multi-column output
                    uint32_t dst_idx = get_dst_index(accumulate);
                    for (uint32_t i = wt; i < chunk_end; ++i) {
                        if constexpr (waits_per_tile(input_policy)) {
                            // One-at-a-time: wait/pop per tile
                            cb_input.wait_front(onetile);
                            reduce_tile<reduce_type, reduce_dim>(
                                input_cb, scaler_cb, 0, 0, dst_idx);
                            cb_input.pop_front(onetile);
                        } else if constexpr (waits_bulk(input_policy)) {
                            // BulkWaitBulkPop: use indexed access
                            uint32_t tile_idx = ht * current_chunk + (i - wt);
                            reduce_tile<reduce_type, reduce_dim>(
                                input_cb, scaler_cb, tile_idx, 0, dst_idx);
                        } else {  // PreloadedPolicy or PersistentPolicy: indexed access
                            uint32_t tile_idx = batch_offset + ht * stride + i;
                            reduce_tile<reduce_type, reduce_dim>(
                                input_cb, scaler_cb, tile_idx, 0, dst_idx);
                        }
                        ++dst_idx;
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
                        cb_output.reserve_back(onetile);
                    }
                    pack_tile(base_dst + i, output_cb);
                    if constexpr (should_pop(input_policy)) {
                        cb_output.push_back(onetile);
                    }
                }
                tile_regs_release();

                // BulkWaitBulkPop: pop all tiles after processing
                if constexpr (waits_bulk(input_policy)) {
                    cb_input.pop_front(tiles_in_chunk);
                }
            }
            // Update batch_offset for indexed modes (PreloadedPolicy and PersistentPolicy)
            if constexpr (!should_pop(input_policy)) {
                batch_offset += tiles_per_bulk;
            }
        }

        // No-pop modes: bulk push output at end
        if constexpr (!should_pop(input_policy)) {
            cb_output.push_back(total_output_tiles);
        }
    }

    // Cleanup
    if constexpr (!use_matmul) {
        reduce_uninit<>();
    }
}

}  // namespace compute_kernel_lib
