// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Implementation file for reduce_helpers_compute.hpp
// Do not include directly - include reduce_helpers_compute.hpp instead

namespace compute_kernel_lib {

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
constexpr bool waits_per_batch(ReduceInputPolicy p) { return p == ReduceInputPolicy::WaitAndPopPerBatch; }
constexpr bool waits_upfront(ReduceInputPolicy p) { return p == ReduceInputPolicy::WaitUpfrontNoPop; }
constexpr bool no_wait(ReduceInputPolicy p) { return p == ReduceInputPolicy::NoWaitNoPop; }
constexpr bool should_pop(ReduceInputPolicy p) {
    return p == ReduceInputPolicy::WaitAndPopPerTile || p == ReduceInputPolicy::WaitAndPopPerBatch;
}
constexpr bool manages_cb(ReduceInputPolicy p) {
    // Returns true if the reduce function manages CB wait/reserve/push (not preloaded)
    return p != ReduceInputPolicy::NoWaitNoPop;
}

// =============================================================================
// Helper Function Implementations
// =============================================================================

template <PoolType reduce_type, ReduceDim reduce_dim, bool enforce_fp32_accumulation>
ALWI void reduce_init_short_with_dt(uint32_t old_cbid, uint32_t input_cb, uint32_t scaler_cb) {
    // Reconfigure SRCA data format from old_cbid to input_cb (similar to copy_tile_to_dst_init_short_with_dt)
    UNPACK((llk_unpack_reconfig_data_format_srca<DST_ACCUM_MODE>(old_cbid, input_cb)));
    MATH((llk_math_reconfig_data_format_srca<DST_ACCUM_MODE>(old_cbid, input_cb)));

    // Reconfigure unpacker for reduce operation (SRCA and SRCB)
    UNPACK((llk_unpack_AB_reduce_init<reduce_dim, BroadcastType::NONE, enforce_fp32_accumulation>(input_cb, scaler_cb)));

    // Reconfigure math for reduce operation
    MATH((llk_math_reduce_init<reduce_type, reduce_dim, DST_ACCUM_MODE, MATH_FIDELITY, enforce_fp32_accumulation>()));

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

template <PoolType reduce_type, ReduceDim reduce_dim, typename AccumulateT, bool enforce_fp32_accumulation>
ALWI void reload_accumulator_if_needed(uint32_t input_cb, uint32_t scaler_cb, const AccumulateT& accumulate) {
    if constexpr (is_accumulate_v<AccumulateT>) {
        if (!accumulate.is_first()) {  // Reload on all iterations except first
            constexpr uint32_t onetile = 1;
            cb_wait_front(accumulate.config.cb_accumulator, onetile);
            copy_tile_to_dst_init_short_with_dt(input_cb, accumulate.config.cb_accumulator);
            copy_tile(accumulate.config.cb_accumulator, 0, accumulate.config.dst_index);
            cb_pop_front(accumulate.config.cb_accumulator, onetile);

            // CRITICAL: Re-init reduce after copy_tile corrupts SRCA config
            // Use short version since packer config is still valid from initial reduce_init
            // Pass accumulator CB as old_cbid to reconfigure data format from accumulator to input CB
            reduce_init_short_with_dt<reduce_type, reduce_dim, enforce_fp32_accumulation>(
                accumulate.config.cb_accumulator, input_cb, scaler_cb);
        }
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
    static_assert(reduce_type != PoolType::MIN, "PoolType::MIN is not supported for reduce operations");
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

    // Apply reconfig based on mode
    if constexpr (reconfig_input(reconfig_mode)) {
        reconfig_data_format(input_cb, scaler_cb);
    }
    if constexpr (reconfig_output(reconfig_mode)) {
        pack_reconfig_data_format(output_cb);
    }

    // Auto-detect FP32 dest accumulation mode from compile-time define
    constexpr bool enforce_fp32_accumulation = get_fp32_dest_acc_enabled();

    // Initialization
    reduce_init<reduce_type, reduce_dim, enforce_fp32_accumulation>(input_cb, scaler_cb, output_cb);
    cb_wait_front(scaler_cb, 1);  // Wait for scaler tile

    constexpr uint32_t onetile = 1;

    // Pattern dispatch based on reduce_dim
    if constexpr (reduce_dim == ReduceDim::REDUCE_SCALAR) {
        // =================================================================
        // REDUCE_SCALAR: HW reduction - all tiles -> 1 output tile per batch
        // =================================================================
        const uint32_t stride = (input_memory_layout.row_stride > 0) ? input_memory_layout.row_stride : Wt;
        const uint32_t tiles_per_batch = Ht * stride;
        const uint32_t total_tiles = tiles_per_batch * num_batches;

        // PreloadedPolicy: bulk reserve output upfront
        if constexpr (no_wait(input_policy)) {
            cb_reserve_back(output_cb, num_batches);
        }

        // PersistentPolicy: wait for all tiles upfront
        if constexpr (waits_upfront(input_policy)) {
            cb_wait_front(input_cb, total_tiles);
        }

        uint32_t batch_offset = 0;
        for (uint32_t nc = 0; nc < num_batches; ++nc) {
            // StreamingBatchedPolicy: wait for all tiles per batch
            if constexpr (waits_per_batch(input_policy)) {
                cb_wait_front(input_cb, tiles_per_batch);
            }

            tile_regs_acquire();

            // Reload accumulator if needed (zero overhead when AccumulateT is NoAccumulation)
            reload_accumulator_if_needed<reduce_type, reduce_dim, AccumulateT, enforce_fp32_accumulation>(
                input_cb, scaler_cb, accumulate);

            const uint32_t dst_idx = get_dst_index(accumulate);
            for (uint32_t ht = 0; ht < Ht; ++ht) {
                for (uint32_t wt = 0; wt < Wt; ++wt) {
                    if constexpr (waits_per_tile(input_policy)) {
                        // One-at-a-time: wait/pop per tile
                        cb_wait_front(input_cb, onetile);
                        reduce_tile<reduce_type, reduce_dim, enforce_fp32_accumulation>(input_cb, scaler_cb, 0, 0, dst_idx);
                        cb_pop_front(input_cb, onetile);
                    } else if constexpr (waits_per_batch(input_policy)) {
                        // Batched: use indexed access
                        uint32_t tile_idx = ht * stride + wt;
                        reduce_tile<reduce_type, reduce_dim, enforce_fp32_accumulation>(
                            input_cb, scaler_cb, tile_idx, 0, dst_idx);
                    } else {  // PreloadedPolicy or PersistentPolicy: indexed access
                        uint32_t tile_idx = batch_offset + ht * stride + wt;
                        reduce_tile<reduce_type, reduce_dim, enforce_fp32_accumulation>(
                            input_cb, scaler_cb, tile_idx, 0, dst_idx);
                    }
                }
            }
            // Not PreloadedPolicy: reserve per-batch
            if constexpr (manages_cb(input_policy)) {
                cb_reserve_back(output_cb, onetile);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(get_dst_index(accumulate), output_cb);
            tile_regs_release();
            if constexpr (manages_cb(input_policy)) {
                cb_push_back(output_cb, onetile);
            }

            // StreamingBatchedPolicy: pop all tiles after processing
            if constexpr (waits_per_batch(input_policy)) {
                cb_pop_front(input_cb, tiles_per_batch);
            }

            // PreloadedPolicy or PersistentPolicy: update batch offset
            if constexpr (!should_pop(input_policy)) {
                batch_offset += tiles_per_batch;
            }
        }

        // PreloadedPolicy: bulk push output at end
        if constexpr (no_wait(input_policy)) {
            cb_push_back(output_cb, num_batches);
        }
    } else if constexpr (reduce_dim == ReduceDim::REDUCE_ROW) {
        // =================================================================
        // REDUCE_ROW: W reduction - each row -> 1 output tile (Ht outputs per batch)
        // =================================================================
        const uint32_t stride = (input_memory_layout.row_stride > 0) ? input_memory_layout.row_stride : Wt;
        const uint32_t total_outputs = Ht * num_batches;
        const uint32_t total_tiles = Ht * stride * num_batches;

        // PreloadedPolicy: bulk reserve output upfront
        if constexpr (no_wait(input_policy)) {
            cb_reserve_back(output_cb, total_outputs);
        }

        // PersistentPolicy: wait for all tiles upfront
        if constexpr (waits_upfront(input_policy)) {
            cb_wait_front(input_cb, total_tiles);
        }

        uint32_t index_offset = 0;
        for (uint32_t nc = 0; nc < num_batches; ++nc) {
            for (uint32_t ht = 0; ht < Ht; ++ht) {
                // StreamingBatchedPolicy: wait for entire row upfront
                if constexpr (waits_per_batch(input_policy)) {
                    cb_wait_front(input_cb, Wt);
                }

                tile_regs_acquire();

                // Reload accumulator if needed (zero overhead when AccumulateT is NoAccumulation)
                reload_accumulator_if_needed<reduce_type, reduce_dim, AccumulateT, enforce_fp32_accumulation>(
                    input_cb, scaler_cb, accumulate);

                const uint32_t dst_idx = get_dst_index(accumulate);
                for (uint32_t wt = 0; wt < Wt; ++wt) {
                    if constexpr (waits_per_tile(input_policy)) {
                        // One-at-a-time: wait/pop per tile
                        cb_wait_front(input_cb, onetile);
                        reduce_tile<reduce_type, reduce_dim, enforce_fp32_accumulation>(input_cb, scaler_cb, 0, 0, dst_idx);
                        cb_pop_front(input_cb, onetile);
                    } else if constexpr (waits_per_batch(input_policy)) {
                        // Batched: use indexed access
                        reduce_tile<reduce_type, reduce_dim, enforce_fp32_accumulation>(
                            input_cb, scaler_cb, wt, 0, dst_idx);
                    } else {  // PreloadedPolicy or PersistentPolicy: indexed access
                        reduce_tile<reduce_type, reduce_dim, enforce_fp32_accumulation>(
                            input_cb, scaler_cb, wt + index_offset, 0, dst_idx);
                    }
                }

                // Call post-reduce operation (e.g., recip_tile for softmax)
                // User's lambda can include reduce_uninit() if needed before custom ops
                post_reduce_op(dst_idx);

                // Not PreloadedPolicy: reserve per-row to avoid deadlock
                if constexpr (manages_cb(input_policy)) {
                    cb_reserve_back(output_cb, onetile);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(dst_idx, output_cb);
                tile_regs_release();
                if constexpr (manages_cb(input_policy)) {
                    cb_push_back(output_cb, onetile);
                }

                // StreamingBatchedPolicy: pop all tiles after processing
                if constexpr (waits_per_batch(input_policy)) {
                    cb_pop_front(input_cb, Wt);
                }

                // PreloadedPolicy or PersistentPolicy: update index offset
                if constexpr (!should_pop(input_policy)) {
                    index_offset += stride;
                }
            }
        }

        // PreloadedPolicy: bulk push output at end
        if constexpr (no_wait(input_policy)) {
            cb_push_back(output_cb, total_outputs);
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
        const uint32_t tiles_per_batch = Ht * stride;
        const uint32_t total_outputs = Wt * num_batches;
        const uint32_t total_tiles = tiles_per_batch * num_batches;

        // PreloadedPolicy: bulk reserve output upfront
        if constexpr (no_wait(input_policy)) {
            cb_reserve_back(output_cb, total_outputs);
        }

        // PersistentPolicy: wait for all tiles upfront
        if constexpr (waits_upfront(input_policy)) {
            cb_wait_front(input_cb, total_tiles);
        }

        uint32_t batch_offset = 0;
        for (uint32_t nc = 0; nc < num_batches; ++nc) {
            for (uint32_t wt = 0; wt < Wt; wt += chunk_size) {
                uint32_t chunk_end = (wt + chunk_size < Wt) ? (wt + chunk_size) : Wt;
                uint32_t current_chunk = chunk_end - wt;
                uint32_t tiles_in_chunk = Ht * current_chunk;

                // StreamingBatchedPolicy: wait for entire chunk upfront
                if constexpr (waits_per_batch(input_policy)) {
                    cb_wait_front(input_cb, tiles_in_chunk);
                }

                tile_regs_acquire();

                // Reload accumulator if needed (zero overhead when AccumulateT is NoAccumulation)
                reload_accumulator_if_needed<reduce_type, reduce_dim, AccumulateT, enforce_fp32_accumulation>(
                    input_cb, scaler_cb, accumulate);

                for (uint32_t ht = 0; ht < Ht; ++ht) {
                    // Base dst_index: from accumulation config or 0 for multi-column output
                    uint32_t dst_idx = get_dst_index(accumulate);
                    for (uint32_t i = wt; i < chunk_end; ++i) {
                        if constexpr (waits_per_tile(input_policy)) {
                            // One-at-a-time: wait/pop per tile
                            cb_wait_front(input_cb, onetile);
                            reduce_tile<reduce_type, reduce_dim, enforce_fp32_accumulation>(
                                input_cb, scaler_cb, 0, 0, dst_idx);
                            cb_pop_front(input_cb, onetile);
                        } else if constexpr (waits_per_batch(input_policy)) {
                            // Batched: use indexed access
                            uint32_t tile_idx = ht * current_chunk + (i - wt);
                            reduce_tile<reduce_type, reduce_dim, enforce_fp32_accumulation>(
                                input_cb, scaler_cb, tile_idx, 0, dst_idx);
                        } else {  // PreloadedPolicy or PersistentPolicy: indexed access
                            uint32_t tile_idx = batch_offset + ht * stride + i;
                            reduce_tile<reduce_type, reduce_dim, enforce_fp32_accumulation>(
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
                    // Not PreloadedPolicy: reserve/push per output tile
                    if constexpr (manages_cb(input_policy)) {
                        cb_reserve_back(output_cb, onetile);
                    }
                    pack_tile(base_dst + i, output_cb);
                    if constexpr (manages_cb(input_policy)) {
                        cb_push_back(output_cb, onetile);
                    }
                }
                tile_regs_release();

                // StreamingBatchedPolicy: pop all tiles after processing
                if constexpr (waits_per_batch(input_policy)) {
                    cb_pop_front(input_cb, tiles_in_chunk);
                }
            }
            // Update batch_offset for indexed modes (PreloadedPolicy and PersistentPolicy)
            if constexpr (!should_pop(input_policy)) {
                batch_offset += tiles_per_batch;
            }
        }

        // PreloadedPolicy: bulk push output at end
        if constexpr (no_wait(input_policy)) {
            cb_push_back(output_cb, total_outputs);
        }
    }

    // Cleanup
    reduce_uninit<enforce_fp32_accumulation>();
}

}  // namespace compute_kernel_lib
