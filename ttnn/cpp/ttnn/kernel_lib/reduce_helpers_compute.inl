// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Implementation file for reduce_helpers_compute.hpp
// Do not include directly - include reduce_helpers_compute.hpp instead

#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/debug/assert.h"
#include "experimental/circular_buffer.h"
#include "experimental/dataflow_buffer.h"
#include "tt-metalium/circular_buffer_constants.h"
#include "ttnn/cpp/ttnn/kernel_lib/buffer_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/cb_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_common.hpp"


namespace compute_kernel_lib {

// HiFi4 fidelity for matmul-based reduce (higher precision than kernel default)
constexpr ckernel::MathFidelity REDUCE_MATMUL_FIDELITY = ckernel::MathFidelity::HiFi4;

// Matmul wrappers that use REDUCE_MATMUL_FIDELITY instead of MATH_FIDELITY
ALWI void reduce_with_matmul_init(uint32_t in0_cb_id, uint32_t in1_cb_id) {
    state_configure(in1_cb_id, in0_cb_id);
    MATH((llk_math_matmul_init<REDUCE_MATMUL_FIDELITY, MM_THROTTLE>(in0_cb_id, in1_cb_id, 0)));
    UNPACK((llk_unpack_AB_matmul_init(in0_cb_id, in1_cb_id, 0)));
}

// `_with_dt` variants reconfigure the unpacker / math SRCA data format when the
// previous op (the accumulator-reload `copy_tile`) ran with a different format.
// They rely on `llk_unpack_reconfig_data_format_srca` / `llk_math_reconfig_data_format_srca`
// LLKs, which are only present on Gen1 (WH/BH). Quasar manages data-format
// reconfiguration internally via the DFB metadata and does not expose these LLKs,
// so the body is empty there. This function is only reachable when the caller
// passes `Accumulate{...}` to `compute_kernel_lib::reduce`; passing
// `NoAccumulation{}` (the default) skips the call entirely. Accumulation-reload
// support on Quasar is a follow-up.
ALWI void reduce_with_matmul_init_with_dt(uint32_t in0_cb_id, uint32_t in1_cb_id, uint32_t c_in_old_srca) {
    UNPACK((llk_unpack_reconfig_data_format_srca<DST_ACCUM_MODE, p_dim_stride_target::IGNORE>(c_in_old_srca, in1_cb_id)));
    MATH((llk_math_reconfig_data_format_srca<DST_ACCUM_MODE>(c_in_old_srca, in1_cb_id)));
    reduce_with_matmul_init(in0_cb_id, in1_cb_id);
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

// See note above on `_with_dt` and Gen1-only LLKs. `-Wtemplate-body` in modern GCC
// checks non-dependent names in template bodies eagerly, so we have to guard the
// body even though this template is only instantiated from the accumulation
// reload path (which our reduce-W kernel doesn't take).
template <PoolType reduce_type, ReduceDim reduce_dim>
ALWI void reduce_init_short_with_dt(uint32_t old_cb_id, uint32_t input_cb_id, uint32_t scaler_cb_id) {
    // Reconfigure SRCA data format from old_cb_id to input_cb_id (similar to copy_tile_to_dst_init_short_with_dt)
    UNPACK((llk_unpack_reconfig_data_format_srca<DST_ACCUM_MODE, p_dim_stride_target::IGNORE>(old_cb_id, input_cb_id)));
    MATH((llk_math_reconfig_data_format_srca<DST_ACCUM_MODE>(old_cb_id, input_cb_id)));

    // Reconfigure unpacker for reduce operation (SRCA and SRCB)
    UNPACK((llk_unpack_AB_reduce_init<reduce_type, reduce_dim>(input_cb_id, scaler_cb_id)));

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
    bool use_matmul = false,
    typename AccumBufRef>
ALWI void reload_accumulator_if_needed(
    AccumBufRef& accum_buf, uint32_t input_cb_id, uint32_t scaler_cb_id, const AccumulateT& accumulate) {
    if constexpr (is_accumulate_v<AccumulateT>) {
        if (!accumulate.is_first()) {  // Reload on all iterations except first
            constexpr uint32_t onetile = 1;
            accum_buf.wait_front(onetile);
            const uint32_t prev_srca_cb = use_matmul ? scaler_cb_id : input_cb_id;
            copy_tile_to_dst_init_short_with_dt(prev_srca_cb, accumulate.config.cb_accumulator);
            copy_tile(accumulate.config.cb_accumulator, 0, accumulate.config.dst_index);
            accum_buf.pop_front(onetile);

            // CRITICAL: Re-init after copy_tile corrupts SRCA config
            // Use short version since packer config is still valid from initial init
            // Pass accumulator buffer id as old_cb_id to reconfigure data format from accumulator to input
            if constexpr (use_matmul) {
                reduce_with_matmul_init_with_dt(input_cb_id, scaler_cb_id, accumulate.config.cb_accumulator);
            } else {
                reduce_init_short_with_dt<reduce_type, reduce_dim>(
                    accumulate.config.cb_accumulator, input_cb_id, scaler_cb_id);
            }
        }
    }
}

template <ReduceInputPolicy input_policy>
ALWI void assert_input_cb_size(uint32_t input_cb_id, uint32_t tiles_per_bulk, uint32_t total_tiles) {
    if constexpr (waits_per_tile(input_policy)) {
        ASSERT(get_cb_num_pages(input_cb_id) >= 1);
    } else if constexpr (waits_bulk(input_policy)) {
        ASSERT(get_cb_num_pages(input_cb_id) >= tiles_per_bulk);
        ASSERT(get_cb_num_pages(input_cb_id) % tiles_per_bulk == 0);
    } else {  // waits_upfront or no_wait
        ASSERT(get_cb_num_pages(input_cb_id) >= total_tiles);
    }
}

template <ReduceInputPolicy input_policy>
ALWI void assert_output_cb_size(uint32_t output_cb_id, uint32_t total_outputs) {
    if constexpr (should_pop(input_policy)) {
        // Per-tile reserve/push: only needs 1 page
        ASSERT(get_cb_num_pages(output_cb_id) >= 1);
    } else {
        // Bulk reserve upfront: needs all outputs
        ASSERT(get_cb_num_pages(output_cb_id) >= total_outputs);
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
    typename InputBuf,
    typename ScalerBuf,
    typename OutputBuf,
    typename AccumulateT,
    typename PostReduceOp>
ALWI void reduce(
    InputBuf&& input_buf_arg,
    ScalerBuf&& scaler_buf_arg,
    OutputBuf&& output_buf_arg,
    ReduceInputBlockShape input_block_shape,
    ReduceInputMemoryLayout input_memory_layout,
    AccumulateT accumulate,
    PostReduceOp post_reduce_op) {
    // =============================================================================
    // Buffer-type adapter: lift each buffer-like argument into a uniform handle.
    // - uint32_t (CB id) -> wraps in a CircularBuffer (Gen1).
    // - experimental::CircularBuffer& -> passthrough.
    // - experimental::DataflowBuffer& -> passthrough (Gen1 with DFB id == CB id, Gen2 real DFB).
    // The handle exposes wait_front/pop_front/reserve_back/push_back uniformly and
    // an id() accessor that returns the underlying uint32_t for the LLK calls below
    // (reduce_init, reduce_tile, pack_tile, copy_tile, ...) which still take raw ids.
    //
    // We use universal references for the buffer arguments so that:
    //   - lvalue buffer objects are handled by reference (critical on Gen2 where a
    //     DataflowBuffer carries per-object transaction state that must not diverge
    //     between the caller and the helper),
    //   - rvalue uint32_t ids (literals) bind via reference collapsing without copy,
    //   - the deduced template types strip any reference qualifier when matching
    //     against `BufferRef`'s kind specializations.
    // =============================================================================
    ::kernel_lib::BufferRef<std::remove_reference_t<InputBuf>> input_buf{input_buf_arg};
    ::kernel_lib::BufferRef<std::remove_reference_t<ScalerBuf>> scaler_buf{scaler_buf_arg};
    ::kernel_lib::BufferRef<std::remove_reference_t<OutputBuf>> output_buf{output_buf_arg};

    const uint32_t input_cb_id = input_buf.id();
    const uint32_t scaler_cb_id = scaler_buf.id();
    const uint32_t output_cb_id = output_buf.id();

    // =============================================================================
    // Static Assertions (compile-time validation)
    // =============================================================================
    static_assert(
        is_accumulation_type_v<AccumulateT>,
        "AccumulateT must be a valid accumulation type (NoAccumulation or Accumulate)");
    static_assert(
        is_post_reduce_op_v<PostReduceOp>,
        "PostReduceOp must be callable with a uint32_t argument");
    static_assert(
        !is_accumulate_v<AccumulateT> || reduce_type != PoolType::MAX || reduce_dim == ReduceDim::REDUCE_COL,
        "Accumulate::at with PoolType::MAX only works for REDUCE_COL. For REDUCE_ROW / REDUCE_SCALAR "
        "the pack reduce edge mask drops the face-row-0 spread that GMPOOL needs as its running "
        "accumulator on the reload pass, so the previous MAX is lost.");

    // =============================================================================
    // Runtime Assertions (parameter validation)
    // =============================================================================
    ASSERT(input_cb_id < NUM_CIRCULAR_BUFFERS);
    ASSERT(scaler_cb_id < NUM_CIRCULAR_BUFFERS);
    ASSERT(output_cb_id < NUM_CIRCULAR_BUFFERS);
    ASSERT(input_cb_id != output_cb_id);
    ASSERT(input_cb_id != scaler_cb_id);
    ASSERT(output_cb_id != scaler_cb_id);
    UNPACK(ASSERT(is_valid_cb_tile_page_size(input_cb_id, (DataFormat)unpack_src_format[input_cb_id])));
    UNPACK(ASSERT(is_valid_cb_tile_page_size(scaler_cb_id, (DataFormat)unpack_src_format[scaler_cb_id])));
    PACK(ASSERT(is_valid_cb_tile_page_size(output_cb_id, (DataFormat)pack_dst_format[output_cb_id])));
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

    // The accumulator path always identifies its buffer by raw id (carried inside
    // AccumulationConfig). We construct a CircularBuffer wrapper here for the
    // reload helper; on Gen1 this is exact, and on Gen2 the same id maps to the
    // intended DFB (callers using the typed-buffer path would pass the DFB
    // accessor's `.id` into AccumulationConfig).
    ::kernel_lib::BufferRef<uint32_t> accum_buf{[&]() -> uint32_t {
        if constexpr (enable_accumulation) { return accumulate.config.cb_accumulator; }
        else { return 0; }
    }()};

    // Apply reconfig based on mode
    if constexpr (reconfig_input(reconfig_mode)) {
        if constexpr (use_matmul) {
            reconfig_data_format(scaler_cb_id, input_cb_id);
        } else {
            reconfig_data_format(input_cb_id, scaler_cb_id);
        }
    }
    if constexpr (reconfig_output(reconfig_mode)) {
        pack_reconfig_data_format(output_cb_id);
    }
    // Initialization
    if constexpr (use_matmul) {
        reduce_with_matmul_init(input_cb_id, scaler_cb_id);
    } else {
        reduce_init<reduce_type, reduce_dim>(input_cb_id, scaler_cb_id, output_cb_id);
    }
    scaler_buf.wait_front(1);  // Wait for scaler tile

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
        UNPACK((assert_input_cb_size<input_policy>(input_cb_id, tiles_per_bulk, total_input_tiles)));
        PACK((assert_output_cb_size<input_policy>(output_cb_id, total_output_tiles)));

        // No-pop modes: bulk reserve output upfront
        if constexpr (!should_pop(input_policy)) {
            output_buf.reserve_back(total_output_tiles);
        }

        // PersistentPolicy: wait for all tiles upfront
        if constexpr (waits_upfront(input_policy)) {
            input_buf.wait_front(total_input_tiles);
        }

        uint32_t batch_offset = 0;
        for (uint32_t nc = 0; nc < num_batches; ++nc) {
            // BulkWaitBulkPop: wait for all Ht×Wt tiles in bulk
            if constexpr (waits_bulk(input_policy)) {
                input_buf.wait_front(tiles_per_bulk);
            }

            tile_regs_acquire();

            // Reload accumulator if needed (zero overhead when AccumulateT is NoAccumulation)
            reload_accumulator_if_needed<reduce_type, reduce_dim, AccumulateT>(
                accum_buf, input_cb_id, scaler_cb_id, accumulate);

            const uint32_t dst_idx = get_dst_index(accumulate);
            for (uint32_t ht = 0; ht < Ht; ++ht) {
                for (uint32_t wt = 0; wt < Wt; ++wt) {
                    if constexpr (waits_per_tile(input_policy)) {
                        // One-at-a-time: wait/pop per tile
                        input_buf.wait_front(onetile);
                        reduce_tile<reduce_type, reduce_dim>(input_cb_id, scaler_cb_id, 0, 0, dst_idx);
                        input_buf.pop_front(onetile);
                    } else if constexpr (waits_bulk(input_policy)) {
                        // BulkWaitBulkPop: use indexed access
                        uint32_t tile_idx = ht * stride + wt;
                        reduce_tile<reduce_type, reduce_dim>(
                            input_cb_id, scaler_cb_id, tile_idx, 0, dst_idx);
                    } else {  // PreloadedPolicy or PersistentPolicy: indexed access
                        uint32_t tile_idx = batch_offset + ht * stride + wt;
                        reduce_tile<reduce_type, reduce_dim>(
                            input_cb_id, scaler_cb_id, tile_idx, 0, dst_idx);
                    }
                }
            }

            // Call post-reduce operation on the single accumulated DST register.
            // No-op when PostReduceOp is the default NoOp.
            post_reduce_op(dst_idx);

            // Pop modes: reserve per-batch
            if constexpr (should_pop(input_policy)) {
                output_buf.reserve_back(onetile);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(get_dst_index(accumulate), output_cb_id);
            tile_regs_release();
            if constexpr (should_pop(input_policy)) {
                output_buf.push_back(onetile);
            }

            // BulkWaitBulkPop: pop all tiles after processing
            if constexpr (waits_bulk(input_policy)) {
                input_buf.pop_front(tiles_per_bulk);
            }

            // PreloadedPolicy or PersistentPolicy: update batch offset
            if constexpr (!should_pop(input_policy)) {
                batch_offset += tiles_per_bulk;
            }
        }

        // No-pop modes: bulk push output at end
        if constexpr (!should_pop(input_policy)) {
            output_buf.push_back(total_output_tiles);
        }
    } else if constexpr (reduce_dim == ReduceDim::REDUCE_ROW) {
        // =================================================================
        // REDUCE_ROW: W reduction - each row -> 1 output tile (Ht outputs per batch)
        // =================================================================
        const uint32_t stride = (input_memory_layout.row_stride > 0) ? input_memory_layout.row_stride : Wt;
        const uint32_t total_output_tiles = Ht * num_batches;
        const uint32_t total_input_tiles = Ht * stride * num_batches;
        UNPACK((assert_input_cb_size<input_policy>(input_cb_id, Wt, total_input_tiles)));
        PACK((assert_output_cb_size<input_policy>(output_cb_id, total_output_tiles)));

        // No-pop modes: bulk reserve output upfront
        if constexpr (!should_pop(input_policy)) {
            output_buf.reserve_back(total_output_tiles);
        }

        // PersistentPolicy: wait for all tiles upfront
        if constexpr (waits_upfront(input_policy)) {
            input_buf.wait_front(total_input_tiles);
        }

        uint32_t index_offset = 0;
        for (uint32_t nc = 0; nc < num_batches; ++nc) {
            for (uint32_t ht = 0; ht < Ht; ++ht) {
                // BulkWaitBulkPop: wait for entire row upfront
                if constexpr (waits_bulk(input_policy)) {
                    input_buf.wait_front(Wt);
                }

                tile_regs_acquire();

                // Reload accumulator if needed (zero overhead when AccumulateT is NoAccumulation)
                reload_accumulator_if_needed<reduce_type, reduce_dim, AccumulateT, use_matmul>(
                    accum_buf, input_cb_id, scaler_cb_id, accumulate);

                const uint32_t dst_idx = get_dst_index(accumulate);
                for (uint32_t wt = 0; wt < Wt; ++wt) {
                    if constexpr (waits_per_tile(input_policy)) {
                        // One-at-a-time: wait/pop per tile
                        input_buf.wait_front(onetile);
                        if constexpr (use_matmul) {
                            reduce_matmul_tiles(input_cb_id, scaler_cb_id, 0, 0, dst_idx);
                        } else {
                            reduce_tile<reduce_type, reduce_dim>(input_cb_id, scaler_cb_id, 0, 0, dst_idx);
                        }
                        input_buf.pop_front(onetile);
                    } else if constexpr (waits_bulk(input_policy)) {
                        // BulkWaitBulkPop: use indexed access
                        if constexpr (use_matmul) {
                            reduce_matmul_tiles(input_cb_id, scaler_cb_id, wt, 0, dst_idx);
                        } else {
                            reduce_tile<reduce_type, reduce_dim>(
                                input_cb_id, scaler_cb_id, wt, 0, dst_idx);
                        }
                    } else {  // PreloadedPolicy or PersistentPolicy: indexed access
                        if constexpr (use_matmul) {
                            reduce_matmul_tiles(input_cb_id, scaler_cb_id, wt + index_offset, 0, dst_idx);
                        } else {
                            reduce_tile<reduce_type, reduce_dim>(
                                input_cb_id, scaler_cb_id, wt + index_offset, 0, dst_idx);
                        }
                    }
                }

                // Call post-reduce operation (e.g., recip_tile for softmax)
                // User's lambda can include reduce_uninit() if needed before custom ops
                post_reduce_op(dst_idx);

                // Pop modes: reserve per-row to avoid deadlock
                if constexpr (should_pop(input_policy)) {
                    output_buf.reserve_back(onetile);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(dst_idx, output_cb_id);
                tile_regs_release();
                if constexpr (should_pop(input_policy)) {
                    output_buf.push_back(onetile);
                }

                // BulkWaitBulkPop: pop all tiles after processing
                if constexpr (waits_bulk(input_policy)) {
                    input_buf.pop_front(Wt);
                }

                // PreloadedPolicy or PersistentPolicy: update index offset
                if constexpr (!should_pop(input_policy)) {
                    index_offset += stride;
                }
            }
        }

        // No-pop modes: bulk push output at end
        if constexpr (!should_pop(input_policy)) {
            output_buf.push_back(total_output_tiles);
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
        UNPACK((assert_input_cb_size<input_policy>(input_cb_id, Ht * chunk_size, total_input_tiles)));
        PACK((assert_output_cb_size<input_policy>(output_cb_id, total_output_tiles)));

        // No-pop modes: bulk reserve output upfront
        if constexpr (!should_pop(input_policy)) {
            output_buf.reserve_back(total_output_tiles);
        }

        // PersistentPolicy: wait for all tiles upfront
        if constexpr (waits_upfront(input_policy)) {
            input_buf.wait_front(total_input_tiles);
        }

        uint32_t batch_offset = 0;
        for (uint32_t nc = 0; nc < num_batches; ++nc) {
            for (uint32_t wt = 0; wt < Wt; wt += chunk_size) {
                uint32_t chunk_end = (wt + chunk_size < Wt) ? (wt + chunk_size) : Wt;
                uint32_t current_chunk = chunk_end - wt;
                uint32_t tiles_in_chunk = Ht * current_chunk;

                // BulkWaitBulkPop: wait for entire chunk upfront
                if constexpr (waits_bulk(input_policy)) {
                    input_buf.wait_front(tiles_in_chunk);
                }

                tile_regs_acquire();

                // Reload accumulator if needed (zero overhead when AccumulateT is NoAccumulation)
                reload_accumulator_if_needed<reduce_type, reduce_dim, AccumulateT>(
                    accum_buf, input_cb_id, scaler_cb_id, accumulate);

                for (uint32_t ht = 0; ht < Ht; ++ht) {
                    // Base dst_index: from accumulation config or 0 for multi-column output
                    uint32_t dst_idx = get_dst_index(accumulate);
                    for (uint32_t i = wt; i < chunk_end; ++i) {
                        if constexpr (waits_per_tile(input_policy)) {
                            // One-at-a-time: wait/pop per tile
                            input_buf.wait_front(onetile);
                            reduce_tile<reduce_type, reduce_dim>(
                                input_cb_id, scaler_cb_id, 0, 0, dst_idx);
                            input_buf.pop_front(onetile);
                        } else if constexpr (waits_bulk(input_policy)) {
                            // BulkWaitBulkPop: use indexed access
                            uint32_t tile_idx = ht * current_chunk + (i - wt);
                            reduce_tile<reduce_type, reduce_dim>(
                                input_cb_id, scaler_cb_id, tile_idx, 0, dst_idx);
                        } else {  // PreloadedPolicy or PersistentPolicy: indexed access
                            uint32_t tile_idx = batch_offset + ht * stride + i;
                            reduce_tile<reduce_type, reduce_dim>(
                                input_cb_id, scaler_cb_id, tile_idx, 0, dst_idx);
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
                        output_buf.reserve_back(onetile);
                    }
                    pack_tile(base_dst + i, output_cb_id);
                    if constexpr (should_pop(input_policy)) {
                        output_buf.push_back(onetile);
                    }
                }
                tile_regs_release();

                // BulkWaitBulkPop: pop all tiles after processing
                if constexpr (waits_bulk(input_policy)) {
                    input_buf.pop_front(tiles_in_chunk);
                }
            }
            // Update batch_offset for indexed modes (PreloadedPolicy and PersistentPolicy)
            if constexpr (!should_pop(input_policy)) {
                batch_offset += tiles_per_bulk;
            }
        }

        // No-pop modes: bulk push output at end
        if constexpr (!should_pop(input_policy)) {
            output_buf.push_back(total_output_tiles);
        }
    }

    // Cleanup
    if constexpr (!use_matmul) {
        reduce_uninit<>();
    }
}

}  // namespace compute_kernel_lib
