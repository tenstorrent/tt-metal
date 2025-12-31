// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/cb_api.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/pack.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"

/**
 * @file reduce_helpers.hpp
 * @brief Single unified reduce function with automatic dispatch
 *
 * Provides ONE function that handles all reduce operations:
 * - Row reduction (REDUCE_ROW): Reduces W dimension, outputs Ht tiles per batch
 * - Column reduction (REDUCE_COL): Reduces H dimension, outputs Wt tiles per batch
 * - Scalar reduction (REDUCE_SCALAR): Reduces both H and W, outputs 1 tile per batch
 *
 * This library hides the complexity of:
 * - tile_regs_acquire/commit/wait/release DST register management
 * - reduce_init/reduce_uninit initialization
 * - Circular buffer manipulation (cb_wait_front, cb_pop_front, cb_reserve_back, cb_push_back)
 * - pack_tile for writing results to output CB
 * - Auto-batched STREAMING mode for optimal performance (waits/pops tiles in bulk)
 *
 * DEST register capacity is automatically detected via dest_helpers.hpp.
 *
 * IMPORTANT: Requires compute kernel hardware initialization.
 * Call compute_kernel_hw_startup() before using.
 *
 * IMPORTANT: The scaler CB must contain the scaling factor tile BEFORE calling reduce().
 *
 * Usage:
 *   #include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers.hpp"
 *
 *   compute_kernel_hw_startup(cb_in, cb_scaler, cb_out);
 *
 *   // STREAMING mode (default) - auto-batched for efficiency:
 *   // Library automatically waits for all tiles, processes them, and pops them.
 *   // No manual CB management needed!
 *
 *   // Reduce entire HxW grid to single tile (REDUCE_SCALAR)
 *   compute_kernel_lib::reduce<SUM, REDUCE_SCALAR>(cb_in, cb_scaler, cb_out, Ht, Wt, NC);
 *
 *   // Reduce each row (W dimension) - output has Ht tiles per batch
 *   compute_kernel_lib::reduce<SUM, REDUCE_ROW>(cb_in, cb_scaler, cb_out, Ht, Wt, NC);
 *
 *   // Reduce each column (H dimension) - output has Wt tiles per batch
 *   compute_kernel_lib::reduce<SUM, REDUCE_COL>(cb_in, cb_scaler, cb_out, Ht, Wt, NC);
 *
 *   // Using defines for reduce type/dim (REDUCE_OP and REDUCE_DIM must be defined)
 *   compute_kernel_lib::reduce(cb_in, cb_scaler, cb_out, Ht, Wt, NC);
 */

namespace compute_kernel_lib {

// get_dest_limit() and DEST_AUTO_LIMIT are provided by dest_helpers.hpp

/**
 * @brief Input mode for reduce operations
 *
 * STREAMING: Adaptive batching mode - batches tiles up to CB capacity, processes via indexing (default)
 *            Automatically adapts batch size to avoid deadlock with small CBs.
 * PRELOADED: All tiles already present in CB, accessed via indexing (caller manages wait/pop)
 * PERSISTENT: Wait for all tiles upfront, indexed access, NO pop (tiles persist for reuse)
 */
enum class ReduceInputMode {
    STREAMING,  // Adaptive batching: waits for min(CB_capacity, tiles_needed), processes, pops
    PRELOADED,  // All tiles ready upfront, caller manages wait/pop (for when tiles must persist)
    PERSISTENT  // Wait for all tiles, indexed access, no pop (tiles persist for reuse)
};

/**
 * @brief Default no-op functor for post_reduce_op parameter
 *
 * When no custom post-reduce operation is needed, this empty functor is used.
 * It compiles away completely due to inlining.
 */
struct NoOp {
    ALWI void operator()() const {}
};

// =============================================================================
// Single Unified Reduce Function
// =============================================================================

/**
 * @brief Unified reduce function handling all reduction patterns
 *
 * This single function handles:
 * - Row reduction (REDUCE_ROW): Reduces W dimension, outputs Ht tiles per batch
 * - Column reduction (REDUCE_COL): Reduces H dimension, outputs Wt tiles per batch
 * - Scalar reduction (REDUCE_SCALAR): Reduces both H and W, outputs 1 tile per batch
 *
 * IMPORTANT - HARDWARE INITIALIZATION REQUIREMENT:
 * Before calling this function, you MUST initialize the compute kernel hardware by
 * calling compute_kernel_hw_startup() at the start of your kernel.
 *
 * IMPORTANT - SCALER CB REQUIREMENT:
 * The scaler CB (icb_scaler) must contain the scaling factor tile BEFORE calling
 * this function. The function will wait for it automatically when init=true.
 *
 * IMPORTANT - REDUCE_COL DATA LAYOUT:
 * - STREAMING mode: Tiles are batched per chunk. Library waits for all Ht*chunk_size tiles
 *   per chunk, processes via indexed access, then pops all. Host can specify row_chunk
 *   for custom chunk sizes (default: 0 = use auto-detected DEST limit).
 * - PRELOADED mode: Tiles in standard row-major order (batch_offset + ht*stride + wt).
 *
 * INPUT MODES:
 * - STREAMING (default): Auto-batched mode. Library waits for all tiles per row/batch/chunk,
 *                        processes them via indexed access, then pops all. Most efficient for
 *                        typical use cases. Equivalent to PRELOADED but library handles CB lifecycle.
 * - PRELOADED: All tiles already present in CB, accessed via indexing. Caller manages wait/pop.
 *              Use when tiles must persist in CB after reduction. Use input_stride
 *              to specify the stride between rows (for non-contiguous layouts).
 *
 * @tparam reduce_type The type of reduce operation (SUM, AVG, MAX) - defaults to REDUCE_OP define
 * @tparam reduce_dim The dimension to reduce (REDUCE_ROW, REDUCE_COL, REDUCE_SCALAR) - defaults to REDUCE_DIM define
 * @tparam input_mode Input handling mode (STREAMING or PRELOADED) - defaults to STREAMING
 * @tparam init If true, calls reduce_init before processing (default: true)
 * @tparam uninit If true, calls reduce_uninit after processing (default: true)
 * @tparam enforce_fp32_accumulation Enable FP32 accumulation (default: false)
 *
 * @param icb Input circular buffer containing tiles to reduce
 * @param icb_scaler Circular buffer containing scaler tile
 * @param ocb Output circular buffer for reduced tiles
 * @param Ht Height in tiles (number of tile rows)
 * @param Wt Width in tiles (number of tile columns)
 * @param num_batches Number of batches to process (NC dimension)
 * @param row_chunk Chunk size for REDUCE_COL (default: 0 = use auto-detected DEST limit)
 *                  For REDUCE_ROW and REDUCE_SCALAR, this parameter is ignored.
 *                  For REDUCE_COL, if the host arranges tiles with a specific chunk size,
 *                  pass that value here to ensure correct data interpretation.
 * @param input_stride Stride between row groups for PRELOADED mode (default: 0 = use Wt)
 *                     Only used when input_mode is PRELOADED.
 * @param max_batch_tiles Maximum tiles STREAMING mode can batch at once (default: 1 for compatibility)
 *                        Set to CB capacity to enable batching. Use 1 for one-tile-at-a-time behavior.
 *
 * @example
 *   // Reduce entire HxW grid to single tile (REDUCE_SCALAR)
 *   compute_kernel_lib::reduce<SUM, REDUCE_SCALAR>(cb_in, cb_scaler, cb_out, Ht, Wt, NC);
 *
 * @example
 *   // Reduce each row (W dimension) - output has Ht tiles per batch
 *   compute_kernel_lib::reduce<SUM, REDUCE_ROW>(cb_in, cb_scaler, cb_out, Ht, Wt, NC);
 *
 * @example
 *   // Reduce each column (H dimension) - output has Wt tiles per batch, auto chunk size
 *   compute_kernel_lib::reduce<SUM, REDUCE_COL>(cb_in, cb_scaler, cb_out, Ht, Wt, NC);
 *
 * @example
 *   // Reduce each column with host-specified chunk size
 *   compute_kernel_lib::reduce<SUM, REDUCE_COL>(cb_in, cb_scaler, cb_out, Ht, Wt, NC, row_chunk);
 *
 * @example
 *   // Using defines for reduce type/dim
 *   compute_kernel_lib::reduce(cb_in, cb_scaler, cb_out, Ht, Wt, NC);
 *
 * @example
 *   // PRELOADED mode: tiles already in CB, with custom stride between rows
 *   compute_kernel_lib::reduce<SUM, REDUCE_ROW, ReduceInputMode::PRELOADED>(
 *       cb_in, cb_scaler, cb_out, Ht, Wt, NC, 0, input_stride);
 *
 * @example
 *   // PRELOADED mode for REDUCE_COL: tiles in row-major order
 *   compute_kernel_lib::reduce<SUM, REDUCE_COL, ReduceInputMode::PRELOADED>(
 *       cb_in, cb_scaler, cb_out, Ht, Wt, NC);
 *
 * @example
 *   // PRELOADED mode for REDUCE_SCALAR: all tiles pre-loaded
 *   compute_kernel_lib::reduce<SUM, REDUCE_SCALAR, ReduceInputMode::PRELOADED>(
 *       cb_in, cb_scaler, cb_out, Ht, Wt, NC);
 *
 * @example
 *   // STREAMING mode with auto-batching (efficient for multi-tile rows)
 *   // Library waits for all Wt tiles per row, processes them, then pops all Wt tiles
 *   compute_kernel_lib::reduce<SUM, REDUCE_ROW>(cb_in, cb_scaler, cb_out, Ht, Wt, NC);
 *   // Same as PRELOADED but library handles wait/pop automatically!
 *
 * @example
 *   // Post-reduce operation: softmax pattern with recip_tile after SUM reduce
 *   // Set uninit=false since lambda calls reduce_uninit() before recip
 *   compute_kernel_lib::reduce<SUM, REDUCE_ROW, ReduceInputMode::PRELOADED,
 *       true, false, ENABLE_FP32_DEST_ACC>(
 *       cb_exps, cb_scaler, cb_out, 1, Wt, 1, 0, 0,
 *       []() {
 *           reduce_uninit();
 *           recip_tile_init();
 *           recip_tile(0);
 *       });
 */
template <
    PoolType reduce_type = REDUCE_OP,
    ReduceDim reduce_dim = REDUCE_DIM,
    ReduceInputMode input_mode = ReduceInputMode::STREAMING,
    bool init = true,
    bool uninit = true,
    typename PostReduceOp = NoOp>
ALWI void reduce(
    uint32_t icb,
    uint32_t icb_scaler,
    uint32_t ocb,
    uint32_t Ht,
    uint32_t Wt,
    uint32_t num_batches,
    uint32_t row_chunk = 0,
    uint32_t input_stride = 0,
    uint32_t max_batch_tiles = 1,
    PostReduceOp post_reduce_op = {}) {
// Auto-detect FP32 dest accumulation mode from compile-time define
#ifdef ENABLE_FP32_DEST_ACC
    constexpr bool enforce_fp32_accumulation = (ENABLE_FP32_DEST_ACC == 1);
#else
    constexpr bool enforce_fp32_accumulation = false;
#endif

    // Initialization
    if constexpr (init) {
        reduce_init<reduce_type, reduce_dim, enforce_fp32_accumulation>(icb, icb_scaler, ocb);
    }
    cb_wait_front(icb_scaler, 1);  // Wait for scaler tile

    constexpr uint32_t onetile = 1;

    // Pattern dispatch based on reduce_dim
    if constexpr (reduce_dim == ReduceDim::REDUCE_SCALAR) {
        // =================================================================
        // REDUCE_SCALAR: HW reduction - all tiles -> 1 output tile per batch
        // =================================================================
        const uint32_t stride = (input_stride > 0) ? input_stride : Wt;
        const uint32_t tiles_per_batch = Ht * stride;
        const uint32_t total_tiles = tiles_per_batch * num_batches;

        // PRELOADED: bulk reserve output upfront
        if constexpr (input_mode == ReduceInputMode::PRELOADED) {
            cb_reserve_back(ocb, num_batches);
        }

        // PERSISTENT: wait for all tiles upfront
        if constexpr (input_mode == ReduceInputMode::PERSISTENT) {
            cb_wait_front(icb, total_tiles);
        }

        uint32_t batch_offset = 0;
        for (uint32_t nc = 0; nc < num_batches; ++nc) {
            // STREAMING adaptive batching: batch entire batch if CB allows
            if constexpr (input_mode == ReduceInputMode::STREAMING) {
                if (max_batch_tiles >= tiles_per_batch) {
                    // CB can hold entire batch - batch all tiles
                    cb_wait_front(icb, tiles_per_batch);
                }
                // else: batch_size=1, wait/pop per tile below
            }

            tile_regs_acquire();
            for (uint32_t ht = 0; ht < Ht; ++ht) {
                for (uint32_t wt = 0; wt < Wt; ++wt) {
                    if constexpr (input_mode == ReduceInputMode::STREAMING) {
                        if (max_batch_tiles >= tiles_per_batch) {
                            // Batched: use indexed access
                            uint32_t tile_idx = ht * stride + wt;
                            reduce_tile<reduce_type, reduce_dim, enforce_fp32_accumulation>(
                                icb, icb_scaler, tile_idx, 0, 0);
                        } else {
                            // One-at-a-time: wait/pop per tile (backward compatible)
                            cb_wait_front(icb, onetile);
                            reduce_tile<reduce_type, reduce_dim, enforce_fp32_accumulation>(icb, icb_scaler, 0, 0, 0);
                            cb_pop_front(icb, onetile);
                        }
                    } else {  // PRELOADED or PERSISTENT: indexed access
                        uint32_t tile_idx = batch_offset + ht * stride + wt;
                        reduce_tile<reduce_type, reduce_dim, enforce_fp32_accumulation>(
                            icb, icb_scaler, tile_idx, 0, 0);
                    }
                }
            }
            // STREAMING/PERSISTENT: reserve per-batch
            if constexpr (input_mode == ReduceInputMode::STREAMING || input_mode == ReduceInputMode::PERSISTENT) {
                cb_reserve_back(ocb, onetile);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, ocb);
            tile_regs_release();
            if constexpr (input_mode == ReduceInputMode::STREAMING || input_mode == ReduceInputMode::PERSISTENT) {
                cb_push_back(ocb, onetile);
            }

            // STREAMING batched: pop all tiles after processing
            if constexpr (input_mode == ReduceInputMode::STREAMING) {
                if (max_batch_tiles >= tiles_per_batch) {
                    cb_pop_front(icb, tiles_per_batch);
                }
            }

            // PRELOADED or PERSISTENT: update batch offset
            if constexpr (input_mode != ReduceInputMode::STREAMING) {
                batch_offset += tiles_per_batch;
            }
        }

        // PRELOADED: bulk push output at end
        if constexpr (input_mode == ReduceInputMode::PRELOADED) {
            cb_push_back(ocb, num_batches);
        }
    } else if constexpr (reduce_dim == ReduceDim::REDUCE_ROW) {
        // =================================================================
        // REDUCE_ROW: W reduction - each row -> 1 output tile (Ht outputs per batch)
        // =================================================================
        const uint32_t stride = (input_stride > 0) ? input_stride : Wt;
        const uint32_t total_outputs = Ht * num_batches;
        const uint32_t total_tiles = Ht * stride * num_batches;

        // PRELOADED: bulk reserve output upfront
        if constexpr (input_mode == ReduceInputMode::PRELOADED) {
            cb_reserve_back(ocb, total_outputs);
        }

        // PERSISTENT: wait for all tiles upfront
        if constexpr (input_mode == ReduceInputMode::PERSISTENT) {
            cb_wait_front(icb, total_tiles);
        }

        uint32_t index_offset = 0;
        for (uint32_t nc = 0; nc < num_batches; ++nc) {
            for (uint32_t ht = 0; ht < Ht; ++ht) {
                // STREAMING adaptive batching: batch entire row if CB allows
                if constexpr (input_mode == ReduceInputMode::STREAMING) {
                    if (max_batch_tiles >= Wt) {
                        // CB can hold entire row - batch all tiles
                        cb_wait_front(icb, Wt);
                    }
                    // else: batch_size=1, wait/pop per tile below
                }

                tile_regs_acquire();
                for (uint32_t wt = 0; wt < Wt; ++wt) {
                    if constexpr (input_mode == ReduceInputMode::STREAMING) {
                        if (max_batch_tiles >= Wt) {
                            // Batched: use indexed access
                            reduce_tile<reduce_type, reduce_dim, enforce_fp32_accumulation>(icb, icb_scaler, wt, 0, 0);
                        } else {
                            // One-at-a-time: wait/pop per tile (backward compatible)
                            cb_wait_front(icb, onetile);
                            reduce_tile<reduce_type, reduce_dim, enforce_fp32_accumulation>(icb, icb_scaler, 0, 0, 0);
                            cb_pop_front(icb, onetile);
                        }
                    } else {  // PRELOADED or PERSISTENT: indexed access
                        reduce_tile<reduce_type, reduce_dim, enforce_fp32_accumulation>(
                            icb, icb_scaler, wt + index_offset, 0, 0);
                    }
                }

                // Call post-reduce operation (e.g., recip_tile for softmax)
                // User's lambda can include reduce_uninit() if needed before custom ops
                post_reduce_op();

                // STREAMING/PERSISTENT: reserve per-row to avoid deadlock
                if constexpr (input_mode == ReduceInputMode::STREAMING || input_mode == ReduceInputMode::PERSISTENT) {
                    cb_reserve_back(ocb, onetile);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, ocb);
                tile_regs_release();
                if constexpr (input_mode == ReduceInputMode::STREAMING || input_mode == ReduceInputMode::PERSISTENT) {
                    cb_push_back(ocb, onetile);
                }

                // STREAMING batched: pop all tiles after processing
                if constexpr (input_mode == ReduceInputMode::STREAMING) {
                    if (max_batch_tiles >= Wt) {
                        cb_pop_front(icb, Wt);
                    }
                }

                // PRELOADED or PERSISTENT: update index offset
                if constexpr (input_mode != ReduceInputMode::STREAMING) {
                    index_offset += stride;
                }
            }
        }

        // PRELOADED: bulk push output at end
        if constexpr (input_mode == ReduceInputMode::PRELOADED) {
            cb_push_back(ocb, total_outputs);
        }
    } else {
        // =================================================================
        // REDUCE_COL: H reduction - each column -> 1 output tile (Wt outputs per batch)
        // Need chunking due to DEST register limits
        // STREAMING: Tiles arrive in N C W_skip H W_chunk order (chunked by chunk_size)
        // PRELOADED: Tiles in row-major order, indexed as batch_offset + ht*stride + wt
        // =================================================================

        // Use provided row_chunk if > 0, otherwise use auto-detected DEST limit
        const uint32_t chunk_size = (row_chunk > 0) ? row_chunk : DEST_AUTO_LIMIT;
        const uint32_t stride = (input_stride > 0) ? input_stride : Wt;
        const uint32_t tiles_per_batch = Ht * stride;
        const uint32_t total_outputs = Wt * num_batches;
        const uint32_t total_tiles = tiles_per_batch * num_batches;

        // PRELOADED: bulk reserve output upfront
        if constexpr (input_mode == ReduceInputMode::PRELOADED) {
            cb_reserve_back(ocb, total_outputs);
        }

        // PERSISTENT: wait for all tiles upfront
        if constexpr (input_mode == ReduceInputMode::PERSISTENT) {
            cb_wait_front(icb, total_tiles);
        }

        uint32_t batch_offset = 0;
        for (uint32_t nc = 0; nc < num_batches; ++nc) {
            for (uint32_t wt = 0; wt < Wt; wt += chunk_size) {
                uint32_t chunk_end = (wt + chunk_size < Wt) ? (wt + chunk_size) : Wt;
                uint32_t current_chunk = chunk_end - wt;
                uint32_t tiles_in_chunk = Ht * current_chunk;

                // PERSISTENT: wait for all tiles upfront
                if constexpr (input_mode == ReduceInputMode::PERSISTENT) {
                    cb_wait_front(icb, tiles_in_chunk);
                }

                // STREAMING adaptive batching: batch tiles up to max_batch_tiles
                if constexpr (input_mode == ReduceInputMode::STREAMING) {
                    if (max_batch_tiles >= tiles_in_chunk) {
                        // CB can hold entire chunk - batch all tiles
                        cb_wait_front(icb, tiles_in_chunk);
                    }
                    // else: batch_size=1, wait/pop per tile below
                }

                tile_regs_acquire();
                uint32_t tiles_processed = 0;
                for (uint32_t ht = 0; ht < Ht; ++ht) {
                    uint32_t dst_idx = 0;
                    for (uint32_t i = wt; i < chunk_end; ++i) {
                        if constexpr (input_mode == ReduceInputMode::STREAMING) {
                            if (max_batch_tiles >= tiles_in_chunk) {
                                // Batched: use indexed access
                                uint32_t tile_idx = ht * current_chunk + (i - wt);
                                reduce_tile<reduce_type, reduce_dim, enforce_fp32_accumulation>(
                                    icb, icb_scaler, tile_idx, 0, dst_idx);
                            } else {
                                // One-at-a-time: wait/pop per tile (backward compatible)
                                cb_wait_front(icb, onetile);
                                reduce_tile<reduce_type, reduce_dim, enforce_fp32_accumulation>(
                                    icb, icb_scaler, 0, 0, dst_idx);
                                cb_pop_front(icb, onetile);
                            }
                        } else {  // PRELOADED or PERSISTENT: indexed access
                            uint32_t tile_idx = batch_offset + ht * stride + i;
                            reduce_tile<reduce_type, reduce_dim, enforce_fp32_accumulation>(
                                icb, icb_scaler, tile_idx, 0, dst_idx);
                        }
                        ++dst_idx;
                        ++tiles_processed;
                    }
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < current_chunk; ++i) {
                    // STREAMING/PERSISTENT: reserve/push per output tile
                    if constexpr (
                        input_mode == ReduceInputMode::STREAMING || input_mode == ReduceInputMode::PERSISTENT) {
                        cb_reserve_back(ocb, onetile);
                    }
                    pack_tile(i, ocb);
                    if constexpr (
                        input_mode == ReduceInputMode::STREAMING || input_mode == ReduceInputMode::PERSISTENT) {
                        cb_push_back(ocb, onetile);
                    }
                }
                tile_regs_release();

                // STREAMING batched: pop all tiles after processing
                if constexpr (input_mode == ReduceInputMode::STREAMING) {
                    if (max_batch_tiles >= tiles_in_chunk) {
                        cb_pop_front(icb, tiles_in_chunk);
                    }
                }
            }
            // Update batch_offset for indexed modes (PRELOADED and PERSISTENT)
            if constexpr (input_mode != ReduceInputMode::STREAMING) {
                batch_offset += tiles_per_batch;
            }
        }

        // PRELOADED: bulk push output at end
        if constexpr (input_mode == ReduceInputMode::PRELOADED) {
            cb_push_back(ocb, total_outputs);
        }
    }

    // Cleanup
    if constexpr (uninit) {
        reduce_uninit();
    }
}

}  // namespace compute_kernel_lib
