// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>

#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/cb_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/common_types.hpp"

/**
 * @file binary_op_helpers.hpp
 * @brief Single unified binary operation functions (add, sub, mul) with automatic dispatch
 *
 * Provides unified functions that handle all binary operations:
 * - Element-wise add/sub/mul
 * - Broadcast variants (row, col, scalar)
 * - Multiple input modes (streaming, batched, preloaded, persistent)
 *
 * This library hides the complexity of:
 * - tile_regs_acquire/commit/wait/release DST register management
 * - DEST register capacity limits (auto-detected via dest_helpers.hpp)
 * - add_tiles_init/sub_tiles_init/mul_tiles_init initialization
 * - Circular buffer manipulation (cb_wait_front, cb_pop_front, cb_reserve_back, cb_push_back)
 * - pack_tile for writing results to output CB
 * - Broadcast init functions (add_bcast_rows_init_short, etc.)
 *
 * DEST register capacity is automatically detected via dest_helpers.hpp:
 * - SyncFull + 16-bit: 16 tiles
 * - SyncFull + 32-bit: 8 tiles
 * - SyncHalf + 16-bit: 8 tiles
 * - SyncHalf + 32-bit: 4 tiles
 *
 * IMPORTANT: Requires compute kernel hardware initialization.
 * Call compute_kernel_hw_startup() before using.
 *
 * Usage:
 *   #include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"
 *
 *   compute_kernel_hw_startup(cb_in0, cb_in1, cb_out);
 *
 *   // Simple single tile add:
 *   compute_kernel_lib::add(cb_in0, cb_in1, cb_out,
 *       compute_kernel_lib::BinaryTileShape::single());
 *
 *   // Add N tiles in streaming mode:
 *   compute_kernel_lib::add(cb_in0, cb_in1, cb_out,
 *       compute_kernel_lib::BinaryTileShape::block(N));
 *
 *   // Add with row broadcast:
 *   compute_kernel_lib::add<compute_kernel_lib::BroadcastDim::ROW>(
 *       cb_data, cb_bias, cb_out,
 *       compute_kernel_lib::BinaryTileShape::grid(Ht, Wt));
 */

namespace compute_kernel_lib {

// =============================================================================
// Enumerations
// =============================================================================

/**
 * @brief Binary operation type
 */
enum class BinaryOpType {
    ADD,  // A + B
    SUB,  // A - B
    MUL   // A * B
};

/**
 * @brief Broadcast dimension for second operand
 *
 * NONE: Standard element-wise (no broadcast)
 * ROW: Broadcast single row across all rows (B has filled 0-row, C[h,w] = A[h,w] op B[h])
 * COL: Broadcast single column across all columns (B has filled 0-col, C[h,w] = A[h,w] op B[w])
 * SCALAR: Broadcast single value (B[0,0] applied to all elements)
 */
enum class BroadcastDim { NONE, ROW, COL, SCALAR };

/**
 * @brief Input handling mode for binary operations
 *
 * STREAMING: One-at-a-time mode - waits/pops each tile individually (default, safest)
 *            Safe for any CB size, works with external producers.
 *
 * STREAMING_BATCHED: Batched mode - waits for all tiles in batch, indexed access, pops all
 *                    Optimal when tiles are pre-loaded in CB.
 *
 * PRELOADED: All tiles already present in CB, accessed via indexing (caller manages wait/pop)
 *            Use for asymmetric wait/pop patterns or when tiles are pre-fetched.
 *
 * PERSISTENT: Wait for all tiles upfront, indexed access, NO pop (tiles persist for reuse)
 *             Use when same input is needed for subsequent operations.
 */
enum class BinaryInputMode { STREAMING, STREAMING_BATCHED, PRELOADED, PERSISTENT };

/**
 * @brief Data format reconfiguration mode
 *
 * Controls whether the library automatically reconfigures the unpacker and packer
 * data formats before executing the binary operation.
 */
enum class BinaryDataFormatReconfig {
    NONE = 0,    // No reconfig - formats already match
    INPUT = 1,   // Reconfig unpacker only
    OUTPUT = 2,  // Reconfig packer only
    BOTH = 3     // Reconfig both (default)
};

// =============================================================================
// Configuration Types
// =============================================================================

/**
 * @brief Tile shape specification for binary operations
 *
 * Specifies the tile grid dimensions (rows x cols x batches).
 * For most binary ops, both inputs have same shape (or one is broadcast).
 */
struct BinaryTileShape {
    uint32_t rows;
    uint32_t cols;
    uint32_t batches;

    // Full grid specification
    static constexpr BinaryTileShape grid(uint32_t r, uint32_t c, uint32_t b = 1) { return {r, c, b}; }

    // Single tile (1x1x1)
    static constexpr BinaryTileShape single() { return {1, 1, 1}; }

    // Single row of tiles (1 x cols x batches)
    static constexpr BinaryTileShape row(uint32_t c, uint32_t b = 1) { return {1, c, b}; }

    // Single column of tiles (rows x 1 x batches)
    static constexpr BinaryTileShape col(uint32_t r, uint32_t b = 1) { return {r, 1, b}; }

    // Linear block of tiles (1 x tiles x batches) - for streaming N tiles
    static constexpr BinaryTileShape block(uint32_t tiles, uint32_t b = 1) { return {1, tiles, b}; }
};

/**
 * @brief Tile memory layout specification for indexed access modes
 *
 * Specifies the stride pattern for accessing tiles in non-contiguous layouts.
 * Used when input_mode is PRELOADED or PERSISTENT.
 */
struct BinaryTileLayout {
    uint32_t row_stride_a = 0;    // 0 = auto-detect from cols (contiguous row-major)
    uint32_t row_stride_b = 0;    // Separate stride for second input (for broadcast patterns)
    uint32_t batch_stride_a = 0;  // Reserved for future use
    uint32_t batch_stride_b = 0;

    // Factory methods
    static constexpr BinaryTileLayout contiguous() { return {}; }
    static constexpr BinaryTileLayout with_stride_a(uint32_t s) { return {s, 0, 0, 0}; }
    static constexpr BinaryTileLayout with_stride_b(uint32_t s) { return {0, s, 0, 0}; }
    static constexpr BinaryTileLayout with_strides(uint32_t sa, uint32_t sb) { return {sa, sb, 0, 0}; }
};

// NoAccumulation is defined in common_types.hpp

/**
 * @brief Accumulation configuration for iterative accumulation patterns
 *
 * Used when the result of binary operation accumulates into a persistent CB.
 * The iteration index determines reload behavior:
 * - iteration == 0: skip reload (first call, no accumulated value yet)
 * - iteration > 0: reload from accumulator CB before operating
 */
struct BinaryAccumulate {
    uint32_t cb_accumulator = 0;  // CB for accumulator
    uint32_t dst_index = 0;       // DST register for accumulation (default: 0)
    uint32_t iteration = 0;       // Current iteration (0 = first, skip reload)

    constexpr BinaryAccumulate(uint32_t cb, uint32_t dst, uint32_t iter) :
        cb_accumulator(cb), dst_index(dst), iteration(iter) {}

    constexpr BinaryAccumulate(uint32_t cb, uint32_t iter = 0) : cb_accumulator(cb), dst_index(0), iteration(iter) {}

    // Factory for concise call sites
    static constexpr BinaryAccumulate at(uint32_t cb, uint32_t iter, uint32_t dst = 0) {
        return BinaryAccumulate(cb, dst, iter);
    }

    // Convenience: check if this is first iteration (skip reload)
    constexpr bool is_first() const { return iteration == 0; }
};

// NoOp is defined in common_types.hpp

// =============================================================================
// Type Traits
// =============================================================================

template <typename T>
struct is_binary_accumulate : std::false_type {};

template <>
struct is_binary_accumulate<BinaryAccumulate> : std::true_type {};

template <typename T>
inline constexpr bool is_binary_accumulate_v = is_binary_accumulate<T>::value;

/**
 * @brief Helper to extract dst_index from accumulation type
 */
template <typename AccumT>
ALWI constexpr uint32_t get_binary_dst_index(const AccumT& accum) {
    if constexpr (is_binary_accumulate_v<AccumT>) {
        return accum.dst_index;
    } else {
        return 0;
    }
}

// =============================================================================
// Internal Helper Functions - Init Dispatch
// =============================================================================

/**
 * @brief Initialize binary operation for BroadcastDim::NONE
 */
template <BinaryOpType op_type>
ALWI void binary_init_none(uint32_t icb_a, uint32_t icb_b) {
    if constexpr (op_type == BinaryOpType::ADD) {
        add_tiles_init(icb_a, icb_b);
    } else if constexpr (op_type == BinaryOpType::SUB) {
        sub_tiles_init(icb_a, icb_b);
    } else {
        mul_tiles_init(icb_a, icb_b);
    }
}

/**
 * @brief Initialize binary operation for BroadcastDim::ROW
 */
template <BinaryOpType op_type>
ALWI void binary_init_row(uint32_t icb_a, uint32_t icb_b) {
    if constexpr (op_type == BinaryOpType::ADD) {
        add_bcast_rows_init_short(icb_a, icb_b);
    } else if constexpr (op_type == BinaryOpType::MUL) {
        mul_bcast_rows_init_short(icb_a, icb_b);
    }
    // Note: sub_bcast_rows not available in current API
}

/**
 * @brief Initialize binary operation for BroadcastDim::COL
 */
template <BinaryOpType op_type>
ALWI void binary_init_col(uint32_t icb_a, uint32_t icb_b) {
    if constexpr (op_type == BinaryOpType::ADD) {
        add_bcast_cols_init_short(icb_a, icb_b);
    } else if constexpr (op_type == BinaryOpType::SUB) {
        sub_bcast_cols_init_short(icb_a, icb_b);
    } else {
        mul_bcast_cols_init_short(icb_a, icb_b);
    }
}

/**
 * @brief Initialize binary operation for BroadcastDim::SCALAR
 */
template <BinaryOpType op_type>
ALWI void binary_init_scalar(uint32_t icb_a, uint32_t icb_b) {
    if constexpr (op_type == BinaryOpType::ADD) {
        add_bcast_scalar_init_short(icb_a, icb_b);
    } else if constexpr (op_type == BinaryOpType::SUB) {
        sub_tiles_bcast_scalar_init_short(icb_a, icb_b);
    } else {
        mul_tiles_bcast_scalar_init_short(icb_a, icb_b);
    }
}

/**
 * @brief Compile-time dispatch to correct init function based on op type and broadcast dim
 */
template <BinaryOpType op_type, BroadcastDim bcast_dim>
ALWI void binary_init_dispatch(uint32_t icb_a, uint32_t icb_b) {
    if constexpr (bcast_dim == BroadcastDim::NONE) {
        binary_init_none<op_type>(icb_a, icb_b);
    } else if constexpr (bcast_dim == BroadcastDim::ROW) {
        binary_init_row<op_type>(icb_a, icb_b);
    } else if constexpr (bcast_dim == BroadcastDim::COL) {
        binary_init_col<op_type>(icb_a, icb_b);
    } else {
        binary_init_scalar<op_type>(icb_a, icb_b);
    }
}

// =============================================================================
// Internal Helper Functions - Operation Execution
// =============================================================================

/**
 * @brief Execute binary operation for BroadcastDim::NONE
 */
template <BinaryOpType op_type>
ALWI void binary_exec_none(uint32_t icb_a, uint32_t icb_b, uint32_t itile_a, uint32_t itile_b, uint32_t idst) {
    if constexpr (op_type == BinaryOpType::ADD) {
        add_tiles(icb_a, icb_b, itile_a, itile_b, idst);
    } else if constexpr (op_type == BinaryOpType::SUB) {
        sub_tiles(icb_a, icb_b, itile_a, itile_b, idst);
    } else {
        mul_tiles(icb_a, icb_b, itile_a, itile_b, idst);
    }
}

/**
 * @brief Execute binary operation for BroadcastDim::ROW
 */
template <BinaryOpType op_type>
ALWI void binary_exec_row(uint32_t icb_a, uint32_t icb_b, uint32_t itile_a, uint32_t itile_b, uint32_t idst) {
    if constexpr (op_type == BinaryOpType::ADD) {
        add_tiles_bcast_rows(icb_a, icb_b, itile_a, itile_b, idst);
    } else if constexpr (op_type == BinaryOpType::SUB) {
        sub_tiles_bcast<BroadcastType::ROW>(icb_a, icb_b, itile_a, itile_b, idst);
    } else {
        mul_tiles_bcast_rows(icb_a, icb_b, itile_a, itile_b, idst);
    }
}

/**
 * @brief Execute binary operation for BroadcastDim::COL
 */
template <BinaryOpType op_type>
ALWI void binary_exec_col(uint32_t icb_a, uint32_t icb_b, uint32_t itile_a, uint32_t itile_b, uint32_t idst) {
    if constexpr (op_type == BinaryOpType::ADD) {
        add_tiles_bcast_cols(icb_a, icb_b, itile_a, itile_b, idst);
    } else if constexpr (op_type == BinaryOpType::SUB) {
        sub_tiles_bcast_cols(icb_a, icb_b, itile_a, itile_b, idst);
    } else {
        mul_tiles_bcast_cols(icb_a, icb_b, itile_a, itile_b, idst);
    }
}

/**
 * @brief Execute binary operation for BroadcastDim::SCALAR
 */
template <BinaryOpType op_type>
ALWI void binary_exec_scalar(uint32_t icb_a, uint32_t icb_b, uint32_t itile_a, uint32_t itile_b, uint32_t idst) {
    if constexpr (op_type == BinaryOpType::ADD) {
        add_tiles_bcast_scalar(icb_a, icb_b, itile_a, itile_b, idst);
    } else if constexpr (op_type == BinaryOpType::SUB) {
        sub_tiles_bcast_scalar(icb_a, icb_b, itile_a, itile_b, idst);
    } else {
        mul_tiles_bcast_scalar(icb_a, icb_b, itile_a, itile_b, idst);
    }
}

/**
 * @brief Compile-time dispatch to correct operation function
 */
template <BinaryOpType op_type, BroadcastDim bcast_dim>
ALWI void binary_exec(uint32_t icb_a, uint32_t icb_b, uint32_t itile_a, uint32_t itile_b, uint32_t idst) {
    if constexpr (bcast_dim == BroadcastDim::NONE) {
        binary_exec_none<op_type>(icb_a, icb_b, itile_a, itile_b, idst);
    } else if constexpr (bcast_dim == BroadcastDim::ROW) {
        binary_exec_row<op_type>(icb_a, icb_b, itile_a, itile_b, idst);
    } else if constexpr (bcast_dim == BroadcastDim::COL) {
        binary_exec_col<op_type>(icb_a, icb_b, itile_a, itile_b, idst);
    } else {
        binary_exec_scalar<op_type>(icb_a, icb_b, itile_a, itile_b, idst);
    }
}

// =============================================================================
// Internal Helper Functions - Accumulator Reload
// =============================================================================

/**
 * @brief Reload accumulator tile into DST register if needed
 *
 * When AccumT is BinaryAccumulate and iteration > 0:
 * 1. Loads accumulator tile from cb_accumulator to DST[dst_index]
 * 2. Re-initializes binary operation (critical after copy_tile corrupts SRCA config)
 *
 * When AccumT is NoAccumulation: compiles to nothing (zero overhead)
 */
template <BinaryOpType op_type, BroadcastDim bcast_dim, typename AccumT>
ALWI void reload_accumulator_if_needed_binary(uint32_t icb_a, uint32_t icb_b, const AccumT& accum) {
    if constexpr (is_binary_accumulate_v<AccumT>) {
        if (!accum.is_first()) {
            constexpr uint32_t onetile = 1;
            cb_wait_front(accum.cb_accumulator, onetile);
            copy_tile_to_dst_init_short_with_dt(icb_a, accum.cb_accumulator);
            copy_tile(accum.cb_accumulator, 0, accum.dst_index);
            cb_pop_front(accum.cb_accumulator, onetile);

            // CRITICAL: Re-init binary op after copy_tile corrupts SRCA config
            binary_init_dispatch<op_type, bcast_dim>(icb_a, icb_b);
        }
    }
}

// =============================================================================
// Internal Core Logic - Separated by BroadcastDim for Debugging
// =============================================================================

/**
 * @brief Core binary operation logic for BroadcastDim::NONE
 *
 * Element-wise operation: C[i] = A[i] op B[i]
 * Both inputs have identical shape.
 */
template <BinaryOpType op_type, BinaryInputMode input_mode, bool init, typename AccumT, typename PostOp>
ALWI void binary_op_none(
    uint32_t icb_a,
    uint32_t icb_b,
    uint32_t ocb,
    BinaryTileShape shape,
    BinaryTileLayout layout,
    AccumT accum,
    PostOp post_op) {
    constexpr uint32_t dest_limit = DEST_AUTO_LIMIT;
    constexpr uint32_t onetile = 1;

    const uint32_t Ht = shape.rows;
    const uint32_t Wt = shape.cols;
    const uint32_t num_batches = shape.batches;
    const uint32_t total_tiles = Ht * Wt * num_batches;
    const uint32_t stride_a = (layout.row_stride_a > 0) ? layout.row_stride_a : Wt;
    const uint32_t stride_b = (layout.row_stride_b > 0) ? layout.row_stride_b : Wt;

    // Initialization
    if constexpr (init) {
        binary_init_none<op_type>(icb_a, icb_b);
    }

    // PERSISTENT: wait for all tiles upfront
    if constexpr (input_mode == BinaryInputMode::PERSISTENT) {
        cb_wait_front(icb_a, total_tiles);
        cb_wait_front(icb_b, total_tiles);
    }

    // PRELOADED: bulk reserve output upfront
    if constexpr (input_mode == BinaryInputMode::PRELOADED) {
        cb_reserve_back(ocb, total_tiles);
    }

    uint32_t index_offset_a = 0;
    uint32_t index_offset_b = 0;
    uint32_t output_offset = 0;

    // Account for base_dst offset when calculating effective DEST capacity
    const uint32_t base_dst = get_binary_dst_index(accum);
    const uint32_t effective_dest_limit = dest_limit - base_dst;

    for (uint32_t nc = 0; nc < num_batches; ++nc) {
        for (uint32_t ht = 0; ht < Ht; ++ht) {
            // Process row in chunks that fit within DEST limit
            for (uint32_t wt_base = 0; wt_base < Wt; wt_base += effective_dest_limit) {
                const uint32_t chunk_size =
                    (wt_base + effective_dest_limit <= Wt) ? effective_dest_limit : (Wt - wt_base);

                // STREAMING_BATCHED: wait for chunk upfront
                if constexpr (input_mode == BinaryInputMode::STREAMING_BATCHED) {
                    cb_wait_front(icb_a, chunk_size);
                    cb_wait_front(icb_b, chunk_size);
                }

                tile_regs_acquire();

                // Reload accumulator if needed
                reload_accumulator_if_needed_binary<op_type, BroadcastDim::NONE>(icb_a, icb_b, accum);

                for (uint32_t wt = 0; wt < chunk_size; ++wt) {
                    uint32_t tile_a, tile_b, dst_idx;

                    if constexpr (input_mode == BinaryInputMode::STREAMING) {
                        cb_wait_front(icb_a, onetile);
                        cb_wait_front(icb_b, onetile);
                        tile_a = 0;
                        tile_b = 0;
                        dst_idx = base_dst;
                    } else if constexpr (input_mode == BinaryInputMode::STREAMING_BATCHED) {
                        tile_a = wt;
                        tile_b = wt;
                        dst_idx = base_dst + wt;
                    } else {
                        // PRELOADED or PERSISTENT
                        tile_a = index_offset_a + wt_base + wt;
                        tile_b = index_offset_b + wt_base + wt;
                        dst_idx = base_dst + wt;
                    }

                    binary_exec_none<op_type>(icb_a, icb_b, tile_a, tile_b, dst_idx);

                    if constexpr (input_mode == BinaryInputMode::STREAMING) {
                        // Immediate commit/pack for streaming
                        tile_regs_commit();
                        tile_regs_wait();

                        cb_reserve_back(ocb, onetile);
                        pack_tile(base_dst, ocb);
                        cb_push_back(ocb, onetile);

                        cb_pop_front(icb_a, onetile);
                        cb_pop_front(icb_b, onetile);

                        tile_regs_release();

                        if (wt < chunk_size - 1) {
                            tile_regs_acquire();
                        }
                    }
                }

                // Post-op callback
                if constexpr (!std::is_same_v<PostOp, NoOp> && input_mode != BinaryInputMode::STREAMING) {
                    for (uint32_t i = 0; i < chunk_size; ++i) {
                        post_op(base_dst + i);
                    }
                }

                if constexpr (input_mode != BinaryInputMode::STREAMING) {
                    tile_regs_commit();
                    tile_regs_wait();

                    if constexpr (input_mode != BinaryInputMode::PRELOADED) {
                        cb_reserve_back(ocb, chunk_size);
                    }

                    for (uint32_t i = 0; i < chunk_size; ++i) {
                        pack_tile(base_dst + i, ocb, output_offset + i);
                    }

                    if constexpr (input_mode != BinaryInputMode::PRELOADED) {
                        cb_push_back(ocb, chunk_size);
                    }

                    tile_regs_release();
                }

                // STREAMING_BATCHED: pop after chunk
                if constexpr (input_mode == BinaryInputMode::STREAMING_BATCHED) {
                    cb_pop_front(icb_a, chunk_size);
                    cb_pop_front(icb_b, chunk_size);
                }

                output_offset += chunk_size;
            }

            // Update indices for indexed modes
            if constexpr (input_mode == BinaryInputMode::PRELOADED || input_mode == BinaryInputMode::PERSISTENT) {
                index_offset_a += stride_a;
                index_offset_b += stride_b;
            }
        }
    }

    // PRELOADED: bulk push at end
    if constexpr (input_mode == BinaryInputMode::PRELOADED) {
        cb_push_back(ocb, total_tiles);
    }
}

/**
 * @brief Core binary operation logic for BroadcastDim::ROW
 *
 * Row broadcast: C[h,w] = A[h,w] op B[w]
 * B has shape [1, Wt] and broadcasts across all rows of A.
 */
template <BinaryOpType op_type, BinaryInputMode input_mode, bool init, typename AccumT, typename PostOp>
ALWI void binary_op_row(
    uint32_t icb_a,
    uint32_t icb_b,
    uint32_t ocb,
    BinaryTileShape shape,
    BinaryTileLayout layout,
    AccumT accum,
    PostOp post_op) {
    constexpr uint32_t dest_limit = DEST_AUTO_LIMIT;
    constexpr uint32_t onetile = 1;

    const uint32_t Ht = shape.rows;
    const uint32_t Wt = shape.cols;
    const uint32_t num_batches = shape.batches;
    const uint32_t total_tiles_a = Ht * Wt * num_batches;
    const uint32_t stride_a = (layout.row_stride_a > 0) ? layout.row_stride_a : Wt;

    // Initialization
    if constexpr (init) {
        binary_init_row<op_type>(icb_a, icb_b);
    }

    // PERSISTENT: wait for all tiles upfront
    if constexpr (input_mode == BinaryInputMode::PERSISTENT) {
        cb_wait_front(icb_a, total_tiles_a);
        cb_wait_front(icb_b, Wt);  // Only Wt tiles for broadcast
    }

    // PRELOADED: bulk reserve output upfront
    if constexpr (input_mode == BinaryInputMode::PRELOADED) {
        cb_reserve_back(ocb, total_tiles_a);
    }

    // For row broadcast, B tiles persist - wait once at start for non-PERSISTENT modes
    if constexpr (input_mode == BinaryInputMode::STREAMING || input_mode == BinaryInputMode::STREAMING_BATCHED) {
        cb_wait_front(icb_b, Wt);
    }

    uint32_t index_offset_a = 0;
    uint32_t output_offset = 0;

    // Account for base_dst offset when calculating effective DEST capacity
    const uint32_t base_dst = get_binary_dst_index(accum);
    const uint32_t effective_dest_limit = dest_limit - base_dst;

    for (uint32_t nc = 0; nc < num_batches; ++nc) {
        for (uint32_t ht = 0; ht < Ht; ++ht) {
            // Process row in chunks that fit within DEST limit
            for (uint32_t wt_base = 0; wt_base < Wt; wt_base += effective_dest_limit) {
                const uint32_t chunk_size =
                    (wt_base + effective_dest_limit <= Wt) ? effective_dest_limit : (Wt - wt_base);

                // STREAMING_BATCHED: wait for A chunk
                if constexpr (input_mode == BinaryInputMode::STREAMING_BATCHED) {
                    cb_wait_front(icb_a, chunk_size);
                }

                tile_regs_acquire();

                reload_accumulator_if_needed_binary<op_type, BroadcastDim::ROW>(icb_a, icb_b, accum);

                for (uint32_t wt = 0; wt < chunk_size; ++wt) {
                    uint32_t tile_a, tile_b, dst_idx;

                    if constexpr (input_mode == BinaryInputMode::STREAMING) {
                        cb_wait_front(icb_a, onetile);
                        tile_a = 0;
                        tile_b = wt_base + wt;  // Index into persisted B tiles
                        dst_idx = base_dst;
                    } else if constexpr (input_mode == BinaryInputMode::STREAMING_BATCHED) {
                        tile_a = wt;
                        tile_b = wt_base + wt;
                        dst_idx = base_dst + wt;
                    } else {
                        tile_a = index_offset_a + wt_base + wt;
                        tile_b = wt_base + wt;  // B always indexed from 0
                        dst_idx = base_dst + wt;
                    }

                    binary_exec_row<op_type>(icb_a, icb_b, tile_a, tile_b, dst_idx);

                    if constexpr (input_mode == BinaryInputMode::STREAMING) {
                        tile_regs_commit();
                        tile_regs_wait();

                        cb_reserve_back(ocb, onetile);
                        pack_tile(base_dst, ocb);
                        cb_push_back(ocb, onetile);

                        cb_pop_front(icb_a, onetile);

                        tile_regs_release();

                        if (wt < chunk_size - 1) {
                            tile_regs_acquire();
                        }
                    }
                }

                if constexpr (!std::is_same_v<PostOp, NoOp> && input_mode != BinaryInputMode::STREAMING) {
                    for (uint32_t i = 0; i < chunk_size; ++i) {
                        post_op(base_dst + i);
                    }
                }

                if constexpr (input_mode != BinaryInputMode::STREAMING) {
                    tile_regs_commit();
                    tile_regs_wait();

                    if constexpr (input_mode != BinaryInputMode::PRELOADED) {
                        cb_reserve_back(ocb, chunk_size);
                    }

                    for (uint32_t i = 0; i < chunk_size; ++i) {
                        pack_tile(base_dst + i, ocb, output_offset + i);
                    }

                    if constexpr (input_mode != BinaryInputMode::PRELOADED) {
                        cb_push_back(ocb, chunk_size);
                    }

                    tile_regs_release();
                }

                if constexpr (input_mode == BinaryInputMode::STREAMING_BATCHED) {
                    cb_pop_front(icb_a, chunk_size);
                }

                output_offset += chunk_size;
            }

            if constexpr (input_mode == BinaryInputMode::PRELOADED || input_mode == BinaryInputMode::PERSISTENT) {
                index_offset_a += stride_a;
            }
        }
    }

    // B tiles are NOT popped - caller may need them for multiple operations
    if constexpr (input_mode == BinaryInputMode::PRELOADED) {
        cb_push_back(ocb, total_tiles_a);
    }
}

/**
 * @brief Core binary operation logic for BroadcastDim::COL
 *
 * Column broadcast: C[h,w] = A[h,w] op B[h]
 * B has shape [Ht, 1] and broadcasts across all columns of A.
 */
template <BinaryOpType op_type, BinaryInputMode input_mode, bool init, typename AccumT, typename PostOp>
ALWI void binary_op_col(
    uint32_t icb_a,
    uint32_t icb_b,
    uint32_t ocb,
    BinaryTileShape shape,
    BinaryTileLayout layout,
    AccumT accum,
    PostOp post_op) {
    constexpr uint32_t dest_limit = DEST_AUTO_LIMIT;
    constexpr uint32_t onetile = 1;

    const uint32_t Ht = shape.rows;
    const uint32_t Wt = shape.cols;
    const uint32_t num_batches = shape.batches;
    const uint32_t total_tiles_a = Ht * Wt * num_batches;
    const uint32_t stride_a = (layout.row_stride_a > 0) ? layout.row_stride_a : Wt;

    if constexpr (init) {
        binary_init_col<op_type>(icb_a, icb_b);
    }

    if constexpr (input_mode == BinaryInputMode::PERSISTENT) {
        cb_wait_front(icb_a, total_tiles_a);
        cb_wait_front(icb_b, Ht * num_batches);  // Ht tiles per batch for col broadcast
    }

    if constexpr (input_mode == BinaryInputMode::PRELOADED) {
        cb_reserve_back(ocb, total_tiles_a);
    }

    uint32_t index_offset_a = 0;
    uint32_t index_offset_b = 0;
    uint32_t output_offset = 0;

    // Account for base_dst offset when calculating effective DEST capacity
    const uint32_t base_dst = get_binary_dst_index(accum);
    const uint32_t effective_dest_limit = dest_limit - base_dst;

    for (uint32_t nc = 0; nc < num_batches; ++nc) {
        for (uint32_t ht = 0; ht < Ht; ++ht) {
            // For col broadcast, wait for one B tile per row
            if constexpr (
                input_mode == BinaryInputMode::STREAMING || input_mode == BinaryInputMode::STREAMING_BATCHED) {
                cb_wait_front(icb_b, onetile);
            }

            for (uint32_t wt_base = 0; wt_base < Wt; wt_base += effective_dest_limit) {
                const uint32_t chunk_size =
                    (wt_base + effective_dest_limit <= Wt) ? effective_dest_limit : (Wt - wt_base);

                if constexpr (input_mode == BinaryInputMode::STREAMING_BATCHED) {
                    cb_wait_front(icb_a, chunk_size);
                }

                tile_regs_acquire();

                reload_accumulator_if_needed_binary<op_type, BroadcastDim::COL>(icb_a, icb_b, accum);

                for (uint32_t wt = 0; wt < chunk_size; ++wt) {
                    uint32_t tile_a, tile_b, dst_idx;

                    if constexpr (input_mode == BinaryInputMode::STREAMING) {
                        cb_wait_front(icb_a, onetile);
                        tile_a = 0;
                        tile_b = 0;  // Same B tile for entire row
                        dst_idx = base_dst;
                    } else if constexpr (input_mode == BinaryInputMode::STREAMING_BATCHED) {
                        tile_a = wt;
                        tile_b = 0;
                        dst_idx = base_dst + wt;
                    } else {
                        tile_a = index_offset_a + wt_base + wt;
                        tile_b = index_offset_b;
                        dst_idx = base_dst + wt;
                    }

                    binary_exec_col<op_type>(icb_a, icb_b, tile_a, tile_b, dst_idx);

                    if constexpr (input_mode == BinaryInputMode::STREAMING) {
                        tile_regs_commit();
                        tile_regs_wait();

                        cb_reserve_back(ocb, onetile);
                        pack_tile(base_dst, ocb);
                        cb_push_back(ocb, onetile);

                        cb_pop_front(icb_a, onetile);

                        tile_regs_release();

                        if (wt < chunk_size - 1) {
                            tile_regs_acquire();
                        }
                    }
                }

                if constexpr (!std::is_same_v<PostOp, NoOp> && input_mode != BinaryInputMode::STREAMING) {
                    for (uint32_t i = 0; i < chunk_size; ++i) {
                        post_op(base_dst + i);
                    }
                }

                if constexpr (input_mode != BinaryInputMode::STREAMING) {
                    tile_regs_commit();
                    tile_regs_wait();

                    if constexpr (input_mode != BinaryInputMode::PRELOADED) {
                        cb_reserve_back(ocb, chunk_size);
                    }

                    for (uint32_t i = 0; i < chunk_size; ++i) {
                        pack_tile(base_dst + i, ocb, output_offset + i);
                    }

                    if constexpr (input_mode != BinaryInputMode::PRELOADED) {
                        cb_push_back(ocb, chunk_size);
                    }

                    tile_regs_release();
                }

                if constexpr (input_mode == BinaryInputMode::STREAMING_BATCHED) {
                    cb_pop_front(icb_a, chunk_size);
                }

                output_offset += chunk_size;
            }

            // Pop B tile after processing entire row
            if constexpr (
                input_mode == BinaryInputMode::STREAMING || input_mode == BinaryInputMode::STREAMING_BATCHED) {
                cb_pop_front(icb_b, onetile);
            }

            if constexpr (input_mode == BinaryInputMode::PRELOADED || input_mode == BinaryInputMode::PERSISTENT) {
                index_offset_a += stride_a;
                index_offset_b += 1;  // One B tile per row
            }
        }
    }

    if constexpr (input_mode == BinaryInputMode::PRELOADED) {
        cb_push_back(ocb, total_tiles_a);
    }
}

/**
 * @brief Core binary operation logic for BroadcastDim::SCALAR
 *
 * Scalar broadcast: C[h,w] = A[h,w] op B[0,0]
 * B has shape [1, 1] and broadcasts to all elements of A.
 */
template <BinaryOpType op_type, BinaryInputMode input_mode, bool init, typename AccumT, typename PostOp>
ALWI void binary_op_scalar(
    uint32_t icb_a,
    uint32_t icb_b,
    uint32_t ocb,
    BinaryTileShape shape,
    BinaryTileLayout layout,
    AccumT accum,
    PostOp post_op) {
    constexpr uint32_t dest_limit = DEST_AUTO_LIMIT;
    constexpr uint32_t onetile = 1;

    const uint32_t Ht = shape.rows;
    const uint32_t Wt = shape.cols;
    const uint32_t num_batches = shape.batches;
    const uint32_t total_tiles_a = Ht * Wt * num_batches;
    const uint32_t stride_a = (layout.row_stride_a > 0) ? layout.row_stride_a : Wt;

    if constexpr (init) {
        binary_init_scalar<op_type>(icb_a, icb_b);
    }

    if constexpr (input_mode == BinaryInputMode::PERSISTENT) {
        cb_wait_front(icb_a, total_tiles_a);
        cb_wait_front(icb_b, onetile);
    }

    if constexpr (input_mode == BinaryInputMode::PRELOADED) {
        cb_reserve_back(ocb, total_tiles_a);
    }

    // For scalar broadcast, wait for single B tile at start
    if constexpr (input_mode == BinaryInputMode::STREAMING || input_mode == BinaryInputMode::STREAMING_BATCHED) {
        cb_wait_front(icb_b, onetile);
    }

    uint32_t index_offset_a = 0;
    uint32_t output_offset = 0;

    // Account for base_dst offset when calculating effective DEST capacity
    const uint32_t base_dst = get_binary_dst_index(accum);
    const uint32_t effective_dest_limit = dest_limit - base_dst;

    for (uint32_t nc = 0; nc < num_batches; ++nc) {
        for (uint32_t ht = 0; ht < Ht; ++ht) {
            for (uint32_t wt_base = 0; wt_base < Wt; wt_base += effective_dest_limit) {
                const uint32_t chunk_size =
                    (wt_base + effective_dest_limit <= Wt) ? effective_dest_limit : (Wt - wt_base);

                if constexpr (input_mode == BinaryInputMode::STREAMING_BATCHED) {
                    cb_wait_front(icb_a, chunk_size);
                }

                tile_regs_acquire();

                reload_accumulator_if_needed_binary<op_type, BroadcastDim::SCALAR>(icb_a, icb_b, accum);

                for (uint32_t wt = 0; wt < chunk_size; ++wt) {
                    uint32_t tile_a, tile_b, dst_idx;

                    if constexpr (input_mode == BinaryInputMode::STREAMING) {
                        cb_wait_front(icb_a, onetile);
                        tile_a = 0;
                        tile_b = 0;  // Always same scalar tile
                        dst_idx = base_dst;
                    } else if constexpr (input_mode == BinaryInputMode::STREAMING_BATCHED) {
                        tile_a = wt;
                        tile_b = 0;
                        dst_idx = base_dst + wt;
                    } else {
                        tile_a = index_offset_a + wt_base + wt;
                        tile_b = 0;
                        dst_idx = base_dst + wt;
                    }

                    binary_exec_scalar<op_type>(icb_a, icb_b, tile_a, tile_b, dst_idx);

                    if constexpr (input_mode == BinaryInputMode::STREAMING) {
                        tile_regs_commit();
                        tile_regs_wait();

                        cb_reserve_back(ocb, onetile);
                        pack_tile(base_dst, ocb);
                        cb_push_back(ocb, onetile);

                        cb_pop_front(icb_a, onetile);

                        tile_regs_release();

                        if (wt < chunk_size - 1) {
                            tile_regs_acquire();
                        }
                    }
                }

                if constexpr (!std::is_same_v<PostOp, NoOp> && input_mode != BinaryInputMode::STREAMING) {
                    for (uint32_t i = 0; i < chunk_size; ++i) {
                        post_op(base_dst + i);
                    }
                }

                if constexpr (input_mode != BinaryInputMode::STREAMING) {
                    tile_regs_commit();
                    tile_regs_wait();

                    if constexpr (input_mode != BinaryInputMode::PRELOADED) {
                        cb_reserve_back(ocb, chunk_size);
                    }

                    for (uint32_t i = 0; i < chunk_size; ++i) {
                        pack_tile(base_dst + i, ocb, output_offset + i);
                    }

                    if constexpr (input_mode != BinaryInputMode::PRELOADED) {
                        cb_push_back(ocb, chunk_size);
                    }

                    tile_regs_release();
                }

                if constexpr (input_mode == BinaryInputMode::STREAMING_BATCHED) {
                    cb_pop_front(icb_a, chunk_size);
                }

                output_offset += chunk_size;
            }

            if constexpr (input_mode == BinaryInputMode::PRELOADED || input_mode == BinaryInputMode::PERSISTENT) {
                index_offset_a += stride_a;
            }
        }
    }

    // B scalar tile is NOT popped - caller may need it
    if constexpr (input_mode == BinaryInputMode::PRELOADED) {
        cb_push_back(ocb, total_tiles_a);
    }
}

// =============================================================================
// Main Binary Operation Function
// =============================================================================

/**
 * @brief Unified binary operation function handling all patterns
 *
 * This single function handles:
 * - Element-wise add/sub/mul
 * - Broadcast variants (row, col, scalar)
 * - All input modes (streaming, batched, preloaded, persistent)
 * - Optional accumulation
 * - Data format reconfiguration
 * - Post-operation callbacks
 * - DEST register limit management (auto-detected via DEST_AUTO_LIMIT)
 *
 * @tparam op_type Binary operation type (ADD, SUB, MUL)
 * @tparam bcast_dim Broadcast dimension (NONE, ROW, COL, SCALAR)
 * @tparam input_mode Input handling mode (STREAMING, STREAMING_BATCHED, etc.)
 * @tparam reconfig Data format reconfiguration mode
 * @tparam init If true, calls init before processing
 * @tparam uninit If true, calls uninit after processing (reserved for future)
 * @tparam AccumT Accumulation type (NoAccumulation or BinaryAccumulate)
 * @tparam PostOp Post-operation callback type
 *
 * @param icb_a Input CB for first operand
 * @param icb_b Input CB for second operand
 * @param ocb Output CB for result
 * @param shape Tile grid dimensions
 * @param layout Tile memory layout (default: contiguous)
 * @param accum Accumulation config (default: no accumulation)
 * @param post_op Post-operation callback (default: no-op)
 */
template <
    BinaryOpType op_type,
    BroadcastDim bcast_dim = BroadcastDim::NONE,
    BinaryInputMode input_mode = BinaryInputMode::STREAMING,
    BinaryDataFormatReconfig reconfig = BinaryDataFormatReconfig::BOTH,
    bool init = true,
    bool uninit = true,
    typename AccumT = NoAccumulation,
    typename PostOp = NoOp>
ALWI void binary_op(
    uint32_t icb_a,
    uint32_t icb_b,
    uint32_t ocb,
    BinaryTileShape shape,
    BinaryTileLayout layout = {},
    AccumT accum = {},
    PostOp post_op = {}) {
    // Apply data format reconfiguration
    if constexpr (reconfig == BinaryDataFormatReconfig::INPUT || reconfig == BinaryDataFormatReconfig::BOTH) {
        reconfig_data_format(icb_a, icb_b);
    }
    if constexpr (reconfig == BinaryDataFormatReconfig::OUTPUT || reconfig == BinaryDataFormatReconfig::BOTH) {
        pack_reconfig_data_format(ocb);
    }

    // Dispatch to broadcast-specific implementation
    // Each branch is a separate function for easier debugging
    if constexpr (bcast_dim == BroadcastDim::NONE) {
        binary_op_none<op_type, input_mode, init, AccumT, PostOp>(icb_a, icb_b, ocb, shape, layout, accum, post_op);
    } else if constexpr (bcast_dim == BroadcastDim::ROW) {
        binary_op_row<op_type, input_mode, init, AccumT, PostOp>(icb_a, icb_b, ocb, shape, layout, accum, post_op);
    } else if constexpr (bcast_dim == BroadcastDim::COL) {
        binary_op_col<op_type, input_mode, init, AccumT, PostOp>(icb_a, icb_b, ocb, shape, layout, accum, post_op);
    } else {
        binary_op_scalar<op_type, input_mode, init, AccumT, PostOp>(icb_a, icb_b, ocb, shape, layout, accum, post_op);
    }

    // Cleanup (binary ops don't have explicit uninit, but maintain hook for consistency)
    if constexpr (uninit) {
        // Reserved for future use
    }
}

// =============================================================================
// Convenience Aliases
// =============================================================================

/**
 * @brief Add tiles helper - alias for binary_op<ADD, ...>
 */
template <
    BroadcastDim bcast_dim = BroadcastDim::NONE,
    BinaryInputMode input_mode = BinaryInputMode::STREAMING,
    BinaryDataFormatReconfig reconfig = BinaryDataFormatReconfig::BOTH,
    bool init = true,
    bool uninit = true,
    typename AccumT = NoAccumulation,
    typename PostOp = NoOp>
ALWI void add(
    uint32_t icb_a,
    uint32_t icb_b,
    uint32_t ocb,
    BinaryTileShape shape,
    BinaryTileLayout layout = {},
    AccumT accum = {},
    PostOp post_op = {}) {
    binary_op<BinaryOpType::ADD, bcast_dim, input_mode, reconfig, init, uninit, AccumT, PostOp>(
        icb_a, icb_b, ocb, shape, layout, accum, post_op);
}

/**
 * @brief Sub tiles helper - alias for binary_op<SUB, ...>
 */
template <
    BroadcastDim bcast_dim = BroadcastDim::NONE,
    BinaryInputMode input_mode = BinaryInputMode::STREAMING,
    BinaryDataFormatReconfig reconfig = BinaryDataFormatReconfig::BOTH,
    bool init = true,
    bool uninit = true,
    typename AccumT = NoAccumulation,
    typename PostOp = NoOp>
ALWI void sub(
    uint32_t icb_a,
    uint32_t icb_b,
    uint32_t ocb,
    BinaryTileShape shape,
    BinaryTileLayout layout = {},
    AccumT accum = {},
    PostOp post_op = {}) {
    binary_op<BinaryOpType::SUB, bcast_dim, input_mode, reconfig, init, uninit, AccumT, PostOp>(
        icb_a, icb_b, ocb, shape, layout, accum, post_op);
}

/**
 * @brief Mul tiles helper - alias for binary_op<MUL, ...>
 */
template <
    BroadcastDim bcast_dim = BroadcastDim::NONE,
    BinaryInputMode input_mode = BinaryInputMode::STREAMING,
    BinaryDataFormatReconfig reconfig = BinaryDataFormatReconfig::BOTH,
    bool init = true,
    bool uninit = true,
    typename AccumT = NoAccumulation,
    typename PostOp = NoOp>
ALWI void mul(
    uint32_t icb_a,
    uint32_t icb_b,
    uint32_t ocb,
    BinaryTileShape shape,
    BinaryTileLayout layout = {},
    AccumT accum = {},
    PostOp post_op = {}) {
    binary_op<BinaryOpType::MUL, bcast_dim, input_mode, reconfig, init, uninit, AccumT, PostOp>(
        icb_a, icb_b, ocb, shape, layout, accum, post_op);
}

}  // namespace compute_kernel_lib
