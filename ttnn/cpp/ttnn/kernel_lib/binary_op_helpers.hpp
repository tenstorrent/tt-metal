// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>

#include "api/compute/eltwise_binary.h"
#include "ttnn/cpp/ttnn/kernel_lib/cb_policies.hpp"
#include "api/compute/bcast.h"
#include "api/compute/cb_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/common_types.hpp"

/**
 * @file binary_op_helpers.hpp
 * @brief Unified binary operations with direct LLK calls - massively simplified from 1399 to ~500 lines
 *
 * ## Broadcast Dimension Reference
 *
 * BroadcastDim specifies the SHAPE of operand B and how it broadcasts to match A:
 *
 * | BroadcastDim | B Shape   | B Tiles | B Replicates | What Gets Broadcast                    |
 * |--------------|-----------|---------|--------------|----------------------------------------|
 * | NONE         | [Ht, Wt]  | Ht * Wt | (none)       | All elements used as-is                |
 * | ROW          | [1, Wt]   | Wt      | Down (↓)     | Top row of each tile                   |
 * | COL          | [Ht, 1]   | Ht      | Right (→)    | Leftmost column of each tile           |
 * | SCALAR       | [1, 1]    | 1       | Both (↓→)    | Top-left element of the single tile    |
 *
 * ## Relationship to Reduce Operations
 *
 * After reduction, use the corresponding broadcast to apply the result:
 *
 * | Reduce Operation | Output Shape | Use Broadcast | Example                     |
 * |------------------|--------------|---------------|-----------------------------|
 * | REDUCE_ROW       | [Ht, 1]      | COL           | Subtract row-wise mean      |
 * | REDUCE_COL       | [1, Wt]      | ROW           | Subtract column-wise mean   |
 * | REDUCE_SCALAR    | [1, 1]       | SCALAR        | Subtract global mean        |
 *
 * Note: REDUCE_ROW produces COL-shaped output (this is correct but counterintuitive).
 * "REDUCE_ROW" means "reduce along row direction" = sum across width = column output.
 */

namespace compute_kernel_lib {

enum class BinaryOpType { ADD, SUB, MUL, SQUARE };
enum class BroadcastDim { NONE, ROW, COL, SCALAR };
enum class BinaryDataFormatReconfig { NONE = 0, INPUT = 1, OUTPUT = 2, BOTH = 3 };

struct BinaryTileShape {
    uint32_t rows, cols;
    static constexpr BinaryTileShape single() { return {1, 1}; }
    static constexpr BinaryTileShape row(uint32_t cols) { return {1, cols}; }
    static constexpr BinaryTileShape col(uint32_t rows) { return {rows, 1}; }
    static constexpr BinaryTileShape grid(uint32_t rows, uint32_t cols) { return {rows, cols}; }
};

struct BinaryTileLayout {
    uint32_t row_major_a = 1, row_major_b = 1;
};

// Use types from common_types.hpp (NoAccumulation, NoOp already defined there)
struct BinaryAccumulate {
    uint32_t cb_accumulator, dst_index = 0;
};

template <BinaryOpType op_type>
constexpr EltwiseBinaryType map_to_eltwise_type() {
    return op_type == BinaryOpType::ADD ? ELWADD : op_type == BinaryOpType::SUB ? ELWSUB : ELWMUL;
}

template <BroadcastDim bcast_dim>
constexpr BroadcastType map_to_broadcast_type() {
    return bcast_dim == BroadcastDim::NONE  ? BroadcastType::NONE
           : bcast_dim == BroadcastDim::ROW ? BroadcastType::ROW
           : bcast_dim == BroadcastDim::COL ? BroadcastType::COL
                                            : BroadcastType::SCALAR;
}

template <BroadcastDim bcast_dim>
ALWI constexpr uint32_t get_b_tile_count(uint32_t Ht, uint32_t Wt) {
    return bcast_dim == BroadcastDim::SCALAR ? 1
           : bcast_dim == BroadcastDim::ROW  ? Wt
           : bcast_dim == BroadcastDim::COL  ? Ht
                                             : Ht * Wt;
}

template <typename AccumT>
ALWI constexpr bool is_accumulator_enabled() {
    return !std::is_same_v<AccumT, NoAccumulation>;
}

template <typename AccumT>
ALWI constexpr uint32_t get_binary_dst_index(AccumT accum) {
    if constexpr (is_accumulator_enabled<AccumT>()) {
        return accum.dst_index;
    } else {
        return 0;
    }
}

// =============================================================================
// Unified LLK Calls - Single Init and Exec for All Broadcast Modes
// =============================================================================

template <BinaryOpType op_type, BroadcastDim bcast_dim>
ALWI void binary_init(uint32_t icb_a, uint32_t icb_b) {
    constexpr EltwiseBinaryType elt_type = map_to_eltwise_type<op_type>();
    constexpr BroadcastType bcast_type = map_to_broadcast_type<bcast_dim>();

    MATH((llk_math_eltwise_binary_init_with_operands<elt_type, bcast_type, MATH_FIDELITY>(icb_a, icb_b)));
    UNPACK((llk_unpack_AB_init<bcast_type>(icb_a, icb_b)));
}

template <BinaryOpType op_type, BroadcastDim bcast_dim>
ALWI void binary_exec(uint32_t icb_a, uint32_t icb_b, uint32_t itile_a, uint32_t itile_b, uint32_t idst) {
    constexpr EltwiseBinaryType elt_type = map_to_eltwise_type<op_type>();
    constexpr BroadcastType bcast_type = map_to_broadcast_type<bcast_dim>();

    UNPACK((llk_unpack_AB<bcast_type>(icb_a, icb_b, itile_a, itile_b)));
    MATH(
        (llk_math_eltwise_binary<elt_type, bcast_type, DST_ACCUM_MODE, MATH_FIDELITY, EltwiseBinaryReuseDestType::NONE>(
            icb_a, icb_b, idst, true)));
}

// =============================================================================
// Unified Binary Operation - All Broadcast Dimensions in One Function
// =============================================================================

template <
    BinaryOpType op_type,
    BroadcastDim bcast_dim = BroadcastDim::NONE,
    typename InputAPolicy = cb_policies::Streaming,
    typename InputBPolicy = InputAPolicy,
    typename OutputPolicy = cb_policies::OutputPerTile,
    BinaryDataFormatReconfig reconfig = BinaryDataFormatReconfig::BOTH,
    bool init = true,
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
    constexpr uint32_t onetile = 1;
    constexpr uint32_t dest_limit = DEST_AUTO_LIMIT;
    constexpr bool is_square = (op_type == BinaryOpType::SQUARE);

    const uint32_t Ht = shape.rows;
    const uint32_t Wt = shape.cols;
    const uint32_t total_tiles_a = Ht * Wt;
    const uint32_t b_tile_count = get_b_tile_count<bcast_dim>(Ht, Wt);

    // Data format reconfiguration (copy original logic exactly)
    if constexpr (reconfig == BinaryDataFormatReconfig::INPUT || reconfig == BinaryDataFormatReconfig::BOTH) {
        reconfig_data_format(icb_a, icb_b);
    }
    if constexpr (reconfig == BinaryDataFormatReconfig::OUTPUT || reconfig == BinaryDataFormatReconfig::BOTH) {
        pack_reconfig_data_format(ocb);
    }

    // Initialization
    if constexpr (init) {
        binary_init<op_type, bcast_dim>(icb_a, icb_b);
    }

    // Upfront waits
    if constexpr (InputAPolicy::waits_upfront) {
        cb_wait_front(icb_a, total_tiles_a);
    }

    // B policy: ROW and SCALAR always wait upfront
    if constexpr (!is_square) {
        if constexpr (bcast_dim == BroadcastDim::ROW || bcast_dim == BroadcastDim::SCALAR) {
            if constexpr (!InputBPolicy::waits_caller_managed) {
                cb_wait_front(icb_b, b_tile_count);
            }
        } else if constexpr (InputBPolicy::waits_upfront) {
            cb_wait_front(icb_b, b_tile_count);
        }
    }

    // Upfront output reserve
    if constexpr (OutputPolicy::bulk) {
        cb_reserve_back(ocb, total_tiles_a);
    }

    const uint32_t base_dst = get_binary_dst_index(accum);
    const uint32_t effective_dest_limit = dest_limit - base_dst;
    uint32_t tiles_processed = 0;

    for (uint32_t ht = 0; ht < Ht; ++ht) {
        for (uint32_t wt_base = 0; wt_base < Wt; wt_base += effective_dest_limit) {
            const uint32_t chunk_size = (wt_base + effective_dest_limit <= Wt) ? effective_dest_limit : (Wt - wt_base);

            // Per-chunk waits
            if constexpr (InputAPolicy::waits_per_chunk) {
                cb_wait_front(icb_a, chunk_size);
            }
            if constexpr (!is_square && InputBPolicy::waits_per_chunk) {
                if constexpr (bcast_dim == BroadcastDim::NONE) {
                    cb_wait_front(icb_b, chunk_size);
                } else if constexpr (bcast_dim == BroadcastDim::COL) {
                    cb_wait_front(icb_b, onetile);
                }
            }

            if constexpr (OutputPolicy::per_chunk) {
                cb_reserve_back(ocb, chunk_size);
            }

            tile_regs_acquire();

            // Accumulator reload if needed
            if constexpr (is_accumulator_enabled<AccumT>()) {
                cb_wait_front(accum.cb_accumulator, 1);
                copy_tile(accum.cb_accumulator, 0, accum.dst_index);
                cb_pop_front(accum.cb_accumulator, 1);
                binary_init<op_type, bcast_dim>(icb_a, icb_b);
            }

            for (uint32_t wt = 0; wt < chunk_size; ++wt) {
                // Per-tile waits
                if constexpr (InputAPolicy::waits_per_tile) {
                    cb_wait_front(icb_a, onetile);
                }
                if constexpr (!is_square && InputBPolicy::waits_per_tile) {
                    if constexpr (bcast_dim == BroadcastDim::NONE || bcast_dim == BroadcastDim::COL) {
                        cb_wait_front(icb_b, onetile);
                    }
                }

                // Tile indices
                uint32_t tile_a, tile_b, dst_idx;

                if constexpr (InputAPolicy::waits_per_tile) {
                    tile_a = 0;
                    dst_idx = base_dst;
                } else if constexpr (InputAPolicy::waits_per_chunk) {
                    tile_a = wt;
                    dst_idx = base_dst + wt;
                } else {
                    tile_a = ht * Wt + wt_base + wt;
                    dst_idx = base_dst + wt;
                }

                if constexpr (is_square) {
                    tile_b = tile_a;
                } else if constexpr (bcast_dim == BroadcastDim::SCALAR) {
                    tile_b = 0;
                } else if constexpr (bcast_dim == BroadcastDim::ROW) {
                    tile_b = wt_base + wt;
                } else if constexpr (bcast_dim == BroadcastDim::COL) {
                    tile_b = InputBPolicy::waits_per_tile ? 0 : ht;
                } else {  // NONE
                    if constexpr (InputBPolicy::waits_per_tile) {
                        tile_b = 0;
                    } else if constexpr (InputBPolicy::waits_per_chunk) {
                        tile_b = wt;
                    } else {
                        tile_b = ht * Wt + wt_base + wt;
                    }
                }

                // Execute (unified LLK call)
                binary_exec<op_type, bcast_dim>(icb_a, icb_b, tile_a, tile_b, dst_idx);

                // Per-tile streaming
                if constexpr (InputAPolicy::waits_per_tile) {
                    tile_regs_commit();
                    tile_regs_wait();

                    if constexpr (OutputPolicy::per_tile) {
                        cb_reserve_back(ocb, onetile);
                        pack_tile(base_dst, ocb);
                        cb_push_back(ocb, onetile);
                    } else if constexpr (OutputPolicy::per_chunk) {
                        pack_tile(base_dst, ocb, wt);
                    } else {
                        pack_tile(base_dst, ocb, tiles_processed);
                    }

                    if constexpr (InputAPolicy::pops_per_tile) {
                        cb_pop_front(icb_a, onetile);
                    }
                    if constexpr (!is_square && InputBPolicy::pops_per_tile) {
                        if constexpr (bcast_dim == BroadcastDim::NONE || bcast_dim == BroadcastDim::COL) {
                            cb_pop_front(icb_b, onetile);
                        }
                    }

                    tile_regs_acquire();
                    tiles_processed++;
                }
            }

            // Per-chunk commit/pack/pop
            if constexpr (!InputAPolicy::waits_per_tile) {
                tile_regs_commit();
                tile_regs_wait();

                if constexpr (OutputPolicy::per_chunk) {
                    for (uint32_t wt = 0; wt < chunk_size; ++wt) {
                        pack_tile(base_dst + wt, ocb, wt);
                    }
                    cb_push_back(ocb, chunk_size);
                } else if constexpr (OutputPolicy::bulk) {
                    for (uint32_t wt = 0; wt < chunk_size; ++wt) {
                        pack_tile(base_dst + wt, ocb, tiles_processed + wt);
                    }
                } else {
                    for (uint32_t wt = 0; wt < chunk_size; ++wt) {
                        cb_reserve_back(ocb, onetile);
                        pack_tile(base_dst + wt, ocb);
                        cb_push_back(ocb, onetile);
                    }
                }

                tiles_processed += chunk_size;
            }

            if constexpr (InputAPolicy::pops_per_chunk) {
                cb_pop_front(icb_a, chunk_size);
            }
            if constexpr (!is_square && InputBPolicy::pops_per_chunk) {
                if constexpr (bcast_dim == BroadcastDim::NONE) {
                    cb_pop_front(icb_b, chunk_size);
                } else if constexpr (bcast_dim == BroadcastDim::COL) {
                    cb_pop_front(icb_b, onetile);
                }
            }

            tile_regs_release();
        }
    }

    // Bulk output push
    if constexpr (OutputPolicy::bulk) {
        cb_push_back(ocb, total_tiles_a);
    }

    // At-end pops
    if constexpr (InputAPolicy::pops_at_end) {
        cb_pop_front(icb_a, total_tiles_a);
    }
    if constexpr (!is_square && InputBPolicy::pops_at_end) {
        cb_pop_front(icb_b, b_tile_count);
    }

    // B pop for ROW/SCALAR (unless caller-managed or never)
    if constexpr (!is_square) {
        if constexpr (bcast_dim == BroadcastDim::ROW || bcast_dim == BroadcastDim::SCALAR) {
            if constexpr (!InputBPolicy::pops_caller_managed && !InputBPolicy::pops_never) {
                cb_pop_front(icb_b, b_tile_count);
            }
        }
    }
}

// =============================================================================
// Convenience Aliases
// =============================================================================

template <BroadcastDim bcast_dim = BroadcastDim::NONE, typename... Args>
ALWI void add(uint32_t icb_a, uint32_t icb_b, uint32_t ocb, BinaryTileShape shape, Args&&... args) {
    binary_op<BinaryOpType::ADD, bcast_dim>(icb_a, icb_b, ocb, shape, std::forward<Args>(args)...);
}

template <BroadcastDim bcast_dim = BroadcastDim::NONE, typename... Args>
ALWI void sub(uint32_t icb_a, uint32_t icb_b, uint32_t ocb, BinaryTileShape shape, Args&&... args) {
    binary_op<BinaryOpType::SUB, bcast_dim>(icb_a, icb_b, ocb, shape, std::forward<Args>(args)...);
}

template <BroadcastDim bcast_dim = BroadcastDim::NONE, typename... Args>
ALWI void mul(uint32_t icb_a, uint32_t icb_b, uint32_t ocb, BinaryTileShape shape, Args&&... args) {
    binary_op<BinaryOpType::MUL, bcast_dim>(icb_a, icb_b, ocb, shape, std::forward<Args>(args)...);
}

template <typename... Args>
ALWI void square(uint32_t icb, uint32_t ocb, BinaryTileShape shape, Args&&... args) {
    binary_op<BinaryOpType::SQUARE, BroadcastDim::NONE>(icb, icb, ocb, shape, std::forward<Args>(args)...);
}

// Advanced with full policy control
template <
    BroadcastDim bcast_dim,
    typename InputAPolicy,
    typename InputBPolicy = InputAPolicy,
    typename OutputPolicy = cb_policies::OutputPerTile,
    typename... Args>
ALWI void add(uint32_t icb_a, uint32_t icb_b, uint32_t ocb, BinaryTileShape shape, Args&&... args) {
    binary_op<BinaryOpType::ADD, bcast_dim, InputAPolicy, InputBPolicy, OutputPolicy>(
        icb_a, icb_b, ocb, shape, std::forward<Args>(args)...);
}

template <
    BroadcastDim bcast_dim,
    typename InputAPolicy,
    typename InputBPolicy = InputAPolicy,
    typename OutputPolicy = cb_policies::OutputPerTile,
    typename... Args>
ALWI void sub(uint32_t icb_a, uint32_t icb_b, uint32_t ocb, BinaryTileShape shape, Args&&... args) {
    binary_op<BinaryOpType::SUB, bcast_dim, InputAPolicy, InputBPolicy, OutputPolicy>(
        icb_a, icb_b, ocb, shape, std::forward<Args>(args)...);
}

template <
    BroadcastDim bcast_dim,
    typename InputAPolicy,
    typename InputBPolicy = InputAPolicy,
    typename OutputPolicy = cb_policies::OutputPerTile,
    typename... Args>
ALWI void mul(uint32_t icb_a, uint32_t icb_b, uint32_t ocb, BinaryTileShape shape, Args&&... args) {
    binary_op<BinaryOpType::MUL, bcast_dim, InputAPolicy, InputBPolicy, OutputPolicy>(
        icb_a, icb_b, ocb, shape, std::forward<Args>(args)...);
}

}  // namespace compute_kernel_lib
