// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file eltwise_helpers.inl
 * @brief eltwise_op implementation. Included only by eltwise_helpers.hpp.
 */

namespace compute_kernel_lib {

template <
    uint32_t cb_out,
    Dst pack_slot,
    EltwiseOutputPolicy output_policy,
    EltwisePackReconfig pack_reconfig,
    typename Chain>
ALWI void eltwise_op(Chain chain, EltwiseTileShape shape) {
    static_assert(
        static_cast<uint32_t>(pack_slot) < Chain::stride,
        "pack_slot must be less than Chain::stride (max DEST slot used by chain + 1)");

    if constexpr (pack_reconfig == EltwisePackReconfig::Reconfig) {
        pack_reconfig_data_format(cb_out);
    }

    // DestReuseOp clobbers the copy_tile unpack MOP (binary_dest_reuse_tiles_init
    // reconfigures the unpacker each call). Two modes:
    //
    // No DestReuseOp: hoist copy_tile_to_dst_init_short + chain_init_all once before loop.
    // With DestReuseOp: re-call copy_tile_to_dst_init_short + chain_init_all every tile so
    //   each tile starts with a fresh copy MOP before Load::exec, then DestReuseOp::exec
    //   re-inits for dest-reuse at the end of the tile (as existing kernels already do).
    constexpr bool has_dest_reuse = detail::chain_has_dest_reuse_v<Chain>;
    constexpr bool has_load = detail::chain_has_load_v<Chain>;

    if constexpr (!has_dest_reuse) {
        // Standard path: hoist all inits once
        if constexpr (has_load) {
            copy_tile_to_dst_init_short(detail::FirstLoadCB<Chain>::value);
        }
        chain_init_all(chain);
    }

    // Upfront waits for B inputs (broadcast FpuOp / DestReuseOp upfront policies)
    chain_wait_b_upfront(chain, shape);

    // Upfront waits for A inputs (non-streaming FpuOp PolicyA)
    chain_wait_a_upfront(chain, shape);

    const uint32_t total = shape.rows * shape.cols;

    if constexpr (output_policy == EltwiseOutputPolicy::Bulk) {
        cb_reserve_back(cb_out, total);
    }

    for (uint32_t ht = 0; ht < shape.rows; ++ht) {
        for (uint32_t wt = 0; wt < shape.cols; ++wt) {
            if constexpr (has_dest_reuse) {
                // Per-tile path: re-init copy MOP before Load elements run, then
                // DestReuseOp::exec re-inits for dest-reuse at end of tile.
                if constexpr (has_load) {
                    copy_tile_to_dst_init_short(detail::FirstLoadCB<Chain>::value);
                }
                chain_init_all(chain);
            }

            tile_regs_acquire();
            chain_exec_eltwise(chain, ht, wt, shape.cols);
            tile_regs_commit();
            tile_regs_wait();

            if constexpr (output_policy == EltwiseOutputPolicy::PerTile) {
                cb_reserve_back(cb_out, 1);
            }
            pack_tile(static_cast<uint32_t>(pack_slot), cb_out);
            if constexpr (output_policy == EltwiseOutputPolicy::PerTile) {
                cb_push_back(cb_out, 1);
            }

            tile_regs_release();
        }
    }

    if constexpr (output_policy == EltwiseOutputPolicy::Bulk) {
        cb_push_back(cb_out, total);
    }

    // Upfront pops for B and A (consume-mode policies)
    chain_pop_b_upfront(chain, shape);
    chain_pop_a_upfront(chain, shape);

    // Reset advancing tile indices in DestReuseOp (and FpuOp WaitUpfront) for reuse
    chain_reset_tile_idx(chain);
}

}  // namespace compute_kernel_lib
