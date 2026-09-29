// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/bcast.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tilize.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

template <uint32_t block_ct, uint32_t Mt>
TT_KERNEL void compute(uint32_t wi_start, uint32_t wi_count) {
    // Kimi-K3 uses a fixed four-tap causal convolution, with three preceding rows supplied by history.
    constexpr uint32_t tap_count = 4;
    // Group up to four channel tiles per destination acquire; the host sizes partial for two blocks, so a
    // dst_tiles reservation never waits on the previous tap's block that this loop is still reading.
    constexpr uint32_t dst_tiles = block_ct % 4 == 0 ? 4 : (block_ct % 2 == 0 ? 2 : 1);
    static_assert(dst_tiles <= compute_kernel_lib::DEST_AUTO_LIMIT, "dst_tiles must fit the destination registers");
#ifdef KDA_CONV_TILED_INPUT
    // The reader delivers each shifted tap view already tiled.
    compute_kernel_hw_startup(dfb::act_tile, dfb::weights, dfb::output);
#else
    compute_kernel_hw_startup(dfb::act_rm, dfb::act_tile, dfb::output);
#endif
    DataflowBuffer activation(dfb::act_tile);
    DataflowBuffer weights(dfb::weights);
    DataflowBuffer partial(dfb::partial);
    DataflowBuffer output(dfb::output);
    silu_tile_init();

    // Work items are channel-block-major; the reader queues tap weights once per run of items
    // that share a channel block.
    for (uint32_t item = 0; item < wi_count; ++item) {
        const uint32_t work = wi_start + item;
        const uint32_t block = work / Mt;
        if (item == 0 || (work - 1) / Mt != block) {
            weights.wait_front(tap_count * block_ct);
        }
        for (uint32_t tap = 0; tap < tap_count; ++tap) {
#ifndef KDA_CONV_TILED_INPUT
            compute_kernel_lib::tilize<block_ct, dfb::act_rm, dfb::act_tile>(1);
#endif
            activation.wait_front(block_ct);

            const bool is_final_tap = tap + 1 == tap_count;
            const uint32_t destination_dfb = is_final_tap ? dfb::output : dfb::partial;
            DataflowBuffer& destination = is_final_tap ? output : partial;
            if (tap != 0) {
                partial.wait_front(block_ct);
            }

            // Process dst_tiles channel tiles per destination acquire; every tile still sees the
            // same multiply, BF16 partial add and final SiLU as a one-tile loop.
            for (uint32_t ct = 0; ct < block_ct; ct += dst_tiles) {
                destination.reserve_back(dst_tiles);
                tile_regs_acquire();
                reconfig_data_format_srca(dfb::act_tile);
                reconfig_data_format_srcb(dfb::weights);
                mul_bcast_rows_init(dfb::act_tile, dfb::weights);
                for (uint32_t i = 0; i < dst_tiles; ++i) {
                    mul_tiles_bcast_rows(dfb::act_tile, dfb::weights, ct + i, tap * block_ct + ct + i, i);
                }
                if (tap != 0) {
                    reconfig_data_format_srca(dfb::partial);
                    add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(dfb::partial);
                    for (uint32_t i = 0; i < dst_tiles; ++i) {
                        add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(dfb::partial, ct + i, i);
                    }
                }
                if (is_final_tap) {
                    for (uint32_t i = 0; i < dst_tiles; ++i) {
                        silu_tile(i);
                    }
                }
                tile_regs_commit();

                tile_regs_wait();
                for (uint32_t i = 0; i < dst_tiles; ++i) {
                    pack_tile(i, destination_dfb);
                }
                destination.push_back(dst_tiles);
                tile_regs_release();
            }
            if (tap != 0) {
                partial.pop_front(block_ct);
            }
            activation.pop_front(block_ct);
        }
        if (item + 1 == wi_count || (work + 1) / Mt != block) {
            weights.pop_front(tap_count * block_ct);
        }
    }
}
