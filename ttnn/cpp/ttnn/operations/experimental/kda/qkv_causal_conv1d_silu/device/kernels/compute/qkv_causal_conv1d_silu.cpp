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

template <uint32_t block_ct, uint32_t num_blocks>
TT_KERNEL void compute(uint32_t wi_count) {
    // Kimi-K3 uses a fixed four-tap causal convolution, with three preceding rows supplied by history.
    constexpr uint32_t tap_count = 4;
    compute_kernel_hw_startup(dfb::act_rm, dfb::act_tile, dfb::output);
    DataflowBuffer activation(dfb::act_tile);
    DataflowBuffer weights(dfb::weights);
    DataflowBuffer partial(dfb::partial);
    DataflowBuffer output(dfb::output);
    silu_tile_init();

    if constexpr (num_blocks == 1) {
        weights.wait_front(tap_count * block_ct);
    }
    for (uint32_t item = 0; item < wi_count; ++item) {
        if constexpr (num_blocks > 1) {
            weights.wait_front(tap_count * block_ct);
        }
        for (uint32_t tap = 0; tap < tap_count; ++tap) {
            compute_kernel_lib::tilize<block_ct, dfb::act_rm, dfb::act_tile>(1);
            activation.wait_front(block_ct);

            const bool is_final_tap = tap + 1 == tap_count;
            const uint32_t destination_dfb = is_final_tap ? dfb::output : dfb::partial;
            if (tap != 0) {
                partial.wait_front(block_ct);
            }

            reconfig_data_format_srca(dfb::act_tile);
            reconfig_data_format_srcb(dfb::weights);
            if (tap == 0) {
                mul_bcast_rows_init(dfb::act_tile, dfb::weights);
            }
            for (uint32_t ct = 0; ct < block_ct; ++ct) {
                if (is_final_tap) {
                    output.reserve_back(1);
                } else {
                    partial.reserve_back(1);
                }
                tile_regs_acquire();
                if (tap != 0) {
                    mul_bcast_rows_init(dfb::act_tile, dfb::weights);
                }
                mul_tiles_bcast_rows(dfb::act_tile, dfb::weights, ct, tap * block_ct + ct, 0);

                if (tap != 0) {
                    reconfig_data_format_srca(dfb::partial);
                    add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(dfb::partial);
                    add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(dfb::partial, 0, 0);
                    // The partial add binds srcA to the accumulator; restore activation for the next multiply.
                    reconfig_data_format_srca(dfb::act_tile);
                }
                if (is_final_tap) {
                    silu_tile(0);
                }
                tile_regs_commit();

                tile_regs_wait();
                pack_tile(0, destination_dfb);
                if (is_final_tap) {
                    output.push_back(1);
                } else {
                    partial.push_back(1);
                }
                if (tap != 0) {
                    partial.pop_front(1);
                }
                tile_regs_release();
            }
            activation.pop_front(block_ct);
        }
        if constexpr (num_blocks > 1) {
            weights.pop_front(tap_count * block_ct);
        }
    }
}
