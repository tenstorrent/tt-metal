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
    compute_kernel_hw_startup(dfb::act_rm, dfb::act_tile, dfb::output);
    DataflowBuffer activation(dfb::act_tile);
    DataflowBuffer weights(dfb::weights);
    DataflowBuffer partial_a(dfb::partial_a);
    DataflowBuffer partial_b(dfb::partial_b);
    DataflowBuffer output(dfb::output);
    silu_tile_init();

    if constexpr (num_blocks == 1) {
        weights.wait_front(4 * block_ct);
    }
    for (uint32_t item = 0; item < wi_count; ++item) {
        if constexpr (num_blocks > 1) {
            weights.wait_front(4 * block_ct);
        }
        for (uint32_t tap = 0; tap < 4; ++tap) {
            compute_kernel_lib::tilize<block_ct, dfb::act_rm, dfb::act_tile>(1);
            activation.wait_front(block_ct);

            const uint32_t source_partial_dfb = (tap == 1 || tap == 3) ? dfb::partial_a : dfb::partial_b;
            const uint32_t destination_dfb =
                (tap == 0 || tap == 2) ? dfb::partial_a : (tap == 1 ? dfb::partial_b : dfb::output);
            if (tap == 1 || tap == 3) {
                partial_a.wait_front(block_ct);
            } else if (tap == 2) {
                partial_b.wait_front(block_ct);
            }

            reconfig_data_format_srca(dfb::act_tile);
            reconfig_data_format_srcb(dfb::weights);
            if (tap == 0) {
                mul_bcast_rows_init(dfb::act_tile, dfb::weights);
            }
            for (uint32_t ct = 0; ct < block_ct; ++ct) {
                tile_regs_acquire();
                if (tap != 0) {
                    mul_bcast_rows_init(dfb::act_tile, dfb::weights);
                }
                mul_tiles_bcast_rows(dfb::act_tile, dfb::weights, ct, tap * block_ct + ct, 0);

                if (tap != 0) {
                    reconfig_data_format_srca(source_partial_dfb);
                    add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(source_partial_dfb);
                    add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(source_partial_dfb, 0, 0);
                    if (tap == 1 || tap == 3) {
                        partial_a.pop_front(1);
                    } else {
                        partial_b.pop_front(1);
                    }
                    reconfig_data_format_srca(dfb::act_tile);
                }
                if (tap == 3) {
                    silu_tile(0);
                }
                tile_regs_commit();

                if (tap == 0 || tap == 2) {
                    partial_a.reserve_back(1);
                } else if (tap == 1) {
                    partial_b.reserve_back(1);
                } else {
                    output.reserve_back(1);
                }
                tile_regs_wait();
                pack_tile(0, destination_dfb);
                if (tap == 0 || tap == 2) {
                    partial_a.push_back(1);
                } else if (tap == 1) {
                    partial_b.push_back(1);
                } else {
                    output.push_back(1);
                }
                tile_regs_release();
            }
            activation.pop_front(block_ct);
        }
        if constexpr (num_blocks > 1) {
            weights.pop_front(4 * block_ct);
        }
    }
}
