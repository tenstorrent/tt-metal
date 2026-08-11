// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/tilize.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

void kernel_main() {
    constexpr uint32_t block_ct = get_compile_time_arg_val(0);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(1);
    constexpr uint32_t act_rm_cb = 0;
    constexpr uint32_t act_tile_cb = 1;
    constexpr uint32_t weights_cb = 2;
    constexpr uint32_t partial_a_cb = 3;
    constexpr uint32_t partial_b_cb = 4;
    constexpr uint32_t output_cb = 5;
    const uint32_t mt_count = get_arg_val<uint32_t>(0);

    DataflowBuffer activation(act_tile_cb);
    DataflowBuffer weights(weights_cb);
    DataflowBuffer partial_a(partial_a_cb);
    DataflowBuffer partial_b(partial_b_cb);
    DataflowBuffer output(output_cb);
    binary_op_init_common(act_tile_cb, weights_cb, partial_a_cb);
    silu_tile_init();

    if constexpr (num_blocks == 1) {
        weights.wait_front(4 * block_ct);
    }
    for (uint32_t item = 0; item < mt_count; ++item) {
        if constexpr (num_blocks > 1) {
            weights.wait_front(4 * block_ct);
        }
        for (uint32_t tap = 0; tap < 4; ++tap) {
            compute_kernel_lib::tilize<block_ct, act_rm_cb, act_tile_cb>(1);
            activation.wait_front(block_ct);

            DataflowBuffer source_partial = (tap == 1 || tap == 3) ? partial_a : partial_b;
            DataflowBuffer destination = tap == 0 || tap == 2 ? partial_a : (tap == 1 ? partial_b : output);
            const uint32_t source_partial_cb = source_partial.get_id();
            const uint32_t destination_cb = destination.get_id();
            if (tap != 0) {
                source_partial.wait_front(block_ct);
            }

            for (uint32_t ct = 0; ct < block_ct; ++ct) {
                tile_regs_acquire();
                reconfig_data_format_srca(act_tile_cb);
                reconfig_data_format_srcb(weights_cb);
                mul_bcast_rows_init_short(act_tile_cb, weights_cb);
                mul_tiles_bcast_rows(act_tile_cb, weights_cb, ct, tap * block_ct + ct, 0);

                if (tap != 0) {
                    reconfig_data_format_srca(source_partial_cb);
                    binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
                        source_partial_cb);
                    binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
                        source_partial_cb, 0, 0);
                    source_partial.pop_front(1);
                    reconfig_data_format_srca(act_tile_cb);
                }
                if (tap == 3) {
                    silu_tile(0);
                }
                tile_regs_commit();

                destination.reserve_back(1);
                tile_regs_wait();
                pack_tile(0, destination_cb);
                destination.push_back(1);
                tile_regs_release();
            }
            activation.pop_front(block_ct);
        }
        if constexpr (num_blocks > 1) {
            weights.pop_front(4 * block_ct);
        }
    }
}
