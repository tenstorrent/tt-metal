// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tilize.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

void kernel_main() {
    constexpr uint32_t Ct = get_compile_time_arg_val(0);
    constexpr uint32_t act_rm_cb = 0;
    constexpr uint32_t act_tile_cb = 1;
    constexpr uint32_t weights_cb = 2;
    constexpr uint32_t partial_a_cb = 3;
    constexpr uint32_t partial_b_cb = 4;
    constexpr uint32_t output_cb = 5;
    constexpr uint32_t projected_tiles_cb = 8;
    constexpr uint32_t projected_rm_cb = 9;
    const uint32_t mt_count = get_arg_val<uint32_t>(0);

    DataflowBuffer activation(act_tile_cb);
    DataflowBuffer weights(weights_cb);
    DataflowBuffer partial_a(partial_a_cb);
    DataflowBuffer partial_b(partial_b_cb);
    DataflowBuffer output(output_cb);
    weights.wait_front(4 * Ct);
    compute_kernel_hw_startup(projected_tiles_cb, projected_rm_cb);

    for (uint32_t item = 0; item < mt_count; ++item) {
        compute_kernel_lib::untilize<Ct, projected_tiles_cb, projected_rm_cb>(1);
        binary_op_init_common(act_tile_cb, weights_cb, partial_a_cb);
        ckernel::silu_tile_init();
        for (uint32_t tap = 0; tap < 4; ++tap) {
            compute_kernel_lib::tilize<Ct, act_rm_cb, act_tile_cb>(1);
            activation.wait_front(Ct);
            DataflowBuffer source_partial = (tap == 1 || tap == 3) ? partial_a : partial_b;
            DataflowBuffer destination = tap == 0 || tap == 2 ? partial_a : (tap == 1 ? partial_b : output);
            const uint32_t source_partial_cb = source_partial.get_id();
            const uint32_t destination_cb = destination.get_id();
            if (tap != 0) {
                source_partial.wait_front(Ct);
            }
            for (uint32_t ct = 0; ct < Ct; ++ct) {
                tile_regs_acquire();
                reconfig_data_format_srca(act_tile_cb);
                reconfig_data_format_srcb(weights_cb);
                mul_bcast_rows_init_short(act_tile_cb, weights_cb);
                mul_tiles_bcast_rows(act_tile_cb, weights_cb, ct, tap * Ct + ct, 0);
                if (tap != 0) {
                    reconfig_data_format_srca(source_partial_cb);
                    binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
                        source_partial_cb);
                    binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
                        source_partial_cb, 0, 0);
                    source_partial.pop_front(1);
                }
                if (tap == 3) {
                    ckernel::silu_tile(0);
                }
                tile_regs_commit();
                destination.reserve_back(1);
                tile_regs_wait();
                pack_tile(0, destination_cb);
                destination.push_back(1);
                tile_regs_release();
            }
            activation.pop_front(Ct);
        }
    }
}
