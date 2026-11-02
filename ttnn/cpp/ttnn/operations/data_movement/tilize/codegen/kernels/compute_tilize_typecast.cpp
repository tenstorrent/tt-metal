// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/compute/tilize.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_mid = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t num_col_chunks = get_compile_time_arg_val(3);
    constexpr uint32_t chunk_Wt = get_compile_time_arg_val(4);
    constexpr uint32_t BLOCK = get_compile_time_arg_val(5);
    constexpr uint32_t in_data_format = get_compile_time_arg_val(6);
    constexpr uint32_t out_data_format = get_compile_time_arg_val(7);

    const uint32_t num_tile_rows = get_arg_val<uint32_t>(0);
    CircularBuffer cb_in_exp(cb_in);
    CircularBuffer cb_mid_exp(cb_mid);
    CircularBuffer cb_out_exp(cb_out);

    // Hardware startup mid-kernel corrupted the last tiles packed for the previous chunk on WH, so
    // per-chunk operand switches use reconfig.
    compute_kernel_hw_startup(cb_in, cb_mid);
    bool first = true;
    for (uint32_t b = 0; b < num_tile_rows; ++b) {
        for (uint32_t c = 0; c < num_col_chunks; ++c) {
            if (!first) {
                reconfig_data_format_srca(cb_mid, cb_in);
                pack_reconfig_data_format(cb_out, cb_mid);
            }
            first = false;
            tilize_init(cb_in, chunk_Wt, cb_mid);
            cb_in_exp.wait_front(chunk_Wt);
            cb_mid_exp.reserve_back(chunk_Wt);
            tilize_block(cb_in, chunk_Wt, cb_mid);
            cb_mid_exp.push_back(chunk_Wt);
            cb_in_exp.pop_front(chunk_Wt);
            tilize_uninit(cb_in, cb_mid);

            reconfig_data_format_srca(cb_in, cb_mid);
            pack_reconfig_data_format(cb_mid, cb_out);
            copy_init(cb_mid);
            uint32_t remaining = chunk_Wt;
            while (remaining > 0) {
                const uint32_t count = remaining < BLOCK ? remaining : BLOCK;
                cb_mid_exp.wait_front(count);
                cb_out_exp.reserve_back(count);
                tile_regs_acquire();
                for (uint32_t t = 0; t < count; ++t) {
                    copy_tile(cb_mid, t, t);
                    typecast_tile_init<in_data_format, out_data_format>();
                    typecast_tile<in_data_format, out_data_format>(t);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t t = 0; t < count; ++t) {
                    pack_tile(t, cb_out);
                }
                tile_regs_release();
                cb_out_exp.push_back(count);
                cb_mid_exp.pop_front(count);
                remaining -= count;
            }
        }
    }
}
