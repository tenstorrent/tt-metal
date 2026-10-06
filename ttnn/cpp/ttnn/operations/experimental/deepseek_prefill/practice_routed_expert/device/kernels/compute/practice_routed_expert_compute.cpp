// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/compute/situ_glu.h"
#include "api/dataflow/circular_buffer.h"

// For each row of x, two passes:
//   1. each H tile: the gate and up matmuls reduce over K in DEST, then SiTU-GLU combines them in place.
//      H collects in L1, never in DRAM.
//   2. each output tile: the down matmul reduces that H row over N.
void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w_gate = get_compile_time_arg_val(1);
    constexpr uint32_t cb_w_up = get_compile_time_arg_val(2);
    constexpr uint32_t cb_w_down = get_compile_time_arg_val(3);
    constexpr uint32_t cb_h = get_compile_time_arg_val(4);
    constexpr uint32_t cb_out = get_compile_time_arg_val(5);
    constexpr uint32_t m_tiles = get_compile_time_arg_val(6);
    constexpr uint32_t k_tiles = get_compile_time_arg_val(7);
    constexpr uint32_t n_tiles = get_compile_time_arg_val(8);

    constexpr uint32_t dst_gate = 0;
    constexpr uint32_t dst_up = 1;

    CircularBuffer x_cb(cb_x);
    CircularBuffer w_gate_cb(cb_w_gate);
    CircularBuffer w_up_cb(cb_w_up);
    CircularBuffer w_down_cb(cb_w_down);
    CircularBuffer h_cb(cb_h);
    CircularBuffer out_cb(cb_out);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w_gate, cb_h);

    for (uint32_t m = 0; m < m_tiles; ++m) {
        reconfig_data_format<SrcOrder::Reverse>(cb_x, cb_w_gate);
        pack_reconfig_data_format(cb_h);
        for (uint32_t n = 0; n < n_tiles; ++n) {
            // The matmul and SiTU-GLU both program the math engine, so each re-inits before its turn.
            matmul_init(cb_x, cb_w_gate);
            tile_regs_acquire();
            for (uint32_t k = 0; k < k_tiles; ++k) {
                x_cb.wait_front(1);
                w_gate_cb.wait_front(1);
                w_up_cb.wait_front(1);
                matmul_tiles(cb_x, cb_w_gate, 0, 0, dst_gate);
                matmul_tiles(cb_x, cb_w_up, 0, 0, dst_up);
                x_cb.pop_front(1);
                w_gate_cb.pop_front(1);
                w_up_cb.pop_front(1);
            }
            situ_glu_tile_init();
            situ_glu_tile(dst_gate, dst_up, dst_gate);
            tile_regs_commit();

            h_cb.reserve_back(1);
            tile_regs_wait();
            pack_tile(dst_gate, cb_h);
            tile_regs_release();
            h_cb.push_back(1);
        }

        reconfig_data_format<SrcOrder::Reverse>(cb_h, cb_w_down);
        pack_reconfig_data_format(cb_out);
        matmul_init(cb_h, cb_w_down);
        h_cb.wait_front(n_tiles);
        for (uint32_t j = 0; j < k_tiles; ++j) {
            tile_regs_acquire();
            for (uint32_t n = 0; n < n_tiles; ++n) {
                w_down_cb.wait_front(1);
                matmul_tiles(cb_h, cb_w_down, n, 0, 0);
                w_down_cb.pop_front(1);
            }
            tile_regs_commit();

            out_cb.reserve_back(1);
            tile_regs_wait();
            pack_tile(0, cb_out);
            tile_regs_release();
            out_cb.push_back(1);
        }
        h_cb.pop_front(n_tiles);
    }
}
