// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"

using std::uint32_t;

// Block-diagonal batched matmul per core. full_in0 is sender-major; matmul_block does not reduce over kt_dim.
using namespace ckernel;
void kernel_main() {
    constexpr uint32_t M_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t K_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t Nc_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t Bc = get_compile_time_arg_val(3);
    constexpr uint32_t inA_K_tiles_per_core = get_compile_time_arg_val(4);

    constexpr uint32_t full_in0_cb_id = tt::CBIndex::c_3;
    constexpr uint32_t in1_cb_id = tt::CBIndex::c_1;
    constexpr uint32_t out_cb_id = tt::CBIndex::c_2;

    constexpr uint32_t full_in0_num_tiles = Bc * M_tiles * K_tiles;
    constexpr uint32_t in1_num_tiles = Bc * K_tiles * Nc_tiles;
    constexpr uint32_t out_num_tiles = Bc * M_tiles * Nc_tiles;
    constexpr uint32_t sender_slice_tiles = Bc * M_tiles * inA_K_tiles_per_core;

    constexpr uint32_t out_block_h = M_tiles;
    constexpr uint32_t out_block_w = 1;
    constexpr uint32_t in0_block_w = inA_K_tiles_per_core;

    CircularBuffer full_in0_cb(full_in0_cb_id);
    CircularBuffer in1_cb(in1_cb_id);
    CircularBuffer out_cb(out_cb_id);

    compute_kernel_hw_startup<SrcOrder::Reverse>(full_in0_cb_id, in1_cb_id, out_cb_id);

    full_in0_cb.wait_front(full_in0_num_tiles);
    in1_cb.wait_front(in1_num_tiles);

    matmul_block_init(full_in0_cb_id, in1_cb_id, false, out_block_w, out_block_h, in0_block_w);

    out_cb.reserve_back(out_num_tiles);
    for (uint32_t bc_i = 0; bc_i < Bc; ++bc_i) {
        const uint32_t in0_batch_base = (bc_i * M_tiles) * inA_K_tiles_per_core;
        const uint32_t in1_batch_base = bc_i * K_tiles * Nc_tiles;
        for (uint32_t nc = 0; nc < Nc_tiles; ++nc) {
            tile_regs_acquire();
            for (uint32_t kt = 0; kt < K_tiles; ++kt) {
                const uint32_t sender = kt / inA_K_tiles_per_core;
                const uint32_t kc_local = kt - sender * inA_K_tiles_per_core;
                const uint32_t in0_tile = sender * sender_slice_tiles + in0_batch_base + kc_local;
                const uint32_t in1_tile = in1_batch_base + kt * Nc_tiles + nc;
                matmul_block(
                    full_in0_cb_id, in1_cb_id, in0_tile, in1_tile, 0, false, out_block_w, out_block_h, in0_block_w);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t mt = 0; mt < out_block_h; ++mt) {
                pack_tile<true>(mt, out_cb_id, (bc_i * M_tiles + mt) * Nc_tiles + nc);
            }
            tile_regs_release();
        }
    }
    out_cb.push_back(out_num_tiles);
    full_in0_cb.pop_front(full_in0_num_tiles);
}
