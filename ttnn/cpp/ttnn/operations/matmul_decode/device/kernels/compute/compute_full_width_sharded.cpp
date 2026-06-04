// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/matmul.h"

using std::uint32_t;

// Full-width-sharded matmul compute: C = A @ B per core.
//
// Inputs (published by reader_full_width_sharded):
//   full_in0_cb (c_3): full gathered A in sender-major K layout
//   in1_cb (c_1):      this core's B shard (buffer-backed, already in L1)
// Output:
//   out_cb (c_2):      this core's output shard (buffer-backed)
//
// Blocking: in0_block_w (K) = 1, out_block_h (M) = 8, out_block_w (N) = 1.
void kernel_main() {
    constexpr uint32_t in0_block_w = 1;
    constexpr uint32_t out_block_h = 8;
    constexpr uint32_t out_block_w = 1;

    constexpr uint32_t M_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t K_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t N_tiles_per_core = get_compile_time_arg_val(2);
    constexpr uint32_t inA_K_tiles_per_core = get_compile_time_arg_val(3);
    constexpr uint32_t last_out_block_h = get_compile_time_arg_val(4);
    constexpr uint32_t num_blocks_h = get_compile_time_arg_val(5);

    constexpr uint32_t in0_cb_id = tt::CBIndex::c_3;
    constexpr uint32_t in1_cb_id = tt::CBIndex::c_1;
    constexpr uint32_t out_cb_id = tt::CBIndex::c_2;

    constexpr uint32_t in0_num_tiles = M_tiles * K_tiles;
    constexpr uint32_t shard_num_tiles = M_tiles * inA_K_tiles_per_core;

    // Gathered A is a regular CB (reader publishes via push_back).
    cb_wait_front(in0_cb_id, in0_num_tiles);
    // in1/out are buffer-backed: data is already in L1 (see vecadd_sharding).

    mm_init(in0_cb_id, in1_cb_id, out_cb_id);

    for (uint32_t bh = 0; bh < num_blocks_h; ++bh) {
        const uint32_t block_h = (bh == num_blocks_h - 1) ? last_out_block_h : out_block_h;
        for (uint32_t bw = 0; bw < N_tiles_per_core; bw += out_block_w) {
            for (uint32_t h = 0; h < block_h; ++h) {
                const uint32_t mt = bh * out_block_h + h;
                for (uint32_t w = 0; w < out_block_w; ++w) {
                    const uint32_t nt = bw + w;

                    tile_regs_acquire();
                    for (uint32_t kt = 0; kt < K_tiles; ++kt) {
                        const uint32_t sender_id = kt / inA_K_tiles_per_core;
                        const uint32_t kt_local = kt % inA_K_tiles_per_core;
                        const uint32_t in0_tile_index =
                            sender_id * shard_num_tiles + mt * inA_K_tiles_per_core + kt_local;
                        const uint32_t in1_tile_index = kt * N_tiles_per_core + nt;
                        matmul_tiles(in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index, 0);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_tile(0, out_cb_id);
                    tile_regs_release();
                    cb_push_back(out_cb_id, 1);
                }
            }
        }
    }

    cb_pop_front(in0_cb_id, in0_num_tiles);
}
