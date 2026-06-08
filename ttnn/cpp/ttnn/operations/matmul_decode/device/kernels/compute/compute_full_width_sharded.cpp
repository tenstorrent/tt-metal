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
// Blocking: in0_block_w (K) = 1, out_block_h (M) = 1, out_block_w (N) = 1.
//
// M_tiles>1 fix: the original kernel hardcoded a single M-tile (out_block_h=1,
// in0 index without an M offset, pack always to out slot 0), so it only ever
// computed M-tile 0 and left M-rows 1..M_tiles-1 unwritten/garbage. We now loop
// over every M-tile (bh) x N-tile (bw), accumulating over K, and pack each
// (bh,bw) tile to its sequential output slot.
//
// full_in0 layout (built by reader_full_width_sharded): A is width(K)-sharded
// across `num_senders` sender cores, each holding a contiguous [M_tiles*32,
// inA_K_tiles_per_core*32] slice in TILE row-major order, and the reader copies
// each sender's whole slice into full_in0 at offset sender*shard_num_tiles. So
// full_in0 is SENDER-MAJOR: the tile for global (m, k_global) lives at
//   sender * (M_tiles * inA_K_tiles_per_core)  // sender base
//   + m * inA_K_tiles_per_core                 // M-row within the sender slice
//   + kc_local                                 // K-col within the sender slice
// where sender = k_global / inA_K_tiles_per_core, kc_local = k_global % ...
// (For M_tiles==1 this collapses to k_global, matching the old behaviour.)

using namespace ckernel;
void kernel_main() {
    constexpr uint32_t in0_block_w = 1;
    constexpr uint32_t out_block_h = 1;
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

    constexpr uint32_t num_K_blocks = K_tiles / in0_block_w;
    // tiles per sender slice in full_in0 (== reader's shard_num_tiles)
    constexpr uint32_t sender_slice_tiles = M_tiles * inA_K_tiles_per_core;

    // Gathered A is a regular CB (reader publishes via push_back).
    cb_wait_front(in0_cb_id, in0_num_tiles);
    // in1/out are buffer-backed: data is already in L1 (see vecadd_sharding).

    // mm_init(in0_cb_id, in1_cb_id, out_cb_id);
    mm_block_init(in0_cb_id, in1_cb_id, out_cb_id, false, out_block_w, out_block_h, in0_block_w);

    // Reserve the whole output shard ([M_tiles x N_tiles_per_core] tiles, row-major).
    cb_reserve_back(out_cb_id, M_tiles * N_tiles_per_core);
    for (uint32_t mt = 0; mt < M_tiles; ++mt) {
        for (uint32_t bw = 0; bw < N_tiles_per_core; ++bw) {
            tile_regs_acquire();
            for (uint32_t k = 0; k < num_K_blocks; ++k) {
                // Translate global K-tile index k into its sender-major slot in full_in0.
                const uint32_t sender = k / inA_K_tiles_per_core;
                const uint32_t kc_local = k - sender * inA_K_tiles_per_core;
                const uint32_t in0_tile = sender * sender_slice_tiles + mt * inA_K_tiles_per_core + kc_local;
                matmul_block(
                    in0_cb_id,
                    in1_cb_id,
                    in0_tile,
                    k * N_tiles_per_core + bw,  // B is [K_tiles x N_tiles_per_core] row-major
                    0,
                    false,
                    out_block_w,
                    out_block_h,
                    in0_block_w);
            }
            tile_regs_commit();
            tile_regs_wait();
            // Pack to the row-major output slot for (mt, bw).
            pack_tile<true>(0, out_cb_id, mt * N_tiles_per_core + bw);
            tile_regs_release();
        }
    }
    cb_push_back(out_cb_id, M_tiles * N_tiles_per_core);

    cb_pop_front(in0_cb_id, in0_num_tiles);
}
