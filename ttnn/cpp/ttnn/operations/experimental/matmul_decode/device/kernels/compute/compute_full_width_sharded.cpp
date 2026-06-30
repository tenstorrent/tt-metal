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
// Blocking: in0_block_w (K) = inA_K_tiles_per_core, out_block_h (M) = M_tiles,
// out_block_w (N) = 1.
//
// out_block_h = M_tiles: we process the entire M dimension of the output shard
// in a single DST block, so matmul_block produces all M-tiles for a given N-tile
// at once (DST holds out_block_h * out_block_w == M_tiles tiles). The program
// factory asserts M < 256, which keeps M_tiles <= 8 so the block fits in DST.
//
// full_in0 layout (built by reader_full_width_sharded): A is width(K)-sharded
// across `num_senders` sender cores, each holding a contiguous [M_tiles*32,
// inA_K_tiles_per_core*32] slice in TILE row-major order, and the reader copies
// each sender's whole slice into full_in0 at offset sender*shard_num_tiles. So
// within one sender's slice the tiles are laid out as M_tiles rows x
// inA_K_tiles_per_core cols (row-major), so the sender slice is an in0 block of
// rt_dim=M_tiles, kt_dim=inA_K_tiles_per_core.
//
// matmul_block accumulation: matmul_block does NOT internally reduce over kt_dim
// K-tiles. kt_dim (= in0_block_w) is only the in0 *row stride* (K-tiles per
// M-row in the in0 block layout). Each matmul_block call multiplies one K-column
// slice (rt_dim in0 tiles x ct_dim in1 tiles) into DST; the reduction over K is
// done by the explicit K loop, advancing in0_index by 1 (next K-col of the row)
// and in1_index by the in1 row stride (N_tiles_per_core). We therefore loop over
// every global K-tile (across all senders and all kc within a sender),
// accumulating into the same DST tiles.
//
// in1 (B) is [K_tiles x N_tiles_per_core] row-major. With out_block_w (ct_dim) = 1
// each call consumes exactly one in1 tile at in1_index = k_global * N_tiles_per_core
// + bw, so this is correct for any N_tiles_per_core.

using namespace ckernel;
void kernel_main() {
    constexpr uint32_t out_block_w = 1;

    constexpr uint32_t M_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t K_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t N_tiles_per_core = get_compile_time_arg_val(2);
    constexpr uint32_t inA_K_tiles_per_core = get_compile_time_arg_val(3);

    // The whole M dimension of the output shard is computed in one DST block.
    constexpr uint32_t out_block_h = M_tiles;
    // Each sender's K-slice is consumed as the inner (kt) dimension of one block.
    constexpr uint32_t in0_block_w = inA_K_tiles_per_core;

    constexpr uint32_t in0_cb_id = tt::CBIndex::c_3;
    constexpr uint32_t in1_cb_id = tt::CBIndex::c_1;
    constexpr uint32_t out_cb_id = tt::CBIndex::c_2;

    constexpr uint32_t in0_num_tiles = M_tiles * K_tiles;

    // Number of sender slices in full_in0 (== number of A-holding cores).
    constexpr uint32_t num_senders = K_tiles / inA_K_tiles_per_core;
    // tiles per sender slice in full_in0 (== reader's shard_num_tiles)
    constexpr uint32_t sender_slice_tiles = M_tiles * inA_K_tiles_per_core;

    // Gathered A is a regular CB (reader publishes via push_back).
    cb_wait_front(in0_cb_id, in0_num_tiles);
    // in1/out are buffer-backed (sharded) CBs: their data is already resident in L1.

    mm_block_init(in0_cb_id, in1_cb_id, out_cb_id, false, out_block_w, out_block_h, in0_block_w);

    // Reserve the whole output shard ([M_tiles x N_tiles_per_core] tiles, row-major).
    cb_reserve_back(out_cb_id, M_tiles * N_tiles_per_core);
    for (uint32_t bw = 0; bw < N_tiles_per_core; ++bw) {
        tile_regs_acquire();
        for (uint32_t sender = 0; sender < num_senders; ++sender) {
            // Base of this sender's [M_tiles x inA_K_tiles_per_core] block in full_in0.
            const uint32_t in0_base = sender * sender_slice_tiles;
            for (uint32_t kc = 0; kc < inA_K_tiles_per_core; ++kc) {
                // in0: K-column kc within the sender's block (row stride == in0_block_w).
                const uint32_t in0_tile = in0_base + kc;
                // in1: the single B tile for this global K-row at N-tile column bw.
                const uint32_t k_global = sender * inA_K_tiles_per_core + kc;
                const uint32_t in1_tile = k_global * N_tiles_per_core + bw;
                matmul_block(in0_cb_id, in1_cb_id, in0_tile, in1_tile, 0, false, out_block_w, out_block_h, in0_block_w);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        // Pack each of the M_tiles output rows to its row-major output slot.
        for (uint32_t mt = 0; mt < out_block_h; ++mt) {
            pack_tile<true>(mt, out_cb_id, mt * N_tiles_per_core + bw);
        }
        tile_regs_release();
    }
    cb_push_back(out_cb_id, M_tiles * N_tiles_per_core);

    cb_pop_front(in0_cb_id, in0_num_tiles);
}
