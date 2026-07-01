// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"

using std::uint32_t;

// Partial-width-sharded matmul compute.
//
// Phase 1 (every B core): compute a partial product over this core's K-slice.
//   full_in0_cb (c_3): full gathered A  [M_tiles x K_tiles]   (published by reader)
//   in1_cb (c_1):      this core's B block [Kc_tiles x Nc_tiles] (resident in L1)
//   -> partial_cb (c_4): partial [M_tiles x Nc_tiles]
//
// Phase 2 (base cores only, k_idx == 0): sum the K_blocks partials gathered by the
// writer into reduce_cb (c_5) and write the final output shard.
//   reduce_cb (c_5): K_blocks * [M_tiles x Nc_tiles] partials (block k == k_idx)
//   -> out_cb (c_2): output shard [M_tiles x Nc_tiles]
//
// full A (full_in0) layout (built by reader): A is width(K)-sharded across
// `num_senders = K_tiles / inA_K_tiles_per_core` sender cores, each holding a
// contiguous [M_tiles*32, inA_K_tiles_per_core*32] slice in TILE row-major order,
// and the reader copies each sender's whole slice into full_in0 at offset
// sender*M_tiles*inA_K_tiles_per_core. So full_in0 is SENDER-MAJOR: global tile
// (m, k_global) lives at
//   sender * (M_tiles * inA_K_tiles_per_core) + m * inA_K_tiles_per_core + kc_local
// with sender = k_global / inA_K_tiles_per_core, kc_local = k_global % ...
// (For M_tiles==1 this collapses to k_global, matching the old behaviour.)
//
// M_tiles>1: phase 1 computes the entire M dimension of the partial in a single
// DST block (out_block_h = M_tiles), so each matmul_block produces all M-tiles of
// a given N-tile at once. The program factory asserts M < 256, keeping M_tiles <= 8
// so the block fits in DST. Phase 2 reduces per (mt,nc) across the K_blocks slabs
// in reduce_cb (each slab is one [M_tiles x Nc_tiles] block).
//
// matmul_block accumulation: matmul_block does NOT internally reduce over kt_dim.
// kt_dim (= inA_K_tiles_per_core) is only the in0 *row stride* (K-tiles per M-row
// in full_in0's per-sender slice). Each call multiplies one K-column slice
// (rt_dim=M_tiles in0 tiles x ct_dim=1 in1 tile) into DST; the reduction over this
// core's K-slice is done by the explicit kc loop, accumulating into the same DST.

using namespace ckernel;
void kernel_main() {
    constexpr uint32_t M_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t K_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t Kc_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t Nc_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t K_blocks = get_compile_time_arg_val(4);
    constexpr uint32_t inA_K_tiles_per_core = get_compile_time_arg_val(5);

    const uint32_t k_idx = get_arg_val<uint32_t>(0);
    const uint32_t is_base = get_arg_val<uint32_t>(1);

    constexpr uint32_t full_in0_cb_id = tt::CBIndex::c_3;
    constexpr uint32_t in1_cb_id = tt::CBIndex::c_1;
    constexpr uint32_t out_cb_id = tt::CBIndex::c_2;
    constexpr uint32_t partial_cb_id = tt::CBIndex::c_4;
    constexpr uint32_t reduce_cb_id = tt::CBIndex::c_5;

    constexpr uint32_t full_in0_num_tiles = M_tiles * K_tiles;
    constexpr uint32_t in1_num_tiles = Kc_tiles * Nc_tiles;
    constexpr uint32_t block_num_tiles = M_tiles * Nc_tiles;
    constexpr uint32_t reduce_num_tiles = K_blocks * block_num_tiles;
    constexpr uint32_t sender_slice_tiles = M_tiles * inA_K_tiles_per_core;

    // The whole M dimension of the partial is computed in one DST block.
    constexpr uint32_t out_block_h = M_tiles;
    constexpr uint32_t out_block_w = 1;
    // in0 row stride within a sender slice (K-tiles per M-row) == matmul kt_dim.
    constexpr uint32_t in0_block_w = inA_K_tiles_per_core;

    CircularBuffer full_in0_cb(full_in0_cb_id);
    CircularBuffer in1_cb(in1_cb_id);
    CircularBuffer out_cb(out_cb_id);
    CircularBuffer partial_cb(partial_cb_id);
    CircularBuffer reduce_cb(reduce_cb_id);

    compute_kernel_hw_startup<SrcOrder::Reverse>(full_in0_cb_id, in1_cb_id, partial_cb_id);

    // ---- Phase 1: partial matmul ----
    full_in0_cb.wait_front(full_in0_num_tiles);
    in1_cb.wait_front(in1_num_tiles);

    matmul_block_init(full_in0_cb_id, in1_cb_id, false, out_block_w, out_block_h, in0_block_w);

    const uint32_t k_offset = k_idx * Kc_tiles;  // this core's K-slice start (global K-tile)
    partial_cb.reserve_back(block_num_tiles);
    for (uint32_t nc = 0; nc < Nc_tiles; ++nc) {
        tile_regs_acquire();
        for (uint32_t kc = 0; kc < Kc_tiles; ++kc) {
            const uint32_t k_global = k_offset + kc;
            const uint32_t sender = k_global / inA_K_tiles_per_core;
            const uint32_t kc_local = k_global - sender * inA_K_tiles_per_core;
            // in0: K-column kc_local of this sender's block; matmul reads + r*in0_block_w
            // for r in [0, M_tiles), i.e. every M-row of that K-column.
            const uint32_t in0_tile = sender * sender_slice_tiles + kc_local;
            const uint32_t in1_tile = kc * Nc_tiles + nc;
            matmul_block(
                full_in0_cb_id, in1_cb_id, in0_tile, in1_tile, 0, false, out_block_w, out_block_h, in0_block_w);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t mt = 0; mt < out_block_h; ++mt) {
            pack_tile<true>(mt, partial_cb_id, mt * Nc_tiles + nc);
        }
        tile_regs_release();
    }
    partial_cb.push_back(block_num_tiles);
    full_in0_cb.pop_front(full_in0_num_tiles);

    if (is_base == 0) {
        return;
    }
    // ---- Phase 2: reduce K_blocks partials (base cores only) ----
    // reduce_cb holds K_blocks contiguous [M_tiles x Nc_tiles] slabs; for each
    // (mt,nc) pairwise accumulate the matching tile across all K_blocks slabs.
    reduce_cb.wait_front(reduce_num_tiles);

    binary_op_init_common(reduce_cb_id, reduce_cb_id, out_cb_id);
    add_tiles_init(reduce_cb_id, reduce_cb_id, true /* acc_to_dest */);

    out_cb.reserve_back(block_num_tiles);
    for (uint32_t mt = 0; mt < M_tiles; ++mt) {
        for (uint32_t nc = 0; nc < Nc_tiles; ++nc) {
            const uint32_t tile_in_block = mt * Nc_tiles + nc;
            tile_regs_acquire();
            for (uint32_t block = 0; block < K_blocks; block += 2) {
                add_tiles(
                    reduce_cb_id,
                    reduce_cb_id,
                    block * block_num_tiles + tile_in_block,
                    (block + 1) * block_num_tiles + tile_in_block,
                    0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile<true>(0, out_cb_id, tile_in_block);
            tile_regs_release();
        }
    }
    out_cb.push_back(block_num_tiles);
    reduce_cb.pop_front(reduce_num_tiles);
}
