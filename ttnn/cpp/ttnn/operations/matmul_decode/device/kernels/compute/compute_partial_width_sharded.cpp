// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"

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
// M_tiles>1 fix: phase 1 now loops over every M-tile (mt) and packs each (mt,nc)
// partial to its row-major slot in partial_cb ([M_tiles x Nc_tiles]); phase 2
// reduces per (mt,nc) across the K_blocks slabs in reduce_cb (each slab is one
// [M_tiles x Nc_tiles] block).

using namespace ckernel;
void kernel_main() {
    constexpr uint32_t M_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t K_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t Kc_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t Nc_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t K_blocks = get_compile_time_arg_val(4);
    constexpr uint32_t inA_K_tiles_per_core = get_compile_time_arg_val(5);
    // deep-plan_13 Phase 3: out_w fat-fill on the phase-1 partial loop (out_h clamped to 1;
    // M-fill P0-A gated). out_subblock_w divides Nc_tiles.
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(6);

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

    // ---- Phase 1: partial matmul ----
    cb_wait_front(full_in0_cb_id, full_in0_num_tiles);
    cb_wait_front(in1_cb_id, in1_num_tiles);

    mm_block_init(full_in0_cb_id, in1_cb_id, partial_cb_id, false, out_subblock_w, 1, 1);

    const uint32_t k_offset = k_idx * Kc_tiles;  // this core's K-slice start (global K-tile)
    cb_reserve_back(partial_cb_id, block_num_tiles);
    // out_w fat-fill: out_h clamped to 1 (M-major A non-contiguous, P0-A); fatten on Nc.
    for (uint32_t mt = 0; mt < M_tiles; ++mt) {
        for (uint32_t nc0 = 0; nc0 < Nc_tiles; nc0 += out_subblock_w) {
            tile_regs_acquire();
            for (uint32_t kc = 0; kc < Kc_tiles; ++kc) {
                const uint32_t k_global = k_offset + kc;
                const uint32_t sender = k_global / inA_K_tiles_per_core;
                const uint32_t kc_local = k_global - sender * inA_K_tiles_per_core;
                const uint32_t in0_tile = sender * sender_slice_tiles + mt * inA_K_tiles_per_core + kc_local;
                // B is [Kc_tiles x Nc_tiles] row-major: out_w in1 tiles (kc, nc0..) are contiguous.
                matmul_block(
                    full_in0_cb_id, in1_cb_id, in0_tile, kc * Nc_tiles + nc0, 0, false, out_subblock_w, 1, 1);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < out_subblock_w; ++j) {
                pack_tile<true>(j, partial_cb_id, mt * Nc_tiles + (nc0 + j));
            }
            tile_regs_release();
        }
    }
    cb_push_back(partial_cb_id, block_num_tiles);
    cb_pop_front(full_in0_cb_id, full_in0_num_tiles);

    if (is_base == 0) {
        return;
    }
    // ---- Phase 2: reduce K_blocks partials (base cores only) ----
    // reduce_cb holds K_blocks contiguous [M_tiles x Nc_tiles] slabs; for each
    // (mt,nc) sum the matching tile across ALL K_blocks slabs.
    //
    // deep-plan_13 sec 6.4 FIX: the original loop did `for (block=0; block<K_blocks; block+=2)
    // add_tiles(reduce, reduce, block, block+1, 0)` with acc_to_dest -- two REAL defects:
    //   (A) stale DST-init: slot 0 was never zeroed before the first acc_to_dest add;
    //   (B) odd-K_blocks OOB: `block+1` reads reduce_cb[K_blocks] (out of bounds) for odd
    //       K_blocks AND silently DROPS the last block's contribution.
    // Replaced with a clean accumulation: copy slab 0 into DST slot 0, then add each
    // remaining slab one at a time via acc_to_dest. Handles ARBITRARY K_blocks (odd + even)
    // with no OOB and no stale DST. (Selector still prefers even K_blocks; M3 tests {2,3,4}.)
    cb_wait_front(reduce_cb_id, reduce_num_tiles);

    binary_op_init_common(reduce_cb_id, reduce_cb_id, out_cb_id);

    // deep-plan_13 sec 6.4 reduce fix. Two defects in the original phase-2 loop:
    //   (A) stale DST-init: slot 0 was never seeded before the first acc_to_dest add;
    //   (B) odd-K_blocks OOB: `block += 2; add(block, block+1)` read reduce_cb[K_blocks].
    // add_tiles(reduce, reduce, a, b, 0) with acc_to_dest = dst += reduce[a] + reduce[b]
    // (DST zeroed on tile_regs_acquire). FIX (handles ARBITRARY K_blocks, exact values):
    //   * ODD  K_blocks: copy_tile seeds the unpaired tail slab (K_blocks-1) into DST, then the
    //     pair loop accumulates the even prefix [0..K_blocks-2] -> exact total, no OOB.
    //   * EVEN K_blocks: the pair loop alone sums all K_blocks/2 distinct pairs from a zeroed DST.
    // The selector still prefers EVEN K_blocks; M3 tests {2,3,4} to exercise both paths.
    constexpr bool odd_k = (K_blocks & 1u) != 0u;

    cb_reserve_back(out_cb_id, block_num_tiles);
    for (uint32_t mt = 0; mt < M_tiles; ++mt) {
        for (uint32_t nc = 0; nc < Nc_tiles; ++nc) {
            const uint32_t tile_in_block = mt * Nc_tiles + nc;
            tile_regs_acquire();
            if (odd_k) {
                // Seed DST with the unpaired tail slab so it is counted exactly once.
                copy_tile_init(reduce_cb_id);
                copy_tile(reduce_cb_id, (K_blocks - 1) * block_num_tiles + tile_in_block, 0);
            }
            add_tiles_init(reduce_cb_id, reduce_cb_id, true /* acc_to_dest */);
            for (uint32_t block = 0; block + 1 < K_blocks; block += 2) {
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
    cb_push_back(out_cb_id, block_num_tiles);
    cb_pop_front(reduce_cb_id, reduce_num_tiles);
}
