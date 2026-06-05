// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"

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
// Decode-only: M_tiles == 1, so full A is laid out as K_tiles tiles and this core's
// K-slice is the contiguous run [k_idx * Kc_tiles, (k_idx + 1) * Kc_tiles).

using namespace ckernel;
void kernel_main() {
    constexpr uint32_t M_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t K_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t Kc_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t Nc_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t K_blocks = get_compile_time_arg_val(4);

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

    // ---- Phase 1: partial matmul ----
    cb_wait_front(full_in0_cb_id, full_in0_num_tiles);
    cb_wait_front(in1_cb_id, in1_num_tiles);

    mm_block_init(full_in0_cb_id, in1_cb_id, partial_cb_id, false, 1, 1, 1);

    const uint32_t k_offset = k_idx * Kc_tiles;  // this core's K-slice start (M_tiles == 1)
    cb_reserve_back(partial_cb_id, block_num_tiles);
    for (uint32_t nc = 0; nc < Nc_tiles; ++nc) {
        tile_regs_acquire();
        for (uint32_t kc = 0; kc < Kc_tiles; ++kc) {
            matmul_block(full_in0_cb_id, in1_cb_id, k_offset + kc, kc * Nc_tiles + nc, 0, false, 1, 1, 1);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, partial_cb_id, nc);
        tile_regs_release();
    }
    cb_push_back(partial_cb_id, block_num_tiles);
    cb_pop_front(full_in0_cb_id, full_in0_num_tiles);

    if (is_base == 0) {
        return;
    }

    // ---- Phase 2: reduce K_blocks partials (base cores only) ----
    // reduce_cb holds K_blocks contiguous blocks; pairwise accumulate into DST.
    cb_wait_front(reduce_cb_id, reduce_num_tiles);

    binary_op_init_common(reduce_cb_id, reduce_cb_id, out_cb_id);
    add_tiles_init(reduce_cb_id, reduce_cb_id, true /* acc_to_dest */);

    cb_reserve_back(out_cb_id, block_num_tiles);
    for (uint32_t nc = 0; nc < Nc_tiles; ++nc) {
        tile_regs_acquire();
        for (uint32_t block = 0; block < K_blocks; block += 2) {
            add_tiles(reduce_cb_id, reduce_cb_id, block * Nc_tiles + nc, (block + 1) * Nc_tiles + nc, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, out_cb_id, nc);
        tile_regs_release();
    }
    cb_push_back(out_cb_id, block_num_tiles);
    cb_pop_front(reduce_cb_id, reduce_num_tiles);
}
