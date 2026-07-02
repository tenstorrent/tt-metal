// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Shared compute phases for the partial-width-sharded matmul_decode compute kernels:
// phase1_partial (K-slice matmul -> partial), phase2_reduce (sum K_blocks partials, optional gelu),
// phase3_multiply (GeGLU gate*up). compute_partial_width_sharded uses phase 1 (then its own
// residual-aware phase 2); compute_gate_up_partial_width_sharded uses all three.

#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/gelu.h"

// Partial matmul of this core's K-slice of A with in1 -> partial. The caller cb_pop_front()s
// full_in0 after the last call (A may be shared by multiple weights).
template <
    uint32_t M_tiles,
    uint32_t Kc_tiles,
    uint32_t Nc_tiles,
    uint32_t inA_K_tiles_per_core,
    uint32_t out_block_w,
    uint32_t out_block_h,
    uint32_t in0_block_w,
    uint32_t block_num_tiles,
    uint32_t sender_slice_tiles>
inline void phase1_partial(uint32_t full_in0_cb_id, uint32_t in1_cb_id, uint32_t partial_cb_id, uint32_t k_offset) {
    using namespace ckernel;
    mm_block_init(full_in0_cb_id, in1_cb_id, partial_cb_id, false, out_block_w, out_block_h, in0_block_w);
    cb_reserve_back(partial_cb_id, block_num_tiles);
    for (uint32_t nc = 0; nc < Nc_tiles; ++nc) {
        tile_regs_acquire();
        for (uint32_t kc = 0; kc < Kc_tiles; ++kc) {
            const uint32_t k_global = k_offset + kc;
            const uint32_t sender = k_global / inA_K_tiles_per_core;
            const uint32_t kc_local = k_global - sender * inA_K_tiles_per_core;
            // in0: K-column kc_local of this sender's block; matmul reads + r*in0_block_w for r in
            // [0, M_tiles), i.e. every M-row of that K-column.
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
    cb_push_back(partial_cb_id, block_num_tiles);
}

// Base cores: pairwise-reduce the K_blocks partials into out_cb, optionally fusing gelu.
template <
    uint32_t M_tiles,
    uint32_t Nc_tiles,
    uint32_t K_blocks,
    uint32_t block_num_tiles,
    bool do_gelu,
    bool gelu_approx>
inline void phase2_reduce(uint32_t reduce_cb_id, uint32_t out_cb_id) {
    using namespace ckernel;
    constexpr uint32_t reduce_num_tiles = K_blocks * block_num_tiles;
    cb_wait_front(reduce_cb_id, reduce_num_tiles);

    binary_op_init_common(reduce_cb_id, reduce_cb_id, out_cb_id);
    add_tiles_init(reduce_cb_id, reduce_cb_id, true /* acc_to_dest */);
    if (do_gelu) {
        gelu_tile_init<gelu_approx>();
    }
    cb_reserve_back(out_cb_id, block_num_tiles);
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
            if (do_gelu) {
                gelu_tile<gelu_approx>(0);
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

// Base cores: elementwise multiply of two reduced results a * b -> out (the GeGLU gate*up).
template <uint32_t block_num_tiles>
inline void phase3_multiply(uint32_t a_cb_id, uint32_t b_cb_id, uint32_t out_cb_id) {
    using namespace ckernel;
    cb_wait_front(a_cb_id, block_num_tiles);
    cb_wait_front(b_cb_id, block_num_tiles);

    binary_op_init_common(a_cb_id, b_cb_id, out_cb_id);
    mul_tiles_init(a_cb_id, b_cb_id);
    cb_reserve_back(out_cb_id, block_num_tiles);
    for (uint32_t t = 0; t < block_num_tiles; ++t) {
        tile_regs_acquire();
        mul_tiles(a_cb_id, b_cb_id, t, t, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, out_cb_id, t);
        tile_regs_release();
    }
    cb_push_back(out_cb_id, block_num_tiles);
    cb_pop_front(a_cb_id, block_num_tiles);
    cb_pop_front(b_cb_id, block_num_tiles);
}
