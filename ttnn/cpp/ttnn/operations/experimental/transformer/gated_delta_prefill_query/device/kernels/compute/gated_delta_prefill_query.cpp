// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Compute for the gated-delta prefill-then-query op — K @ K^T + unit-lower-triangular mask (WIP).
//
// The core's K section is Height(seq) x Hidden(d) tiles, laid out hidden-major in cb_k. All
// matmuls are done first, then all the masking, so the matmul <-> eltwise mode switch happens
// once (no per-tile data-format reformats; everything is bf16).
//
//   Phase 1 (matmul): K @ K^T, one G x G output block at a time (G = gram_block). Per block,
//     read a G x 1 chunk and accumulate the partial product over the hidden dim (kt = 1). All
//     blocks' tiles are packed into cb_gram.
//   Phase 2 (mask + ident, fused): each Gram tile (*) strict-lower mask leaves masked = strictly
//     lower part in DST; then binary_dest_reuse adds the identity to DST in place (no pack/unpack
//     round trip), giving a unit lower-triangular tile straight into cb_kkt.
//
// The mask/identity tiles (cb_mask, cb_ident) are hand-built by the writer kernel. cb_k is NOT
// popped (kept resident for later stages). NOTE: mask+ident are applied uniformly to every tile,
// which is correct for diagonal tiles (the seq_tile_count == 1 case); off-diagonal tiles of a
// multi-tile block will need position-aware handling later.

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/copy_dest_values.h"
#include "api/compute/common.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    const uint32_t seq_tile_count = get_arg_val<uint32_t>(0);  // seq-tiles (height) owned by this core

    constexpr uint32_t d_tiles = get_compile_time_arg_val(0);     // hidden-dim tiles (contraction)
    constexpr uint32_t gram_block = get_compile_time_arg_val(1);  // G: output block is G x G

    constexpr uint32_t cb_k = tt::CBIndex::c_0;      // K section, hidden-major, resident
    constexpr uint32_t cb_kkt = tt::CBIndex::c_1;    // unit lower-triangular result
    constexpr uint32_t cb_mask = tt::CBIndex::c_2;   // strict-lower mask (0 on/above diag, 1 below)
    constexpr uint32_t cb_ident = tt::CBIndex::c_3;  // identity (1 on diag)
    constexpr uint32_t cb_gram = tt::CBIndex::c_4;   // raw K @ K^T
    constexpr uint32_t cb_solve = tt::CBIndex::c_5;  // triangle-solve scaffold output (DST[2])

    // Total result tiles for this core = sum over G x G blocks of n*n.
    uint32_t total = 0;
    for (uint32_t b = 0; b < seq_tile_count; b += gram_block) {
        const uint32_t n = (seq_tile_count - b) < gram_block ? (seq_tile_count - b) : gram_block;
        total += n * n;
    }

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_k, cb_k, cb_gram);

    CircularBuffer cb_k_o(cb_k);
    CircularBuffer cb_kkt_o(cb_kkt);
    CircularBuffer cb_mask_o(cb_mask);
    CircularBuffer cb_ident_o(cb_ident);
    CircularBuffer cb_gram_o(cb_gram);
    CircularBuffer cb_solve_o(cb_solve);

    cb_k_o.wait_front(seq_tile_count * d_tiles);  // whole section resident; not popped
    cb_mask_o.wait_front(1);
    cb_ident_o.wait_front(1);

    // ---- Phase 1: all matmuls -> cb_gram ----
    cb_gram_o.reserve_back(total);
    uint32_t out = 0;
    for (uint32_t b = 0; b < seq_tile_count; b += gram_block) {
        const uint32_t n = (seq_tile_count - b) < gram_block ? (seq_tile_count - b) : gram_block;
        matmul_block_init(cb_k, cb_k, /*transpose=*/1, /*ct_dim=*/n, /*rt_dim=*/n, /*kt_dim=*/1);
        tile_regs_acquire();
        for (uint32_t kd = 0; kd < d_tiles; ++kd) {
            const uint32_t in_tile = kd * seq_tile_count + b;
            matmul_block(cb_k, cb_k, in_tile, in_tile, /*idst=*/0, /*transpose=*/1, n, n, /*kt_dim=*/1);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < n * n; ++i) {
            pack_tile<true>(i, cb_gram, out + i);
        }
        tile_regs_release();
        out += n * n;
    }
    cb_gram_o.push_back(total);

    // ---- Phase 2 (fused): masked = gram (*) strict-lower mask (in DST), then += identity via
    //      dest reuse (no pack/unpack round trip) -> unit lower-triangular tile -> cb_kkt ----
    cb_gram_o.wait_front(total);
    cb_kkt_o.reserve_back(total);
    for (uint32_t i = 0; i < total; ++i) {
        tile_regs_acquire();
        mul_tiles_init(cb_gram, cb_mask);
        mul_tiles(cb_gram, cb_mask, i, 0, 0);  // DST[0] = gram_i (*) strict-lower mask
        // DST[0] = identity + DST[0]  (identity is a single tile, reused; DST loaded into SrcB)
        binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_ident);
        binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_ident, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, cb_kkt, i);
        tile_regs_release();
    }
    cb_kkt_o.push_back(total);
    cb_gram_o.pop_front(total);

    // ---- Phase 3 (triangle-solve scaffold): DST[0] = masked matmul output (unit lower-tri),
    //      DST[1] = a K tile on the k/hidden dim, dummy SFPU (copy_dest_values, placeholder for
    //      the real triangle solve) -> DST[2], packed out to cb_solve. Each dummy call is its own
    //      tight tile_regs cycle (load DST[0]/DST[1], compute, pack DST[2]), so one output tile is
    //      produced per (output tile, k-dim) step. K-tile indexing assumes one output tile per
    //      seq-tile (the seq_tile_count==1 case); the multi-tile-block mapping comes with the real
    //      solve. ----
    cb_kkt_o.wait_front(total);
    cb_solve_o.reserve_back(total * d_tiles);
    uint32_t solve_out = 0;
    for (uint32_t i = 0; i < total; ++i) {
        for (uint32_t kd = 0; kd < d_tiles; ++kd) {
            tile_regs_acquire();
            copy_tile_init(cb_kkt);
            copy_tile(cb_kkt, i, /*dst=*/0);  // DST[0] = masked matmul output (unit lower-tri)
            copy_tile_init(cb_k);
            copy_tile(cb_k, kd * seq_tile_count + i, /*dst=*/1);  // DST[1] = K value on the k dim
            copy_dest_values_init();
            copy_dest_values(/*idst_in=*/0, /*idst_out=*/2);  // dummy SFPU -> DST[2]
            tile_regs_commit();
            tile_regs_wait();
            pack_tile<true>(2, cb_solve, solve_out++);  // pack DST[2] every iteration
            tile_regs_release();
        }
    }
    cb_solve_o.push_back(total * d_tiles);
    cb_kkt_o.pop_front(total);
}
