// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute for indexer_score. Per causal-valid output tile (s, t):
//   acc(fp32) = sum_h relu(q[h,s,:] @ k[t,:]^T) * w[h,s]   (bcast col)
//   diagonal tile (t == chunk_t + s): acc += -inf strict upper triangle
//   pack_untilize acc -> bf16 row-major out
// Heads run in passes of 8 (fp32 DEST, full sync). q/w rows stay resident.

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/relu.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/cb_api.h"
#include "api/compute/tile_move_copy.h"

#include "indexer_score_common.hpp"

constexpr uint32_t cb_q = tt::CBIndex::c_0;
constexpr uint32_t cb_k = tt::CBIndex::c_1;
constexpr uint32_t cb_w = tt::CBIndex::c_2;
constexpr uint32_t cb_mask = tt::CBIndex::c_3;
constexpr uint32_t cb_qk = tt::CBIndex::c_24;
constexpr uint32_t cb_mul = tt::CBIndex::c_25;
constexpr uint32_t cb_acc = tt::CBIndex::c_26;
constexpr uint32_t cb_out = tt::CBIndex::c_16;

constexpr uint32_t HP = 8;  // heads per DEST pass

// phase 1: per head, relu(q_row . k_col^T) -> cb_qk (fp32), HP heads per pass
inline void relu_qk_all_heads() {
    cb_wait_front(cb_k, Dt);
    reconfig_data_format(cb_q, cb_k);
    pack_reconfig_data_format(cb_qk);
    mm_init_short(cb_q, cb_k, 1);
    for (uint32_t hp = 0; hp < Hi; hp += HP) {
        tile_regs_acquire();
        for (uint32_t h8 = 0; h8 < HP; ++h8) {
            const uint32_t qbase = (hp + h8) * Dt;
            for (uint32_t d = 0; d < Dt; ++d) {
                matmul_tiles(cb_q, cb_k, qbase + d, d, h8);
            }
            relu_tile(h8);
        }
        tile_regs_commit();
        cb_reserve_back(cb_qk, HP);
        tile_regs_wait();
        for (uint32_t h8 = 0; h8 < HP; ++h8) {
            pack_tile(h8, cb_qk);
        }
        tile_regs_release();
        cb_push_back(cb_qk, HP);
    }
    cb_pop_front(cb_k, Dt);
    cb_wait_front(cb_qk, Hi);
}

// phase 2a: one HP pass of qk[h] * bcast_col(w[h]) -> cb_mul
inline void mul_gates_pass(uint32_t hp) {
    reconfig_data_format(cb_qk, cb_w);
    mul_bcast_cols_init_short(cb_qk, cb_w);
    pack_reconfig_data_format(cb_mul);
    cb_reserve_back(cb_mul, HP);
    tile_regs_acquire();
    for (uint32_t h8 = 0; h8 < HP; ++h8) {
        mul_tiles_bcast_cols(cb_qk, cb_w, hp + h8, hp + h8, h8);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t h8 = 0; h8 < HP; ++h8) {
        pack_tile(h8, cb_mul);
    }
    tile_regs_release();
    cb_push_back(cb_mul, HP);
}

// phase 2b: fold one HP pass of cb_mul into the ping-pong fp32 accumulator
inline void accumulate_pass(uint32_t hp) {
    reconfig_data_format(cb_acc, cb_mul);
    add_tiles_init(cb_acc, cb_mul);
    pack_reconfig_data_format(cb_acc);
    cb_wait_front(cb_mul, HP);
    for (uint32_t h8 = 0; h8 < HP; ++h8) {
        if (hp == 0 && h8 == 0) {
            // prime acc with the first contribution
            reconfig_data_format_srca(cb_mul);
            copy_tile_to_dst_init_short(cb_mul);
            tile_regs_acquire();
            copy_tile(cb_mul, 0, 0);
            tile_regs_commit();
            reconfig_data_format(cb_acc, cb_mul);
            add_tiles_init(cb_acc, cb_mul);
        } else {
            cb_wait_front(cb_acc, 1);
            tile_regs_acquire();
            add_tiles(cb_acc, cb_mul, 0, h8, 0);
            tile_regs_commit();
            cb_pop_front(cb_acc, 1);
        }
        cb_reserve_back(cb_acc, 1);
        tile_regs_wait();
        pack_tile(0, cb_acc);
        tile_regs_release();
        cb_push_back(cb_acc, 1);
    }
    cb_pop_front(cb_mul, HP);
}

// diagonal tile: mask strictly-future columns with -inf
inline void add_diag_mask() {
    reconfig_data_format(cb_acc, cb_mask);
    add_tiles_init(cb_acc, cb_mask);
    pack_reconfig_data_format(cb_acc);
    cb_wait_front(cb_acc, 1);
    tile_regs_acquire();
    add_tiles(cb_acc, cb_mask, 0, 0, 0);
    tile_regs_commit();
    cb_pop_front(cb_acc, 1);
    cb_reserve_back(cb_acc, 1);
    tile_regs_wait();
    pack_tile(0, cb_acc);
    tile_regs_release();
    cb_push_back(cb_acc, 1);
}

// untilize acc -> bf16 row-major out
inline void untilize_to_out() {
    reconfig_data_format_srca(cb_acc);
    pack_reconfig_data_format(cb_out);
    pack_untilize_init<1, 1>(cb_acc, cb_out);
    cb_wait_front(cb_acc, 1);
    cb_reserve_back(cb_out, 1);
    pack_untilize_block<1, 1>(cb_acc, 1, cb_out);
    cb_push_back(cb_out, 1);
    cb_pop_front(cb_acc, 1);
    pack_untilize_uninit(cb_out);
}

void kernel_main() {
    const uint32_t flat_start = get_arg_val<uint32_t>(0);
    const uint32_t flat_count = get_arg_val<uint32_t>(1);
    if (flat_count == 0) {
        return;
    }

    mm_init(cb_q, cb_k, cb_qk, 1 /*transpose k*/);
    relu_tile_init();
    cb_wait_front(cb_mask, 1);

    ValidTileSpan span;
    span.start(flat_start);

    bool have_row = false;
    for (uint32_t i = 0; i < flat_count; ++i) {
        if (!have_row) {
            cb_wait_front(cb_q, Hi * Dt);
            cb_wait_front(cb_w, Hi);
            have_row = true;
        }

        relu_qk_all_heads();
        for (uint32_t hp = 0; hp < Hi; hp += HP) {
            mul_gates_pass(hp);
            accumulate_pass(hp);
        }
        cb_pop_front(cb_qk, Hi);

        if (span.on_diagonal()) {
            add_diag_mask();
        }
        untilize_to_out();

        if (span.advance()) {
            cb_pop_front(cb_q, Hi * Dt);
            cb_pop_front(cb_w, Hi);
            have_row = false;
        }
    }
}
