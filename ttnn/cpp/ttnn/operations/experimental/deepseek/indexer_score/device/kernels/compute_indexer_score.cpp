// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute for indexer_score. Per causal-valid output tile (s, t):
//   acc(fp32) = sum_h relu(q[h,s,:] @ k[t,:]^T) * w[h,s]   (bcast col)
//   diagonal tile (t == chunk_t + s): acc += -inf strict upper triangle
//   pack_untilize acc -> bf16 row-major out
// Heads run in HP-row matmul subblocks (fp32 DEST, half sync; HP from the
// host via determine_largest_subblock_size). q/w rows stay resident.

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

// qk matmul subblock: heads are output rows, k column is 1 tile wide.
// Host picks HP via determine_largest_subblock_size(Hi, 1, dst_size), SDPA-style.
constexpr uint32_t HP = get_compile_time_arg_val(5);
constexpr uint32_t num_subblocks = Hi / HP;

/**
 * qk_cb = relu(q_cb @ k_cb^T): M=Hi (head rows), N=1, K=Dt.
 * Subblock loop mirrors SDPA's matmul_blocks; one matmul_block per inner-dim
 * step accumulates HP head rows in DEST, relu fused before pack.
 */
template <uint32_t q_cb, uint32_t k_cb, uint32_t qk_cb>
void relu_qk_all_heads() {
    // Precondition: q_cb has Hi*Dt produced (resident), k_cb has Dt produced
    // Postcondition: qk_cb has Hi produced, k_cb has Dt consumed
    cb_wait_front(k_cb, Dt);
    reconfig_data_format(q_cb, k_cb);
    pack_reconfig_data_format(qk_cb);
    mm_block_init_short(q_cb, k_cb, 1 /*transpose k*/, 1 /*ct_dim*/, HP /*rt_dim*/, Dt /*kt_dim*/);
    uint32_t in0_offset = 0;
    for (uint32_t sb = 0; sb < num_subblocks; ++sb) {
        tile_regs_acquire();
        for (uint32_t d = 0; d < Dt; ++d) {
            matmul_block(q_cb, k_cb, in0_offset + d, d, 0, 1 /*transpose k*/, 1, HP, Dt);
        }
        for (uint32_t h = 0; h < HP; ++h) {
            relu_tile(h);
        }
        tile_regs_commit();
        cb_reserve_back(qk_cb, HP);
        tile_regs_wait();
        for (uint32_t h = 0; h < HP; ++h) {
            pack_tile(h, qk_cb);
        }
        tile_regs_release();
        cb_push_back(qk_cb, HP);
        in0_offset += HP * Dt;
    }
    cb_pop_front(k_cb, Dt);
    cb_wait_front(qk_cb, Hi);
}

/**
 * mul_cb[h] = qk_cb[hp + h] * bcast_cols(w_cb[hp + h]) for one HP pass
 */
template <uint32_t qk_cb, uint32_t w_cb, uint32_t mul_cb>
void mul_gates_pass(uint32_t hp) {
    // Precondition: qk_cb has Hi produced (resident), w_cb has Hi produced (resident)
    // Postcondition: mul_cb has HP produced
    reconfig_data_format(qk_cb, w_cb);
    pack_reconfig_data_format(mul_cb);
    mul_bcast_cols_init_short(qk_cb, w_cb);
    cb_reserve_back(mul_cb, HP);
    tile_regs_acquire();
    for (uint32_t h = 0; h < HP; ++h) {
        mul_tiles_bcast_cols(qk_cb, w_cb, hp + h, hp + h, h);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t h = 0; h < HP; ++h) {
        pack_tile(h, mul_cb);
    }
    tile_regs_release();
    cb_push_back(mul_cb, HP);
}

/**
 * acc_cb += mul_cb[0..HP), one tile at a time through the 2-slot ping-pong;
 * when first, acc_cb is primed from mul_cb[0] instead of added.
 */
template <uint32_t mul_cb, uint32_t acc_cb>
void accumulate_pass(bool first) {
    // Precondition: mul_cb has HP produced, acc_cb has 1 produced (unless first)
    // Postcondition: mul_cb has HP consumed, acc_cb has 1 produced
    cb_wait_front(mul_cb, HP);
    pack_reconfig_data_format(acc_cb);
    uint32_t h = 0;
    if (first) {
        reconfig_data_format_srca(mul_cb);
        copy_tile_to_dst_init_short(mul_cb);
        tile_regs_acquire();
        copy_tile(mul_cb, 0, 0);
        tile_regs_commit();
        cb_reserve_back(acc_cb, 1);
        tile_regs_wait();
        pack_tile(0, acc_cb);
        tile_regs_release();
        cb_push_back(acc_cb, 1);
        h = 1;
    }
    reconfig_data_format(acc_cb, mul_cb);
    add_tiles_init(acc_cb, mul_cb);
    for (; h < HP; ++h) {
        cb_wait_front(acc_cb, 1);
        tile_regs_acquire();
        add_tiles(acc_cb, mul_cb, 0, h, 0);
        tile_regs_commit();
        cb_pop_front(acc_cb, 1);
        cb_reserve_back(acc_cb, 1);
        tile_regs_wait();
        pack_tile(0, acc_cb);
        tile_regs_release();
        cb_push_back(acc_cb, 1);
    }
    cb_pop_front(mul_cb, HP);
}

/**
 * acc_cb += mask_cb[0]  (-inf on the strict upper triangle of the diagonal tile)
 */
template <uint32_t acc_cb, uint32_t mask_cb>
void add_diag_mask() {
    // Precondition: acc_cb has 1 produced, mask_cb has 1 produced (resident)
    // Postcondition: acc_cb has 1 produced
    reconfig_data_format(acc_cb, mask_cb);
    pack_reconfig_data_format(acc_cb);
    add_tiles_init(acc_cb, mask_cb);
    cb_wait_front(acc_cb, 1);
    tile_regs_acquire();
    add_tiles(acc_cb, mask_cb, 0, 0, 0);
    tile_regs_commit();
    cb_pop_front(acc_cb, 1);
    cb_reserve_back(acc_cb, 1);
    tile_regs_wait();
    pack_tile(0, acc_cb);
    tile_regs_release();
    cb_push_back(acc_cb, 1);
}

/**
 * out_cb = untilize(acc_cb), fp32 tile -> bf16 row-major
 */
template <uint32_t acc_cb, uint32_t out_cb>
void untilize_to_out() {
    // Precondition: acc_cb has 1 produced
    // Postcondition: acc_cb has 1 consumed, out_cb has 1 produced
    reconfig_data_format_srca(acc_cb);
    pack_reconfig_data_format(out_cb);
    pack_untilize_init<1, 1>(acc_cb, out_cb);
    cb_wait_front(acc_cb, 1);
    cb_reserve_back(out_cb, 1);
    pack_untilize_block<1, 1>(acc_cb, 1, out_cb);
    cb_push_back(out_cb, 1);
    cb_pop_front(acc_cb, 1);
    pack_untilize_uninit(out_cb);
}

void kernel_main() {
    const uint32_t flat_start = get_arg_val<uint32_t>(0);
    const uint32_t flat_count = get_arg_val<uint32_t>(1);
    if (flat_count == 0) {
        return;
    }

    mm_block_init(cb_q, cb_k, cb_qk, 1 /*transpose k*/, 1 /*ct_dim*/, HP /*rt_dim*/, Dt /*kt_dim*/);
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

        relu_qk_all_heads<cb_q, cb_k, cb_qk>();
        for (uint32_t hp = 0; hp < Hi; hp += HP) {
            mul_gates_pass<cb_qk, cb_w, cb_mul>(hp);
            accumulate_pass<cb_mul, cb_acc>(hp == 0);
        }
        cb_pop_front(cb_qk, Hi);

        if (span.on_diagonal()) {
            add_diag_mask<cb_acc, cb_mask>();
        }
        untilize_to_out<cb_acc, cb_out>();

        if (span.advance()) {
            cb_pop_front(cb_q, Hi * Dt);
            cb_pop_front(cb_w, Hi);
            have_row = false;
        }
    }
}
