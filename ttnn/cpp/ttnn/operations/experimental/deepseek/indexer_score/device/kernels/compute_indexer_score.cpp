// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute for indexer_score. Per work unit (QC q-tile-rows x kw k-tiles):
//   acc[r,c](fp32) = sum_h relu(q[h,s0+r,:] @ k[c0+c,:]^T) * w[h,s0+r]
//   tile on the causal diagonal: += -inf strict upper triangle; past it: full -inf
//   pack_untilize acc -> bf16 row-major out, (r, c) row-major order
// Heads stream in HB groups, HP rows per DEST subblock (fp32, half sync; HP
// from determine_largest_subblock_size on the host). q/w stay resident per
// group when HB == Hi.

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

constexpr uint32_t HP = get_compile_time_arg_val(8);  // qk subblock height (head rows per DEST pass)
constexpr bool stream_heads = HB < Hi;

/**
 * qk_cb = relu(q_cb[hp..hp+HP) of row r @ k_cb col c^T): one HP-head subblock.
 * q blocks are [QC][HB][Dt] so the HP head rows stride Dt - one matmul_block.
 */
template <uint32_t q_cb, uint32_t k_cb, uint32_t qk_cb>
void relu_qk_subblock(uint32_t hp, uint32_t r, uint32_t c) {
    // Precondition: q_cb has HB*QC*Dt produced (resident), k_cb has kw*Dt produced (resident)
    // Postcondition: qk_cb has HP produced
    reconfig_data_format(q_cb, k_cb);
    pack_reconfig_data_format(qk_cb);
    mm_block_init_short(q_cb, k_cb, 1 /*transpose k*/, 1 /*ct_dim*/, HP /*rt_dim*/, Dt /*kt_dim*/);
    tile_regs_acquire();
    const uint32_t in0_base = (r * HB + hp) * Dt;
    for (uint32_t d = 0; d < Dt; ++d) {
        matmul_block(q_cb, k_cb, in0_base + d, c * Dt + d, 0, 1 /*transpose k*/, 1, HP, Dt);
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
}

/**
 * mul_cb[h] = qk_cb[h] * bcast_cols(w_cb[r*Hi + hb + hp + h]) for one HP pass
 */
template <uint32_t qk_cb, uint32_t w_cb, uint32_t mul_cb>
void mul_gates_pass(uint32_t w_base) {
    // Precondition: qk_cb has HP produced, w_cb has Hi*QC produced (resident, [QC][Hi])
    // Postcondition: qk_cb has HP consumed, mul_cb has HP produced
    cb_wait_front(qk_cb, HP);
    reconfig_data_format(qk_cb, w_cb);
    pack_reconfig_data_format(mul_cb);
    mul_bcast_cols_init_short(qk_cb, w_cb);
    cb_reserve_back(mul_cb, HP);
    tile_regs_acquire();
    for (uint32_t h = 0; h < HP; ++h) {
        mul_tiles_bcast_cols(qk_cb, w_cb, h, w_base + h, h);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t h = 0; h < HP; ++h) {
        pack_tile(h, mul_cb);
    }
    tile_regs_release();
    cb_push_back(mul_cb, HP);
    cb_pop_front(qk_cb, HP);
}

/**
 * acc_cb (ring of unit tiles) += mul_cb[0..HP); when first, primed from mul_cb[0].
 */
template <uint32_t mul_cb, uint32_t acc_cb>
void accumulate_pass(bool first) {
    // Precondition: mul_cb has HP produced, acc front is this unit tile (unless first)
    // Postcondition: mul_cb has HP consumed, acc has this unit tile at back
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
 * acc front += mask_cb[idx] (0 = diag strict-upper -inf, 1 = full -inf)
 */
template <uint32_t acc_cb, uint32_t mask_cb>
void add_mask(uint32_t idx) {
    // Precondition: acc has 1 produced, mask_cb has 2 produced (resident)
    // Postcondition: acc has 1 produced
    reconfig_data_format(acc_cb, mask_cb);
    pack_reconfig_data_format(acc_cb);
    add_tiles_init(acc_cb, mask_cb);
    cb_wait_front(acc_cb, 1);
    tile_regs_acquire();
    add_tiles(acc_cb, mask_cb, 0, idx, 0);
    tile_regs_commit();
    cb_pop_front(acc_cb, 1);
    cb_reserve_back(acc_cb, 1);
    tile_regs_wait();
    pack_tile(0, acc_cb);
    tile_regs_release();
    cb_push_back(acc_cb, 1);
}

/**
 * out_cb = untilize(acc front), fp32 tile -> bf16 row-major
 */
template <uint32_t acc_cb, uint32_t out_cb>
void untilize_to_out() {
    // Precondition: acc has 1 produced
    // Postcondition: acc has 1 consumed, out_cb has 1 produced
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
    cb_wait_front(cb_mask, 2);

    WorkUnitSpan span;
    span.start(flat_start);

    bool have_group = false;
    for (uint32_t i = 0; i < flat_count; ++i) {
        if (!have_group) {
            cb_wait_front(cb_w, Hi * QC);
            if constexpr (!stream_heads) {
                cb_wait_front(cb_q, Hi * QC * Dt);
            }
            have_group = true;
        }
        const uint32_t kw = span.kw();
        cb_wait_front(cb_k, KC * Dt);  // full chunk pushed even when kw < KC (ring alignment)

        // each tile completes all head groups before the next, so cb_acc holds
        // exactly one in-flight tile (the serial L1 acc is the known perf TODO)
        for (uint32_t r = 0; r < QC; ++r) {
            const uint32_t diag = chunk_t + span.s0() + r;
            for (uint32_t c = 0; c < kw; ++c) {
                for (uint32_t hb = 0; hb < Hi; hb += HB) {
                    if constexpr (stream_heads) {
                        cb_wait_front(cb_q, HB * QC * Dt);
                    }
                    for (uint32_t hp = 0; hp < HB; hp += HP) {
                        relu_qk_subblock<cb_q, cb_k, cb_qk>(hp, r, c);
                        mul_gates_pass<cb_qk, cb_w, cb_mul>(r * Hi + hb + hp);
                        accumulate_pass<cb_mul, cb_acc>(hb == 0 && hp == 0);
                    }
                    if constexpr (stream_heads) {
                        cb_pop_front(cb_q, HB * QC * Dt);
                    }
                }
                const uint32_t t = span.c0() + c;
                if (t == diag) {
                    add_mask<cb_acc, cb_mask>(0);
                } else if (t > diag) {
                    add_mask<cb_acc, cb_mask>(1);
                }
                untilize_to_out<cb_acc, cb_out>();
            }
        }
        cb_pop_front(cb_k, KC * Dt);

        if (span.advance()) {
            cb_pop_front(cb_w, Hi * QC);
            if constexpr (!stream_heads) {
                cb_pop_front(cb_q, Hi * QC * Dt);
            }
            have_group = false;
        }
    }
}
