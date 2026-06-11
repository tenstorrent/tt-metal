// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute for indexer_score. Per work unit (q_tiles_per_unit q-tile-rows x
// k_tiles_in_unit k-tiles):
//   acc[r,c] = sum_h relu(q[h,row,:] @ k[col,:]^T) * w[h,row]
//   tile on the causal diagonal: += -inf strict upper triangle; past it: full -inf
//   pack_untilize acc -> bf16 row-major out, (r, c) row-major order
// Heads stream in heads_per_group groups, heads_per_dest_pass rows per DEST
// subblock (bf16 DEST by default, half sync; sized by
// determine_largest_subblock_size on the host). q/w stay resident per group
// when all heads fit.

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/cb_api.h"
#include "api/compute/tile_move_copy.h"

#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "indexer_score_common.hpp"

constexpr uint32_t cb_q = tt::CBIndex::c_0;
constexpr uint32_t cb_k = tt::CBIndex::c_1;
constexpr uint32_t cb_w = tt::CBIndex::c_2;
constexpr uint32_t cb_mask = tt::CBIndex::c_3;
constexpr uint32_t cb_qk = tt::CBIndex::c_24;
constexpr uint32_t cb_mul = tt::CBIndex::c_25;
constexpr uint32_t cb_acc = tt::CBIndex::c_26;
constexpr uint32_t cb_out = tt::CBIndex::c_16;

// qk subblock height (head rows per DEST pass); first per-kernel compile-time arg.
constexpr uint32_t heads_per_dest_pass = get_compile_time_arg_val(num_common_ct_args);

/**
 * qk_cb = relu(q_cb[head_in_group..+heads_per_dest_pass) of row r @ k_cb col c^T): one subblock;
 * relu applied by the packer on the way out of DEST. head_in_group is the head
 * offset within the resident group (0..heads_per_group).
 * q blocks are [q_tiles_per_unit][heads_per_group][head_dim_tiles] so the
 * subblock's head rows stride head_dim_tiles - one matmul_block.
 */
template <uint32_t q_cb, uint32_t k_cb, uint32_t qk_cb>
void relu_qk_subblock(uint32_t head_in_group, uint32_t r, uint32_t c) {
    // Precondition: q_cb has a q group produced (resident), k_cb has the k chunk produced (resident)
    // Postcondition: qk_cb has heads_per_dest_pass produced
    reconfig_data_format(q_cb, k_cb);
    pack_reconfig_data_format(qk_cb);
    mm_block_init_short(
        q_cb, k_cb, 1 /*transpose k*/, 1 /*ct_dim*/, heads_per_dest_pass /*rt_dim*/, head_dim_tiles /*kt_dim*/);
    tile_regs_acquire();
    const uint32_t q_base = (r * heads_per_group + head_in_group) * head_dim_tiles;
    for (uint32_t d = 0; d < head_dim_tiles; ++d) {
        matmul_block(
            q_cb,
            k_cb,
            q_base + d,
            c * head_dim_tiles + d,
            0,
            1 /*transpose k*/,
            1,
            heads_per_dest_pass,
            head_dim_tiles);
    }
    tile_regs_commit();
    cb_reserve_back(qk_cb, heads_per_dest_pass);
    tile_regs_wait();
    // relu in the packer; every other pack must stay linear (negative gates, -inf masks)
    pack_relu_config(ReluConfig::zero());
    for (uint32_t h = 0; h < heads_per_dest_pass; ++h) {
        pack_tile(h, qk_cb);
    }
    pack_relu_config(ReluConfig::none());
    tile_regs_release();
    cb_push_back(qk_cb, heads_per_dest_pass);
}

/**
 * mul_cb[h] = qk_cb[h] * bcast_cols(w_cb[w_base + h]) for one DEST pass
 */
template <uint32_t qk_cb, uint32_t w_cb, uint32_t mul_cb>
void mul_gates_pass(uint32_t w_base) {
    // Precondition: qk_cb has heads_per_dest_pass produced, w_cb has the w group produced (resident)
    // Postcondition: qk_cb has heads_per_dest_pass consumed, mul_cb has heads_per_dest_pass produced
    cb_wait_front(qk_cb, heads_per_dest_pass);
    reconfig_data_format(qk_cb, w_cb);
    pack_reconfig_data_format(mul_cb);
    mul_bcast_cols_init_short(qk_cb, w_cb);
    cb_reserve_back(mul_cb, heads_per_dest_pass);
    tile_regs_acquire();
    for (uint32_t h = 0; h < heads_per_dest_pass; ++h) {
        mul_tiles_bcast_cols(qk_cb, w_cb, h, w_base + h, h);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t h = 0; h < heads_per_dest_pass; ++h) {
        pack_tile(h, mul_cb);
    }
    tile_regs_release();
    cb_push_back(mul_cb, heads_per_dest_pass);
    cb_pop_front(qk_cb, heads_per_dest_pass);
}

/**
 * acc_cb (ring of unit tiles) += mul_cb[0..heads_per_dest_pass); when first, primed from mul_cb[0].
 */
template <uint32_t mul_cb, uint32_t acc_cb>
void accumulate_pass(bool first) {
    // Precondition: mul_cb has heads_per_dest_pass produced, acc front is this unit tile (unless first)
    // Postcondition: mul_cb has heads_per_dest_pass consumed, acc has this unit tile at back
    cb_wait_front(mul_cb, heads_per_dest_pass);
    pack_reconfig_data_format(acc_cb);
    uint32_t h = 0;
    if (first) {
        // no pre-zeroed acc tile exists, so the first head primes the ring by copy instead of add
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
    for (; h < heads_per_dest_pass; ++h) {
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
    cb_pop_front(mul_cb, heads_per_dest_pass);
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

void kernel_main() {
    const uint32_t flat_start = get_arg_val<uint32_t>(0);
    const uint32_t flat_count = get_arg_val<uint32_t>(1);
    if (flat_count == 0) {
        return;
    }

    mm_block_init(
        cb_q, cb_k, cb_qk, 1 /*transpose k*/, 1 /*ct_dim*/, heads_per_dest_pass /*rt_dim*/, head_dim_tiles /*kt_dim*/);
    cb_wait_front(cb_mask, 2);

    WorkUnitSpan span;
    span.start(flat_start);

    bool have_group = false;
    for (uint32_t i = 0; i < flat_count; ++i) {
        if (!have_group) {
            cb_wait_front(cb_w, w_group_tiles);
            if constexpr (!stream_heads) {
                cb_wait_front(cb_q, q_group_tiles);  // all heads form one resident block (heads_per_group == num_heads)
            }
            have_group = true;
        }
        const uint32_t k_tiles_in_unit = span.k_tiles();
        cb_wait_front(cb_k, k_chunk_tiles);  // full chunk pushed even on edge units (ring alignment)

        // each tile completes all head groups before the next, so cb_acc holds
        // exactly one in-flight tile (the serial L1 acc is the known perf TODO)
        for (uint32_t r = 0; r < q_tiles_per_unit; ++r) {
            const uint32_t diag_tile = chunk_start_tiles + span.q_tile_start() + r;
            for (uint32_t c = 0; c < k_tiles_in_unit; ++c) {
                for (uint32_t group_start = 0; group_start < num_heads; group_start += heads_per_group) {
                    if constexpr (stream_heads) {
                        cb_wait_front(cb_q, q_group_tiles);
                    }
                    for (uint32_t head = 0; head < heads_per_group; head += heads_per_dest_pass) {
                        // w is laid out [q_tiles_per_unit][num_heads] (see reader read_w_group)
                        const uint32_t w_base = r * num_heads + group_start + head;
                        relu_qk_subblock<cb_q, cb_k, cb_qk>(head, r, c);
                        mul_gates_pass<cb_qk, cb_w, cb_mul>(w_base);
                        accumulate_pass<cb_mul, cb_acc>(group_start == 0 && head == 0);
                    }
                    if constexpr (stream_heads) {
                        cb_pop_front(cb_q, q_group_tiles);
                    }
                }
                const uint32_t k_tile = span.k_tile_start() + c;
                if (k_tile == diag_tile) {
                    add_mask<cb_acc, cb_mask>(0);
                } else if (k_tile > diag_tile) {
                    add_mask<cb_acc, cb_mask>(1);
                }
                // acc (1 tile) -> bf16 row-major out; go-to untilize helper
                compute_kernel_lib::untilize<1, cb_acc, cb_out>(1);
            }
        }
        cb_pop_front(cb_k, k_chunk_tiles);

        if (span.advance()) {
            cb_pop_front(cb_w, w_group_tiles);
            if constexpr (!stream_heads) {
                cb_pop_front(cb_q, q_group_tiles);
            }
            have_group = false;
        }
    }
}
