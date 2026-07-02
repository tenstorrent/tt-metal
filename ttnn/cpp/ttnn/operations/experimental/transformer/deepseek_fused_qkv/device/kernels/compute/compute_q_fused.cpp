// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"

// Fused Q compute for one Q worker core (correctness-first v1). Each Q core recomputes the full
// q_a locally and owns a contiguous head-slice of q_b:
//   1. q_a = rmsnorm_w(hidden @ Wqa)          (weighted RMSNorm over q_lora)
//   2. q_slice = q_a @ Wqb[:, this core]      (head-slice of the q_b matmul)
//   3. per head: unweighted RMSNorm over Dh, then partial RoPE on the trailing rope_Wt tiles.
// hidden (in0) and normed q_a (in0 for step 2) are resident; the weights stream from DRAM K-blocked.
using namespace ckernel;

namespace {
constexpr uint32_t onetile = 1;

// Blocked matmul with resident in0: out[1, N] += in0[1, K] @ in1[K, N], accumulated over all K
// into `out_cb` (N tiles). in1 is streamed in K-blocks per N-subblock (reader order matches).
FORCE_INLINE void blocked_matmul(
    uint32_t in0_cb,
    uint32_t in1_cb,
    uint32_t out_cb,
    uint32_t N,
    uint32_t in0_block_w,
    uint32_t subblock_w,
    uint32_t num_kb,
    uint32_t num_nsub) {
    CircularBuffer in1_cb_obj(in1_cb);
    CircularBuffer out_cb_obj(out_cb);
    matmul_init(in0_cb, in1_cb);
    out_cb_obj.reserve_back(N);
    for (uint32_t ns = 0; ns < num_nsub; ++ns) {
        tile_regs_acquire();
        for (uint32_t kb = 0; kb < num_kb; ++kb) {
            in1_cb_obj.wait_front(in0_block_w * subblock_w);
            uint32_t in1_idx = 0;
            for (uint32_t kk = 0; kk < in0_block_w; ++kk) {
                const uint32_t in0_idx = kb * in0_block_w + kk;
                for (uint32_t w = 0; w < subblock_w; ++w) {
                    matmul_tiles(in0_cb, in1_cb, in0_idx, in1_idx, w);
                    ++in1_idx;
                }
            }
            in1_cb_obj.pop_front(in0_block_w * subblock_w);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t w = 0; w < subblock_w; ++w) {
            pack_tile(w, out_cb, ns * subblock_w + w);
        }
        tile_regs_release();
    }
    out_cb_obj.push_back(N);
}
}  // namespace

void kernel_main() {
    constexpr uint32_t in0_cb = get_compile_time_arg_val(0);
    constexpr uint32_t in1_cb = get_compile_time_arg_val(1);
    constexpr uint32_t scaler_qa_cb = get_compile_time_arg_val(2);
    constexpr uint32_t scaler_head_cb = get_compile_time_arg_val(3);
    constexpr uint32_t qa_gain_cb = get_compile_time_arg_val(4);
    constexpr uint32_t cos_cb = get_compile_time_arg_val(5);
    constexpr uint32_t sin_cb = get_compile_time_arg_val(6);
    constexpr uint32_t trans_mat_cb = get_compile_time_arg_val(7);
    constexpr uint32_t mm_qa_cb = get_compile_time_arg_val(8);
    constexpr uint32_t qa_cb = get_compile_time_arg_val(9);
    constexpr uint32_t mm_qb_cb = get_compile_time_arg_val(10);
    constexpr uint32_t x2_cb = get_compile_time_arg_val(11);
    constexpr uint32_t recip_cb = get_compile_time_arg_val(12);
    constexpr uint32_t normed_cb = get_compile_time_arg_val(13);
    constexpr uint32_t rotated_cb = get_compile_time_arg_val(14);
    constexpr uint32_t cos_interm_cb = get_compile_time_arg_val(15);
    constexpr uint32_t out_cb = get_compile_time_arg_val(16);
    constexpr uint32_t sin_interm_cb = get_compile_time_arg_val(17);
    constexpr uint32_t Kt_qa = get_compile_time_arg_val(18);
    constexpr uint32_t Nqa = get_compile_time_arg_val(19);
    constexpr uint32_t in0_block_w_qa = get_compile_time_arg_val(20);
    constexpr uint32_t subblock_w_qa = get_compile_time_arg_val(21);
    constexpr uint32_t num_kb_qa = get_compile_time_arg_val(22);
    constexpr uint32_t num_nsub_qa = get_compile_time_arg_val(23);
    constexpr uint32_t Kt_qb = get_compile_time_arg_val(24);
    constexpr uint32_t Nqb_core = get_compile_time_arg_val(25);
    constexpr uint32_t in0_block_w_qb = get_compile_time_arg_val(26);
    constexpr uint32_t subblock_w_qb = get_compile_time_arg_val(27);
    constexpr uint32_t num_kb_qb = get_compile_time_arg_val(28);
    constexpr uint32_t num_nsub_qb = get_compile_time_arg_val(29);
    constexpr uint32_t heads_per_core = get_compile_time_arg_val(30);
    constexpr uint32_t Dht_head = get_compile_time_arg_val(31);
    constexpr uint32_t nope_Wt_head = get_compile_time_arg_val(32);
    constexpr uint32_t rope_Wt = get_compile_time_arg_val(33);
    constexpr uint32_t eps_bits = get_compile_time_arg_val(34);

    CircularBuffer in0_cb_obj(in0_cb);
    CircularBuffer scaler_qa_cb_obj(scaler_qa_cb);
    CircularBuffer scaler_head_cb_obj(scaler_head_cb);
    CircularBuffer qa_gain_cb_obj(qa_gain_cb);
    CircularBuffer cos_cb_obj(cos_cb);
    CircularBuffer sin_cb_obj(sin_cb);
    CircularBuffer trans_mat_cb_obj(trans_mat_cb);
    CircularBuffer mm_qa_cb_obj(mm_qa_cb);
    CircularBuffer qa_cb_obj(qa_cb);
    CircularBuffer mm_qb_cb_obj(mm_qb_cb);
    CircularBuffer x2_cb_obj(x2_cb);
    CircularBuffer recip_cb_obj(recip_cb);
    CircularBuffer normed_cb_obj(normed_cb);
    CircularBuffer rotated_cb_obj(rotated_cb);
    CircularBuffer cos_interm_cb_obj(cos_interm_cb);
    CircularBuffer sin_interm_cb_obj(sin_interm_cb);
    CircularBuffer out_cb_obj(out_cb);

    compute_kernel_hw_startup<SrcOrder::Reverse>(in0_cb, in1_cb, mm_qa_cb);

    // ---- 1) q_a = hidden @ Wqa ----
    in0_cb_obj.wait_front(Kt_qa);
    blocked_matmul(in0_cb, in1_cb, mm_qa_cb, Nqa, in0_block_w_qa, subblock_w_qa, num_kb_qa, num_nsub_qa);
    in0_cb_obj.pop_front(Kt_qa);

    // ---- 2) weighted RMSNorm(q_lora): qa = mm_qa * rsqrt(mean(mm_qa^2)+eps) * gain ----
    mm_qa_cb_obj.wait_front(Nqa);
    mul_tiles_init(mm_qa_cb, mm_qa_cb);
    x2_cb_obj.reserve_back(Nqa);
    for (uint32_t wt = 0; wt < Nqa; ++wt) {
        tile_regs_acquire();
        mul_tiles(mm_qa_cb, mm_qa_cb, wt, wt, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, x2_cb, wt);
        tile_regs_release();
    }
    x2_cb_obj.push_back(Nqa);
    x2_cb_obj.wait_front(Nqa);

    scaler_qa_cb_obj.wait_front(onetile);
    recip_cb_obj.reserve_back(onetile);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(x2_cb, scaler_qa_cb, recip_cb);
    tile_regs_acquire();
    for (uint32_t wt = 0; wt < Nqa; ++wt) {
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(x2_cb, scaler_qa_cb, wt, 0, 0);
    }
    binop_with_scalar_tile_init();
    add_unary_tile(0, eps_bits);
    rsqrt_tile_init();
    rsqrt_tile(0);
    tile_regs_commit();
    reduce_uninit();
    tile_regs_wait();
    pack_tile(0, recip_cb, 0);
    tile_regs_release();
    recip_cb_obj.push_back(onetile);
    x2_cb_obj.pop_front(Nqa);

    recip_cb_obj.wait_front(onetile);
    mul_bcast_cols_init_short(mm_qa_cb, recip_cb);
    normed_cb_obj.reserve_back(Nqa);
    for (uint32_t wt = 0; wt < Nqa; ++wt) {
        tile_regs_acquire();
        mul_tiles_bcast_cols(mm_qa_cb, recip_cb, wt, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, normed_cb, wt);
        tile_regs_release();
    }
    normed_cb_obj.push_back(Nqa);
    recip_cb_obj.pop_front(onetile);
    mm_qa_cb_obj.pop_front(Nqa);

    normed_cb_obj.wait_front(Nqa);
    qa_gain_cb_obj.wait_front(Nqa);
    mul_bcast_rows_init_short(normed_cb, qa_gain_cb);
    qa_cb_obj.reserve_back(Nqa);
    for (uint32_t wt = 0; wt < Nqa; ++wt) {
        tile_regs_acquire();
        mul_tiles_bcast_rows(normed_cb, qa_gain_cb, wt, wt, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, qa_cb, wt);
        tile_regs_release();
    }
    qa_cb_obj.push_back(Nqa);
    normed_cb_obj.pop_front(Nqa);
    qa_gain_cb_obj.pop_front(Nqa);

    // ---- 3) q_slice = qa @ Wqb[:, this core] ----
    qa_cb_obj.wait_front(Nqa);
    blocked_matmul(qa_cb, in1_cb, mm_qb_cb, Nqb_core, in0_block_w_qb, subblock_w_qb, num_kb_qb, num_nsub_qb);
    qa_cb_obj.pop_front(Nqa);

    // ---- 4) per-head unweighted RMSNorm(Dh) + partial RoPE ----
    mm_qb_cb_obj.wait_front(Nqb_core);
    trans_mat_cb_obj.wait_front(onetile);
    cos_cb_obj.wait_front(rope_Wt);
    sin_cb_obj.wait_front(rope_Wt);
    out_cb_obj.reserve_back(Nqb_core);

    for (uint32_t h = 0; h < heads_per_core; ++h) {
        const uint32_t base = h * Dht_head;

        // x2 = head^2
        mul_tiles_init(mm_qb_cb, mm_qb_cb);
        x2_cb_obj.reserve_back(Dht_head);
        for (uint32_t wt = 0; wt < Dht_head; ++wt) {
            tile_regs_acquire();
            mul_tiles(mm_qb_cb, mm_qb_cb, base + wt, base + wt, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, x2_cb, wt);
            tile_regs_release();
        }
        x2_cb_obj.push_back(Dht_head);
        x2_cb_obj.wait_front(Dht_head);

        // recip = 1/sqrt(mean(head^2) + eps)  (scaler = 1/Dh)
        scaler_head_cb_obj.wait_front(onetile);
        recip_cb_obj.reserve_back(onetile);
        reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(x2_cb, scaler_head_cb, recip_cb);
        tile_regs_acquire();
        for (uint32_t wt = 0; wt < Dht_head; ++wt) {
            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(x2_cb, scaler_head_cb, wt, 0, 0);
        }
        binop_with_scalar_tile_init();
        add_unary_tile(0, eps_bits);
        rsqrt_tile_init();
        rsqrt_tile(0);
        tile_regs_commit();
        reduce_uninit();
        tile_regs_wait();
        pack_tile(0, recip_cb, 0);
        tile_regs_release();
        recip_cb_obj.push_back(onetile);
        x2_cb_obj.pop_front(Dht_head);

        // normed = head * recip  (unweighted)
        recip_cb_obj.wait_front(onetile);
        mul_bcast_cols_init_short(mm_qb_cb, recip_cb);
        normed_cb_obj.reserve_back(Dht_head);
        for (uint32_t wt = 0; wt < Dht_head; ++wt) {
            tile_regs_acquire();
            mul_tiles_bcast_cols(mm_qb_cb, recip_cb, base + wt, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, normed_cb, wt);
            tile_regs_release();
        }
        normed_cb_obj.push_back(Dht_head);
        recip_cb_obj.pop_front(onetile);

        // ---- partial RoPE on this head ----
        normed_cb_obj.wait_front(Dht_head);

        // nope passthrough -> out[base + j]
        for (uint32_t j = 0; j < nope_Wt_head; ++j) {
            copy_tile_to_dst_init_short(normed_cb);
            tile_regs_acquire();
            copy_tile(normed_cb, j, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, out_cb, base + j);
            tile_regs_release();
        }

        // rotated = normed_rope @ trans_mat
        matmul_init(normed_cb, trans_mat_cb);
        rotated_cb_obj.reserve_back(rope_Wt);
        tile_regs_acquire();
        for (uint32_t j = 0; j < rope_Wt; ++j) {
            matmul_tiles(normed_cb, trans_mat_cb, nope_Wt_head + j, 0, j);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < rope_Wt; ++j) {
            pack_tile(j, rotated_cb, j);
        }
        tile_regs_release();
        rotated_cb_obj.push_back(rope_Wt);
        rotated_cb_obj.wait_front(rope_Wt);

        // sin_interm = rotated * sin
        mul_bcast_rows_init_short(rotated_cb, sin_cb);
        sin_interm_cb_obj.reserve_back(rope_Wt);
        for (uint32_t j = 0; j < rope_Wt; ++j) {
            tile_regs_acquire();
            mul_tiles_bcast_rows(rotated_cb, sin_cb, j, j, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, sin_interm_cb, j);
            tile_regs_release();
        }
        sin_interm_cb_obj.push_back(rope_Wt);
        rotated_cb_obj.pop_front(rope_Wt);

        // cos_interm = normed_rope * cos
        mul_bcast_rows_init_short(normed_cb, cos_cb);
        cos_interm_cb_obj.reserve_back(rope_Wt);
        for (uint32_t j = 0; j < rope_Wt; ++j) {
            tile_regs_acquire();
            mul_tiles_bcast_rows(normed_cb, cos_cb, nope_Wt_head + j, j, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cos_interm_cb, j);
            tile_regs_release();
        }
        cos_interm_cb_obj.push_back(rope_Wt);

        // out_rope = cos_interm + sin_interm -> out[base + nope_Wt_head + j]
        sin_interm_cb_obj.wait_front(rope_Wt);
        cos_interm_cb_obj.wait_front(rope_Wt);
        add_tiles_init(cos_interm_cb, sin_interm_cb);
        for (uint32_t j = 0; j < rope_Wt; ++j) {
            tile_regs_acquire();
            add_tiles(cos_interm_cb, sin_interm_cb, j, j, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, out_cb, base + nope_Wt_head + j);
            tile_regs_release();
        }
        sin_interm_cb_obj.pop_front(rope_Wt);
        cos_interm_cb_obj.pop_front(rope_Wt);

        normed_cb_obj.pop_front(Dht_head);
    }

    out_cb_obj.push_back(Nqb_core);
    mm_qb_cb_obj.pop_front(Nqb_core);
    cos_cb_obj.pop_front(rope_Wt);
    sin_cb_obj.pop_front(rope_Wt);
    trans_mat_cb_obj.pop_front(onetile);
}
