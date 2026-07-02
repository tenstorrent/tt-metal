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

// Fused KV compute for one KV worker core (correctness-first v1: whole KV row on one core).
//
//   1. matmul: kv_raw[1, Nt] = hidden[1, Kt] @ Wkv[Kt, Nt]     (in0 resident, in1 K-blocked)
//   2. weighted RMSNorm over the full Dh: kv = kv_raw * rsqrt(mean(kv_raw^2)+eps) * gain
//   3. partial RoPE on the trailing rope_Wt tiles (rotate via trans_mat + cos/sin)
//
// hidden (in0) is streamed once and kept resident; the weight (in1) is streamed from DRAM in
// K-blocks per N-subblock so it never has to fit in L1 whole. cos/sin are a single tile-row
// broadcast across all rows (decode: one position shared by every row).
using namespace ckernel;

namespace {
constexpr uint32_t onetile = 1;
}

void kernel_main() {
    constexpr uint32_t in0_cb = get_compile_time_arg_val(0);
    constexpr uint32_t in1_cb = get_compile_time_arg_val(1);
    constexpr uint32_t scaler_cb = get_compile_time_arg_val(2);
    constexpr uint32_t gain_cb = get_compile_time_arg_val(3);
    constexpr uint32_t cos_cb = get_compile_time_arg_val(4);
    constexpr uint32_t sin_cb = get_compile_time_arg_val(5);
    constexpr uint32_t trans_mat_cb = get_compile_time_arg_val(6);
    constexpr uint32_t mm_cb = get_compile_time_arg_val(7);
    constexpr uint32_t x2_cb = get_compile_time_arg_val(8);
    constexpr uint32_t recip_cb = get_compile_time_arg_val(9);
    constexpr uint32_t normed_cb = get_compile_time_arg_val(10);
    constexpr uint32_t normed_g_cb = get_compile_time_arg_val(11);
    constexpr uint32_t rotated_cb = get_compile_time_arg_val(12);
    constexpr uint32_t cos_interm_cb = get_compile_time_arg_val(13);
    constexpr uint32_t sin_interm_cb = get_compile_time_arg_val(14);
    constexpr uint32_t out_cb = get_compile_time_arg_val(15);
    constexpr uint32_t Kt = get_compile_time_arg_val(16);
    constexpr uint32_t Nt = get_compile_time_arg_val(17);
    constexpr uint32_t in0_block_w = get_compile_time_arg_val(18);
    constexpr uint32_t subblock_w = get_compile_time_arg_val(19);
    constexpr uint32_t num_kb = get_compile_time_arg_val(20);
    constexpr uint32_t num_nsub = get_compile_time_arg_val(21);
    constexpr uint32_t nope_Wt = get_compile_time_arg_val(22);
    constexpr uint32_t rope_Wt = get_compile_time_arg_val(23);
    constexpr uint32_t eps_bits = get_compile_time_arg_val(24);

    CircularBuffer in0_cb_obj(in0_cb);
    CircularBuffer in1_cb_obj(in1_cb);
    CircularBuffer scaler_cb_obj(scaler_cb);
    CircularBuffer gain_cb_obj(gain_cb);
    CircularBuffer cos_cb_obj(cos_cb);
    CircularBuffer sin_cb_obj(sin_cb);
    CircularBuffer trans_mat_cb_obj(trans_mat_cb);
    CircularBuffer mm_cb_obj(mm_cb);
    CircularBuffer x2_cb_obj(x2_cb);
    CircularBuffer recip_cb_obj(recip_cb);
    CircularBuffer normed_cb_obj(normed_cb);
    CircularBuffer normed_g_cb_obj(normed_g_cb);
    CircularBuffer rotated_cb_obj(rotated_cb);
    CircularBuffer cos_interm_cb_obj(cos_interm_cb);
    CircularBuffer sin_interm_cb_obj(sin_interm_cb);
    CircularBuffer out_cb_obj(out_cb);

    compute_kernel_hw_startup<SrcOrder::Reverse>(in0_cb, in1_cb, mm_cb);

    // ---- 1) blocked matmul: mm[1, Nt] = hidden[1, Kt] @ Wkv[Kt, Nt] ----
    matmul_init(in0_cb, in1_cb);
    in0_cb_obj.wait_front(Kt);  // hidden resident for the whole matmul
    mm_cb_obj.reserve_back(Nt);
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
            pack_tile(w, mm_cb, ns * subblock_w + w);
        }
        tile_regs_release();
    }
    mm_cb_obj.push_back(Nt);
    in0_cb_obj.pop_front(Kt);

    // ---- 2) weighted RMSNorm over the full Dh ----
    mm_cb_obj.wait_front(Nt);

    // x2 = mm^2
    mul_tiles_init(mm_cb, mm_cb);
    x2_cb_obj.reserve_back(Nt);
    for (uint32_t wt = 0; wt < Nt; ++wt) {
        tile_regs_acquire();
        mul_tiles(mm_cb, mm_cb, wt, wt, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, x2_cb, wt);
        tile_regs_release();
    }
    x2_cb_obj.push_back(Nt);
    x2_cb_obj.wait_front(Nt);

    // recip = 1/sqrt(mean(x^2) + eps); mean via row reduce (scaler = 1/Dh), accumulated over Nt.
    scaler_cb_obj.wait_front(onetile);
    recip_cb_obj.reserve_back(onetile);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(x2_cb, scaler_cb, recip_cb);
    tile_regs_acquire();
    for (uint32_t wt = 0; wt < Nt; ++wt) {
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(x2_cb, scaler_cb, wt, 0, 0);
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
    x2_cb_obj.pop_front(Nt);

    // normed = mm * recip  (recip is per-row at col0; broadcast across columns)
    recip_cb_obj.wait_front(onetile);
    mul_bcast_cols_init_short(mm_cb, recip_cb);
    normed_cb_obj.reserve_back(Nt);
    for (uint32_t wt = 0; wt < Nt; ++wt) {
        tile_regs_acquire();
        mul_tiles_bcast_cols(mm_cb, recip_cb, wt, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, normed_cb, wt);
        tile_regs_release();
    }
    normed_cb_obj.push_back(Nt);
    recip_cb_obj.pop_front(onetile);
    mm_cb_obj.pop_front(Nt);

    // normed_g = normed * gain  (gain is a single row, broadcast across rows)
    normed_cb_obj.wait_front(Nt);
    gain_cb_obj.wait_front(Nt);
    mul_bcast_rows_init_short(normed_cb, gain_cb);
    normed_g_cb_obj.reserve_back(Nt);
    for (uint32_t wt = 0; wt < Nt; ++wt) {
        tile_regs_acquire();
        mul_tiles_bcast_rows(normed_cb, gain_cb, wt, wt, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, normed_g_cb, wt);
        tile_regs_release();
    }
    normed_g_cb_obj.push_back(Nt);
    normed_cb_obj.pop_front(Nt);
    gain_cb_obj.pop_front(Nt);

    // ---- 3) partial RoPE on the trailing rope_Wt tiles ----
    normed_g_cb_obj.wait_front(Nt);
    trans_mat_cb_obj.wait_front(onetile);
    cos_cb_obj.wait_front(rope_Wt);
    sin_cb_obj.wait_front(rope_Wt);
    out_cb_obj.reserve_back(Nt);

    // nope passthrough: out[j] = normed_g[j] for j in [0, nope_Wt)
    for (uint32_t j = 0; j < nope_Wt; ++j) {
        copy_tile_to_dst_init_short(normed_g_cb);
        tile_regs_acquire();
        copy_tile(normed_g_cb, j, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out_cb, j);
        tile_regs_release();
    }

    // rotated = normed_g_rope @ trans_mat  (per rope tile)
    matmul_init(normed_g_cb, trans_mat_cb);
    rotated_cb_obj.reserve_back(rope_Wt);
    tile_regs_acquire();
    for (uint32_t j = 0; j < rope_Wt; ++j) {
        matmul_tiles(normed_g_cb, trans_mat_cb, nope_Wt + j, 0, j);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t j = 0; j < rope_Wt; ++j) {
        pack_tile(j, rotated_cb, j);
    }
    tile_regs_release();
    rotated_cb_obj.push_back(rope_Wt);
    rotated_cb_obj.wait_front(rope_Wt);

    // sin_interm = rotated * sin  (sin single row broadcast across rows)
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

    // cos_interm = normed_g_rope * cos
    mul_bcast_rows_init_short(normed_g_cb, cos_cb);
    cos_interm_cb_obj.reserve_back(rope_Wt);
    for (uint32_t j = 0; j < rope_Wt; ++j) {
        tile_regs_acquire();
        mul_tiles_bcast_rows(normed_g_cb, cos_cb, nope_Wt + j, j, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cos_interm_cb, j);
        tile_regs_release();
    }
    cos_interm_cb_obj.push_back(rope_Wt);

    // out_rope = cos_interm + sin_interm -> out[nope_Wt + j]
    sin_interm_cb_obj.wait_front(rope_Wt);
    cos_interm_cb_obj.wait_front(rope_Wt);
    add_tiles_init(cos_interm_cb, sin_interm_cb);
    for (uint32_t j = 0; j < rope_Wt; ++j) {
        tile_regs_acquire();
        add_tiles(cos_interm_cb, sin_interm_cb, j, j, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out_cb, nope_Wt + j);
        tile_regs_release();
    }
    sin_interm_cb_obj.pop_front(rope_Wt);
    cos_interm_cb_obj.pop_front(rope_Wt);

    out_cb_obj.push_back(Nt);
    normed_g_cb_obj.pop_front(Nt);
    cos_cb_obj.pop_front(rope_Wt);
    sin_cb_obj.pop_front(rope_Wt);
    trans_mat_cb_obj.pop_front(onetile);
}
