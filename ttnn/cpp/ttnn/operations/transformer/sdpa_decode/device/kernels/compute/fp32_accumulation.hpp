// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// FP32 online-softmax state for flash decode (sdpa_flash_decode.cpp), enabled by SDPAProgramConfig::fp32_accumulation.
// The program factory defines SDPA_FP32_ACCUMULATION only when the option is on; without it nothing in this file is
// instantiated and the kernel compiles exactly as before.
//
// Each K chunk rescales the running softmax sum and output accumulator and adds the chunk's contribution, and the
// tree reduction merges the partial states of the cores that split one head's context. The default kernel keeps that
// state in BF16, so every merge rounds it once more and the error grows with context length (~7% relative L2 at 512K
// tokens with 16 cores per head). With the option on, the factory stores scores, sums, outputs, correction factors and
// the tree-exchange buffers in FP32 circular buffers; those read here unpack straight to DEST (UnpackToDestFp32), and
// these helpers update the state with SFPU multiply/add in FP32 DEST. Maxima keep BF16: they are recomputed from the
// scores every chunk.
//
// Include after the SDPA compute helpers (move_block, fused_max_sub_exp_add_tile).

#include "api/compute/eltwise_binary_sfpu.h"

#ifdef SDPA_FP32_ACCUMULATION
constexpr bool sdpa_fp32_accumulation = true;
#else
constexpr bool sdpa_fp32_accumulation = false;
#endif

namespace sdpa_fp32 {

// Unpack tile `tile` of `cb` into DEST register `dst` (lossless for the UnpackToDestFp32 state CBs).
inline void copy_to_dst(uint32_t cb, uint32_t tile, uint32_t dst) {
    reconfig_data_format_srca(cb);
    copy_init(cb);
    copy_tile(cb, tile, dst);
}

// Column-broadcast tile `tile` of `cb` (a column vector of per-row factors) into DEST register `dst`. FP32 CBs
// unpack straight to DEST; 16-bit factors go through srcB, so both unpacker formats are set.
inline void bcast_col_to_dst(uint32_t cb, uint32_t tile, uint32_t dst) {
    reconfig_data_format(cb, cb);
    unary_bcast_init<BroadcastType::COL>(cb);
    unary_bcast<BroadcastType::COL>(cb, tile, dst);
    unary_bcast_uninit<BroadcastType::COL>(cb);
}

// sum[r] = row sums of the FP32 probabilities, in the reduce_c column-0 layout. reduce_tile<SUM> mis-reduces FP32
// tiles, so accumulate P @ col_identity (1.0 in column 0) in FP32 DEST, as prefill's matmul_reduce does. P enters
// srcB exactly as in the PV matmul, so numerator and denominator see the same probabilities.
template <uint32_t rows>
void row_sum(uint32_t scores_cb, uint32_t col_identity_cb, uint32_t sum_cb, uint32_t cols) {
    CircularBuffer scores(scores_cb), identity(col_identity_cb), sums(sum_cb);
    scores.wait_front(rows * cols);
    identity.wait_front(1);
    sums.reserve_back(rows);
    reconfig_data_format(col_identity_cb, scores_cb);  // matmul: in1 -> srcA, in0 -> srcB
    matmul_init(scores_cb, col_identity_cb);
    pack_reconfig_data_format(sum_cb);
    for (uint32_t row = 0; row < rows; ++row) {
        tile_regs_acquire();
        for (uint32_t col = 0; col < cols; ++col) {
            matmul_tiles(scores_cb, col_identity_cb, row * cols + col, 0, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, sum_cb);
        tile_regs_release();
    }
    sums.push_back(rows);
}

// cur_sum = cur_sum + prev_sum * alpha. Consumes prev_sum, keeps alpha.
inline void sum_update(uint32_t cur_sum, uint32_t prev_sum, uint32_t alpha, uint32_t rows) {
    CircularBuffer cur(cur_sum), prev(prev_sum), factors(alpha);
    cur.wait_front(rows);
    prev.wait_front(rows);
    factors.wait_front(rows);
    pack_reconfig_data_format(cur_sum);
    for (uint32_t row = 0; row < rows; ++row) {
        tile_regs_acquire();
        copy_to_dst(prev_sum, row, 0);
        copy_to_dst(alpha, row, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        copy_to_dst(cur_sum, 0, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
        tile_regs_commit();
        cur.pop_front(1);
        cur.reserve_back(1);
        tile_regs_wait();
        pack_tile(0, cur_sum);
        tile_regs_release();
        cur.push_back(1);
    }
    prev.pop_front(rows);
}

// out = out * bcast_col(alpha) + current. Consumes current and alpha.
template <uint32_t rows, uint32_t cols>
void out_update(uint32_t out_cb, uint32_t current_cb, uint32_t alpha) {
    constexpr uint32_t count = rows * cols;
    CircularBuffer out(out_cb), cur(current_cb), factors(alpha);
    out.wait_front(count);
    cur.wait_front(count);
    factors.wait_front(rows);
    pack_reconfig_data_format(out_cb);
    for (uint32_t tile = 0; tile < count; ++tile) {
        tile_regs_acquire();
        copy_to_dst(out_cb, 0, 0);
        bcast_col_to_dst(alpha, tile / cols, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        copy_to_dst(current_cb, tile, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
        tile_regs_commit();
        out.pop_front(1);
        out.reserve_back(1);
        tile_regs_wait();
        pack_tile(0, out_cb);
        tile_regs_release();
        out.push_back(1);
    }
    cur.pop_front(count);
    factors.pop_front(rows);
}

// Tree merge: out = out * bcast_col(e_local) + child * bcast_col(e_child). Consumes child and both factors.
template <uint32_t rows, uint32_t cols>
void out_merge(uint32_t out_cb, uint32_t e_local, uint32_t child_cb, uint32_t e_child) {
    constexpr uint32_t count = rows * cols;
    CircularBuffer out(out_cb), child(child_cb), local_factors(e_local), child_factors(e_child);
    out.wait_front(count);
    child.wait_front(count);
    local_factors.wait_front(rows);
    child_factors.wait_front(rows);
    pack_reconfig_data_format(out_cb);
    for (uint32_t tile = 0; tile < count; ++tile) {
        tile_regs_acquire();
        copy_to_dst(out_cb, 0, 0);
        bcast_col_to_dst(e_local, tile / cols, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        copy_to_dst(child_cb, tile, 1);
        bcast_col_to_dst(e_child, tile / cols, 2);
        mul_binary_tile_init();
        mul_binary_tile(1, 2, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
        tile_regs_commit();
        out.pop_front(1);
        out.reserve_back(1);
        tile_regs_wait();
        pack_tile(0, out_cb);
        tile_regs_release();
        out.push_back(1);
    }
    child.pop_front(count);
    local_factors.pop_front(rows);
    child_factors.pop_front(rows);
}

// Final normalization: out *= bcast_col(reciprocal). Consumes reciprocal.
template <uint32_t rows, uint32_t cols>
void scale_output(uint32_t out_cb, uint32_t reciprocal) {
    constexpr uint32_t count = rows * cols;
    CircularBuffer out(out_cb), scale(reciprocal);
    out.wait_front(count);
    scale.wait_front(rows);
    pack_reconfig_data_format(out_cb);
    for (uint32_t tile = 0; tile < count; ++tile) {
        tile_regs_acquire();
        copy_to_dst(out_cb, 0, 0);
        bcast_col_to_dst(reciprocal, tile / cols, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        tile_regs_commit();
        out.pop_front(1);
        out.reserve_back(1);
        tile_regs_wait();
        pack_tile(0, out_cb);
        tile_regs_release();
        out.push_back(1);
    }
    scale.pop_front(rows);
}

// correction_block for SDPAProgramConfig::fp32_accumulation (decode tree reduction): sums and correction factors
// are FP32 and read losslessly while maxima stay BF16, so every operand gets its own unpack configuration and
// every output its own pack configuration. The fused SFPU arithmetic is unchanged.
template <uint32_t scale_fp32, VectorMode vector_mode = VectorMode::C>
void correction_block(
    uint32_t cb_worker_max,
    uint32_t cb_worker_sum,
    uint32_t cb_cur_max,
    uint32_t cb_prev_max,
    uint32_t cb_cur_sum,
    uint32_t cb_prev_sum,
    uint32_t cb_exp_max_diff,
    uint32_t cb_exp_max_diff_2,
    uint32_t num_head_tiles) {
    CircularBuffer worker_max(cb_worker_max), worker_sum(cb_worker_sum), cur_max(cb_cur_max), prev_max(cb_prev_max),
        cur_sum(cb_cur_sum), prev_sum(cb_prev_sum), exp_diff(cb_exp_max_diff), exp_diff_2(cb_exp_max_diff_2);
    worker_max.wait_front(num_head_tiles);
    worker_sum.wait_front(num_head_tiles);
    prev_max.wait_front(num_head_tiles);
    prev_sum.wait_front(num_head_tiles);
    cur_max.reserve_back(num_head_tiles);
    cur_sum.reserve_back(num_head_tiles);
    exp_diff.reserve_back(num_head_tiles);
    exp_diff_2.reserve_back(num_head_tiles);
    // Same DST layout as correction_block (#56171).
    constexpr uint32_t worker_sum_dst = (DST_ACCUM_MODE && DST_SYNC_MODE == DstSync::SyncHalf) ? 2 : 4;
    static_assert(
        worker_sum_dst < compute_kernel_lib::DEST_AUTO_LIMIT,
        "correction_block DST layout exceeds DEST capacity for this sync/accum mode");
    constexpr uint16_t scale_bf16 = scale_fp32 >> 16;
    for (uint32_t i = 0; i < num_head_tiles; i++) {
        tile_regs_acquire();
        copy_to_dst(cb_prev_max, i, 0);
        copy_to_dst(cb_worker_max, i, 1);
        copy_to_dst(cb_prev_sum, i, 3);
        copy_to_dst(cb_worker_sum, i, worker_sum_dst);
        exp_tile_init<EXP_APPROX_MODE>();
        MATH((fused_max_sub_exp_add_tile<vector_mode>(0, scale_bf16)));
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_exp_max_diff);
        pack_tile(0, cb_exp_max_diff);
        pack_reconfig_data_format(cb_exp_max_diff_2);
        pack_tile(1, cb_exp_max_diff_2);
        pack_reconfig_data_format(cb_cur_max);
        pack_tile(2, cb_cur_max);
        pack_reconfig_data_format(cb_cur_sum);
        pack_tile(3, cb_cur_sum);
        tile_regs_release();
        cur_max.push_back(1);
        cur_sum.push_back(1);
        exp_diff.push_back(1);
        exp_diff_2.push_back(1);
    }
    prev_sum.pop_front(num_head_tiles);
    worker_sum.pop_front(num_head_tiles);
}

// move_block that sets both unpack and pack formats: under fp32_accumulation source and destination formats can
// differ (BF16 maxima moved into FP32 exchange CBs, for example).
template <bool pop_in_cb>
void move_block_reconfig(uint32_t in_cb, uint32_t out_cb, uint32_t num_tiles) {
    reconfig_data_format_srca(in_cb);
    pack_reconfig_data_format(out_cb);
    move_block<pop_in_cb>(in_cb, out_cb, num_tiles);
}

}  // namespace sdpa_fp32
