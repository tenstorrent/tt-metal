// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// FP32 online-softmax state for SDPA prefill (sdpa.cpp) and flash decode (sdpa_flash_decode.cpp), enabled by
// SDPAProgramConfig::fp32_accumulation. The program factories define SDPA_FP32_ACCUMULATION only when the option
// is on; without it nothing in this file is instantiated and the kernels compile exactly as before.
//
// Each K chunk rescales the running softmax sum and output accumulator and adds the chunk's contribution. The
// default kernels keep that state in BF16 (decode) or update it with FPU operations whose source registers hold
// 19-bit TF32 (prefill), so every chunk rounds it once more and the error grows with context length (~7% relative
// L2 at 512K tokens for decode). With the option on, the factories store the state in FP32 circular buffers that
// unpack straight to DEST (UnpackToDestFp32), and these helpers update it with SFPU multiply/add in FP32 DEST.
// Maxima and correction factors keep their formats: they are recomputed from the scores every chunk.
//
// Lossless reads take a (cb, view) pair. In decode, every FP32 state CB is itself UnpackToDestFp32 (no FPU op
// reads it), so view == cb. In prefill the sum and output CBs also feed FPU ops (matmul_reduce, the attention-sink
// rescale), so each has an UnpackToDestFp32 alias over the same SRAM whose read pointer follows the original's,
// as in the matmul partials reload (bmm_large_block_zm_fused_bias_activation.cpp).

#include "api/compute/eltwise_binary_sfpu.h"

#ifdef SDPA_FP32_ACCUMULATION
constexpr bool sdpa_fp32_accumulation = true;
#else
constexpr bool sdpa_fp32_accumulation = false;
#endif

namespace sdpa_fp32 {

// Prefill only: the UnpackToDestFp32 alias of an FP32 state CB (the factory defines the pairs); others map to
// themselves.
inline uint32_t lossless_view(uint32_t cb) {
#ifdef SDPA_FP32_SUM_A_VIEW
    if (cb == SDPA_FP32_SUM_A) {
        return SDPA_FP32_SUM_A_VIEW;
    }
    if (cb == SDPA_FP32_SUM_B) {
        return SDPA_FP32_SUM_B_VIEW;
    }
    if (cb == SDPA_FP32_OUT_A) {
        return SDPA_FP32_OUT_A_VIEW;
    }
    if (cb == SDPA_FP32_OUT_B) {
        return SDPA_FP32_OUT_B_VIEW;
    }
#endif
    return cb;
}

// Unpack tile `tile` of `cb` (through its lossless view) into DEST register `dst`.
inline void copy_to_dst(uint32_t cb, uint32_t tile, uint32_t dst) {
    const uint32_t view = lossless_view(cb);
    if (view != cb) {
        UNPACK((get_local_cb_interface(view).fifo_rd_ptr = get_local_cb_interface(cb).fifo_rd_ptr));
    }
    reconfig_data_format_srca(view);
    copy_init(view);
    copy_tile(view, tile, dst);
}

// Column-broadcast tile `tile` of `cb` (a column vector of per-row factors) into DEST register `dst`. FP32 views
// unpack straight to DEST; 16-bit factors go through srcB, so both unpacker formats are set.
inline void bcast_col_to_dst(uint32_t cb, uint32_t tile, uint32_t dst) {
    const uint32_t view = lossless_view(cb);
    if (view != cb) {
        UNPACK((get_local_cb_interface(view).fifo_rd_ptr = get_local_cb_interface(cb).fifo_rd_ptr));
    }
    reconfig_data_format(view, view);
    unary_bcast_init<BroadcastType::COL>(view);
    unary_bcast<BroadcastType::COL>(view, tile, dst);
    unary_bcast_uninit<BroadcastType::COL>(view);
}

/******************************************************************************
 *                                  Decode                                    *
 ******************************************************************************/

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

/******************************************************************************
 *                                  Prefill                                   *
 ******************************************************************************/

// sum *= bcast_col(alpha), row by row. Keeps alpha.
inline void mul_sum_inplace(uint32_t sum, uint32_t alpha, uint32_t rows) {
    CircularBuffer sums(sum), factors(alpha);
    sums.wait_front(rows);
    factors.wait_front(rows);
    pack_reconfig_data_format(sum);
    for (uint32_t row = 0; row < rows; ++row) {
        tile_regs_acquire();
        copy_to_dst(sum, 0, 0);
        bcast_col_to_dst(alpha, row, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        tile_regs_commit();
        sums.pop_front(1);
        sums.reserve_back(1);
        tile_regs_wait();
        pack_tile(0, sum);
        tile_regs_release();
        sums.push_back(1);
    }
}

// current += previous. Consumes previous.
inline void add_sum_inplace(uint32_t current, uint32_t previous, uint32_t rows) {
    CircularBuffer cur(current), prev(previous);
    cur.wait_front(rows);
    prev.wait_front(rows);
    pack_reconfig_data_format(current);
    for (uint32_t row = 0; row < rows; ++row) {
        tile_regs_acquire();
        copy_to_dst(current, row, 0);
        copy_to_dst(previous, row, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, current);
        tile_regs_release();
    }
    cur.pop_front(rows);
    prev.pop_front(rows);
    cur.reserve_back(rows);
    cur.push_back(rows);
}

// current += previous * bcast_col(alpha). Consumes previous and alpha.
template <uint32_t rows, uint32_t cols>
void accumulate_output(uint32_t previous, uint32_t alpha, uint32_t current) {
    constexpr uint32_t count = rows * cols;
    CircularBuffer prev(previous), factors(alpha), cur(current);
    prev.wait_front(count);
    factors.wait_front(rows);
    cur.wait_front(count);
    pack_reconfig_data_format(current);
    for (uint32_t tile = 0; tile < count; ++tile) {
        tile_regs_acquire();
        copy_to_dst(previous, tile, 0);
        bcast_col_to_dst(alpha, tile / cols, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        copy_to_dst(current, tile, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, current, tile);
        tile_regs_release();
    }
    prev.pop_front(count);
    factors.pop_front(rows);
    cur.pop_front(count);
    cur.reserve_back(count);
    cur.push_back(count);
}

// output = input * bcast_col(reciprocal). Consumes input and reciprocal.
template <uint32_t rows, uint32_t cols>
void normalize_output(uint32_t input, uint32_t reciprocal, uint32_t output) {
    constexpr uint32_t count = rows * cols;
    CircularBuffer in(input), scale(reciprocal), out(output);
    in.wait_front(count);
    scale.wait_front(rows);
    out.reserve_back(count);
    pack_reconfig_data_format(output);
    for (uint32_t tile = 0; tile < count; ++tile) {
        tile_regs_acquire();
        copy_to_dst(input, tile, 0);
        bcast_col_to_dst(reciprocal, tile / cols, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, output);
        tile_regs_release();
    }
    in.pop_front(count);
    scale.pop_front(rows);
    out.push_back(count);
}

}  // namespace sdpa_fp32

// exp((x - rowmax) * scale) for the FP32 prefill softmax, fusing the scale pass into a base-2 exponential.
// Computed as 2^z with z = x * (scale * log2 e). Inputs are scores minus the row maximum (<= ~0) or masked -inf, so
// one max(z, -126) replaces the generic overflow/underflow guards (-inf -> 2^-126). n = round(z) and
// f = z - n in [-0.5, 0.5] are exact in FP32; 2^f is a degree-4 minimax polynomial with p(0) = 1 and 2.9e-6 maximum
// relative error, far below the TF32 rounding the probabilities get on their way into the PV matmul. About 20 SFPU
// instructions per row instead of ~31 for the scale pass plus the guarded FP32 exponential. Blackhole only (the
// architecture it was validated on); elsewhere the FP32 path keeps the stock accurate exponential.
#if defined(ARCH_BLACKHOLE) && defined(TRISC_MATH)
namespace ckernel::sfpu {
inline void sdpa_fp32_exp2_init() {
    sfpi::vConstFloatPrgm0 = 0x1.62e126p-1f;  // c1
    sfpi::vConstFloatPrgm1 = 0x1.ec038cp-3f;  // c2
}

template <int ITERATIONS>
inline void sdpa_fp32_exp2_scaled(const uint32_t log2e_scale_bits) {
    const sfpi::vFloat c = Converter::as_float(log2e_scale_bits);
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat z = sfpi::max(sfpi::dst_reg[0] * c, -126.0f);
        sfpi::vSMag16 sm = sfpi::convert<sfpi::vSMag16>(z, sfpi::RoundMode::Nearest);
        sfpi::vFloat n = sfpi::convert<sfpi::vFloat>(sm, sfpi::RoundMode::Nearest);
        sfpi::vFloat f = z - n;
        sfpi::vFloat p = f * 0x1.3ap-7f + 0x1.cap-5f;  // c4, c3
        sfpi::vInt i = sfpi::abs(sfpi::as<sfpi::vInt>(sm));
        p = p * f + sfpi::vConstFloatPrgm1;
        i = sfpi::as<sfpi::vInt>(sfpi::copysgn(sfpi::as<sfpi::vFloat>(i), n));
        p = p * f + sfpi::vConstFloatPrgm0;
        p = p * f + 1.0f;
        sfpi::vInt e = sfpi::exexp(p, sfpi::ExponentMode::Biased) + i;
        sfpi::dst_reg[0] = sfpi::setexp(p, e);
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif

// scale * log2(e) in FP32, from the FP32 bit pattern of the attention scale.
constexpr uint32_t sdpa_fp32_log2e_scale_bits(uint32_t scale_bits) {
    return __builtin_bit_cast(uint32_t, __builtin_bit_cast(float, scale_bits) * 1.4426950408889634f);
}
