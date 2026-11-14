// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

// Precision contract: the FPU truncates a source-register operand to the multiplier's width (SrcA keeps 8 explicit
// mantissa bits, SrcB 10), which biases a product's magnitude low. So every elementwise product runs on the SFPU in
// FP32 DST, with FP32 operands unpacked to DST losslessly. The one FPU product is the row sum of squares, the reduce
// LLK's REDUCE_ROW SUM: x^2 (exact in FP32 for BF16 x) sits in SrcB, where it is truncated to TF32, against an exact
// 1.0 scaler in SrcA, accumulating in FP32 DST. That leaves a ~2e-4 low bias in the sum of squares, below a BF16 ulp,
// and needs HiFi3+ so SrcB's low mantissa bits take part.

// tmp[i] = x[i]^2 in FP32 (exact for BF16 x).
template <uint32_t Vt, typename DFB>
FORCE_INLINE void square(DFB& tmp) {
    tmp.reserve_back(Vt);
    pack_reconfig_data_format(dfb::tmp);
    reconfig_data_format_srca(dfb::x);
    copy_init(dfb::x);
    square_tile_init();
    for (uint32_t i = 0; i < Vt; i++) {
        tile_regs_acquire();
        copy_tile(dfb::x, i, 0);
        square_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, dfb::tmp, i);
        tile_regs_release();
    }
    tmp.push_back(Vt);
}

// inv = rsqrt(sum_j x_j^2 * inverse_width + epsilon), valid in column 0 (the reduce leaves the other columns
// unspecified, so the epilogue runs on the faces holding column 0, and the COL broadcast reads only that column).
template <uint32_t Vt, uint32_t inverse_width_bits, uint32_t epsilon_bits, typename DFB0, typename DFB1, typename DFB2>
FORCE_INLINE void inverse_rms(DFB0& tmp, DFB1& ones, DFB2& inv) {
    compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, dfb::tmp, dfb::ones, dfb::inv>(
        tmp,
        ones,
        inv,
        compute_kernel_lib::ReduceInputBlockShape::row(Vt),
        compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
        compute_kernel_lib::NoAccumulation{},
        [](uint32_t dst) {
            binop_with_scalar_tile_init();
            mul_unary_tile(dst, inverse_width_bits, VectorMode::C);
            add_unary_tile(dst, epsilon_bits, VectorMode::C);
            rsqrt_tile_init();
            rsqrt_tile(dst, VectorMode::C);
        });
}

// out[i] = x[i] * inv * weight[i] * sigmoid(gate[i]), all products in FP32 DST on the SFPU.
template <uint32_t Vt, typename DFB>
FORCE_INLINE void normalize_and_gate(DFB& out) {
    out.reserve_back(Vt);
    pack_reconfig_data_format(dfb::out);
    for (uint32_t i = 0; i < Vt; i++) {
        tile_regs_acquire();
        reconfig_data_format_srca(dfb::x);
        copy_init(dfb::x);
        copy_tile(dfb::x, i, 0);
        // inv's column 0 broadcasts across the row; FP32 inv goes through unpack-to-DST (the SrcA path), the BF16
        // weight's row 0 through SrcB.
        reconfig_data_format_srca(dfb::inv);
        unary_bcast_init<BroadcastType::COL>(dfb::inv);
        unary_bcast<BroadcastType::COL>(dfb::inv, 0, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        reconfig_data_format_srcb(dfb::weight);
        unary_bcast_init<BroadcastType::ROW>(dfb::weight);
        unary_bcast<BroadcastType::ROW>(dfb::weight, i, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        reconfig_data_format_srca(dfb::gate);
        copy_init(dfb::gate);
        copy_tile(dfb::gate, i, 1);
        sigmoid_tile_init();
        sigmoid_tile(1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, dfb::out, i);
        tile_regs_release();
    }
    out.push_back(Vt);
}

template <uint32_t Vt, uint32_t inverse_width_bits, uint32_t epsilon_bits>
TT_KERNEL void compute(uint32_t wi_count) {
    compute_kernel_hw_startup(dfb::ones, dfb::tmp, dfb::out);
    DataflowBuffer x(dfb::x);
    DataflowBuffer gate(dfb::gate);
    DataflowBuffer weight(dfb::weight);
    DataflowBuffer tmp(dfb::tmp);
    DataflowBuffer inv(dfb::inv);
    DataflowBuffer out(dfb::out);
    DataflowBuffer ones(dfb::ones);
    weight.wait_front(Vt);
    ones.wait_front(1);
    for (uint32_t i = 0; i < wi_count; i++) {
        x.wait_front(Vt);
        gate.wait_front(Vt);
        square<Vt>(tmp);
        inverse_rms<Vt, inverse_width_bits, epsilon_bits>(tmp, ones, inv);
        inv.wait_front(1);
        normalize_and_gate<Vt>(out);
        inv.pop_front(1);
        x.pop_front(Vt);
        gate.pop_front(Vt);
    }
}
