// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/bcast.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

// Precision contract: every multiply whose operands are not both BF16 runs on the SFPU in FP32 DST, with FP32
// operands unpacked to DST losslessly. The FPU truncates a source-register operand to the multiplier's width
// (SrcA keeps 8 explicit mantissa bits, SrcB 10), which biases the product's magnitude low. The only remaining
// FPU product is the row sum of squares: x^2 (exact in FP32 for BF16 x) enters the matmul as in0, i.e. SrcB
// (TF32, 10 bits), against an exact all-ones tile in SrcA, accumulating in FP32 DST.

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

// inv = rsqrt(sum_j x_j^2 * inverse_width + epsilon), the row value replicated across every column.
template <uint32_t Vt, uint32_t inverse_width_bits, uint32_t epsilon_bits, typename DFB>
FORCE_INLINE void inverse_rms(DFB& inv) {
    inv.reserve_back(1);
    pack_reconfig_data_format(dfb::inv);
    reconfig_data_format(dfb::ones, dfb::tmp);
    matmul_init(dfb::tmp, dfb::ones);
    tile_regs_acquire();
    for (uint32_t i = 0; i < Vt; i++) {
        matmul_tiles(dfb::tmp, dfb::ones, i, 0, 0);
    }
    binop_with_scalar_tile_init();
    mul_unary_tile(0, inverse_width_bits);
    add_unary_tile(0, epsilon_bits);
    rsqrt_tile_init();
    rsqrt_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, dfb::inv, 0);
    tile_regs_release();
    inv.push_back(1);
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
        tmp.wait_front(Vt);
        inverse_rms<Vt, inverse_width_bits, epsilon_bits>(inv);
        tmp.pop_front(Vt);
        inv.wait_front(1);
        normalize_and_gate<Vt>(out);
        inv.pop_front(1);
        x.pop_front(Vt);
        gate.pop_front(Vt);
    }
}
