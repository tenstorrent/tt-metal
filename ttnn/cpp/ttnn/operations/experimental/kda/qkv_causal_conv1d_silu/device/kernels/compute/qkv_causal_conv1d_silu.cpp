// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/bcast.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/device/qkv_causal_conv1d_silu_layout.hpp"

#if defined(TRISC_MATH) || defined(TRISC_PACK)
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_recip.h"
#include "llk_math_eltwise_unary_sfpu_params.h"

namespace ckernel::sfpu {

// calculate_silu for a BFLOAT16 destination, operation for operation and so bit for bit: x * sigmoid(x), with
// sigmoid(x) = 1 / (1 + exp(-x)) from _sfpu_exp_21f_bf16_<true> and one reciprocal Newton step. It covers TILES
// whole tiles in one call and keeps its constants in registers instead of reloading them for every row: 1/ln2 and c2
// in the programmable constants (conv_silu_init), c0, c1 and the exponent bias in locals.
template <int TILES>
inline void calculate_conv_silu() {
    const sfpi::vFloat c0 = EXP_21F_BF16_C0;
    const sfpi::vFloat c1 = EXP_21F_BF16_C1;
    const sfpi::vFloat bias = 127.f;
#pragma GCC unroll 0
    for (int face = 0; face < 4 * TILES; face++) {
#pragma GCC unroll 8
        for (int d = 0; d < 8; d++) {
            const sfpi::vFloat x = sfpi::dst_reg[0];
            sfpi::vFloat xlog2 = -x * sfpi::vConstFloatPrgm1 + bias;
            xlog2 = sfpi::clamp(xlog2, 0.0f, 255.0f);
            const sfpi::vFloat z = sfpi::as<sfpi::vFloat>(_float_to_int32_for_exp_21f_(xlog2));
            const sfpi::vInt exponent = sfpi::exexp(z, sfpi::ExponentMode::Biased);
            sfpi::vFloat fraction = sfpi::convert<sfpi::vFloat>(sfpi::exman(z), sfpi::RoundMode::Nearest);
            fraction = PolynomialEvaluator::eval(fraction, c0, c1, sfpi::vConstFloatPrgm2);
            const sfpi::vFloat exp_neg_x = sfpi::setexp(fraction, exponent);
            const sfpi::vFloat sigmoid = sfpu_reciprocal_iter<1>(1.0f + exp_neg_x);
            sfpi::dst_reg[0] = sfpi::convert<sfpi::vFloat16b>(x * sigmoid, sfpi::RoundMode::Nearest);
            sfpi::dst_reg++;
        }
        math::inc_dst_addr<8>();
        math::inc_dst_addr<8>();
    }
}

// silu_init (reciprocal's 2.0 in vConstFloatPrgm0) plus the exponential's 1/ln2 and c2.
inline void conv_silu_init() {
    sfpi::vConstFloatPrgm1 = 1.4426950216293334961f;
    sfpi::vConstFloatPrgm2 = EXP_21F_BF16_C2;
}

}  // namespace ckernel::sfpu
#endif

// The reader supplies, per channel tile, the four tap-shifted input tiles and, per channel block, the tap tiles
// (weights in row 0). Scaling a tile's columns by a tap is a matmul with that tap's diagonal, so the four taps
// accumulate in the destination registers with no partial round trip; SiLU then runs in place. The diagonals are
// built here once per channel block, as the identity times the row-broadcast tap.
template <uint32_t block_ct, uint32_t Mt, uint32_t group, uint32_t defer_first_rows>
TT_KERNEL void compute(uint32_t wi_start, uint32_t wi_count) {
    constexpr uint32_t tap_count = 4;
    constexpr uint32_t weight_tiles = tap_count * block_ct;
    // Diagonals built per destination acquire.
    constexpr uint32_t diag_group = weight_tiles % 8 == 0 ? 8 : (weight_tiles % 4 == 0 ? 4 : 1);
    DataflowBuffer activation(dfb::act_tile);
    DataflowBuffer taps(dfb::taps);
    DataflowBuffer eye(dfb::eye);
    DataflowBuffer weights(dfb::weights);
    DataflowBuffer output(dfb::output);
    compute_kernel_hw_startup<SrcOrder::Reverse>(dfb::act_tile, dfb::weights, dfb::output);
    // SiLU runs on the pack thread, on the destination half the math thread has just committed, so it overlaps the
    // next group's matmuls.
    PACK((ckernel::llk_math_eltwise_unary_sfpu_init<SfpuType::silu>(ckernel::sfpu::silu_init<false>)));
    PACK((ckernel::sfpu::conv_silu_init()));
    eye.wait_front(1);

    constexpr uint32_t no_block = ~0U;
    uint32_t current_block = no_block;
    const auto process = [&](uint32_t work) {
        const uint32_t block = work / Mt;
        if (block != current_block) {
            if (current_block != no_block) {
                weights.pop_front(weight_tiles);
            }
            current_block = block;
            taps.wait_front(weight_tiles);
            weights.reserve_back(weight_tiles);
            mul_bcast_rows_init(dfb::eye, dfb::taps);
            for (uint32_t tile = 0; tile < weight_tiles; tile += diag_group) {
                tile_regs_acquire();
                for (uint32_t i = 0; i < diag_group; ++i) {
                    mul_tiles_bcast_rows(dfb::eye, dfb::taps, 0, tile + i, i);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < diag_group; ++i) {
                    pack_tile<true>(i, dfb::weights, tile + i);
                }
                tile_regs_release();
            }
            weights.push_back(weight_tiles);
            taps.pop_front(weight_tiles);
            weights.wait_front(weight_tiles);
            matmul_init(dfb::act_tile, dfb::weights);
        }
        for (uint32_t ct0 = 0; ct0 < block_ct; ct0 += group) {
            activation.wait_front(group * tap_count);
            output.reserve_back(group);
            tile_regs_acquire();
            for (uint32_t i = 0; i < group; ++i) {
                for (uint32_t tap = 0; tap < tap_count; ++tap) {
                    matmul_tiles(dfb::act_tile, dfb::weights, i * tap_count + tap, tap * block_ct + ct0 + i, i);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
            PACK(SFPU_UNARY_CALL(DST_SYNC_MODE, false, calculate_conv_silu, (group), 0, VectorMode::None));
            PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
            for (uint32_t i = 0; i < group; ++i) {
                pack_tile(i, dfb::output);
            }
            tile_regs_release();
            output.push_back(group);
            activation.pop_front(group * tap_count);
        }
    };
    qkv_conv::for_each_work_item(wi_start, wi_count, Mt, defer_first_rows, process);
    if (current_block != no_block) {
        weights.pop_front(weight_tiles);
    }
}
