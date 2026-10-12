// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/device/qkv_causal_conv1d_silu_layout.hpp"

#ifdef TRISC_PACK
namespace ckernel::sfpu {
// calculate_silu for a BFLOAT16 destination, bit for bit, made fast enough to keep up with the matmuls: one call
// covers `tiles` whole tiles, and the constants stay in registers instead of being reloaded for every row. With only
// eight vector registers, 1/ln2 and the exp polynomial's c2 go in the programmable constants (silu_tiles_init).
template <int tiles>
inline void silu_tiles() {
    const sfpi::vFloat c0 = EXP_21F_BF16_C0;
    const sfpi::vFloat c1 = EXP_21F_BF16_C1;
    const sfpi::vFloat bias = 127.f;
    for (int face = 0; face < 4 * tiles; face++) {
#pragma GCC unroll 8
        for (int row = 0; row < 8; row++) {
            const sfpi::vFloat x = sfpi::dst_reg[0];
            // exp(-x) = 2^(-x / ln2) as _sfpu_exp_21f_bf16_ computes it, then sigmoid = 1 / (1 + exp(-x)).
            const sfpi::vFloat z = sfpi::as<sfpi::vFloat>(
                _float_to_int32_for_exp_21f_(sfpi::clamp(-x * sfpi::vConstFloatPrgm1 + bias, 0.0f, 255.0f)));
            const sfpi::vFloat mantissa = sfpi::convert<sfpi::vFloat>(sfpi::exman(z), sfpi::RoundMode::Nearest);
            const sfpi::vFloat exp_neg_x = sfpi::setexp(
                PolynomialEvaluator::eval(mantissa, c0, c1, sfpi::vConstFloatPrgm2),
                sfpi::exexp(z, sfpi::ExponentMode::Biased));
            const sfpi::vFloat sigmoid = sfpu_reciprocal_iter<1>(1.0f + exp_neg_x);
            sfpi::dst_reg[0] = sfpi::convert<sfpi::vFloat16b>(x * sigmoid, sfpi::RoundMode::Nearest);
            sfpi::dst_reg++;
        }
        math::inc_dst_addr<8>();
        math::inc_dst_addr<8>();
    }
}

// silu_init, which sets the reciprocal's constant, plus 1/ln2 and c2.
inline void silu_tiles_init() {
    sfpi::vConstFloatPrgm1 = 1.4426950216293334961f;
    sfpi::vConstFloatPrgm2 = EXP_21F_BF16_C2;
}
}  // namespace ckernel::sfpu
#endif

// The reader supplies, per channel tile, the four tap-shifted input tiles and, per channel block, the diagonal tap
// weights. Scaling a tile's columns by a tap is a matmul with that tap's diagonal, so the four taps accumulate in the
// destination registers; SiLU then runs there on the pack thread, overlapping the next group's matmuls.
template <uint32_t block_ct, uint32_t Mt, uint32_t group, uint32_t defer_first_rows>
TT_KERNEL void compute(uint32_t wi_start, uint32_t wi_count) {
    constexpr uint32_t tap_count = 4;
    constexpr uint32_t weight_tiles = tap_count * block_ct;
    DataflowBuffer activation(dfb::act_tile);
    DataflowBuffer weights(dfb::weights);
    DataflowBuffer output(dfb::output);
    compute_kernel_hw_startup<SrcOrder::Reverse>(dfb::act_tile, dfb::weights, dfb::output);
    matmul_init(dfb::act_tile, dfb::weights);
    silu_tile_init_pack();
    PACK(ckernel::sfpu::silu_tiles_init());

    uint32_t current_block = ~0U;
    const auto process = [&](uint32_t work) {
        if (work / Mt != current_block) {
            if (current_block != ~0U) {
                weights.pop_front(weight_tiles);
            }
            current_block = work / Mt;
            weights.wait_front(weight_tiles);
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
            PACK(SFPU_UNARY_CALL(DST_SYNC_MODE, false, silu_tiles, (group), 0, VectorMode::None));
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
    if (current_block != ~0U) {
        weights.pop_front(weight_tiles);
    }
}
