// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The accumulator compute's side of a quantized row (common/kernels/quantized_row.hpp): a batch of the front row's
// flat tiles (as quantized_row_stage.hpp staged them) times the row's gate weight into DEST, then (combine_snap) each
// term to its int32 count of q - the accumulator packs DEST with L1 accumulation into the token's sum, as today.
//
//   Bf16     DEST = row x gate (FPU, the gate tile's element 0 broadcast), then round(DEST / q)        (today)
//   Int8     DEST = the row (FPU datacopy of its Bfp8_b tile: the values with their exponents, exact), then
//            round(DEST x gate / q) - the gate on the SFPU (exact: one fp32 rounding), so the FPU's fidelity does not
//            matter (a bcast multiply would need HiFi4 to stay exact)
//   Fp8E4m3  DEST = codes x factor (FPU ELWMUL, srcB's column 0 broadcast row by row, srcA / srcB / DEST stepping
//            together: DEST row r x its own factor, as sparse_sdpa_fused_dequant.hpp; one LoFi pass, exact: e4m3 x a
//            power of two), then round(DEST x gate x 2^-kFp8FactorShift / q) - the gate on the SFPU, exactly
//
// Usage (after compute_kernel_hw_startup(cb_row, cb_b, cb_acc); cb_b: the gate CB for Bf16 (Int8: unused - pass the
// gate CB or cb_row), the factor CB for Fp8E4m3 - declared UInt16 on the host: the JIT takes no FP16 CB next to BF16
// ones, init() makes it FP16):
//
//     using Weigher = quantized_row::RowWeigher<format, cb_row, cb_b, inverse_step_bits>;   // 0: no snap (fp32 sums)
//     Weigher::init();
//     for each row: cb_wait_front(cb_row, tiles) (and cb_b: 1 gate tile, or tiles factor pages)
//         for each batch of DEST tiles:
//             tile_regs_acquire();
//             Weigher::weigh(first, count, gate_bf16_bits);      // DEST 0 .. count - 1
//             tile_regs_commit(); tile_regs_wait(); pack_tile<true>(...) with L1 accumulation; tile_regs_release();
//         pop the row (and cb_b)
//
// The weigh re-inits the FPU op and the SFPU each batch (as the accumulator does: the SFPU's init in between).
#pragma once

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/quantized_row.hpp"

#ifdef TRISC_MATH
#include "sfpi.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"

namespace ckernel::sfpu {
// A DEST face: v x factor, then (snap) round to nearest even as a sign-magnitude int32 count, |count| below 2^31
// (accumulator_compute.cpp's snap_to_sign_magnitude_count, the factor at run time)
template <bool snap>
inline void quantized_row_weigh_face(uint32_t factor_bits) {
    const sfpi::vFloat factor = __builtin_bit_cast(float, factor_bits);
    const sfpi::vFloat largest_count = 2147483520.0f;  // the largest fp32 below 2^31
    const sfpi::vInt two_to_23_bits = 0x4B000000;
#pragma GCC unroll 8
    for (int face_row = 0; face_row < 8; face_row++) {
        const sfpi::vFloat scaled = sfpi::vFloat(sfpi::dst_reg[0]) * factor;
        if constexpr (snap) {
            const sfpi::vFloat magnitude = sfpi::min(sfpi::abs(scaled), largest_count);
            // below 2^23, adding 2^23 rounds the magnitude to an integer: the sum's bits less 2^23's
            sfpi::vInt count_magnitude = sfpi::as<sfpi::vInt>(magnitude + 8388608.0f) - two_to_23_bits;
            const sfpi::vInt mantissa_shift = sfpi::exexp(magnitude) - 23;
            v_if(mantissa_shift >= 0) {  // from 2^23 on the magnitude is an integer: its mantissa, shifted up
                count_magnitude = sfpi::as<sfpi::vInt>(sfpi::shft(
                    sfpi::as<sfpi::vUInt>(sfpi::exman(magnitude, sfpi::MantissaMode::WithUnitBit)), mantissa_shift));
            }
            v_endif;
            sfpi::dst_reg[0] = sfpi::copysgn(sfpi::as<sfpi::vSMag>(count_magnitude), scaled);
        } else {
            sfpi::dst_reg[0] = scaled;
        }
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu

namespace quantized_row::detail {
// Fp8E4m3's FPU op: ELWMUL with srcB's column 0 broadcast over each row; srcA / srcB / DEST rows step together, 8 an
// instruction, 8 instructions a tile
inline void fp8_math_init() {
    addr_mod_t{.srca = {.incr = 8}, .srcb = {.incr = 8}, .dest = {.incr = 8}}.set(ADDR_MOD_2);
    ckernel_template tmp(1, 8, TT_OP_ELWMUL(0, 0, p_elwise::SRCB_BCAST_COL, ADDR_MOD_2, 0));
    tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_AB));
    tmp.program();
    math::reset_counters(p_setrwc::SET_ABD_F);
    math::_configure_src_zero_flag_(false);  // (fp8 zeros widen into srcA with a residual unless flushed)
}
inline void fp8_math_tile(uint32_t dst_index) {
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);
    ckernel_template::run();
    math::clear_dst_reg_addr();
}
}  // namespace quantized_row::detail
#endif

#ifdef TRISC_UNPACK
namespace quantized_row::detail {
// Fp8E4m3: a flat tile of each operand, all 4 faces into source rows 0..63 (face f to slot f), data valid once (the
// math takes the whole tile in one pass); sparse_sdpa_fused_dequant.hpp's unpack without its tilize. Addresses in
// 16-byte units, less one (as the CB's fifo pointers).
inline void fp8_unpack_tile(uint32_t address_a, uint32_t address_b) {
    volatile uint tt_reg_ptr* cfg = get_cfg_pointer();
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
    wait_for_next_context(2);
    _llk_unpack_configure_addresses_(address_a, address_b, cfg);
    semaphore_post(semaphore::UNPACK_SYNC);
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);
    // SETADCZW mask 0b0101 = {ch0_z, ch1_z}: ch0_z the L1 face, ch1_z the source register's face slot
    TTI_SETADCZW(p_setadc::UNP_B, 0, 0, 0, 0, 0b0101);
    TTI_UNPACR(SrcB, 0, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    TTI_SETADCZW(p_setadc::UNP_B, 0, 1, 0, 1, 0b0101);
    TTI_UNPACR(SrcB, 0, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    TTI_SETADCZW(p_setadc::UNP_B, 0, 2, 0, 2, 0b0101);
    TTI_UNPACR(SrcB, 0, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    TTI_SETADCZW(p_setadc::UNP_B, 0, 3, 0, 3, 0b0101);
    TTI_UNPACR(SrcB, 0, 0, 0, 0, 1, 1 /*dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    TTI_SETADCZW(p_setadc::UNP_A, 0, 0, 0, 0, 0b0101);
    TTI_UNPACR(SrcA, 0, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    TTI_SETADCZW(p_setadc::UNP_A, 0, 1, 0, 1, 0b0101);
    TTI_UNPACR(SrcA, 0, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    TTI_SETADCZW(p_setadc::UNP_A, 0, 2, 0, 2, 0b0101);
    TTI_UNPACR(SrcA, 0, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    TTI_SETADCZW(p_setadc::UNP_A, 0, 3, 0, 3, 0b0101);
    TTI_UNPACR(SrcA, 0, 0, 0, 0, 1, 1 /*dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    t6_semaphore_get(semaphore::UNPACK_SYNC);
    switch_config_context(unp_cfg_context);
}
}  // namespace quantized_row::detail
#endif

namespace quantized_row {

template <Format format, uint32_t cb_row, uint32_t cb_b, uint32_t inverse_step_bits>
struct RowWeigher {
    static constexpr bool kSnap = inverse_step_bits != 0;
    // the gate on the SFPU (Fp8E4m3, Int8): its factor is gate x 2^(log2(1 / q) - shift) - an exponent add on the
    // gate's fp32 bits (shift: Fp8E4m3's factor shift, 0 for Int8)
    static constexpr bool kGateOnSfpu = format != Format::Bf16;
    static constexpr int32_t kGateExponentAdd = (kSnap ? static_cast<int32_t>(inverse_step_bits >> 23) - 127 : 0) -
                                                (format == Format::Fp8E4m3 ? kFp8FactorShift : 0);

    // Once, after compute_kernel_hw_startup(cb_row, cb_b, cb_acc)
    static void init() {
        if constexpr (format == Format::Fp8E4m3) {
            // srcB FP16 (the factor CB is UInt16 on the host): fp8 unpacks to FP16 in srcA and the FPU takes one
            // exponent family
            UNPACK((_llk_unpack_reconfig_data_format_srcb_impl_<DST_ACCUM_MODE, p_dim_stride_target::FACE_ROW_MAJOR>(
                static_cast<uint32_t>(DataFormat::Float16),
                static_cast<uint32_t>(DataFormat::Float16),
                get_local_cb_interface(cb_b).fifo_page_size,
                FACE_R_DIM,
                4)));
            MATH((_llk_math_reconfig_data_format_srcb_<DST_ACCUM_MODE>(static_cast<uint32_t>(DataFormat::Float16))));
            UNPACK((llk_unpack_AB_init<BroadcastType::NONE>(cb_row, cb_b, Transpose::None)));
        }
    }

    // DEST tiles 0 .. count - 1 = flat tiles first .. first + count - 1 of the front row x its gate (bf16 bits; Bf16
    // takes the gate from cb_b's front tile instead), snapped to int32 counts of q (fp32 without the snap)
    static void weigh(uint32_t first, uint32_t count, [[maybe_unused]] uint32_t gate_bits) {
        uint32_t factor_bits = inverse_step_bits;
        if constexpr (format == Format::Fp8E4m3) {
            MATH((detail::fp8_math_init()));
            for (uint32_t j = 0; j < count; ++j) {
                UNPACK((detail::fp8_unpack_tile(
                    get_local_cb_interface(cb_row).fifo_rd_ptr - 1 +
                        (first + j) * get_local_cb_interface(cb_row).fifo_page_size,
                    get_local_cb_interface(cb_b).fifo_rd_ptr - 1 +
                        (first + j) * get_local_cb_interface(cb_b).fifo_page_size)));
                MATH((detail::fp8_math_tile(j)));
            }
        } else if constexpr (format == Format::Int8) {
            copy_init(cb_row);
            for (uint32_t j = 0; j < count; ++j) {
                copy_tile(cb_row, first + j, j);
            }
        } else {
            mul_bcast_scalar_init(cb_row, cb_b);
            for (uint32_t j = 0; j < count; ++j) {
                mul_tiles_bcast_scalar(cb_row, cb_b, first + j, 0, j);
            }
        }
        if constexpr (kGateOnSfpu) {
            factor_bits = gate_bits == 0
                              ? 0u
                              : static_cast<uint32_t>(static_cast<int32_t>(gate_bits << 16) + (kGateExponentAdd << 23));
        }
        if constexpr (kSnap || kGateOnSfpu) {
            MATH((SFPU_UNARY_INIT(unused)));
            for (uint32_t j = 0; j < count; ++j) {
                MATH((SFPU_UNARY_CALL(
                    DST_SYNC_MODE, DST_ACCUM_MODE, quantized_row_weigh_face, (kSnap), j, VectorMode::RC, factor_bits)));
            }
        }
    }
};

}  // namespace quantized_row
