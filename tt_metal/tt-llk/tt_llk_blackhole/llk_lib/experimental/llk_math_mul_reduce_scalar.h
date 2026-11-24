// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL port of tt_llk_blackhole/llk_lib/experimental/llk_math_mul_reduce_scalar.h. Same Tensix instruction sequence
// as production, expressed through the HAL. The production MOVD2A_* macros are replaced by
// hal_port::mul_reduce_scalar::movd2a_16_rows (nothing outside this header uses them).

#pragma once

#include <cstdint>
#include <utility>

#include "ckernel_globals.h"
#include "ckernel_include.h"
#include "ckernel_ops.h"
#include "cmath_common.h"
#include "hal/cfg.h"
#include "hal/counters/rwc.h"
#include "hal/dst.h"
#include "hal/move.h"
#include "hal/src.h"
#include "hal/sync.h"
#include "llk_defs.h"
#include "llk_math_common.h"
#include "llk_math_reduce.h"
#include "tensor_shape.h"

using namespace ckernel;

namespace hal_port
{
namespace mul_reduce_scalar
{
namespace move = hal::move;

// TTI_STALLWAIT(STALL_MATH, MATH | SRCA_VLD) (cmath_common.h srca_bank_wait)
inline void srca_bank_wait()
{
    sync::wait::stall<sync::StallTarget::Math, sync::StallCondition::MathIdle | sync::StallCondition::SrcAValid>();
}

// TTI_STALLWAIT(STALL_MATH, WAIT_SFPU)
inline void wait_sfpu()
{
    sync::wait::stall<sync::StallTarget::Math, sync::StallCondition::SfpuIdle>();
}

// MOVD2A_16_ROWS(Base): TTI_MOVD2A(0, MATH_HALO_ROWS + r * 4, ADDR_MOD_0, MOV_4_ROWS, Base + r * 4), r = 0..15
template <std::uint32_t Base, std::uint32_t... R>
inline __attribute__((always_inline)) void movd2a_rows(std::integer_sequence<std::uint32_t, R...>)
{
    (move::run<move::Transfer<move::Dst, move::SrcA> {
         .source_row = Base + R * 4, .destination_row = p_mova2d::MATH_HALO_ROWS + R * 4, .address_mode = ADDR_MOD_0, .number_of_rows = move::rows::Four}>(),
     ...);
}

template <std::uint32_t Base>
inline __attribute__((always_inline)) void movd2a_16_rows()
{
    movd2a_rows<Base>(std::make_integer_sequence<std::uint32_t, 16> {});
}

// TTI_MOVD2B(0, SRC_ZERO_OFFSET + r * 4, ADDR_MOD_1, MOV_4_ROWS, r * 4), r = 0..15
template <std::uint32_t... R>
inline __attribute__((always_inline)) void movd2b_rows(std::integer_sequence<std::uint32_t, R...>)
{
    (move::run<move::Transfer<move::Dst, move::SrcB> {
         .source_row = R * 4, .destination_row = p_movd2b::SRC_ZERO_OFFSET + R * 4, .address_mode = ADDR_MOD_1, .number_of_rows = move::rows::Four}>(),
     ...);
}

// TTI_SETRWC(CLR_NONE, CR_A, 0, 0, 8, SET_A)
// HAL-GAP: rwc::set encodes a lone SrcA AdvanceAndReset as INCRWC; production issues SETRWC, kept as is.
inline __attribute__((always_inline)) void srca_cr_plus_8()
{
    TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_A, 0, 0, 8, p_setrwc::SET_A);
}

} // namespace mul_reduce_scalar
} // namespace hal_port

/*************************************************************************
 * LLK MUL REDUCE SCALAR - Low-level reduce operations for fused mul+reduce
 *************************************************************************/

/**
 * @brief Execute GAPOOL operations with optional high fidelity phases
 *
 * Executes the appropriate number of GAPOOL instructions based on fidelity level.
 * For high fidelity (math_fidelity > LoFi), executes Number of Fidelity Phases - 1
 * iterations with ADDR_MOD_2, followed by one final GAPOOL with ADDR_MOD_0.
 *
 * @tparam math_fidelity: MathFidelity value LoFi, HiFi2, HiFi3, HiFi4
 */
template <MathFidelity math_fidelity>
inline void execute_high_fidelity_gapool()
{
    // TTI_GAPOOL(CLR_NONE, DIM_16X16, ADDR_MOD_2 / ADDR_MOD_0, INDEX_DIS, 0)
    constexpr std::uint32_t gapool_phase = hal_port::reduce_pool(PoolType::SUM, p_setrwc::CLR_NONE, ADDR_MOD_2, 0);
    constexpr std::uint32_t gapool_last  = hal_port::reduce_pool(PoolType::SUM, p_setrwc::CLR_NONE, ADDR_MOD_0, 0);
    if constexpr (math_fidelity >= MathFidelity::HiFi2)
    {
        TTI_INSN(gapool_phase);
    }
    if constexpr (math_fidelity >= MathFidelity::HiFi3)
    {
        TTI_INSN(gapool_phase);
    }
    if constexpr (math_fidelity >= MathFidelity::HiFi4)
    {
        TTI_INSN(gapool_phase);
    }

    TTI_INSN(gapool_last);
}

/**
 * @brief Move destination tile to source registers for mul_reduce_scalar
 *
 * Moves data from destination registers to source A or B registers.
 *
 * @tparam binary_reuse_dest Direction: DEST_TO_SRCA or DEST_TO_SRCB
 * @param idst Destination tile index (0-7)
 */
template <EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
inline void _llk_math_mul_reduce_scalar_move_dest_to_src_([[maybe_unused]] std::uint32_t idst = 0)
{
    namespace mrs = hal_port::mul_reduce_scalar;
    namespace cfg = hal::cfg;
    namespace rwc = hal::counters::rwc;

    if constexpr (binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCA)
    {
        if (idst == 0)
        {
            // TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, get_dest_buffer_base())
            cfg::write<cfg::Access::TensixCfgUnit, cfg::DestTargetRegCfgMath::Offset, cfg::Sec::S0>(get_dest_buffer_base());
            hal_port::clear_dst_reg_addr(); // TTI_SETRWC(CLR_NONE, 0, 0, 0, 0, SET_D)
        }

        mrs::srca_bank_wait();

        // The switch below only encodes tiles 0..7; any higher idst would silently reuse stale SrcA.
        LLK_ASSERT(idst < 8, "mul_reduce_scalar move_dest_to_src: idst must be 0..7");
        switch (idst)
        {
            case 0:
                mrs::movd2a_16_rows<0>();
                break;
            case 1:
                mrs::movd2a_16_rows<64>();
                break;
            case 2:
                mrs::movd2a_16_rows<128>();
                break;
            case 3:
                mrs::movd2a_16_rows<192>();
                break;
            case 4:
                mrs::movd2a_16_rows<256>();
                break;
            case 5:
                mrs::movd2a_16_rows<320>();
                break;
            case 6:
                mrs::movd2a_16_rows<384>();
                break;
            case 7:
                mrs::movd2a_16_rows<448>();
                break;
            default:
                break;
        }
    }
    else if constexpr (binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCB)
    {
        hal_port::srcb_bank_wait();
        // dst may have been written by the SFPU just before this move (e.g. a fill). The MATH bit in the
        // bank wait only tracks FPU writes, so also drain the SFPU before the FPU copies dst into SrcB.
        mrs::wait_sfpu();

        // TTI_SETRWC(CLR_NONE, 0, 0, 0, 0, SET_BD)
        rwc::set<rwc::MathCounter {.src_b = rwc::Set(0), .dest = rwc::Set(0)}>();

        mrs::movd2b_rows(std::make_integer_sequence<std::uint32_t, 16> {});
    }
}

/**
 * @brief Configure address modifiers for mul_reduce_scalar operations
 *
 * Sets up address modifiers for source A, source B, destination, and fidelity
 * registers based on the specified math fidelity level.
 *
 * @tparam MathFidelity math_fidelity -> MathFidelity value LoFi, HiFi2, HiFi3, HiFi4
 */
template <MathFidelity math_fidelity>
inline void mul_reduce_scalar_configure_addrmod()
{
    using hal_port::ReduceAddrMod;
    using hal_port::set_reduce_addr_mod;
    using S = hal::cfg::Sec;

    constexpr std::uint32_t FIDELITY_INCREMENT = 1;
    constexpr bool HIGH_FIDELITY               = is_high_fidelity(math_fidelity);

    set_reduce_addr_mod<S::S0, ReduceAddrMod {.fid_clr = 1}>();
    set_reduce_addr_mod<S::S1, ReduceAddrMod {.srca_incr = 16, .fid_clr = 1}>();

    if constexpr (HIGH_FIDELITY)
    {
        set_reduce_addr_mod<S::S2, ReduceAddrMod {.fid_incr = FIDELITY_INCREMENT}>();
    }
}

/**
 * @brief Initialize reduce for mul_reduce_scalar operation
 *
 * Configures address modifiers and resets counters for the fused
 * multiply-reduce-scalar operation.
 *
 * @tparam is_fp32_dest_acc_en If true, enables FP32 destination accumulation
 * @tparam MATH_FIDELITY_DESC Math fidelity descriptor (0 = default, higher = more precision)
 * @tparam enforce_fp32_accumulation If true, enforces FP32 accumulation (requires is_fp32_dest_acc_en)
 */
template <bool is_fp32_dest_acc_en, MathFidelity math_fidelity, bool enforce_fp32_accumulation = false>
inline void _llk_math_mul_reduce_scalar_init_()
{
    namespace cfg = hal::cfg;

    mul_reduce_scalar_configure_addrmod<math_fidelity>();

    if constexpr (enforce_fp32_accumulation)
    {
        static_assert(is_fp32_dest_acc_en, "FP32 Dest must be enabled for FP32 accumulation");
    }
    // TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0): the whole word, both disable bits
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::ClrDvalid::SrcA_Disable, cfg::Sec::S0, 0>(), cfg::set<cfg::ClrDvalid::SrcB_Disable, cfg::Sec::S0, 0>());
    hal_port::reset_counters_abd_f();
    hal_port::_configure_default_zero_flag_state_();
}

/**
 * @brief Perform column reduction for mul_reduce_scalar (accumulates across tiles)
 *
 * Performs column-wise reduction that accumulates results into the specified
 * destination tile. Used in a loop to accumulate multiple input tiles.
 *
 * @tparam MATH_FIDELITY_DESC Math fidelity descriptor (0 = default, higher = more precision)
 * @param dst_index Destination tile index to accumulate into (0-7)
 * @param tensor_shape Shape of the operand tile (4 faces for 32x32, 2 faces for a 16x32 tiny tile)
 */
template <MathFidelity math_fidelity>
inline void _llk_math_mul_reduce_column_(const std::uint32_t dst_index, const ckernel::TensorShape tensor_shape = ckernel::DEFAULT_TENSOR_SHAPE)
{
    namespace mrs = hal_port::mul_reduce_scalar;
    namespace rwc = hal::counters::rwc;

    LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(tensor_shape), "Invalid tensor shape for tile-dependent op");

    // A 32x16 narrow tile has fewer face-columns than face-rows; mirror generic llk_math_reduce.h.
    const bool is_narrow_tile         = tensor_shape.num_faces_c_dim < tensor_shape.num_faces_r_dim;
    const std::uint32_t num_row_tiles = tensor_shape.num_faces_r_dim;

    // dst[dst_index] may have just been zeroed through the SFPU; GAPOOL accumulates into it, so drain first.
    mrs::wait_sfpu();
    hal_port::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);
    hal_port::clear_dst_reg_addr(); // TTI_SETRWC(CLR_NONE, 0, 0, 0, 0, SET_D)

    for (std::uint32_t row_tile = 0; row_tile < num_row_tiles; row_tile++)
    {
        execute_high_fidelity_gapool<math_fidelity>();

        if ((!is_narrow_tile) && (tensor_shape.total_num_faces() > 1))
        {
            mrs::srca_cr_plus_8();
            mrs::srca_cr_plus_8();

            execute_high_fidelity_gapool<math_fidelity>();
        }

        if (row_tile == 0)
        {
            mrs::srca_cr_plus_8();
            mrs::srca_cr_plus_8();
            hal_port::clear_dst_reg_addr(); // TTI_SETRWC(CLR_NONE, 0, 0, 0, 0, SET_D)
        }
        else
        {
            // TTI_SETRWC(CLR_NONE, 0, 0, 0, 0, SET_AD)
            rwc::set<rwc::MathCounter {.src_a = rwc::Set(0), .dest = rwc::Set(0)}>();
        }
    }
}

/**
 * @brief Perform final scalar reduction for mul_reduce_scalar
 *
 * Collapses a column-reduced tile into a single scalar value stored in the
 * first element of the output tile.
 *
 * @tparam MATH_FIDELITY_DESC Math fidelity descriptor (0 = default, higher = more precision)
 */
template <MathFidelity math_fidelity>
inline void _llk_math_mul_reduce_scalar_()
{
    namespace move = hal::move;

    hal_port::srcb_bank_wait();

    // Copy row 0 from dest to srcB (rows 16-31 as scratch) and transpose
    // TTI_MOVD2B(0, SRC_ROW16_OFFSET, ADDR_MOD_0, MOV_1_ROW, 0)
    move::run<move::Transfer<move::Dst, move::SrcB> {
        .source_row = 0, .destination_row = p_movd2b::SRC_ROW16_OFFSET, .address_mode = ADDR_MOD_0, .number_of_rows = move::rows::One}>();
    hal::src<hal::BothSources>.reset_pipeline_gating(); // TTI_GATESRCRST(0b1, 0b1)
    hal::src<hal::SrcB>.transpose();                    // TTI_TRNSPSRCB
    hal::src<hal::BothSources>.reset_pipeline_gating(); // TTI_GATESRCRST(0b1, 0b1)

    // Copy all 16 rows from srcB to srcA
    // HAL-GAP: hal::move has no SrcB-to-SrcA transfer (MOVB2A); production lines kept.
    TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 0, ADDR_MOD_0, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ROW16_OFFSET + 0);
    TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 4, ADDR_MOD_0, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ROW16_OFFSET + 4);
    TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 8, ADDR_MOD_0, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ROW16_OFFSET + 8);
    TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 12, ADDR_MOD_0, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ROW16_OFFSET + 12);
    hal::src<hal::BothSources>.reset_pipeline_gating(); // TTI_GATESRCRST(0b1, 0b1)

    // TTI_ZEROACC(CLR_SPECIFIC, 0, 0, ADDR_MOD_0, 0)
    hal::dst::run<hal::dst::Zero {.scope = hal::dst::ZeroScope::SingleRow, .index = 0, .width = hal::dst::DestWidth::Bits16, .address_mode = ADDR_MOD_0}>();

    execute_high_fidelity_gapool<math_fidelity>();
}

/**
 * @brief Clear data valid flags after mul_reduce_scalar operation
 *
 * Clears DVALID flags and resets all counters. Should be called after
 * mul_reduce_scalar operations are complete.
 */
inline void _llk_math_mul_reduce_scalar_clear_dvalid_()
{
    // TTI_CLEARDVALID(CLR_AB, 0)
    hal::src<hal::BothSources>.release<hal::src_ops::BankAdvance::Flip>();
    hal_port::reset_counters_abd_f(); // TTI_SETRWC(CLR_NONE, 0, 0, 0, 0, SET_ABD_F)
}
