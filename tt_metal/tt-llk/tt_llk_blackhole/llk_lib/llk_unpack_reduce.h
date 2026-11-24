// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL port of tt_llk_blackhole/llk_lib/llk_unpack_reduce.h. Same instruction sequence and access kinds as production,
// expressed through the HAL.

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cunpack_common.h"
#include "hal/cfg.h"
#include "hal/mop.h"
#include "hal/nop.h"
#include "hal/sync.h"
#include "hal/tensix_counters.h"
#include "hal/unpack.h"
#include "llk_unpack_common.h"

using namespace ckernel;
using namespace ckernel::unpacker;

/**
 * @brief Program the unpacker MOP for a reduce-with-scaler operation.
 *
 * Sets up the UNPACR sequence that zeroes SrcA, unpacks the operand into SrcA, and loads the
 * scaler row into SrcB across the four-unpack halo template.
 *
 * @tparam type: Reduction pooling op, values = <SUM/AVG/MAX>
 * @tparam dim: Reduction dimension, values = <REDUCE_ROW/REDUCE_COL/REDUCE_SCALAR>
 */
template <PoolType type, ReduceDim dim>
inline void _llk_unpack_reduce_mop_config_()
{
    using hal::unpack::Engine;
    using hal::unpack::SourceHandoff;
    // TT_OP_UNPACR(SrcA, 0b1, 0, 0, 0, 1, 1, RAREFYB_DISABLE, 0, 0, 0, 0, 1)
    static constexpr std::uint32_t unpack_srca = hal::unpack::get_operation<hal::unpack::DataTransfer {
        .engine = Engine::Unpacker0, .increments = {.channel0 = {.z = 1}}, .handoff = SourceHandoff::FlipAndSetDataValid}>();
    // HAL-GAP: the clear-source UNPACR_NOP flavor (CLR_SRC_0 / CLR_SRC, WAIT_LIKE_UNPACR) has no HAL descriptor.
    static constexpr std::uint32_t unpack_zerosrca =
        TT_OP_UNPACR_NOP(p_unpacr_nop::UNP0, 0, 0, 0, 0, p_unpacr_nop::WAIT_LIKE_UNPACR, 0, p_unpacr_nop::CLR_SRC_0, p_unpacr_nop::CLR_SRC);
    // TT_OP_UNPACR(SrcB, 0b0, 0, 0, 0, 1, 1, RAREFYB_DISABLE, 0, 0, 0, 0, 1)
    static constexpr std::uint32_t unpack_srcb =
        hal::unpack::get_operation<hal::unpack::DataTransfer {.engine = Engine::Unpacker1, .handoff = SourceHandoff::FlipAndSetDataValid}>();

    // Template 0 with SrcB and halo (four A slots): zero SrcA, unpack SrcA, two NOPs, then SrcB. The skip slots
    // are 0 as in production; the reduce run's mask is 0, so they are never issued.
    hal::mop::program(hal::mop::foreach_mask_bit(
        hal::mop::if_clear(unpack_zerosrca, unpack_srca, TT_OP_NOP, TT_OP_NOP, unpack_srcb), hal::mop::otherwise(0 /*skip_a0*/, 0 /*skip_b*/)));
}

/**
 * @brief Initialize the unpacker for a reduce-with-scaler operation.
 *
 * Configures the SrcB format registers and scaler L1 base address, sets haloize (transpose) mode
 * according to the reduce dimension and the within-face transpose flag, programs the datum count,
 * and programs the reduce MOP.
 *
 * @tparam type: Reduction pooling op, values = <SUM/AVG/MAX>
 * @tparam dim: Reduction dimension, values = <REDUCE_ROW/REDUCE_COL/REDUCE_SCALAR>
 * @param unpB_src_format: Source data format of the scaler operand (SrcB) in L1.
 * @param unpB_dst_format: Destination data format the scaler operand is converted to.
 * @param within_face_16x16_transpose: Nonzero to enable the 16x16 within-face transpose.
 * @ref _llk_unpack_reduce_ is the matching execute call.
 * @ref _llk_math_reduce_init_ is the matching init on the math thread (single-operand unpack pairing).
 */
template <PoolType type, ReduceDim dim>
inline void _llk_unpack_reduce_init_(
    const std::uint32_t unpB_src_format, const std::uint32_t unpB_dst_format, const std::uint32_t within_face_16x16_transpose = 0)
{
    // This path's MOP clears SrcA to zero for every pool type, which a min reduce would lose to.
    static_assert(
        type != PoolType::MIN,
        "The FPU reduce has no MIN: the hardware provides GMPOOL (max) and GAPOOL (average) only. "
        "Use the SFPU reduce instead (ckernel_sfpu_reduce.h::calculate_reduce).");

    namespace cfg          = hal::cfg;
    constexpr auto TileDsc = cfg::Thcon[cfg::Reg0].TileDescriptor;
    static_assert(TileDsc.InDataFormat.addr32(cfg::Sec::S1) == THCON_SEC1_REG0_TileDescriptor_ADDR32, "SrcB tile descriptor word");
    static_assert(cfg::Thcon[cfg::Reg2].Out_data_format.addr32(cfg::Sec::S1) == THCON_SEC1_REG2_Out_data_format_ADDR32, "SrcB Out_data_format word");
    static_assert(cfg::Unpacker[1].Cntx[0].Base.addr32(cfg::Sec::S0) == THCON_SEC1_REG3_Base_address_ADDR32, "SrcB context-0 base word");
    static_assert(cfg::Unpacker[1].Cntx[1].Base.addr32(cfg::Sec::S0) == THCON_SEC1_REG3_Base_cntx1_address_ADDR32, "SrcB context-1 base word");

    // Configure SrcB format registers. The 4-bit format fields take the format's low 4 bits: production RMWs the
    // unmasked format under the same byte mask (the RMWCIB mask drops the upper bits); the HAL asserts the value
    // fits the field, so mask it here.
    cfg::write<cfg::Access::TensixCfgUnit, TileDsc.InDataFormat, cfg::Sec::S1>(masked_data_format(unpB_src_format));

    // Set FP8 E4M3 mode for SrcB; selects the e4m3 (vs e5m2) exponent layout and clears any stale 4b-exp setting.
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg1].Unp_LF8_4b_exp, cfg::Sec::S1>(
        ((unpB_src_format & 0x1F) == static_cast<std::uint32_t>(DataFormat::Fp8_e4m3)) ? 1 : 0);

    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg2].Out_data_format, cfg::Sec::S1>(masked_data_format(unpB_dst_format));

    cfg::write<cfg::Access::TensixCfgUnit, cfg::Unpacker[1].Cntx[0].Base, cfg::Sec::S0, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Deferred>(
        hal::gpr<p_gpr_unpack::L1_BUFFER_ADDR>());
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Unpacker[1].Cntx[1].Base, cfg::Sec::S0, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Deferred>(
        hal::gpr<p_gpr_unpack::L1_BUFFER_ADDR>());
    hal::nop::thread();

    // REDUCE_ROW requires transpose itself; additionally, within_face_16x16_transpose flag could require transpose;
    // if we have the flag set with REDUCE_ROW, we don't need to do anything
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg2].Haloize_mode, cfg::Sec::S0>(
        ReduceDim::REDUCE_ROW == dim ? !within_face_16x16_transpose : within_face_16x16_transpose);

    // TTI_SETADCXX(0b11, FACE_R_DIM * FACE_C_DIM - 1, 0x0)
    hal_port::set_unpacker_x_end<p_setadc::UNP_AB, FACE_R_DIM>();

    _llk_unpack_reduce_mop_config_<type, dim>();
}

/**
 * @brief Unpack a tile for a reduce-with-scaler operation.
 *
 * Programs the operand base address, temporarily narrows SrcB to a single 16-datum row for the
 * scaler, runs the reduce MOP, restores the SrcB datum count, and switches config context.
 *
 * @tparam type: Reduction pooling op, values = <SUM/AVG/MAX>
 * @tparam dim: Reduction dimension, values = <REDUCE_ROW/REDUCE_COL/REDUCE_SCALAR>
 * @param address: L1 address of the source tile.
 * @note Call @ref _llk_unpack_reduce_init_ with matching template args before this function.
 */
template <PoolType type, ReduceDim dim>
inline void _llk_unpack_reduce_(const std::uint32_t address)
{
    namespace sync = hal::sync;
    namespace adc  = hal::counters::adc;

    // Clear z/w start counters: TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111)
    adc::set<adc::AddrCounter {.channel0 = {.z = 0, .w = 0}, .channel1 = {.z = 0, .w = 0}}.for_clients<adc::Client::Unpacker0, adc::Client::Unpacker1>()>();

    // Program srcA and srcB base addresses
    volatile std::uint32_t tt_reg_ptr *cfg = hal::cfg::detail::state_cfg_bank(); // get pointer to registers for current state ID

    // Wait for free context
    hal_port::wait_for_next_context(2);

    // Validate and configure address
    _llk_unpack_configure_single_address_(address, cfg);

    // Trisc::SEMPOST for context acquire
    sync::semaphore::post<sync::Access::MMIO, hal_port::UNPACK_SYNC_SEM>();

    // Load only 16 datums into srcB: TTI_SETADCXX(p_setadc::UNP1, FACE_C_DIM - 1, 0x0)
    hal_port::set_unpacker_x_end<p_setadc::UNP_B, 1>();

    // Stall unpacker until pending CFG writes from Trisc have completed
    sync::wait::stall<sync::StallTarget::Unpack, sync::StallCondition::RiscvAccessProcessed>();

    // Run MOP: mop_run(0, 4)
    hal::mop::Runner<hal::mop::MopTemplate::Template0>::run<4 /*count*/, 0 /*mask*/>();

    // Restore face height: TTI_SETADCXX(p_setadc::UNP1, FACE_R_DIM * FACE_C_DIM - 1, 0x0)
    hal_port::set_unpacker_x_end<p_setadc::UNP_B, FACE_R_DIM>();

    // T6::SEMGET for context release
    sync::semaphore::get<hal_port::UNPACK_SYNC_SEM>();

    // Switch unpacker config context
    hal_port::switch_config_context(unp_cfg_context);
}
