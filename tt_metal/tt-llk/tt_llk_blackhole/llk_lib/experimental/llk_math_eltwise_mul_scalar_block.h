// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// HAL port of tt_llk_blackhole/llk_lib/experimental/llk_math_eltwise_mul_scalar_block.h. Same Tensix instruction
// sequence as production, expressed through the HAL.

#pragma once

#include <cstdint>

#include "ckernel_include.h"
#include "ckernel_ops.h"
#include "cmath_common.h"
#include "hal/cfg.h"
#include "hal/counters/rwc.h"
#include "hal/fpu.h"
#include "llk_math_common.h"
#include "llk_math_reduce.h"
#include "tensor_shape.h"

namespace ckernel
{

inline void _llk_math_eltwise_mul_scalar_block_init_()
{
    namespace cfg = hal::cfg;

    // addr_mod_t {.srca = {.incr = 8}, .srcb = {.incr = 0}, .dest = {.incr = 8}}.set(ADDR_MOD_7)
    hal_port::set_reduce_addr_mod<cfg::Sec::S7, hal_port::ReduceAddrMod {.srca_incr = 8, .dest_incr = 8}>();

    // TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0): the whole word, both disable bits
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::ClrDvalid::SrcA_Disable, cfg::Sec::S0, 0>(), cfg::set<cfg::ClrDvalid::SrcB_Disable, cfg::Sec::S0, 0>());
    hal_port::reset_counters_abd_f();
}

inline void _llk_math_eltwise_mul_scalar_block_(const std::uint32_t dst_index, const std::uint32_t block_size)
{
    namespace rwc = hal::counters::rwc;

    constexpr std::uint32_t kOpsPerTile = 8;
    // TTI_ELWMUL(CLR_NONE, 0, SRCB_BCAST_ALL, ADDR_MOD_7, 0)
    constexpr hal::fpu::Elementwise mul_bcast_scalar {
        .operation        = hal::fpu::ElementwiseOperation::Multiply,
        .broadcast        = hal::fpu::SrcBBroadcast::Scalar,
        .dest_write       = hal::fpu::DestWriteMode::Overwrite,
        .address_modifier = ADDR_MOD_7,
        .dest_row_offset  = 0,
        .release          = hal::fpu::SourceRelease::None};

    for (std::uint32_t i = 0; i < block_size; ++i)
    {
        hal_port::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index + i);
        // TTI_SETRWC(CLR_NONE, 0, 0, 0, 0, SET_ABD)
        rwc::set<rwc::MathCounter {.src_a = rwc::Set(0), .src_b = rwc::Set(0), .dest = rwc::Set(0)}>();
        for (std::uint32_t op = 0; op < kOpsPerTile; ++op)
        {
            hal::fpu::run<mul_bcast_scalar>();
        }
        // TTI_SETRWC(CLR_A, 0, 0, 0, 0, SET_AB)
        rwc::set<rwc::MathCounter {.src_a = rwc::Set(0), .src_b = rwc::Set(0), .flip = rwc::Src::A}>();
    }

    // TTI_SETRWC(CLR_B, 0, 0, 0, 0, SET_ABD)
    rwc::set<rwc::MathCounter {.src_a = rwc::Set(0), .src_b = rwc::Set(0), .dest = rwc::Set(0), .flip = rwc::Src::B}>();
    hal_port::clear_dst_reg_addr();
}

} // namespace ckernel
