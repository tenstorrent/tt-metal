// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_template.h"
#include "ckernel_trisc_common.h"
#include "llk_defs.h"
#include "llk_unpack_common.h"
#include "tensor_shape.h"

using namespace ckernel;

inline void _llk_unpack_gather_compress_mop_config_(const std::uint32_t buf_desc_id)
{
    constexpr std::uint32_t MOP_OUTER_LOOP = 1;
    constexpr std::uint32_t MOP_INNER_LOOP = 1;

    const std::uint32_t unpack_instrn = TT_OP_UNPACR1_STRIDE(
        0 /*Src_Reg_Y_Cntr_Incr*/,
        0 /*L1_Tile_Idx_or_Tile_Idx_Inc*/,
        0 /*Tile_Idx_Inc*/,
        0 /*Row_Mask_Reg_Sel*/,
        0 /*L1_16datums_Row_Index*/,
        buf_desc_id,
        1 /*SetDatValid*/);

    ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, unpack_instrn);
    temp.program_bank0_sw_cntl(instrn_buffer);
}

inline void _llk_unpack_gather_compress_init_(
    const GatherCompressModeSelect unpack_mode, const std::uint32_t unpack_dst_format, const std::uint32_t buf_desc_id)
{
    cfg_rmw(THCON_UNPACKER1_REG0_ENABLE_ARG_FIFO_RMW, 1);                                     // enable index FIFO
    cfg_rmw(THCON_UNPACKER1_REG0_ARG_FIFO_UNPACR_STRIDE_MODE_RMW, (std::uint8_t)unpack_mode); // program the register to describe mode
    cfg_rmw(THCON_UNPACKER1_REG1_UNPACK_STRIDE_VAL_SOURCE_RMW, 0);                            // Use a fixed stride in L1
    cfg_rmw(THCON_UNPACKER1_REG2_UNPACK_STRIDE_OFFSET_0_RMW, 1);                              // Step through contiguous rows
    cfg_rmw(THCON_UNPACKER1_REG0_OUT_DATA_FORMAT_RMW, static_cast<std::uint8_t>(unpack_dst_format));
    cfg_rmw(ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW, 1); // Keeps a zero index in the exponent slot from being reinterpreted as a zero-flag.

    _llk_unpack_gather_compress_mop_config_(buf_desc_id);
}

// Set-off the UNPACR programmed by _llk_unpack_gather_compress_init_; it drains the ARG FIFO, so the
// index vector must have been pushed immediately before.
inline void _llk_unpack_gather_compress_()
{
    ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);
}
