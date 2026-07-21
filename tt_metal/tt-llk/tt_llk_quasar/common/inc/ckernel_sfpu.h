// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <limits>
#include <utility>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "llk_defs.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_exp.h"
#include "sfpu/ckernel_sfpu_recip.h"
#include "sfpu/ckernel_sfpu_relu.h"
#include "sfpu/ckernel_sfpu_sqrt.h"
#include "sfpu/ckernel_sfpu_typecast_fp16b_uint16.h"
#include "sfpu/ckernel_sfpu_typecast_int32_fp32.h"

namespace ckernel
{
using namespace ckernel::math;
using namespace ckernel::trisc;

/**
 * @brief Programs SFPU addrmods
 */
inline void _sfpu_configure_addrmod_()
{
    // NOTE: this kernel is typically used in conjunction with
    // A2D, which is using ADDR_MOD_0 and ADDR_MOD_1, so use one
    // that doesn't conflict!

    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    }
        .set(ADDR_MOD_7);
}

/**
 * @brief Sets up starting index of SFPU, Stalls till all FPU operations are done
 * @param tile_index: Use to index to a tile in Destination register
 */
inline void _llk_math_sfpu_start_(const std::uint32_t tile_index)
{
    _set_dst_write_addr_<DstTileShape::Tile32x32>(tile_index);
    TTI_STALLWAIT(p_stall::STALL_SFPU, 0, 0, p_stall::MATH);
}

/**
 * @brief Resets dest counter to 0
 */
inline void _llk_math_sfpu_done_()
{
    _reset_counters_<p_setrwc::SET_D>();
}

/**
 * @brief Clear SrcS valids
 * @tparam SRCS_RD_DONE: Whether the source reg S read is done
 * @tparam SRCS_WR_DONE: Whether the source reg S write is done
 */
template <bool SRCS_RD_DONE, bool SRCS_WR_DONE>
inline void _llk_math_sfpu_srcs_clear_vlds_()
{
    TTI_SFPNOP(SRCS_WR_DONE, SRCS_RD_DONE, 0);
}

/**
 * @brief Increments dest counter by a face (16x16 default)
 */
inline void _llk_math_sfpu_inc_dst_face_addr_()
{
    _inc_dst_addr_<16>();
}

/**
 * @brief Initialization for SFPU operations
 */
inline void _llk_math_sfpu_init_()
{
    _init_sfpu_config_reg_();
    _sfpu_configure_addrmod_();
    _reset_counters_<p_setrwc::SET_ABD_F>();
}

} // namespace ckernel
