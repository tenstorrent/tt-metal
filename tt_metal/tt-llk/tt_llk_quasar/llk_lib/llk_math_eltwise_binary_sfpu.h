// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>

#include "cmath_common.h"
#include "llk_defs.h"
using namespace ckernel::math;

/**
 * @brief Programs SFPU addrmods for binary SFPU operations
 */
inline void _eltwise_binary_sfpu_configure_addrmod_()
{
    // NOTE: this kernel is typically used in conjunction with
    // A2D, which is using ADDR_MOD_0 and ADDR_MOD_1, so use one
    // that doesn't conflict!

    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    }
        .set(ADDR_MOD_7, csr_read<CSR::TRISC_ID>());
}

/**
 * @brief Sets up starting index of SFPU for binary op, stalls till all FPU operations are done
 * @param tile_index: Use to index to a tile in Destination register
 */
inline void _llk_math_eltwise_binary_sfpu_start_(const std::uint32_t tile_index)
{
    _set_dst_write_addr_<DstTileShape::Tile32x32>(tile_index);
    TTI_STALLWAIT(p_stall::STALL_SFPU, 0, 0, p_stall::MATH);
}

/**
 * @brief Resets dest counter to 0
 */
inline void _llk_math_eltwise_binary_sfpu_done_()
{
    _reset_counters_<p_setrwc::SET_D>();
}

/**
 * @brief Increments dest counter by a face (16x16 default)
 */
inline void _llk_math_eltwise_binary_sfpu_inc_dst_face_addr_()
{
    _inc_dst_addr_<16>();
}

/**
 * @brief Initialization for binary SFPU operations
 */
inline void _llk_math_eltwise_binary_sfpu_init_()
{
    _init_sfpu_config_reg_();
    _eltwise_binary_sfpu_configure_addrmod_();
    _reset_counters_<p_setrwc::SET_ABD_F>();
}

/**
 * @brief Runs binary SFPU operation for a tile (default 32x32)
 * @tparam APPROXIMATE: Selects between approximate (faster) or accurate (slower) mode
 * @param sfpu_func: function pointer to the binary sfpu function to run
 * @param dst_tile_index: Starting tile index in the destination register
 * @param args: variable number of args passed through to sfpu_func
 */
template <bool APPROXIMATE, class F, class... ARGS>
inline void _llk_math_eltwise_binary_sfpu_params_(F&& sfpu_func, std::uint32_t dst_tile_index, ARGS&&... args)
{
    _llk_math_eltwise_binary_sfpu_start_(dst_tile_index);

    for (std::uint32_t face = 0; face < NUM_FACES; face++)
    {
        sfpu_func(static_cast<ARGS&&>(args)...);

        _llk_math_eltwise_binary_sfpu_inc_dst_face_addr_();
    }

    _llk_math_eltwise_binary_sfpu_done_();
}
