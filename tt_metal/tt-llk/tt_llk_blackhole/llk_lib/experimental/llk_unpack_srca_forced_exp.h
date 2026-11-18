// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_ops.h"

using namespace ckernel;

/**
 * @brief Make unpacker A read BFP2_b tiles that hold only mantissas: input format BFP2_b, every datum with the shared
 * exponent exp in place of the tile's exponent section (a tile is then 256 bytes).
 *
 * @param exp: The biased exponent every datum takes, values = <0..255>
 * @note Call after the init that sets unpacker A's format, which this overrides; @ref _llk_unpack_srca_forced_exp_off_
 * clears the forced exponent.
 */
inline void _llk_unpack_srca_forced_exp_bfp2_(const std::uint32_t exp)
{
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::UNPACK0);
    cfg_reg_rmw_tensix<THCON_SEC0_REG0_TileDescriptor_ADDR32, 0, 0x0f>(to_underlying(DataFormat::Bfp2_b));
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Force_shared_exp_RMW>(1);
    cfg_reg_rmw_tensix<UNP0_FORCED_SHARED_EXP_shared_exp_RMW>(exp);
}

/**
 * @brief Clear unpacker A's forced exponent (@ref _llk_unpack_srca_forced_exp_bfp2_).
 */
inline void _llk_unpack_srca_forced_exp_off_()
{
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::UNPACK0);
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Force_shared_exp_RMW>(0);
}
