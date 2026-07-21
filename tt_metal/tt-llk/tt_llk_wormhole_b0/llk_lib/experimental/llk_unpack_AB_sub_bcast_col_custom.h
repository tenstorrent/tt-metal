// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cunpack_common.h"
#include "llk_assert.h"
#include "llk_unpack_common.h"
#include "tensor_shape.h"

using namespace ckernel;
using namespace ckernel::unpacker;

// Custom init for the blocked sub+bcast(col) unpack flow.
// @param tensor_shape Shape of the operand tile (2 faces for 16x32 tiny tiles, 4 faces for full 32x32 tiles).
inline void _llk_unpack_AB_sub_bcast_col_init_custom_(const ckernel::TensorShape& tensor_shape = ckernel::DEFAULT_TENSOR_SHAPE)
{
    LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(tensor_shape), "Invalid tensor shape for tile-dependent op");

    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(0); // transpose within the face

    // Force both unpackers to unpack all faces of the operand for the blocked path. Full 32x32
    // tile = 4 faces = 1024 datums, 16x32 tiny tile = 2 faces = 512 datums.
    const std::uint32_t x_end = tensor_shape.total_tensor_size() - 1;
    TT_SETADCXX(p_setadc::UNP0, x_end, 0x0);
    TT_SETADCXX(p_setadc::UNP1, x_end, 0x0);
}

// Custom blocked unpack: one SrcB tile + ct_dim SrcA tiles.
inline void _llk_unpack_AB_sub_bcast_col_custom_(const std::uint32_t address_a, const std::uint32_t address_b, const std::uint32_t ct_dim = 1)
{
    // Start from a clean A/B tile index for each blocked sub call.
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);

    // Program SrcA and SrcB base addresses.
    volatile std::uint32_t tt_reg_ptr* cfg = get_cfg_pointer();
    // Wait for a free unpack config context before programming the next block.
    wait_for_next_context(2);
    _llk_unpack_configure_addresses_(address_a, address_b, cfg);
    // Trisc::SEMPOST for context acquire.
    semaphore_post(semaphore::UNPACK_SYNC);
    // Stall unpacker until pending CFG writes from Trisc have completed.
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

    constexpr std::uint8_t ADDRMOD_CH1Y_0_CH1Z_0_CH0Y_0_CH0Z_0 = 0b00'00'00'00;

    // Load SrcB once and keep reusing it for all ct_dim SrcA tiles.
    TTI_UNPACR(SrcB, ADDRMOD_CH1Y_0_CH1Z_0_CH0Y_0_CH0Z_0, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);

    for (std::uint32_t tile = 0; tile < ct_dim; tile++)
    {
        TTI_UNPACR(SrcA, ADDRMOD_CH1Y_0_CH1Z_0_CH0Y_0_CH0Z_0, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
        TTI_INCADCZW(p_setadc::UNP_A, 0, 0, 1, 0);
    }

    // T6::SEMGET for context release.
    t6_semaphore_get(semaphore::UNPACK_SYNC);
    // Switch unpacker config context so the next blocked call can program the
    // other side while this one executes.
    switch_config_context(unp_cfg_context);
}

inline void _llk_unpack_AB_sub_bcast_col_uninit_custom_()
{
    // No state to restore: the multi-face unpack X span set in the custom init is transient — every
    // generic unpack op re-establishes its own X span (config_unpacker_x_end / SETADCXX) in its init.
}
