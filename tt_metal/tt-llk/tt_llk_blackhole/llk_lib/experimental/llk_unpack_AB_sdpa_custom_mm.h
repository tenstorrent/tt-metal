// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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
#include "experimental/llk_unpack_AB_custom_mm.h"
#include "llk_assert.h"

using namespace ckernel;
using namespace ckernel::unpacker;

template <bool read_transposed = false, bool configure_mask_extent = false>
inline void _llk_unpack_AB_sdpa_custom_mm_(
    const std::uint32_t base_address_a,
    const std::uint32_t base_address_b,
    const std::uint32_t base_address_mask,
    const std::uint32_t tile_index_a,
    const std::uint32_t tile_index_b,
    const std::uint32_t tile_size_a,
    const std::uint32_t tile_size_b,
    const std::uint32_t kt_dim,
    const std::uint32_t ct_dim              = 1,
    const bool mask_chunk                   = false,
    const std::uint32_t operandB_face_r_dim = 8,
    const std::uint32_t row_stride          = 0)
{
    const std::uint32_t in1_row = row_stride ? row_stride : kt_dim;
    if constexpr (read_transposed)
    {
        LLK_ASSERT(in1_row >= kt_dim, "row_stride must contain the contraction tiles");
    }
    volatile std::uint32_t* cfg         = get_cfg_pointer();
    // With one output tile, the MOP uses only its full-unpack block: consecutive
    // contraction tiles are adjacent, even for the transposed read.
    const std::uint32_t block_increment = read_transposed && ct_dim > 1 ? in1_row * tile_size_a : tile_size_a;
    const std::uint32_t inner_increment = read_transposed ? -(((ct_dim - 1) * in1_row) - 1) * tile_size_a : tile_size_a;

    const std::uint32_t address_a = base_address_a + tile_size_a * tile_index_a;
    const std::uint32_t address_b = base_address_b + tile_size_b * tile_index_b;

    // Wait for all contexts to be free
    wait_for_next_context(1);
    reset_config_context();

    if (mask_chunk)
    {
        cfg[THCON_SEC1_REG3_Base_cntx1_address_ADDR32] = base_address_mask;
        if constexpr (configure_mask_extent)
        {
            LLK_ASSERT(ct_dim * 2 <= 64, "mask_chunk requires ct_dim*2 <= 64 SrcB rows");
            TT_SETADCXX(p_setadc::UNP_B, ct_dim * 2 * FACE_C_DIM - 1, 0x0);
        }
        TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);
        TTI_UNPACR_COMMON_EXPLICIT_CONTEXT(SrcB, 0b00000000, 1, 1);
        if constexpr (configure_mask_extent)
        {
            TT_SETADCXX(p_setadc::UNP_B, operandB_face_r_dim * FACE_C_DIM - 1, 0x0);
        }
    }

    _llk_unpack_AB_custom_mm_run_(cfg, address_a, address_b, block_increment, inner_increment, kt_dim, ct_dim);
}

// OV with one K tile row. Reuse the replay buffer configured by the standard
// OV init, but issue one row directly instead of the MOP's pair of rows.
inline void _llk_unpack_A_sdpa_custom_mm_single_row_(const std::uint32_t address_a, const std::uint32_t tile_size_a, const std::uint32_t nt_dim)
{
    volatile std::uint32_t* cfg = get_cfg_pointer();
    wait_for_next_context(1);
    reset_config_context();
    cfg[THCON_SEC0_REG3_Base_address_ADDR32] = address_a;
    cfg[SCRATCH_SEC0_val_ADDR32]             = tile_size_a;
    cfg[SCRATCH_SEC1_val_ADDR32]             = tile_size_a; // Final address is unused.
    semaphore_post(semaphore::UNPACK_SYNC);
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);
    // Each replay block is [UNPACR SrcA, CFGSHIFTMASK, NOP]; the last block
    // is at offset 27. Split wider rows to fit each replay's instruction limit.
    if (nt_dim == 1)
    {
        lltt::replay(27, 3);
    }
    else
    {
        lltt::replay(0, 3 * ((nt_dim + 1) / 2));
        lltt::replay(30 - 3 * (nt_dim / 2), 3 * (nt_dim / 2));
    }
    t6_semaphore_get(semaphore::UNPACK_SYNC);
    wait_for_next_context(1);
    reset_config_context();
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
    TTI_SETADCXY(0b011, 0, 0, 0, 0, 0b1010);
}
