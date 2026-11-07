// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_ops.h"
#include "cpack_common.h"
#include "llk_assert.h"
#include "llk_pack_common.h"

namespace ckernel
{

/**
 * @brief Pack weighted-reduce results into consecutive 1x32 output rows, one DEST slot per row.
 *
 * @param tile0_address: L1 address of output tile 0 in 16-byte words, minus one.
 * @param first_chunk: Output row of the first chunk, counted across tiles.
 * @note Needs a pack init that leaves ADDR_MOD_3 stepping Z by one, ADDR_MOD_1 with no step and the DEST strides
 *       sdpa_custom_mm's pack init sets (Z 8 rows, W 16 rows). The output format must be 16-bit.
 */
inline void _llk_pack_sdpa_weighted_reduce_block_(
    const std::uint32_t tile0_address, const std::uint32_t tile_size, const std::uint32_t first_chunk, const std::uint32_t num_chunks, const std::uint32_t first_slot)
{
    constexpr std::uint32_t row_size = TILE_C_DIM * 2 / 16;
    constexpr std::uint32_t max_run_rows = 8;
    LLK_ASSERT(num_chunks > 0, "sdpa_weighted_reduce (pack): a block needs at least one chunk");

    const std::uint32_t end = first_chunk + num_chunks;
    for (std::uint32_t chunk = first_chunk; chunk < end;)
    {
        const std::uint32_t tile = chunk / TILE_R_DIM;
        const std::uint32_t row  = chunk % TILE_R_DIM;
        std::uint32_t rows       = end - chunk;
        rows                     = rows < TILE_R_DIM - row ? rows : TILE_R_DIM - row;
        rows                     = rows < max_run_rows ? rows : max_run_rows;
        const std::uint32_t slot = first_slot + chunk - first_chunk;
        set_dst_write_addr(slot);
        program_packer_destination(tile0_address + tile * tile_size + row * row_size);
        for (std::uint32_t i = 0; i < 2 * rows - 1; i++)
        {
            TTI_PACR(
                p_pacr::CFG_CTXT_0,
                p_pacr::NO_ROW_PAD_ZERO,
                p_pacr::DST_ACCESS_NORMAL_MODE,
                ADDR_MOD_3,
                p_pacr::ADDR_CNT_CTXT_0,
                p_pacr::P_ZERO_OUTPUT_DISABLED,
                p_pacr::SINGLE_INTF_ACTIVE,
                0,
                0,
                0,
                0,
                0);
        }
        TTI_PACR(
            p_pacr::CFG_CTXT_0,
            p_pacr::NO_ROW_PAD_ZERO,
            p_pacr::DST_ACCESS_NORMAL_MODE,
            ADDR_MOD_1,
            p_pacr::ADDR_CNT_CTXT_0,
            p_pacr::P_ZERO_OUTPUT_DISABLED,
            p_pacr::SINGLE_INTF_ACTIVE,
            0,
            0,
            0,
            0,
            1);
        TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0101);
        chunk += rows;
    }
}

} // namespace ckernel
