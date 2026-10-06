// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "llk_assert.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "llk_pack.h"
#include "llk_pack_common.h"

using namespace ckernel;
using namespace ckernel::packer;

/*************************************************************************
 * LLK PACK BLOCK CONTIGUOUS (Wormhole B0)
 *
 * Packs num_tiles consecutive 32x32 DEST tiles to equally spaced L1 tiles
 * in a single call. The per-tile loop runs in the MOP, not on the pack
 * RISC-V, so the RISC-V issues a handful of instructions per block.
 *
 * A per-tile RISC-V loop around _llk_pack_ costs a few RISC-V cycles per
 * tile, and its speed depends on code placement: the TRISC branch
 * predictor is a 16-entry untagged table indexed by address bits 2..8, and
 * two loop branches in the same entry mispredict on every tile. When the
 * RISC-V is the slower side, the packers wait for it.
 *
 * MOP structure (the blocked sequence of the multi-tile _llk_pack_mop_config_):
 *   OUTER     = num_tiles - 1 (runtime-patchable via mop_cfg[0])
 *   INNER     = 1
 *   START_OP  = PACR, close tile k
 *   LOOP_OP0  = INCADCZW(W++) to advance to the next DEST tile
 *   LOOP_OP1  = OUTPUT_ADDR += OUTPUT_ADDR_OFFSET (next L1 tile)
 *   END_OP0   = REG2FLOP the new L1 destination address
 *   END_OP1   = PACR resetting the per-tile packer counters
 * The last tile is packed and closed by one PACR after the MOP.
 *
 * Precondition: _llk_pack_hw_configure_ and _llk_pack_init_ (PackMode::Default,
 * 4 faces of 16 rows) must have run. This function only replaces the MOP;
 * call _llk_pack_init_ again before going back to _llk_pack_.
 *************************************************************************/

namespace llk_pack_block_internal
{
static std::uint32_t configured_outer = 0;
static std::uint32_t zero_output_flag = p_pacr::P_ZERO_OUTPUT_DISABLED;
} // namespace llk_pack_block_internal

// Program the MOP for block-contiguous packing. l1_tile_stride_words: L1 distance between output tiles in
// 16-byte words; 0 packs the tiles densely (one packed tile size apart).
template <bool zero_output = false>
inline void _llk_pack_block_contiguous_mop_config_(const std::uint32_t pack_dst_format, const std::uint32_t l1_tile_stride_words = 0)
{
    constexpr std::uint32_t ZERO_OUTPUT_FLAG = zero_output ? p_pacr::P_ZERO_OUTPUT_ENABLED : p_pacr::P_ZERO_OUTPUT_DISABLED;
    constexpr std::uint32_t PACKCNT          = 4;
    constexpr std::uint32_t MEGAROW          = 1;

    const std::uint32_t stride_words = l1_tile_stride_words ? l1_tile_stride_words : _llk_pack_output_addr_offset_words_(pack_dst_format);
    TT_SETDMAREG(p_setdmareg::PAYLOAD_IMMEDIATE, stride_words, p_setdmareg::MODE_IMMEDIATE, LO_16(p_gpr_pack::OUTPUT_ADDR_OFFSET));

    ckernel::ckernel_template tmp(
        1, // OUTER (placeholder, set by _llk_pack_block_contiguous_)
        1, // INNER
        TT_OP_INCADCZW(p_setadc::PAC, 0, 0, 1, 0),
        TT_OP_ADDDMAREG(p_adddmareg::REG_PLUS_REG, p_gpr_pack::OUTPUT_ADDR, p_gpr_pack::OUTPUT_ADDR, p_gpr_pack::OUTPUT_ADDR_OFFSET));
    tmp.set_start_op(TT_OP_PACR(ADDR_MOD_1, ZERO_OUTPUT_FLAG, PACK_SEL(PACKCNT), 0, MEGAROW, 0, 1));
    tmp.set_end_ops(
        TT_OP_REG2FLOP(1, 0, 0, 0, THCON_SEC0_REG1_L1_Dest_addr_ADDR32 - THCON_CFGREG_BASE_ADDR32, p_gpr_pack::OUTPUT_ADDR),
        TT_OP_PACR(ADDR_MOD_2, 0, 0xf, 0, 0, 1, 0));
    tmp.program();

    llk_pack_block_internal::configured_outer = 1;
    llk_pack_block_internal::zero_output_flag = ZERO_OUTPUT_FLAG;
}

// Pack num_tiles tiles from DEST, starting at tile_index, to L1 starting at address (same address form as _llk_pack_).
template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_pack_block_contiguous_(const std::uint32_t tile_index, const std::uint32_t address, const std::uint32_t num_tiles)
{
    LLK_ASSERT(num_tiles >= 1, "num_tiles must be >= 1");
    LLK_ASSERT(is_valid_L1_address(address), "L1 address must be in valid L1 memory region");

    set_dst_write_addr(tile_index);
    const std::uint32_t new_l1_addr = (1 << 31) | address;
    TT_SETDMAREG(0, LOWER_HALFWORD(address), 0, LO_16(p_gpr_pack::OUTPUT_ADDR));
    TT_SETDMAREG(0, UPPER_HALFWORD(new_l1_addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));
    TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC0_REG1_L1_Dest_addr_ADDR32 - THCON_CFGREG_BASE_ADDR32, p_gpr_pack::OUTPUT_ADDR);

    if (num_tiles > 1)
    {
        const std::uint32_t outer = num_tiles - 1;
        if (outer != llk_pack_block_internal::configured_outer)
        {
            // The MOP reads its config when it runs; wait for the previous MOP before changing the count.
            ckernel::mop_sync();
            reinterpret_cast<volatile std::uint32_t*>(TENSIX_MOP_CFG_BASE)[0] = outer;
            llk_pack_block_internal::configured_outer                        = outer;
        }
        mop_run(1, 1);
    }

    // Pack and close the last tile; leaves the packer counters in the single-tile state.
    if (llk_pack_block_internal::zero_output_flag == p_pacr::P_ZERO_OUTPUT_ENABLED)
    {
        TTI_PACR(ADDR_MOD_1, p_pacr::P_ZERO_OUTPUT_ENABLED, 0xf, 0, 1, 0, 1);
    }
    else
    {
        TTI_PACR(ADDR_MOD_1, p_pacr::P_ZERO_OUTPUT_DISABLED, 0xf, 0, 1, 0, 1);
    }
}
