// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL port of tt_llk_blackhole/llk_lib/experimental/llk_pack_block.h. Same instruction sequence and access kinds as
// production, expressed through the HAL.

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "hal/mop.h"
#include "hal/pack.h"
#include "hal/replay.h"
#include "hal/tensix_counters.h"
#include "llk_assert.h"
#include "llk_defs.h"
#include "llk_pack_common.h"

using namespace ckernel;
using namespace ckernel::packer;

namespace hal_port
{
namespace pack_block
{
// TT_OP_SETADCZW(PAC, 0, 0, 0, 0, 0b0001): channel 0 Z = 0 as a masked SETADCZW. The public planner encodes a lone
// counter as SETADC, a different instruction.
// HAL-GAP: no public encoder forces the packed SETADCZW form for a single counter; uses hal::counters::adc::detail::packed.
inline constexpr std::uint32_t setadczw_pac_ch0_z0 = []
{
    namespace adc = hal::counters::adc;
    adc::detail::Group group {};
    group.fields[0] = 0; // channel 0 Z
    return adc::detail::packed(adc::AddrCounter {}.for_clients<adc::Client::Packers>().clients, true /*zw*/, adc::Action::Set, group);
}();
} // namespace pack_block
} // namespace hal_port

/*************************************************************************
 * LLK PACK BLOCK CONTIGUOUS
 *
 * Packs multiple tiny tiles from sparse DEST (Tile32x32 slot convention)
 * to dense L1 (contiguous output) in a single _llk_pack_ call.
 *
 * MOP structure:
 *   OUTER = num_tiles (runtime-patchable via mop_cfg[0])
 *   INNER = 1
 *   START_OP  = REPLAY(all-but-last PACRs for one tile)
 *   last_inner= last PACR with ADDR_MOD_2 (non-last tile)
 *   last_outer= last PACR with ADDR_MOD_1 + Last=1 (final tile)
 *   END_OP0   = INCADCZW(W++) to advance to next Tile32x32 slot
 *   END_OP1   = SETADCZW(Z=0) to reset Z for next tile
 *
 * REPLAY buffer stores num_faces * pacrs_per_face - 1 PACR instructions
 * encoding the complete per-tile face traversal. The final PACR of each
 * tile lives in the MOP template (last_inner / last_outer) so it can
 * carry the Last=1 bit on the final tile.
 *
 * Precondition: _llk_pack_init_ or _llk_pack_configure_addrmod_ +
 * set_packer_strides must have been called to establish the normal
 * pack ADDR_MOD_0/1/2 and strides. This function only replaces the MOP.
 *************************************************************************/

// Program the REPLAY buffer and MOP for block-contiguous packing.
template <bool zero_output = false>
inline void _llk_pack_block_contiguous_mop_config_(const std::uint32_t face_r_dim = FACE_R_DIM, const std::uint32_t num_faces = 4)
{
    using namespace hal::mop;
    using hal::pack::DataTransfer;
    using hal::pack::TileBoundary;
    namespace adc = hal::counters::adc;

    constexpr hal::pack::DatumOverride ZERO_OUTPUT_FLAG = zero_output ? hal::pack::DatumOverride::Zero : hal::pack::DatumOverride::None;

    const std::uint32_t PACK_INTF_SEL = face_r_dim == 1 ? p_pacr::SINGLE_INTF_ACTIVE : (face_r_dim == 2 ? p_pacr::TWO_INTFS_ACTIVE : p_pacr::ALL_INTF_ACTIVE);

    const std::uint32_t pacrs_per_face = (face_r_dim < 4) ? 1 : face_r_dim >> 2;
    const std::uint32_t total_pacrs    = num_faces * pacrs_per_face;

    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    LLK_ASSERT(total_pacrs >= 1, "At least 1 PACR required per tile");
    LLK_ASSERT(total_pacrs <= 16, "Replay buffer overflow: max 16 PACRs per tile");

    // Guard the subtraction to avoid unsigned underflow — the compiler's
    // intrinsic range checker on __builtin_rvtt_ttreplay rejects UINT32_MAX
    // even in dead code paths.
    const std::uint32_t replay_len = (total_pacrs > 1) ? (total_pacrs - 1) : 0;

    // TT_OP_PACR(CFG_CTXT_0, NO_ROW_PAD_ZERO, DST_ACCESS_NORMAL_MODE, AddrMode, ADDR_CNT_CTXT_0, ZERO_OUTPUT_FLAG,
    //            PACK_INTF_SEL, 0, 0, 0, 0, Last)
    DataTransfer transfer {
        .address_modifier = ADDR_MOD_0,
        .interfaces       = static_cast<std::uint8_t>(PACK_INTF_SEL),
        .datum_override   = ZERO_OUTPUT_FLAG,
    };
    const std::uint32_t pacr_mod0 = transfer.get_operation();
    transfer.address_modifier     = ADDR_MOD_2;
    const std::uint32_t pacr_mod2 = transfer.get_operation(); // last_inner: final PACR of a non-last tile (Last=0)
    transfer.address_modifier     = ADDR_MOD_1;
    transfer.boundary             = TileBoundary::Last;
    const std::uint32_t pacr_last = transfer.get_operation(); // last_outer: final PACR of the last tile (Last=1)

    // --- Load REPLAY buffer with the per-tile PACR sequence ---
    // Contains: for each face, (pacrs_per_face - 1) PACRs with ADDR_MOD_0
    // followed by 1 PACR with ADDR_MOD_2 (face transition: Y clear, Z += 1).
    // The very last PACR of the tile is NOT in the replay — it goes in the
    // MOP template as last_inner / last_outer.
    if (replay_len > 0)
    {
        hal::replay::record(
            hal::replay::BufferRange {0, replay_len},
            [&]
            {
                // Emit instructions into replay buffer via instrn_buffer (runtime-safe)
                for (std::uint32_t face = 0; face < num_faces; face++)
                {
                    const bool is_last_face          = (face == num_faces - 1);
                    const std::uint32_t replay_pacrs = is_last_face ? (pacrs_per_face - 1) : pacrs_per_face;

                    for (std::uint32_t p = 0; p < replay_pacrs; p++)
                    {
                        const bool is_last_pacr_of_face = (p == pacrs_per_face - 1);
                        // HAL-GAP: no HAL API issues a pre-encoded runtime instruction word; TT_INSN kept.
                        TT_INSN(is_last_pacr_of_face ? pacr_mod2 : pacr_mod0);
                    }
                }
            });
    }

    // --- Program MOP ---
    // OUTER = num_tiles (placeholder 1, overwritten in _llk_pack_block_contiguous_), INNER = 1
    // START_OP: replay the per-tile PACR sequence (all but last PACR), or NOP if the tile has only 1 PACR
    // last_inner / last_outer: the final PACR of a non-last / the last tile
    // END_OP0: W++ (next Tile32x32 slot): TT_OP_INCADCZW(PAC, 0, 0, 1, 0)
    // END_OP1: Z=0 (reset for next tile's face traversal): TT_OP_SETADCZW(PAC, 0, 0, 0, 0, 0b0001)
    const std::uint32_t start_op = (replay_len > 0) ? hal::replay::get_operation(hal::replay::BufferRange {0, replay_len}) : TT_OP_NOP;
    constexpr std::uint32_t next_slot =
        adc::AddrCounter {.action = adc::Action::Increment, .channel0 = {.w = 1}}.for_clients<adc::Client::Packers>().operation();
    constexpr std::uint32_t z_to_zero = hal_port::pack_block::setadczw_pac_ch0_z0;
    static_assert(next_slot == TT_OP_INCADCZW(p_setadc::PAC, 0, 0, 1, 0), "END_OP0 must stay INCADCZW");
    static_assert(z_to_zero == TT_OP_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0001), "END_OP1 must stay SETADCZW");

    auto config = loop(1, start_op, loop(1, TT_OP_NOP), next_slot, z_to_zero).patch(last_outer, pacr_last);
    // HAL-GAP: cfg::patch(last_inner, ...) rejects an outer count of 1, but this MOP's outer count is a placeholder that
    // _llk_pack_block_contiguous_ overwrites with num_tiles, so the word is set directly, as production stores it.
    config.last_inner_op = pacr_mod2;
    program(config);
}

// Execute the block-contiguous pack.
// tile_index: starting tile in DEST (sets W counter)
// address: L1 destination address for the contiguous output block
// num_tiles: number of tiles to pack (1-8, runtime parameter)
template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_pack_block_contiguous_(const std::uint32_t tile_index, const std::uint32_t address, const std::uint32_t num_tiles)
{
    namespace adc = hal::counters::adc;

    set_dst_write_addr(tile_index);

    TTI_INSN(hal_port::pack_block::setadczw_pac_ch0_z0); // Z = 0: TTI_SETADCZW(PAC, 0, 0, 0, 0, 0b0001)

    hal_port::program_packer_destination(address);

    // Patch outer loop count to num_tiles before running the MOP (mop_sync, then mop_cfg[0] = num_tiles).
    hal::mop::update(hal::mop::outer_count, num_tiles);
    hal::mop::Runner<hal::mop::MopTemplate::Template1>::run(); // TTI_MOP(1, 0, 0)

    // reset Z/W: TTI_SETADCZW(PAC, 0, 0, 0, 0, 0b0101)
    adc::set<adc::AddrCounter {.channel0 = {.z = 0}, .channel1 = {.z = 0}}.for_clients<adc::Client::Packers>()>();
}
