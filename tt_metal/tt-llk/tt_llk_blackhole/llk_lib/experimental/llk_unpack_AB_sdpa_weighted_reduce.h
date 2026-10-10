// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "cunpack_common.h"
#include "llk_assert.h"
#include "llk_unpack_common.h"

using namespace ckernel;
using namespace ckernel::unpacker;

/**
 * @brief Unpack num_chunks chunks in one context: per chunk the next back-to-back qk tile's first two faces into SrcA and
 *        the weights tile into SrcB.
 *
 * @param address_qk: L1 address of the first qk tile in 16-byte words, minus one (the CB read pointer convention).
 * @param address_weights: L1 address of the weights tile in 16-byte words, minus one.
 * @param qk_num_faces: Faces per qk tile, 2 or 4. The walk steps from a tile's second face to the next tile's first.
 * @param num_chunks: Number of qk tiles, one chunk each.
 * @note Call weighted_reduce_init_short first: it leaves unpacker 0 at one face per UNPACR and no haloize.
 * @note The qk tiles must be dense and back to back, qk_num_faces faces each with no padding, and not block float.
 */
inline void _llk_unpack_AB_sdpa_weighted_reduce_block_(
    const std::uint32_t address_qk, const std::uint32_t address_weights, const std::uint32_t qk_num_faces, const std::uint32_t num_chunks)
{
    // The unpacker's input address generator sees only the low eight bits of Ch0.Z.
    constexpr std::uint32_t ch0_z_range = 256;
    LLK_ASSERT(qk_num_faces == 2 || qk_num_faces == 4, "sdpa_weighted_reduce (unpack): qk tiles must have 2 or 4 faces");
    LLK_ASSERT(num_chunks > 0, "sdpa_weighted_reduce (unpack): a block needs at least one chunk");
    LLK_ASSERT(num_chunks * qk_num_faces <= ch0_z_range, "sdpa_weighted_reduce (unpack): the qk faces must fit the 8-bit Z counter");
    volatile std::uint32_t tt_reg_ptr* cfg = get_cfg_pointer();

    TTI_SETADCZW(p_setadc::UNP_AB, 0, 0, 0, 0, 0b1111 /*Z0, W0, Z1, W1*/);
    wait_for_next_context(2);
    _llk_unpack_configure_addresses_(address_qk, address_weights, cfg);
    semaphore_post(semaphore::UNPACK_SYNC);
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

    // The second face's Ch0 Z increment steps the L1 read to the next qk tile's first face.
    const std::uint32_t next_tile_addr_mode = qk_num_faces - 1;
    for (std::uint32_t i = 0; i < num_chunks; i++)
    {
        TTI_UNPACR(SrcA, 0b00010001 /*Ch0.Z+=1, Ch1.Z+=1*/, 0, 0, 0, 1 /*OvrdThreadId*/, 0 /*no Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
        TTI_UNPACR(SrcB, 0b00000000 /*no counter step*/, 0, 0, 0, 1 /*OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
        TT_UNPACR(SrcA, next_tile_addr_mode /*Ch0.Z+=qk_num_faces-1*/, 0, 0, 0, 1 /*OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
        // Back to SrcA face 0 for the next chunk's bank.
        TTI_SETADCZW(p_setadc::UNP_A, 0, 0, 0, 0, 0b0100 /*Z1*/);
    }

    t6_semaphore_get(semaphore::UNPACK_SYNC);
    switch_config_context(unp_cfg_context);
    TTI_SETADCZW(p_setadc::UNP_AB, 0, 0, 0, 0, 0b1111 /*Z0, W0, Z1, W1*/);
}
