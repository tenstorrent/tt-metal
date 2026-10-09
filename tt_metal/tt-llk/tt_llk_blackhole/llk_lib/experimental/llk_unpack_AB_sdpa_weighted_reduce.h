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
 * @brief Unpack num_chunks weighted-reduce chunks in one context transaction: per chunk the first two faces of the next
 *        qk tile into SrcA and the weights into SrcB, each source set valid once loaded.
 *
 * @param address_qk: L1 address of the first qk tile, in the 16-byte-word encoding of L1_ADDRESS(). The qk tiles sit back
 *                    to back, qk_num_faces faces each.
 * @param address_weights: L1 address of the weights tile, read again for every chunk.
 * @param qk_num_faces: Faces per qk tile, 2 to 4.
 * @param num_chunks: Number of chunks; num_chunks x qk_num_faces at most 256 (the Z counter's address bits).
 * @note The unpacker X ends are the caller's (weighted_reduce_init_short sets unpacker 0 to one face per UNPACR).
 */
inline void _llk_unpack_AB_sdpa_weighted_reduce_block_(
    const std::uint32_t address_qk, const std::uint32_t address_weights, const std::uint32_t qk_num_faces, const std::uint32_t num_chunks)
{
    LLK_ASSERT(qk_num_faces >= 2 && qk_num_faces <= 4, "sdpa_weighted_reduce (unpack): qk tiles must have 2 to 4 faces");
    LLK_ASSERT(num_chunks * qk_num_faces <= 256, "sdpa_weighted_reduce (unpack): the qk faces must fit the 8-bit Z counter");
    volatile std::uint32_t tt_reg_ptr* cfg = get_cfg_pointer();

    TTI_SETADCZW(p_setadc::UNP_AB, 0, 0, 0, 0, 0b1111);
    wait_for_next_context(2);
    _llk_unpack_configure_addresses_(address_qk, address_weights, cfg);
    semaphore_post(semaphore::UNPACK_SYNC);
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

    // The second face's Ch0 Z increment steps the L1 read to the next qk tile's first face.
    const std::uint32_t next_tile_addr_mode = qk_num_faces - 1;
    for (std::uint32_t i = 0; i < num_chunks; i++)
    {
        TTI_UNPACR(SrcA, 0b00010001, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
        TTI_UNPACR(SrcB, 0b00000000, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
        TT_UNPACR(SrcA, next_tile_addr_mode, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
        // Back to SrcA face 0 for the next chunk's bank.
        TTI_SETADCZW(p_setadc::UNP_A, 0, 0, 0, 0, 0b0100);
    }

    t6_semaphore_get(semaphore::UNPACK_SYNC);
    switch_config_context(unp_cfg_context);
    TTI_SETADCZW(p_setadc::UNP_AB, 0, 0, 0, 0, 0b1111);
}
