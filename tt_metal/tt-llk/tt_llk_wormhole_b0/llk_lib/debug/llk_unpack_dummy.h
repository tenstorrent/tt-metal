// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Debug-only SrcA flush for Wormhole B0.
//
// Unlike Quasar -- where llk_unpack_dummy is a required drain primitive that orders a POP_TILES after
// its WAIT_TILES (TEN-4746) -- Wormhole B0 has no such WAIT/POP ordering requirement, so this call has
// no functional role in a kernel. It is provided only so the shared compute-API dummy_unpack() resolves
// on every architecture; use it directly only for debug, when an explicit SrcA flush is wanted.

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_ops.h"

/**
 * @brief Flush the SrcA bank: STALLWAIT on SrcA-clear then a clear-SrcA UNPACR_NOP (no CB read, no DEST
 *        write). Reads nothing from L1.
 *
 * The STALLWAIT ensures SrcA is free before the clear, so this cannot clobber a SrcA bank still owned by an
 * in-flight op; SrcA is cleared only (the next op re-unpacks it). WH/BH have no WAIT/POP ordering
 * requirement, so this is a debug-only flush. (On Quasar the same call is a required drain primitive; see
 * that arch's llk_unpack_dummy.)
 *
 * @param dfb_id  Unused on WH/BH; accepted only to match the Quasar signature, where it names the drained
 *                dataflow buffer.
 */
inline void llk_unpack_dummy([[maybe_unused]] const std::uint32_t dfb_id)
{
    TTI_STALLWAIT(ckernel::p_stall::STALL_UNPACK, ckernel::p_stall::SRCA_CLR);
    TTI_UNPACR_NOP(ckernel::SrcA, ckernel::p_unpacr_nop::UNP_ZEROSRC);
}
