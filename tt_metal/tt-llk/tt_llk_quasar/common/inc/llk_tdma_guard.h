// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// TEN-4746 (Quasar-only) tile-counter guard.
//
// HW constraint: a tile-counter WAIT (TT_WAIT_TILES / TT_WAIT_FREE) must be followed by at least one
// real TDMA (UNPACR / PACR) that touches the same dataflow buffer before the matching counter retire
// (TT_POP_TILES / TT_PUSH_TILES). If a POP/PUSH follows a WAIT on the same dfb with no intervening
// TDMA on that dfb, the WAIT can resolve before tiles/space are actually available.
//
// This guard tracks, per TRISC and per dfb id, whether a WAIT is still "armed" (issued with no TDMA
// on that dfb since). Arm in the WAIT, disarm in the data-moving llk_unpack_*/llk_pack_* executes,
// assert-disarmed in the POP/PUSH. It is a debug-only aid: when LLK asserts are disabled the macros
// compile to nothing (zero cost).
//
// dfb ids are in [0..31], so a single uint32_t bitmask covers every dataflow buffer.

#include <cstdint>

#include "llk_assert.h"

#if defined(ENV_LLK_INFRA) || defined(ENABLE_LLK_ASSERT_ONLY) || defined(ENABLE_LLK_ASSERT)

namespace llk_tdma_guard
{
// One mask per TRISC (the unpack TRISC owns the unpack copy, the pack TRISC owns the pack copy).
// Bit d set  => a WAIT armed dfb d and no TDMA has touched dfb d since.
inline std::uint32_t& armed_mask()
{
    static std::uint32_t mask = 0;
    return mask;
}

inline void note_wait(const std::uint32_t dfb)
{
    armed_mask() |= (static_cast<std::uint32_t>(1) << dfb);
}

inline void note_tdma(const std::uint32_t dfb)
{
    armed_mask() &= ~(static_cast<std::uint32_t>(1) << dfb);
}

inline bool armed(const std::uint32_t dfb)
{
    return (armed_mask() >> dfb) & static_cast<std::uint32_t>(1);
}
} // namespace llk_tdma_guard

#define LLK_TDMA_GUARD_NOTE_WAIT(dfb)            llk_tdma_guard::note_wait(dfb)
#define LLK_TDMA_GUARD_NOTE_TDMA(dfb)            llk_tdma_guard::note_tdma(dfb)
#define LLK_TDMA_GUARD_ASSERT_DISARMED(dfb, msg) LLK_ASSERT(!llk_tdma_guard::armed(dfb), msg)

#else

#define LLK_TDMA_GUARD_NOTE_WAIT(dfb)            ((void)0)
#define LLK_TDMA_GUARD_NOTE_TDMA(dfb)            ((void)0)
#define LLK_TDMA_GUARD_ASSERT_DISARMED(dfb, msg) ((void)0)

#endif
