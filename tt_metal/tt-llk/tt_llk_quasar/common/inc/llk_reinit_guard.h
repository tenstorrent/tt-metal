// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Quasar-only stale-MOP guard (#44071).
//
// The Buffer Descriptor (BFD) ID is baked into the pack/unpack MOP at *init* time
// (_llk_pack_mop_config_ -> TT_OP_PACR0_TILE_INC(..., buf_desc_id, ...)).
// Switching a kernel's pack or unpack target Dataflow Buffer (DFB) without re-running the matching *_init
// leaves the MOP aimed at the previous DFB, and the op silently touches the wrong buffer.
//
// This guard records, per TRISC and per BFD id (one fixed id per TDMA engine), the DFB ID that engine's descriptor
// was last programmed from, and asserts at execute time that the caller's DFB matches.

#include <cstdint>

#include "ckernel_buf_desc.h"
#include "llk_assert.h"

#if defined(ENV_LLK_INFRA) || defined(ENABLE_LLK_ASSERT_ONLY) || defined(ENABLE_LLK_ASSERT)

namespace llk_reinit_guard
{

#ifdef ENV_LLK_INFRA
inline std::uint8_t* slots()
{
    static std::uint8_t s[ckernel::trisc::BD_TABLE_NUM_ENTRIES] = {};
    return s;
}
#else
extern thread_local std::uint8_t reinit_guard_slots[ckernel::trisc::BD_TABLE_NUM_ENTRIES]; // defined in tt_metal/hw/firmware/src/tt-2xx/trisc.cc

inline std::uint8_t* slots()
{
    return reinit_guard_slots;
}
#endif

inline void note_programmed(const std::uint8_t bfd_id, const std::uint32_t dfb)
{
    slots()[bfd_id] = static_cast<std::uint8_t>(dfb + 1);
}

inline bool matches(const std::uint8_t bfd_id, const std::uint32_t dfb)
{
    return slots()[bfd_id] == static_cast<std::uint8_t>(dfb + 1);
}
} // namespace llk_reinit_guard

#define LLK_REINIT_GUARD_NOTE_PROGRAMMED(engine, dfb)     llk_reinit_guard::note_programmed(engine, dfb)
#define LLK_REINIT_GUARD_ASSERT_MATCHES(engine, dfb, msg) LLK_ASSERT(llk_reinit_guard::matches(engine, dfb), msg)

#else

#define LLK_REINIT_GUARD_NOTE_PROGRAMMED(engine, dfb)     ((void)0)
#define LLK_REINIT_GUARD_ASSERT_MATCHES(engine, dfb, msg) ((void)0)

#endif
