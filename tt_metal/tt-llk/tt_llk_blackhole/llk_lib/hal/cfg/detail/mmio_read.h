// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "../access_types.h"
#include "ckernel.h"
#include "register_layout.h"

namespace hal::cfg::detail
{

template <ThreadTarget Target>
inline constexpr std::uint32_t thread_target_index()
{
    if constexpr (Target == ThreadTarget::Current)
    {
#if defined(COMPILE_FOR_TRISC)
        static_assert(COMPILE_FOR_TRISC >= 0 && COMPILE_FOR_TRISC <= 2, "COMPILE_FOR_TRISC must select TRISC0, TRISC1, or TRISC2");
        return COMPILE_FOR_TRISC;
#else
        static_assert(Target != ThreadTarget::Current, "BRISC thread-CFG reads must explicitly select ThreadTarget::T0, T1, or T2");
        return 0;
#endif
    }
    else
    {
        return static_cast<std::uint32_t>(Target) - static_cast<std::uint32_t>(ThreadTarget::T0);
    }
}

template <ThreadTarget Target, std::uint32_t Addr>
inline __attribute__((always_inline)) std::uint32_t read_thread_word_mmio()
{
    constexpr std::uint32_t thread_index = thread_target_index<Target>();
    constexpr std::uint32_t creg_addr    = ThreadCfgBase + thread_index * ThreadCfgWordCount + Addr;
    static_assert(creg_addr <= 0x7ffu, "thread CFG address exceeds the RISC CREG selector");

    ckernel::reg_write(RISCV_DEBUG_REG_TENSIX_CREG_READ, creg_addr);
    ckernel::wait(1);
    return ckernel::reg_read(RISCV_DEBUG_REG_TENSIX_CREG_RDDATA);
}

} // namespace hal::cfg::detail
