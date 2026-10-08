// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "../access_types.h"
#include "register_layout.h"
#include "state_bank.h"

namespace hal::cfg::detail
{

// These duplicate ckernel::reg_read, reg_write, and wait.
// TODO(njokovic) issue #58443: Remove ckernel:: implementations when HAL is applied to all kernels.
inline std::uint32_t reg_read(const std::uint32_t addr)
{
    volatile std::uint32_t tt_reg_ptr* reg = reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(addr);
    return reg[0];
}

inline void reg_write(const std::uint32_t addr, const std::uint32_t data)
{
    volatile std::uint32_t tt_reg_ptr* reg = reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(addr);
    reg[0]                                 = data;
}

inline void wait(const std::uint32_t cycles)
{
    volatile std::uint32_t tt_reg_ptr* clock_lo = reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    volatile std::uint32_t tt_reg_ptr* clock_hi = reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(RISCV_DEBUG_REG_WALL_CLOCK_H);
    const std::uint64_t start                   = clock_lo[0] | (static_cast<std::uint64_t>(clock_hi[0]) << 32);
    std::uint64_t now                           = 0;
    do
    {
        now = clock_lo[0] | (static_cast<std::uint64_t>(clock_hi[0]) << 32);
    } while (now < start + cycles);
}

template <ThreadTarget Target>
inline constexpr std::uint32_t compute_thread_index()
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
    constexpr std::uint32_t thread_index = compute_thread_index<Target>();
    constexpr std::uint32_t creg_addr    = ThreadCfgBase + thread_index * ThreadCfgWordCount + Addr;
    static_assert(creg_addr <= 0x7ffu, "thread CFG address exceeds the RISC CREG selector");

    reg_write(RISCV_DEBUG_REG_TENSIX_CREG_READ, creg_addr);
    wait(1 /*cycles*/);
    return reg_read(RISCV_DEBUG_REG_TENSIX_CREG_RDDATA);
}

/**
 * @brief Read one complete state-CFG word from the active bank.
 */
template <std::uint32_t Addr>
inline std::uint32_t read_state_word_mmio()
{
    return state_cfg_bank()[Addr];
}

} // namespace hal::cfg::detail
