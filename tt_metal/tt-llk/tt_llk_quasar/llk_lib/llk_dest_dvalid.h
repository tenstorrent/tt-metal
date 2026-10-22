// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "cfg_defines.h"
#include "ckernel.h"
#include "llk_defs.h"

namespace ckernel
{

namespace dest_dvalid
{

constexpr dest_dvalid_client NONE = static_cast<dest_dvalid_client>(4);

#if defined(LLK_TRISC_UNPACK) || defined(TRISC_UNPACK)
constexpr dest_dvalid_client THIS_UNIT = dest_dvalid_client::UNPACK;
#elif defined(LLK_TRISC_MATH) || defined(TRISC_MATH)
constexpr dest_dvalid_client THIS_UNIT = dest_dvalid_client::FPU;
#elif defined(LLK_TRISC_ISOLATE_SFPU) || defined(TRISC_ISOLATE_SFPU)
constexpr dest_dvalid_client THIS_UNIT = dest_dvalid_client::SFPU;
#elif defined(LLK_TRISC_PACK) || defined(TRISC_PACK)
constexpr dest_dvalid_client THIS_UNIT = dest_dvalid_client::PACK;
#else
constexpr dest_dvalid_client THIS_UNIT = NONE;
#endif

struct UnitInfo
{
    std::uint32_t bit;
    std::uint32_t ctrl_addr32;
    std::uint32_t wait_res;
    std::uint32_t stall_class;
};

constexpr UnitInfo UNITS[] = {
    {p_cleardvalid::UNPACK_TO_DEST, UNPACK_TO_DEST_DVALID_CTRL_wait_mask_ADDR32, p_stall::UNPACK0, p_stall::STALL_UNPACK},
    {p_cleardvalid::FPU, MATH_DEST_DVALID_CTRL_wait_mask_ADDR32, p_stall::MATH, p_stall::STALL_MATH},
    {p_cleardvalid::SFPU, SFPU_DEST_DVALID_CTRL_wait_mask_ADDR32, p_stall::WAIT_SFPU, p_stall::STALL_SFPU | p_stall::STALL_MATH},
    {p_cleardvalid::PACK, PACK_DEST_DVALID_CTRL_wait_mask_ADDR32, p_stall::PACK, p_stall::STALL_PACK},
};

constexpr UnitInfo info_of(const dest_dvalid_client unit)
{
    return UNITS[static_cast<int>(unit)];
}

constexpr std::uint32_t bit_of(const dest_dvalid_client unit)
{
    return unit == NONE ? 0 : info_of(unit).bit;
}

constexpr std::uint32_t ALL_BITS = 0xF;

constexpr std::uint32_t CTRL_WORD_MASK = UNPACK_TO_DEST_DVALID_CTRL_wait_mask_MASK | UNPACK_TO_DEST_DVALID_CTRL_wait_polarity_MASK |
                                         UNPACK_TO_DEST_DVALID_CTRL_toggle_mask_MASK | UNPACK_TO_DEST_DVALID_CTRL_disable_auto_bank_id_toggle_MASK;
constexpr std::uint32_t RELEASE_MASK = CTRL_WORD_MASK >> 8;
static_assert((CTRL_WORD_MASK & 0xFF) == 0xFF && CTRL_WORD_MASK < (1u << 16));

static_assert(MATH_DEST_DVALID_CTRL_wait_mask_SHAMT == UNPACK_TO_DEST_DVALID_CTRL_wait_mask_SHAMT);
static_assert(SFPU_DEST_DVALID_CTRL_wait_polarity_SHAMT == UNPACK_TO_DEST_DVALID_CTRL_wait_polarity_SHAMT);
static_assert(PACK_DEST_DVALID_CTRL_toggle_mask_SHAMT == UNPACK_TO_DEST_DVALID_CTRL_toggle_mask_SHAMT);
static_assert(PACK_DEST_DVALID_CTRL_disable_auto_bank_id_toggle_SHAMT == UNPACK_TO_DEST_DVALID_CTRL_disable_auto_bank_id_toggle_SHAMT);

constexpr std::uint32_t wait_byte(const std::uint32_t wait_mask, const std::uint32_t wait_polarity)
{
    return (wait_mask << UNPACK_TO_DEST_DVALID_CTRL_wait_mask_SHAMT) | (wait_polarity << UNPACK_TO_DEST_DVALID_CTRL_wait_polarity_SHAMT);
}

constexpr std::uint32_t release_byte(const std::uint32_t toggle_mask, const bool hold_bank)
{
    return ((toggle_mask << UNPACK_TO_DEST_DVALID_CTRL_toggle_mask_SHAMT) |
            (static_cast<std::uint32_t>(hold_bank) << UNPACK_TO_DEST_DVALID_CTRL_disable_auto_bank_id_toggle_SHAMT)) >>
           8;
}

} // namespace dest_dvalid

/**
 * @brief Change only the wait byte, leaving the toggle of a pulse that may still be in flight alone.
 * @param stall: hold UNIT's dest accesses until the word is applied, needed when the old word could admit one early
 */
template <dest_dvalid_client UNIT, std::uint32_t MASK, std::uint32_t POLARITY>
inline void _llk_dest_dvalid_write_wait_(const bool stall)
{
    constexpr dest_dvalid::UnitInfo INFO = dest_dvalid::info_of(UNIT);
    static_assert(INFO.ctrl_addr32 < 256, "control address must fit RMWCIB");
    TTI_RMWCIB0(INFO.ctrl_addr32, 0xFF, dest_dvalid::wait_byte(MASK, POLARITY));
    if (stall)
    {
        TTI_STALLWAIT(INFO.stall_class, p_stall::NOTHING, p_stall::NOTHING, p_stall::CFGEXU);
    }
}

/**
 * @brief Change only the release byte, right before the pulse that uses it.
 */
template <dest_dvalid_client UNIT, std::uint32_t TOGGLE, bool HOLD_BANK>
inline void _llk_dest_dvalid_write_release_()
{
    TTI_RMWCIB1(dest_dvalid::info_of(UNIT).ctrl_addr32, dest_dvalid::RELEASE_MASK, dest_dvalid::release_byte(TOGGLE, HOLD_BANK));
}

/**
 * @brief Waits for UNIT's engine and pulses its `last`.
 */
template <dest_dvalid_client UNIT>
inline void _llk_dest_dvalid_pulse_()
{
    constexpr dest_dvalid::UnitInfo INFO = dest_dvalid::info_of(UNIT);
    TTI_STALLWAIT(p_stall::STALL_THREAD, p_stall::NOTHING, p_stall::CFGEXU, INFO.wait_res);
    TTI_CLEARDVALID(0, 0, 0, 0, INFO.bit, 0);
}

/**
 * @brief Takes UNIT out of the handshake: its dest accesses are no longer gated.
 */
template <dest_dvalid_client UNIT>
inline void _llk_dest_dvalid_leave_()
{
    constexpr dest_dvalid::UnitInfo INFO = dest_dvalid::info_of(UNIT);
    TTI_STALLWAIT(p_stall::STALL_THREAD, p_stall::NOTHING, p_stall::NOTHING, INFO.wait_res);
    _llk_dest_dvalid_write_wait_<UNIT, 0 /*MASK*/, 0 /*POLARITY*/>(false /*stall*/);
}

/**
 * @brief Hand dest to NEXT after UNIT finishes.
 * @tparam HAS_BIT: false when UNIT was admitted on an empty bank, so it has no own bit to clear
 * @tparam HOLD_BANK: keep UNIT on its bank, otherwise the pulse moves it to the other one
 */
template <dest_dvalid_client UNIT, dest_dvalid_client NEXT, bool HAS_BIT, bool HOLD_BANK>
inline void _llk_dest_dvalid_signal_()
{
    static_assert(UNIT != dest_dvalid::NONE && NEXT != dest_dvalid::NONE && UNIT != NEXT, "a signal needs an owner and a successor");
    constexpr std::uint32_t OWN = HAS_BIT ? dest_dvalid::bit_of(UNIT) : 0;
    _llk_dest_dvalid_write_release_<UNIT, OWN ^ dest_dvalid::bit_of(NEXT), HOLD_BANK>();
    _llk_dest_dvalid_pulse_<UNIT>();
}

/**
 * @brief Ends a bank use on PACK's thread: clears PACK's bit and moves PACK to the other bank.
 */
inline void _llk_dest_dvalid_end_use_()
{
    _llk_dest_dvalid_write_release_<dest_dvalid_client::PACK, p_cleardvalid::PACK, false /*HOLD_BANK*/>();
    _llk_dest_dvalid_pulse_<dest_dvalid_client::PACK>();
}

/**
 * @brief Moves UNIT to the other bank while it has no access in flight.
 *
 * Pulse the same nonzero mask twice, holding the bank on the first pulse, to preserve all ownership bits.
 */
template <dest_dvalid_client UNIT>
inline void _llk_dest_dvalid_bank_flip_()
{
    _llk_dest_dvalid_write_release_<UNIT, dest_dvalid::bit_of(UNIT), true /*HOLD_BANK*/>();
    _llk_dest_dvalid_pulse_<UNIT>();
    _llk_dest_dvalid_write_release_<UNIT, dest_dvalid::bit_of(UNIT), false /*HOLD_BANK*/>();
    _llk_dest_dvalid_pulse_<UNIT>();
}

/**
 * @brief Clears every dvalid bit and every bank id. For firmware boot, when no unit can be active.
 */
inline void _llk_dest_dvalid_boot_reset_()
{
    TTI_CLEARDVALID(0, 0, dest_dvalid::ALL_BITS, dest_dvalid::ALL_BITS, 0, 0);
}

} // namespace ckernel
