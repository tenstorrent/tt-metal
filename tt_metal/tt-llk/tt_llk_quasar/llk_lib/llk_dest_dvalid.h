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

constexpr std::uint32_t ctrl_word(const std::uint32_t wait_mask, const std::uint32_t wait_polarity, const std::uint32_t toggle_mask, const bool hold_bank)
{
    return (wait_mask << UNPACK_TO_DEST_DVALID_CTRL_wait_mask_SHAMT) | (wait_polarity << UNPACK_TO_DEST_DVALID_CTRL_wait_polarity_SHAMT) |
           (toggle_mask << UNPACK_TO_DEST_DVALID_CTRL_toggle_mask_SHAMT) |
           (static_cast<std::uint32_t>(hold_bank) << UNPACK_TO_DEST_DVALID_CTRL_disable_auto_bank_id_toggle_SHAMT);
}

constexpr std::uint32_t PULSE_STALL = p_stall::STALL_THREAD;

// Semaphore that carries "the previous section on this bank is finished" to a section head, per head unit and bank.
// PACK never heads a section. 0 and 3 are the unpack to dest semaphores, unused by a chain kernel.
constexpr std::uint8_t TOKEN_SEMS[4][2] = {{12, 13}, {14, 15}, {0, 3}, {0, 0}};

constexpr std::uint8_t token_sem(const dest_dvalid_client head, const std::uint32_t bank)
{
    return TOKEN_SEMS[static_cast<int>(head)][bank];
}

constexpr std::uint32_t sem_bank(const std::uint8_t sem)
{
    return sem >> 3;
}

constexpr std::uint32_t sem_sel(const std::uint8_t sem)
{
    return 1u << (sem & 7);
}

} // namespace dest_dvalid

/**
 * @brief Writes UNIT's control word as a Tensix instruction in the thread's own stream.
 *
 * The write lands after every access the thread pushed before it has issued. UNIT's dest accesses then wait
 * until the CFG unit has applied the word, so the next one is judged by the new word.
 */
template <dest_dvalid_client UNIT, std::uint32_t WORD>
inline void _llk_dest_dvalid_write_ctrl_()
{
    constexpr dest_dvalid::UnitInfo INFO = dest_dvalid::info_of(UNIT);
    static_assert((WORD & ~dest_dvalid::CTRL_WORD_MASK) == 0, "control word out of range");
    static_assert(INFO.ctrl_addr32 < 256, "control address must fit RMWCIB");
    TTI_RMWCIB0(INFO.ctrl_addr32, 0xFF, WORD & 0xFF);
    TTI_RMWCIB1(INFO.ctrl_addr32, dest_dvalid::RELEASE_MASK, WORD >> 8);
    TTI_STALLWAIT(INFO.stall_class, p_stall::NOTHING, p_stall::NOTHING, p_stall::CFGEXU);
}

/**
 * @brief Change only the release byte, preserving admission of preceding dest accesses.
 * @note Order the following pulse after CFGEXU with @ref _llk_dest_dvalid_pulse_.
 */
template <dest_dvalid_client UNIT, std::uint32_t TOGGLE, bool HOLD_BANK>
inline void _llk_dest_dvalid_write_release_()
{
    constexpr std::uint32_t WORD = dest_dvalid::ctrl_word(0, 0, TOGGLE, HOLD_BANK);
    TTI_RMWCIB1(dest_dvalid::info_of(UNIT).ctrl_addr32, dest_dvalid::RELEASE_MASK, WORD >> 8);
}

/**
 * @brief Gate UNIT's dest accesses for its turn and preset the toggle of its section end pulse.
 *
 * A head is admitted by its token semaphore, never sets its own bit, and so ends with an empty toggle.
 * A unit admitted by a signal waits for its own bit and clears that bit when it ends the section.
 */
template <dest_dvalid_client UNIT, bool HEAD, DstSync DST>
inline void _llk_dest_dvalid_admit_()
{
    constexpr std::uint32_t OWN = HEAD ? 0 : dest_dvalid::bit_of(UNIT);
    _llk_dest_dvalid_write_ctrl_<UNIT, dest_dvalid::ctrl_word(OWN, OWN, OWN, DST == DstSync::SyncFull)>();
}

/**
 * @brief Block UNIT's thread until the previous section on `bank` has finished, then consume the token.
 */
inline void _llk_dest_dvalid_token_wait_(const std::uint8_t sem)
{
    TT_SEMWAIT(p_stall::STALL_THREAD, p_stall::STALL_ON_ZERO, dest_dvalid::sem_bank(sem), dest_dvalid::sem_sel(sem));
    TT_SEMGET(dest_dvalid::sem_bank(sem), dest_dvalid::sem_sel(sem));
}

/**
 * @brief Hand the token of a finished bank to the head that actually starts the next section on it.
 * @note Runs on the thread of the unit the token was posted to, so its wait orders the post after the finish.
 */
inline void _llk_dest_dvalid_token_forward_(const std::uint8_t from_sem, const std::uint8_t to_sem)
{
    TT_SEMWAIT(p_stall::STALL_SYNC, p_stall::STALL_ON_ZERO, dest_dvalid::sem_bank(from_sem), dest_dvalid::sem_sel(from_sem));
    TT_SEMGET(dest_dvalid::sem_bank(from_sem), dest_dvalid::sem_sel(from_sem));
    TT_SEMPOST(dest_dvalid::sem_bank(to_sem), dest_dvalid::sem_sel(to_sem));
}

/**
 * @brief Seed every token semaphore. For firmware boot, from one thread, when no section is active.
 */
inline void _llk_dest_dvalid_token_init_()
{
    for (std::uint32_t head = 0; head < 3; ++head)
    {
        for (std::uint32_t bank = 0; bank < 2; ++bank)
        {
            const std::uint8_t sem = dest_dvalid::TOKEN_SEMS[head][bank];
            TT_SEMINIT(1 /*max_value*/, 0 /*init_value*/, dest_dvalid::sem_bank(sem), dest_dvalid::sem_sel(sem));
        }
    }
}

/**
 * @brief Waits for UNIT's engine and pulses its `last`.
 */
template <dest_dvalid_client UNIT>
inline void _llk_dest_dvalid_pulse_()
{
    constexpr dest_dvalid::UnitInfo INFO = dest_dvalid::info_of(UNIT);
    TTI_STALLWAIT(dest_dvalid::PULSE_STALL, p_stall::NOTHING, p_stall::CFGEXU, INFO.wait_res);
    TTI_CLEARDVALID(0, 0, 0, 0, INFO.bit, 0);
}

/**
 * @brief Reset UNIT to bank 0 before programming its admission word.
 */
template <dest_dvalid_client UNIT>
inline void _llk_dest_dvalid_join_()
{
    TTI_CLEARDVALID(0, 0, 0, dest_dvalid::bit_of(UNIT), 0, 0);
}

/**
 * @brief Takes UNIT out of the handshake: its dest accesses are no longer gated.
 */
template <dest_dvalid_client UNIT>
inline void _llk_dest_dvalid_leave_()
{
    constexpr dest_dvalid::UnitInfo INFO = dest_dvalid::info_of(UNIT);
    TTI_STALLWAIT(dest_dvalid::PULSE_STALL, p_stall::NOTHING, p_stall::NOTHING, INFO.wait_res);
    _llk_dest_dvalid_write_ctrl_<UNIT, 0>();
}

/**
 * @brief Hand dest to NEXT after UNIT finishes, doing nothing when UNIT already owns it.
 * @tparam HEAD: true when UNIT holds dest as the section head, so it has no own bit to clear
 */
template <dest_dvalid_client UNIT, dest_dvalid_client NEXT, DstSync DST, bool HEAD>
inline void _llk_dest_dvalid_signal_()
{
    static_assert(UNIT != dest_dvalid::NONE && NEXT != dest_dvalid::NONE, "a signal needs an owner and a successor");
    if constexpr (UNIT != NEXT)
    {
        constexpr std::uint32_t OWN = HEAD ? 0 : dest_dvalid::bit_of(UNIT);
        _llk_dest_dvalid_write_release_<UNIT, OWN ^ dest_dvalid::bit_of(NEXT), DST == DstSync::SyncFull>();
        _llk_dest_dvalid_pulse_<UNIT>();
    }
}

/**
 * @brief Finish a section once UNIT's reads are done: clear UNIT's bit, then post the token the next head waits for.
 * @param token_sem: semaphore of this section's head, see @ref dest_dvalid::token_sem
 * @note The pulse uses the toggle preset by @ref _llk_dest_dvalid_admit_, so no control write is needed here.
 */
template <dest_dvalid_client UNIT>
inline void _llk_dest_dvalid_finish_(const std::uint8_t token_sem)
{
    _llk_dest_dvalid_pulse_<UNIT>();
    TT_SEMPOST(dest_dvalid::sem_bank(token_sem), dest_dvalid::sem_sel(token_sem));
}

/**
 * @brief Moves UNIT to the other bank, for a unit that sat out an odd number of sections.
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
