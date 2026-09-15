// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "cpack_common.h"
#include "llk_defs.h"

namespace ckernel
{

struct dest_dvalid_config
{
    std::uint32_t ctrl_addr32;
    std::uint32_t bit;
    std::uint32_t drain_res;
};

constexpr dest_dvalid_config dest_dvalid_configs[] = {
    // drain_res is the only unit each client waits on before handing dest over. With the SFPU on its own thread,
    // a producer must not wait for SFPU idle: the SFPU may be sitting on the very section this handover releases.
    {UNPACK_TO_DEST_DVALID_CTRL_wait_mask_ADDR32, p_cleardvalid::UNPACK_TO_DEST, p_stall::UNPACK0},
    {MATH_DEST_DVALID_CTRL_wait_mask_ADDR32, p_cleardvalid::FPU, p_stall::MATH},
    {SFPU_DEST_DVALID_CTRL_wait_mask_ADDR32, p_cleardvalid::SFPU, p_stall::WAIT_SFPU},
    {PACK_DEST_DVALID_CTRL_wait_mask_ADDR32, p_cleardvalid::PACK, p_stall::PACK},
};

template <dest_dvalid_client CLIENT>
constexpr dest_dvalid_config dest_dvalid_config_of = dest_dvalid_configs[to_underlying(CLIENT)];

constexpr std::uint32_t DEST_DVALID_ALL_CLIENTS =
    dest_dvalid_configs[0].bit | dest_dvalid_configs[1].bit | dest_dvalid_configs[2].bit | dest_dvalid_configs[3].bit;

constexpr std::uint32_t DEST_DVALID_CTRL_MASK = UNPACK_TO_DEST_DVALID_CTRL_wait_mask_MASK | UNPACK_TO_DEST_DVALID_CTRL_wait_polarity_MASK |
                                                UNPACK_TO_DEST_DVALID_CTRL_toggle_mask_MASK | UNPACK_TO_DEST_DVALID_CTRL_disable_auto_bank_id_toggle_MASK;

static_assert(MATH_DEST_DVALID_CTRL_wait_polarity_SHAMT == UNPACK_TO_DEST_DVALID_CTRL_wait_polarity_SHAMT);
static_assert(MATH_DEST_DVALID_CTRL_toggle_mask_SHAMT == UNPACK_TO_DEST_DVALID_CTRL_toggle_mask_SHAMT);
static_assert(PACK_DEST_DVALID_CTRL_toggle_mask_MASK == UNPACK_TO_DEST_DVALID_CTRL_toggle_mask_MASK);

template <dest_dvalid_client CLIENT>
inline void dest_dvalid_wait_client_idle()
{
    wait_mop_idle();

    if constexpr (CLIENT == dest_dvalid_client::UNPACK)
    {
        wait_unpack_idle();
    }
    else if constexpr (CLIENT == dest_dvalid_client::FPU)
    {
        wait_fpu_idle();
    }
    else if constexpr (CLIENT == dest_dvalid_client::SFPU)
    {
        wait_sfpu_idle();
    }
    else
    {
        wait_pack_idle();
    }
}

inline void dest_dvalid_wait_chain_idle()
{
    bstatus_u busy;
    busy.val         = 0;
    busy.global_fpu  = 1;
    busy.global_sfpu = 1;
    busy.global_pack = 1;
    wait_bstatus_low(busy.val);
}

constexpr std::uint32_t dest_dvalid_lowest(std::uint32_t mask)
{
    return mask & (~mask + 1);
}

constexpr std::uint32_t dest_dvalid_next(std::uint32_t chain, std::uint32_t bit)
{
    const std::uint32_t above = chain & ~((bit << 1) - 1);
    return dest_dvalid_lowest(above != 0 ? above : chain);
}

/**
 * @brief Writes CLIENT's dest dvalid control word.
 *
 * The control words are global CFG registers and are written straight over MMIO, as every proven Quasar dvalid
 * kernel does. Drain this thread's Tensix instructions first so an in-flight pulse still uses the old word.
 */
template <dest_dvalid_client CLIENT>
inline void dest_dvalid_write_ctrl(std::uint32_t ctrl)
{
    tensix_sync();
    volatile std::uint32_t* const cfg_regs              = reinterpret_cast<volatile std::uint32_t*>(TENSIX_CFG_BASE);
    cfg_regs[dest_dvalid_config_of<CLIENT>.ctrl_addr32] = ctrl;
}

/**
 * @brief Resets CLIENT to dest bank 0 with its handshake switched off, so the reset itself cannot stall on a stale wait mask.
 */
template <dest_dvalid_client CLIENT>
inline void dest_dvalid_reset_client_bank()
{
    constexpr dest_dvalid_config CFG = dest_dvalid_config_of<CLIENT>;

    dest_dvalid_write_ctrl<CLIENT>(0);
    TTI_CLEARDVALID(0, 0, 0, CFG.bit, 0, 0);
    if constexpr (CLIENT == dest_dvalid_client::PACK)
    {
        ckernel::pack::clear_dest_bank_id = 0;
    }
}

/**
 * @brief Enables the dest dvalid handshake for CLIENT within the chain of clients that touch dest in this operation.
 *
 * The lowest client of CHAIN produces the first dest section and waits for every chain client to release dest;
 * every other client waits for the section handed over by its predecessor. Each pulse hands the section to the
 * next higher client of CHAIN, the highest hands back to the lowest. All clients start the operation on bank 0.
 *
 * @tparam CLIENT: dest client owned by the calling thread
 * @tparam CHAIN: bitmask (p_cleardvalid bits) of all clients in the operation's dest chain, including CLIENT
 *
 * @note Pair with @ref _llk_dest_dvalid_signal_ per section and @ref _llk_dest_dvalid_disable_ on clients
 *       that do not touch dest during the operation.
 */
template <dest_dvalid_client CLIENT, std::uint32_t CHAIN>
inline void _llk_dest_dvalid_enable_()
{
    constexpr dest_dvalid_config CFG = dest_dvalid_config_of<CLIENT>;
    static_assert((CHAIN & CFG.bit) != 0, "CLIENT must be part of CHAIN");
    constexpr bool IS_FIRST = dest_dvalid_lowest(CHAIN) == CFG.bit;

    dest_dvalid_wait_client_idle<CLIENT>();
    dest_dvalid_reset_client_bank<CLIENT>();

    if constexpr (IS_FIRST)
    {
        dest_dvalid_wait_chain_idle();
    }

    constexpr std::uint32_t WAIT_MASK     = IS_FIRST ? CHAIN : CFG.bit;
    constexpr std::uint32_t WAIT_POLARITY = IS_FIRST ? 0u : CFG.bit;
    constexpr std::uint32_t TOGGLE_MASK   = CFG.bit | dest_dvalid_next(CHAIN, CFG.bit);

    constexpr std::uint32_t ctrl = (WAIT_MASK << UNPACK_TO_DEST_DVALID_CTRL_wait_mask_SHAMT) |
                                   (WAIT_POLARITY << UNPACK_TO_DEST_DVALID_CTRL_wait_polarity_SHAMT) |
                                   (TOGGLE_MASK << UNPACK_TO_DEST_DVALID_CTRL_toggle_mask_SHAMT);

    dest_dvalid_write_ctrl<CLIENT>(ctrl);
}

/**
 * @brief Hands the current dest section from CLIENT to the next client of its chain.
 *
 * @tparam CLIENT: dest client owned by the calling thread
 * @tparam DST: Destination synchronization mode, values = <SyncFull/SyncHalf>
 * @tparam EN_32BIT_DEST: True if dest is in 32-bit mode
 *
 * @note Call once per dest section after @ref _llk_dest_dvalid_enable_ for the operation.
 */
template <dest_dvalid_client CLIENT, DstSync DST, bool EN_32BIT_DEST = false>
inline void _llk_dest_dvalid_signal_()
{
    constexpr dest_dvalid_config CFG = dest_dvalid_config_of<CLIENT>;

    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::NOTHING, p_stall::NOTHING, CFG.drain_res);

    if constexpr (CLIENT == dest_dvalid_client::PACK)
    {
        constexpr std::uint32_t CLR_MODE = (DST == DstSync::SyncHalf) ? p_zeroacc::CLR_HALF : p_zeroacc::CLR_ALL;
        if constexpr (DST == DstSync::SyncFull)
        {
            TTI_ZEROACC(CLR_MODE, EN_32BIT_DEST, 0, ADDR_MOD_0, 0);
        }
        else
        {
            TT_ZEROACC(CLR_MODE, EN_32BIT_DEST, 0, ADDR_MOD_0, ckernel::pack::clear_dest_bank_id);
        }
    }

    TTI_CLEARDVALID(0, 0, 0, 0, CFG.bit, 0);
    if constexpr (DST == DstSync::SyncFull)
    {
        TTI_CLEARDVALID(0, 0, 0, CFG.bit, CFG.bit, 0);
    }

    if constexpr (CLIENT == dest_dvalid_client::PACK && DST == DstSync::SyncHalf)
    {
        ckernel::pack::_update_clear_dest_bank_id_();
    }
}

/**
 * @brief Switches CLIENT's dest dvalid handshake off for an operation in which it does not touch dest.
 */
template <dest_dvalid_client CLIENT>
inline void _llk_dest_dvalid_disable_()
{
    constexpr dest_dvalid_config CFG = dest_dvalid_config_of<CLIENT>;

    TTI_STALLWAIT(p_stall::STALL_MATH | p_stall::STALL_CFG, p_stall::NOTHING, p_stall::NOTHING, CFG.drain_res);
    dest_dvalid_wait_client_idle<CLIENT>();
    dest_dvalid_reset_client_bank<CLIENT>();
}

} // namespace ckernel
