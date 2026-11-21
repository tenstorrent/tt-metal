// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_trisc_common.h"
#include "llk_assert.h"
#include "llk_sync.h"

namespace ckernel
{

enum class DstClient : std::uint8_t
{
    UNPACK = 0,
    MATH   = 1,
    PACK   = 2,
    SFPU   = 3,
};

namespace dest_borrow
{

template <DstClient From, DstClient To>
constexpr std::uint8_t semaphore_id()
{
    static_assert(From == DstClient::MATH || From == DstClient::PACK, "DST lender must be MATH or PACK");
    static_assert(To == DstClient::UNPACK || To == DstClient::SFPU, "DST borrower must be UNPACK or SFPU");
    constexpr std::uint8_t semaphores[2][2] = {
        {trisc::semaphore::MATH_UNPACK, trisc::semaphore::MATH_SFPU},
        {trisc::semaphore::PACK_UNPACK, trisc::semaphore::PACK_SFPU},
    };
    return semaphores[From == DstClient::PACK][To == DstClient::SFPU];
}

template <DstClient Client>
inline void drain()
{
    if constexpr (Client == DstClient::MATH)
    {
        TTI_STALLWAIT(p_stall::STALL_SYNC, 0, p_stall::MATH, p_stall::WAIT_SFPU);
    }
    else if constexpr (Client == DstClient::PACK)
    {
        TTI_STALLWAIT(p_stall::STALL_SYNC, 0, 0, p_stall::PACK0);
    }
    else if constexpr (Client == DstClient::UNPACK)
    {
        TTI_STALLWAIT(p_stall::STALL_SYNC, 0, 0, p_stall::UNPACK0);
    }
    else
    {
        TTI_STALLWAIT(p_stall::STALL_SYNC, 0, 0, p_stall::WAIT_SFPU);
    }
}

} // namespace dest_borrow
} // namespace ckernel

/**
 * @brief Let another TRISC borrow the current Dest section for exclusive use.
 *
 * The lender thread (From) and borrower thread (To) both call this function to let the borrower use
 * the lender's Dest bank. The lender must already own the bank and wait for its return before using
 * it again, committing it or releasing it.
 * Each thread can hold only one loan at a time.
 *
 * @tparam From Lender that owns the section, either DstClient::MATH or DstClient::PACK.
 * @tparam To Borrower, either DstClient::UNPACK or DstClient::SFPU. SFPU runs on TRISC3.
 */
template <ckernel::DstClient From, ckernel::DstClient To>
inline void _llk_tile_regs_borrow_()
{
    using namespace ckernel;
    constexpr auto sem = dest_borrow::semaphore_id<From, To>();
    if constexpr (TRISC_ID == static_cast<std::uint8_t>(From))
    {
        const auto base = trisc::_get_dest_buffer_base_();
        LLK_ASSERT(base == 0 || base == 256 || base == 512, "Invalid lender DST section base");
        mailbox_write(static_cast<std::uint8_t>(To), base);
        dest_borrow::drain<From>();
        _llk_sync_post_(sem);
    }
    else if constexpr (TRISC_ID == static_cast<std::uint8_t>(To))
    {
        constexpr auto lender    = static_cast<std::uint8_t>(From);
        const auto base          = mailbox_read(lender);
        constexpr auto stall_res = p_stall::STALL_SYNC | (To == DstClient::UNPACK ? p_stall::STALL_UNPACK : p_stall::STALL_SFPU);
        _llk_sync_wait_<stall_res, p_stall::STALL_ON_ZERO>(sem);
        LLK_ASSERT(base == 0 || base == 256 || base == 512, "Invalid borrowed DST section base");
        trisc::dest_register_offset = base;
        trisc::_set_dest_section_base_<TRISC_ID>(trisc::_get_dest_buffer_base_());
    }
}

/**
 * @brief Wait for the borrower's work to finish and return exclusive access to the lender.
 *
 * Call on both threads with the same From and To as the matching borrow call.
 *
 * @tparam From Lender that owns the section, either DstClient::MATH or DstClient::PACK.
 * @tparam To Borrower, either DstClient::UNPACK or DstClient::SFPU. SFPU runs on TRISC3.
 */
template <ckernel::DstClient From, ckernel::DstClient To>
inline void _llk_tile_regs_return_()
{
    using namespace ckernel;
    constexpr auto sem = dest_borrow::semaphore_id<From, To>();
    if constexpr (TRISC_ID == static_cast<std::uint8_t>(To))
    {
        dest_borrow::drain<To>();
        _llk_sync_get_(sem);
    }
    else if constexpr (TRISC_ID == static_cast<std::uint8_t>(From))
    {
        constexpr auto stall_res = p_stall::STALL_SYNC | (From == DstClient::PACK ? p_stall::STALL_PACK : (p_stall::STALL_MATH | p_stall::STALL_SFPU));
        _llk_sync_wait_<stall_res, p_stall::STALL_ON_MAX>(sem);
    }
}
