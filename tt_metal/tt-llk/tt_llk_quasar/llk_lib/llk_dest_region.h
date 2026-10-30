// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_trisc_common.h"
#include "llk_defs.h"

namespace ckernel
{

template <std::uint32_t COUNT, bool EN_32BIT_DEST>
class DestRegion
{
    static_assert(COUNT == 1 || COUNT == 2, "ZEROACC clears the whole dest or one half; a finer layout needs its own clear sequence");

    static constexpr std::uint32_t CLEAR_MODE = COUNT == 2 ? p_zeroacc::CLR_HALF : p_zeroacc::CLR_ALL;

public:
    static constexpr std::uint32_t REGIONS = COUNT;
    static constexpr std::uint32_t ROWS    = (trisc::DEST_REGISTER_FULL_SIZE / COUNT) >> EN_32BIT_DEST;

    static constexpr std::uint32_t base(const std::uint32_t index)
    {
        return index * ROWS;
    }

    static constexpr std::uint32_t next(const std::uint32_t index)
    {
        return COUNT == 2 ? index ^ 1u : (index + 1) % COUNT;
    }

    static constexpr std::uint32_t clear_word(const std::uint32_t index)
    {
        return TT_OP_ZEROACC(CLEAR_MODE, EN_32BIT_DEST, 0, ADDR_MOD_7, index);
    }
};

template <DstSync DST>
constexpr std::uint32_t dest_region_count()
{
    static_assert(DST == DstSync::SyncHalf || DST == DstSync::SyncFull, "Unknown destination synchronization mode");
    return DST == DstSync::SyncHalf ? 2 : 1;
}

template <DstSync DST, bool EN_32BIT_DEST>
using DestRegionOf = DestRegion<dest_region_count<DST>(), EN_32BIT_DEST>;

} // namespace ckernel
