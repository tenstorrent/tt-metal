// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_ops.h"
#include "llk_assert.h"

namespace ckernel
{
namespace sfpu
{

// Rebase the Dst write pointer for subsequent SFPSTOREs. Shared by the
// experimental sort kernels (topk_xl, deepseek_top32_rm) to switch between
// column groups / tile offsets of their multi-tile DST regions.
inline void set_dst_write_addr_offset(std::uint32_t addr)
{
    LLK_ASSERT(addr < DEST_REGISTER_HALF_SIZE, "Address overflow in set_dst_write_addr_offset");
    std::uint32_t dst_index = addr + get_dest_buffer_base();
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, dst_index);
}

} // namespace sfpu
} // namespace ckernel
