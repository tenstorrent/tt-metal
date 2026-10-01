// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "sfpi.h"

namespace ckernel::sfpu
{

constexpr int CSA_LOCAL_ROW_LOW_MASK   = 0x1F;
constexpr int CSA_BANK_DEVICE_SHIFT    = 9;
constexpr int CSA_BANK_DEVICE_MASK     = 0x7E0;
constexpr int CSA_LOCAL_ROW_HIGH_MASK  = 0x3FE0;
constexpr int CSA_LOCAL_ROW_HIGH_SHIFT = 6;
constexpr int CSA_OUTPUT_BANK_MASK     = 0x7;
constexpr int CSA_OUTPUT_BANK_SHIFT    = 14;
constexpr int CSA_OUTPUT_ROW_SHIFT     = 3;

template <int ITERATIONS, std::uint32_t ROW_OFFSET>
inline void _csa_index_remap_()
{
    using namespace sfpi;
    // The masks are loaded once per tile rather than once per row; the unrolled loop is recorded once and replayed.
    const vInt local_row_low_mask  = CSA_LOCAL_ROW_LOW_MASK;
    const vInt bank_device_mask    = CSA_BANK_DEVICE_MASK;
    const vInt local_row_high_mask = CSA_LOCAL_ROW_HIGH_MASK;
    const vInt output_bank_mask    = CSA_OUTPUT_BANK_MASK;
#pragma GCC unroll 32
    for (int d = 0; d < ITERATIONS; ++d)
    {
        // Input: [device:3 | bank:3 | bank-local row:14].
        // Convert the distributed 256-row chunks into a global compressed
        // position, add the window offset, then encode [bank | row-in-bank].
        vInt packed = dst_reg[0];
        vInt cpos   = (packed & local_row_low_mask) | ((packed >> CSA_BANK_DEVICE_SHIFT) & bank_device_mask) |
                    ((packed & local_row_high_mask) << CSA_LOCAL_ROW_HIGH_SHIFT);
        vInt row   = cpos + static_cast<int>(ROW_OFFSET);
        dst_reg[0] = ((row & output_bank_mask) << CSA_OUTPUT_BANK_SHIFT) | (row >> CSA_OUTPUT_ROW_SHIFT);
        dst_reg++;
    }
}

} // namespace ckernel::sfpu
