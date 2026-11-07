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
constexpr int CSA_OUTPUT_ROW_MASK      = (1 << CSA_OUTPUT_BANK_SHIFT) - 1;
// Output word: [bank:3 at bit 14 | row-in-bank:14]. The consumer reads the bank
// field at bit 14 and the row from the low 14 bits, so the window holds
// 8 banks x 16384 rows = 2^17 compressed positions.
constexpr std::uint32_t CSA_OUTPUT_CAPACITY = 1u << (CSA_OUTPUT_BANK_SHIFT + CSA_OUTPUT_ROW_SHIFT);

// Contract: every window-relative position (compressed position + ROW_OFFSET)
// must be below CSA_OUTPUT_CAPACITY. The producer configuration bounds the
// window; this kernel does not check the inputs. A position beyond the
// capacity aliases onto row-in-bank modulo 16384 but keeps the correct bank.
template <int ITERATIONS, std::uint32_t ROW_OFFSET>
inline void _csa_index_remap_()
{
    using namespace sfpi;
    static_assert(ROW_OFFSET < CSA_OUTPUT_CAPACITY, "csa_index_remap: ROW_OFFSET must leave room in the 2^17-position window");
    for (int d = 0; d < ITERATIONS; ++d)
    {
        // Input: [device:3 | bank:3 | bank-local row:14].
        // Convert the distributed 256-row chunks into a global compressed
        // position, add the window offset, then encode [bank | row-in-bank].
        vInt packed = dst_reg[0];
        vInt cpos   = (packed & CSA_LOCAL_ROW_LOW_MASK) | ((packed >> CSA_BANK_DEVICE_SHIFT) & CSA_BANK_DEVICE_MASK) |
                    ((packed & CSA_LOCAL_ROW_HIGH_MASK) << CSA_LOCAL_ROW_HIGH_SHIFT);
        vInt row   = cpos + static_cast<int>(ROW_OFFSET);
        // Mask the row field so a position beyond the capacity cannot spill
        // into the bank field.
        dst_reg[0] = ((row & CSA_OUTPUT_BANK_MASK) << CSA_OUTPUT_BANK_SHIFT) | ((row >> CSA_OUTPUT_ROW_SHIFT) & CSA_OUTPUT_ROW_MASK);
        dst_reg++;
    }
}

} // namespace ckernel::sfpu
