// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/compute/common.h"

namespace kda {

FORCE_INLINE constexpr uint32_t largest_divisor_at_most(uint32_t value, uint32_t limit) {
    for (uint32_t divisor = limit; divisor > 1; --divisor) {
        if (value % divisor == 0) {
            return divisor;
        }
    }
    return 1;
}

template <uint32_t Rows, uint32_t Columns>
struct MatmulSubblock {
    static constexpr uint32_t dst_tiles =
        ckernel::get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, ckernel::DstTileShape::Tile32x32>();
    static constexpr uint32_t columns = largest_divisor_at_most(Columns, dst_tiles);
    static constexpr uint32_t rows = largest_divisor_at_most(Rows, dst_tiles / columns);
    static_assert(rows * columns <= dst_tiles);
};

}  // namespace kda
