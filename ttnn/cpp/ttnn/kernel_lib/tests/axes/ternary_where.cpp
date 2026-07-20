// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Ternary element coverage: y = where(cond, a, b). Three CopyTile loads (cond->D0, a->D1, b->D2),
// then Where selects per element (cond != 0 ? a : b) -> D0, packed out. Exercises a 3-input
// DEST-only SFPU element.

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_special.hpp"

void kernel_main() {
    constexpr uint32_t cb_cond = tt::CBIndex::c_0;
    constexpr uint32_t cb_a = tt::CBIndex::c_1;
    constexpr uint32_t cb_b = tt::CBIndex::c_2;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t n = get_compile_time_arg_val(0);

    compute_kernel_hw_startup(cb_cond, cb_a, cb_out);

    using namespace compute_kernel_lib;
    eltwise_chain(
        EltwiseShape::tiles(n),
        CopyTile<cb_cond>{},
        CopyTile<cb_a, Dst::D1>{},
        CopyTile<cb_b, Dst::D2>{},
        Where<DataFormat::Float16_b, Dst::D0, Dst::D1, Dst::D2, Dst::D0>{},
        PackTile<cb_out>{});
}
