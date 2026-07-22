// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Runtime conditional coverage:
//   mode 0: runtime_if first arm            -> -x
//   mode 1: two-element else_if arm         -> -(x * x)
//   mode 2: otherwise arm + bare if         -> abs(x)
//   mode 3: grouped bare if                 -> abs(-x)
//   mode 4: bare if                         -> x * x
//   mode 5: bare if / else-if first arm     -> x * x
//   mode 6: bare if / else-if second arm    -> -x

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_optional.hpp"

void kernel_main() {
    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t n = get_compile_time_arg_val(0);
    const uint32_t mode = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_in, cb_out);

    using namespace compute_kernel_lib;
    eltwise_chain(
        EltwiseShape::tiles(n),
        CopyTile<input(cb_in)>{},
        runtime_if(mode == 0, Negative<Dst::D0>{})
            .else_if(mode == 1, Square<Dst::D0>{}, Negative<Dst::D0>{})
            .otherwise(CopyDest<Dst::D0, Dst::D0>{}),
        runtime_if(mode == 2, Abs<Dst::D0>{}),
        runtime_if(mode == 3, Negative<Dst::D0>{}, Abs<Dst::D0>{}),
        runtime_if(mode == 4, Square<Dst::D0>{}),
        runtime_if(mode == 5, Square<Dst::D0>{}).else_if(mode == 6, Negative<Dst::D0>{}),
        PackTile<output(cb_out)>{});
}
