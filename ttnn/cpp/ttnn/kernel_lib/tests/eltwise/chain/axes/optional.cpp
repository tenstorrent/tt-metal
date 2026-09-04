// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Optional element scenarios. CT args: [n, scenario, enabled].
// scenario 0 gates a unary, 1 gates a pack, and 2 runs runtime conditionals.

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/misc.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/optional.hpp"

void kernel_main() {
    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t n = get_compile_time_arg_val(0);
    constexpr uint32_t scenario = get_compile_time_arg_val(1);
    constexpr bool enabled = get_compile_time_arg_val(2) != 0;
    static_assert(scenario < 3);

    using namespace compute_kernel_lib;
    compute_kernel_hw_startup(cb_in, cb_out);
    if constexpr (scenario == 0) {
        eltwise_chain(
            IterationShape::tiles(n),
            CopyTile<input(cb_in)>{},
            Optional<enabled, Negative<Dst::D0>>{},
            PackTile<output(cb_out)>{});
    } else if constexpr (scenario == 1) {
        constexpr uint32_t cb_out_2 = tt::CBIndex::c_17;
        eltwise_chain(
            IterationShape::tiles(n),
            CopyTile<input(cb_in)>{},
            PackTile<output(cb_out)>{},
            Optional<enabled, PackTile<output(cb_out_2)>>{});
    } else {
        const uint32_t mode = get_arg_val<uint32_t>(0);
        eltwise_chain(
            IterationShape::tiles(n),
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
}
