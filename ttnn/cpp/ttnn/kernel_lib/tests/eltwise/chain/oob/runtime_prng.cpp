// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// A PRNG seeder cannot be guarded by runtime_if: the chain seeds PRNG state once
// before its walk, while a conditional branch is selected within that walk.

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/optional.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/generators/rand.hpp"

void kernel_main() {
    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t total_tiles = get_compile_time_arg_val(0);

    compute_kernel_hw_startup(cb_in, cb_out);

    using namespace compute_kernel_lib;
    eltwise_chain(
        IterationShape::tiles(total_tiles),
        CopyTile<input(cb_in), Dst::D0>{},
        runtime_if(true, RandTile<Dst::D0>{0, 1, 17, 0}),
        PackTile<output(cb_out)>{});
}
