// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_helpers.hpp"

// y = x * exp(x) — fan-out via two CopyTiles on the same CB.
// CopyTile<WaitNoPop> waits but does not pop; CopyTile<NoWaitPop> skips wait
// and pops. Net: one wait + one pop per tile, two physical copy_tile calls.

void kernel_main() {
    constexpr uint32_t num_tiles = get_compile_time_arg_val(0);

    constexpr uint32_t cb_in = 0;
    constexpr uint32_t cb_out = 16;

    using namespace compute_kernel_lib::eltwise;

    init_sfpu(cb_in, cb_out);

    auto chain = eltwise_chain(
        CopyTile<cb_in, Dst::D0, CopyTilePolicy::WaitNoPop>{},
        CopyTile<cb_in, Dst::D1, CopyTilePolicy::NoWaitPop>{},
        Exp<Approx::Exact, Approx::Fast, Dst::D0>{},
        SfpuMul<Dst::D0, Dst::D1, Dst::D0>{});
    eltwise_pipeline(chain, cb_out, num_tiles);
}
