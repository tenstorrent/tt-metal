// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Fan-out validation kernel — single CB read, two SFPU ops fed to two output CBs in one window.
// Uses WaitNoPop on the first CopyTile (waits, doesn't pop) + NoWaitPop on the second
// (no wait — already waited — but pops). That's the canonical fan-out lifecycle.

#include <cstdint>

#include "api/compute/common.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_trig.hpp"

void kernel_main() {
    using namespace compute_kernel_lib;

    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_outA = tt::CBIndex::c_16;
    constexpr uint32_t cb_outB = tt::CBIndex::c_17;

    const uint32_t per_core_block_count = get_compile_time_arg_val(0);
    const uint32_t per_core_block_dim = get_compile_time_arg_val(1);
    const uint32_t num_tiles = per_core_block_count * per_core_block_dim;

    using Chain = EltwiseChain<
        CopyTile<cb_in, Dst::D0, CopyTilePolicy::WaitNoPop>,
        CopyTile<cb_in, Dst::D1, CopyTilePolicy::NoWaitPop>,
        Exp<Approx::Exact, Approx::Fast, Dst::D0>,
        Sin<Dst::D1>,
        PackTile<cb_outA, Dst::D0, PackTilePolicy::PerTileReserveAndPush>,
        PackTile<cb_outB, Dst::D1, PackTilePolicy::PerTileReserveAndPush>>;
    eltwise_pipeline_init<Chain>();

    eltwise_chain(
        num_tiles,
        CopyTile<cb_in, Dst::D0, CopyTilePolicy::WaitNoPop>{},
        CopyTile<cb_in, Dst::D1, CopyTilePolicy::NoWaitPop>{},
        Exp<Approx::Exact, Approx::Fast, Dst::D0>{},
        Sin<Dst::D1>{},
        PackTile<cb_outA, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{},
        PackTile<cb_outB, Dst::D1, PackTilePolicy::PerTileReserveAndPush>{});
}
