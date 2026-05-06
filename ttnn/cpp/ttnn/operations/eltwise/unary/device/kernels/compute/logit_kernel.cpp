// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"     // Log
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_scalar.hpp"   // RsubUnary, Clamp
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu.hpp"  // DivBinary

void kernel_main() {
    using namespace compute_kernel_lib;

    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    const uint32_t packed_scalar1 = get_arg_val<uint32_t>(1);
    const uint32_t packed_scalar2 = get_arg_val<uint32_t>(2);

    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_output = tt::CBIndex::c_2;
    constexpr auto cb_tmp0 = tt::CBIndex::c_1;

    // Stage 1: copy input → DEST[0] (with optional clamp), pack to cb_tmp0.
#ifdef CLAMP
    using Stage1Chain = EltwiseChain<
        CopyTile<cb_input, Dst::D0, CopyTilePolicy::WaitAndPop>,
        Clamp<Dst::D0>,
        PackTile<cb_tmp0, Dst::D0, PackTilePolicy::PerTileReserveAndPush>>;
    eltwise_pipeline_init<Stage1Chain>();
    eltwise_chain(
        num_tiles,
        CopyTile<cb_input, Dst::D0, CopyTilePolicy::WaitAndPop>{},
        Clamp<Dst::D0>{packed_scalar1, packed_scalar2},
        PackTile<cb_tmp0, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});
#else
    using Stage1Chain = EltwiseChain<
        CopyTile<cb_input, Dst::D0, CopyTilePolicy::WaitAndPop>,
        PackTile<cb_tmp0, Dst::D0, PackTilePolicy::PerTileReserveAndPush>>;
    eltwise_pipeline_init<Stage1Chain>();
    eltwise_chain(
        num_tiles,
        CopyTile<cb_input, Dst::D0, CopyTilePolicy::WaitAndPop>{},
        PackTile<cb_tmp0, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});
#endif

    // Stage 2: load cb_tmp0 → DEST[0] and DEST[1], compute log(x / (1 - x)), pack to cb_output.
    //   D0 = 1 - x   (RsubUnary with 1.0f)
    //   D0 = D1 / D0
    //   D0 = log(D0)
    using Stage2Chain = EltwiseChain<
        CopyTile<cb_tmp0, Dst::D1, CopyTilePolicy::WaitNoPop>,
        CopyTile<cb_tmp0, Dst::D0, CopyTilePolicy::NoWaitPop>,
        RsubUnary<Dst::D0>,
        DivBinary<Dst::D1, Dst::D0, Dst::D0>,
        Log<Dst::D0>,
        PackTile<cb_output, Dst::D0, PackTilePolicy::PerTileReserveAndPush>>;
    eltwise_pipeline_init<Stage2Chain>();
    eltwise_chain(
        num_tiles,
        CopyTile<cb_tmp0, Dst::D1, CopyTilePolicy::WaitNoPop>{},
        CopyTile<cb_tmp0, Dst::D0, CopyTilePolicy::NoWaitPop>{},
        RsubUnary<Dst::D0>{0x3F800000u},  // 1.0f - x
        DivBinary<Dst::D1, Dst::D0, Dst::D0>{},
        Log<Dst::D0>{},
        PackTile<cb_output, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});
}
