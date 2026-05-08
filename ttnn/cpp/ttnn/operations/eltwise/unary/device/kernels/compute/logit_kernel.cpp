// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"     // Log
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_scalar.hpp"   // RsubUnary, Clamp
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu.hpp"  // DivBinary
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_optional.hpp"     // OptionalChainElement

// U5: collapse the #ifdef CLAMP branch into a constexpr bool that gates an
// OptionalChainElement<DO_CLAMP, Clamp<...>> inside the stage-1 chain.
#ifdef CLAMP
constexpr bool DO_CLAMP = true;
#else
constexpr bool DO_CLAMP = false;
#endif

void kernel_main() {
    using namespace compute_kernel_lib;

    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    const uint32_t packed_scalar1 = get_arg_val<uint32_t>(1);
    const uint32_t packed_scalar2 = get_arg_val<uint32_t>(2);

    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_output = tt::CBIndex::c_2;
    constexpr auto cb_tmp0 = tt::CBIndex::c_1;

    // D5 row 4: multi-stage kernel. Boot the engine for stage 1's CB triple
    // at the top of MAIN(); re-boot for stage 2's CB triple immediately
    // before stage 2's chain call.
    compute_kernel_hw_startup(cb_input, cb_input, cb_tmp0);

    // Stage 1: copy input → DEST[0] (with optional clamp), pack to cb_tmp0.
    // The Clamp element is wrapped in OptionalChainElement<DO_CLAMP, ...> —
    // when DO_CLAMP is false the wrapper inherits the inner's tag (DestOnly via
    // the SfpuOp CRTP base) but every hook is a no-op (zero runtime cost).
    eltwise_chain(
        num_tiles,
        CopyTile<cb_input, Dst::D0, CopyTilePolicy::WaitAndPop>{},
        OptionalChainElement<DO_CLAMP, Clamp<Dst::D0>>{packed_scalar1, packed_scalar2},
        PackTile<cb_tmp0, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});

    // D5 row 4: re-boot for stage 2's CB triple (cb_tmp0 → cb_output).
    compute_kernel_hw_startup(cb_tmp0, cb_tmp0, cb_output);

    // Stage 2: load cb_tmp0 → DEST[0] and DEST[1], compute log(x / (1 - x)), pack to cb_output.
    //   D0 = 1 - x   (RsubUnary with 1.0f)
    //   D0 = D1 / D0
    //   D0 = log(D0)
    eltwise_chain(
        num_tiles,
        CopyTile<cb_tmp0, Dst::D1, CopyTilePolicy::WaitNoPop>{},
        CopyTile<cb_tmp0, Dst::D0, CopyTilePolicy::NoWaitPop>{},
        RsubUnary<Dst::D0>{0x3F800000u},  // 1.0f - x
        DivBinary<Dst::D1, Dst::D0, Dst::D0>{},
        Log<Dst::D0>{},
        PackTile<cb_output, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});
}
