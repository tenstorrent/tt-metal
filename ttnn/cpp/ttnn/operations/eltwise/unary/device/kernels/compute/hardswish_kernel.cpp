// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_activations.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu.hpp"

void kernel_main() {
    using namespace compute_kernel_lib;

    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_output = tt::CBIndex::c_2;

#ifdef INP_FLOAT32
    // hardswish(x) = x * hardsigmoid(x) — both operands from DEST.
    using Chain = EltwiseChain<
        CopyTile<cb_input, Dst::D0, CopyTilePolicy::WaitNoPop>,
        CopyTile<cb_input, Dst::D1, CopyTilePolicy::NoWaitPop>,
        Hardsigmoid<Dst::D0>,
        MulBinary<Dst::D0, Dst::D1, Dst::D0>,
        PackTile<cb_output, Dst::D0, PackTilePolicy::PerTileReserveAndPush>
    >;
    eltwise_pipeline_init<Chain>();
    eltwise_chain(
        num_tiles,
        CopyTile<cb_input, Dst::D0, CopyTilePolicy::WaitNoPop>{},
        CopyTile<cb_input, Dst::D1, CopyTilePolicy::NoWaitPop>{},
        Hardsigmoid<Dst::D0>{},
        MulBinary<Dst::D0, Dst::D1, Dst::D0>{},
        PackTile<cb_output, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{}
    );
#endif
#ifdef INP_FLOAT
    // hardswish(x) = hardsigmoid(x) * x — DEST_TO_SRCA reuse, CB→srcb, DEST→srca, multiply.
    using Chain = EltwiseChain<
        CopyTile<cb_input, Dst::D0, CopyTilePolicy::WaitNoPop>,
        Hardsigmoid<Dst::D0>,
        DestReuseBinary<cb_input, BinaryFpuOp::Mul, DestReuseType::DEST_TO_SRCA,
                        Dst::D0, Dst::D0, DestReuseReconfig::None,
                        CopyTilePolicy::NoWaitPop, CbIndexMode::FirstTile>,
        PackTile<cb_output, Dst::D0, PackTilePolicy::PerTileReserveAndPush>
    >;
    eltwise_pipeline_init<Chain>();
    eltwise_chain(
        num_tiles,
        CopyTile<cb_input, Dst::D0, CopyTilePolicy::WaitNoPop>{},
        Hardsigmoid<Dst::D0>{},
        DestReuseBinary<cb_input, BinaryFpuOp::Mul, DestReuseType::DEST_TO_SRCA,
                        Dst::D0, Dst::D0, DestReuseReconfig::None,
                        CopyTilePolicy::NoWaitPop, CbIndexMode::FirstTile>{},
        PackTile<cb_output, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{}
    );
#endif
}
