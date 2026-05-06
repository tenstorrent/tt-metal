// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_trig.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu.hpp"

namespace cklib = compute_kernel_lib;

template <bool USE_APPROX>
inline void run_mish(uint32_t num_tiles) {
    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_output = tt::CBIndex::c_2;
    constexpr cklib::Approx approx = USE_APPROX ? cklib::Approx::Fast : cklib::Approx::Exact;

#ifdef INP_FLOAT32
    using Chain = cklib::EltwiseChain<
        cklib::CopyTile<cb_input, cklib::Dst::D0, cklib::CopyTilePolicy::WaitNoPop>,
        cklib::Exp<approx, approx, cklib::Dst::D0>,
        cklib::Log1p<approx, cklib::Dst::D0>,
        cklib::Tanh<cklib::Dst::D0>,
        cklib::CopyTile<cb_input, cklib::Dst::D1, cklib::CopyTilePolicy::NoWaitPop>,
        cklib::MulBinary<cklib::Dst::D0, cklib::Dst::D1, cklib::Dst::D0>,
        cklib::PackTile<cb_output, cklib::Dst::D0, cklib::PackTilePolicy::PerTileReserveAndPush>
    >;
    cklib::eltwise_pipeline_init<Chain>();
    cklib::eltwise_chain(
        num_tiles,
        cklib::CopyTile<cb_input, cklib::Dst::D0, cklib::CopyTilePolicy::WaitNoPop>{},
        cklib::Exp<approx, approx, cklib::Dst::D0>{},
        cklib::Log1p<approx, cklib::Dst::D0>{},
        cklib::Tanh<cklib::Dst::D0>{},
        cklib::CopyTile<cb_input, cklib::Dst::D1, cklib::CopyTilePolicy::NoWaitPop>{},
        cklib::MulBinary<cklib::Dst::D0, cklib::Dst::D1, cklib::Dst::D0>{},
        cklib::PackTile<cb_output, cklib::Dst::D0, cklib::PackTilePolicy::PerTileReserveAndPush>{}
    );
#endif
#ifdef INP_FLOAT
    using Chain = cklib::EltwiseChain<
        cklib::CopyTile<cb_input, cklib::Dst::D0, cklib::CopyTilePolicy::WaitNoPop>,
        cklib::Exp<approx, approx, cklib::Dst::D0>,
        cklib::Log1p<approx, cklib::Dst::D0>,
        cklib::Tanh<cklib::Dst::D0>,
        cklib::DestReuseBinary<cb_input, cklib::BinaryFpuOp::Mul, cklib::DestReuseType::DEST_TO_SRCA,
                               cklib::Dst::D0, cklib::Dst::D0, cklib::DestReuseReconfig::None,
                               cklib::CopyTilePolicy::NoWaitPop, cklib::CbIndexMode::FirstTile>,
        cklib::PackTile<cb_output, cklib::Dst::D0, cklib::PackTilePolicy::PerTileReserveAndPush>
    >;
    cklib::eltwise_pipeline_init<Chain>();
    cklib::eltwise_chain(
        num_tiles,
        cklib::CopyTile<cb_input, cklib::Dst::D0, cklib::CopyTilePolicy::WaitNoPop>{},
        cklib::Exp<approx, approx, cklib::Dst::D0>{},
        cklib::Log1p<approx, cklib::Dst::D0>{},
        cklib::Tanh<cklib::Dst::D0>{},
        cklib::DestReuseBinary<cb_input, cklib::BinaryFpuOp::Mul, cklib::DestReuseType::DEST_TO_SRCA,
                               cklib::Dst::D0, cklib::Dst::D0, cklib::DestReuseReconfig::None,
                               cklib::CopyTilePolicy::NoWaitPop, cklib::CbIndexMode::FirstTile>{},
        cklib::PackTile<cb_output, cklib::Dst::D0, cklib::PackTilePolicy::PerTileReserveAndPush>{}
    );
#endif
}

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    const uint32_t approx_arg = get_arg_val<uint32_t>(1);
    if (approx_arg != 0u) {
        run_mish<true>(num_tiles);
    } else {
        run_mish<false>(num_tiles);
    }
}
