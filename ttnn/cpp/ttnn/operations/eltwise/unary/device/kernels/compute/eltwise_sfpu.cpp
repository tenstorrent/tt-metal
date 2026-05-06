// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
// SFPU op headers stay included so SFPU_OP_CHAIN_0 (host-injected) compiles.
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/trigonometry.h"
#include "api/compute/mul_int_sfpu.h"
#include "api/compute/eltwise_unary/rpow.h"
#include "api/compute/eltwise_unary/rdiv.h"
#include "api/compute/eltwise_unary/fill.h"

namespace {

// Wraps the host-injected SFPU_OP_CHAIN_0 macro inside a chain element.
struct SfpuOpChain : compute_kernel_lib::DestOnlyTag {
    static ALWI void init() {}
    ALWI void exec(uint32_t /*i*/) const {
#ifdef SFPU_OP_CHAIN_0
        SFPU_OP_CHAIN_0
#endif
    }
};

}  // namespace

void kernel_main() {
    using namespace compute_kernel_lib;

    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_output = tt::CBIndex::c_2;

    using Chain = EltwiseChain<
        CopyTile<cb_input, Dst::D0, CopyTilePolicy::WaitAndPop>,
        SfpuOpChain,
        PackTile<cb_output, Dst::D0, PackTilePolicy::PerTileReserveAndPush>
    >;
    eltwise_pipeline_init<Chain>();
    eltwise_chain(
        num_tiles,
        CopyTile<cb_input, Dst::D0, CopyTilePolicy::WaitAndPop>{},
        SfpuOpChain{},
        PackTile<cb_output, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{}
    );
}
