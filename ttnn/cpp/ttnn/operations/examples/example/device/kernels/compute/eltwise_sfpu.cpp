// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"

namespace {
struct SfpuOpChain : compute_kernel_lib::DestOnlyTag {
    static ALWI void init() {}
    ALWI void exec(uint32_t /*i*/, uint32_t /*slot_offset*/) const {
#ifdef SFPU_OP_CHAIN_0
        SFPU_OP_CHAIN_0
#endif
    }
};
}  // namespace

void kernel_main() {
    using namespace compute_kernel_lib;

    constexpr uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    constexpr uint32_t per_core_block_dim = get_compile_time_arg_val(1);
    constexpr uint32_t num_tiles = per_core_block_cnt * per_core_block_dim;

    // D5/D8: caller-side BIG init at the top of MAIN().
    eltwise_chain_with_init(
        num_tiles,
        CopyTile<tt::CBIndex::c_0, Dst::D0, CopyTilePolicy::WaitAndPop>{},
        SfpuOpChain{},
        PackTile<tt::CBIndex::c_2, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});
}
