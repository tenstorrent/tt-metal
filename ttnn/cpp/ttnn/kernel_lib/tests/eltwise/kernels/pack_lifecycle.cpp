// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// PackTilePolicy validation kernel — selects policy via define PACK_POLICY.
// Uses CopyTile + Exp + PackTile chain.

#include <cstdint>

#include "api/compute/common.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"

#ifndef PACK_POLICY
#define PACK_POLICY PerTileReserveAndPush
#endif

void kernel_main() {
    using namespace compute_kernel_lib;

    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    constexpr PackTilePolicy policy = PackTilePolicy::PACK_POLICY;

    const uint32_t per_core_block_count = get_compile_time_arg_val(0);
    const uint32_t per_core_block_dim = get_compile_time_arg_val(1);
    const uint32_t num_tiles = per_core_block_count * per_core_block_dim;

    // D5/D8: caller-side BIG init at the top of MAIN().
    eltwise_chain_with_init(
        num_tiles,
        CopyTile<cb_in, Dst::D0, CopyTilePolicy::WaitAndPop>{},
        Exp<>{},
        PackTile<cb_out, Dst::D0, policy>{});
}
