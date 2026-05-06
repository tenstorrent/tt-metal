// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// CopyTile WaitUpfrontPopAtEnd + BlockIter validation.
// Caller supplies UPFRONT_N tiles, helper waits once, reads tile-i each iteration, pops at end.
// num_tiles MUST equal UPFRONT_N for this kernel.

#include <cstdint>

#include "api/compute/common.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"

#ifndef UPFRONT_N
#define UPFRONT_N 8
#endif

void kernel_main() {
    using namespace compute_kernel_lib;

    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t upfront_n = UPFRONT_N;

    const uint32_t per_core_block_count = get_compile_time_arg_val(0);
    const uint32_t per_core_block_dim = get_compile_time_arg_val(1);
    const uint32_t num_tiles = per_core_block_count * per_core_block_dim;

    using CopyElt = CopyTile<cb_in, Dst::D0, CopyTilePolicy::WaitUpfrontPopAtEnd, CbIndexMode::BlockIter>;
    using PackElt = PackTile<cb_out, Dst::D0, PackTilePolicy::UpfrontReservePushAtEnd, PackTileIndexMode::BlockIter>;
    using Chain = EltwiseChain<CopyElt, Exp<>, PackElt>;

    eltwise_pipeline_init<Chain>();

    constexpr EltwiseChainOptions opts = []() {
        EltwiseChainOptions o{};
        o.upfront_block_size = upfront_n;
        return o;
    }();
    eltwise_chain<opts>(num_tiles, CopyElt{}, Exp<>{}, PackElt{});
}
