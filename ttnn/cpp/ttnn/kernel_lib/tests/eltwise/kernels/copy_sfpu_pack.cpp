// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Templated single-SFPU validation kernel — selects op via define ELTWISE_OP_NAME.
// Used by test plan §3 (SFPU unary by family) and §18 (convenience wrappers).

#include <cstdint>

#include "api/compute/common.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_helpers.hpp"

#ifndef ELTWISE_OP_NAME
#define ELTWISE_OP_NAME Exp
#endif

#ifndef ELTWISE_OP_BRACED
#define ELTWISE_OP_BRACED \
    ELTWISE_OP_NAME<> {}
#endif

void kernel_main() {
    using namespace compute_kernel_lib;

    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    const uint32_t per_core_block_count = get_compile_time_arg_val(0);
    const uint32_t per_core_block_dim = get_compile_time_arg_val(1);
    const uint32_t num_tiles = per_core_block_count * per_core_block_dim;

    // D5/D8: caller-side BIG init at the top of MAIN().
    eltwise_chain_with_init(
        num_tiles,
        CopyTile<cb_in, Dst::D0, CopyTilePolicy::WaitAndPop, CbIndexMode::FirstTile, CopyTileReconfig::None>{},
        ELTWISE_OP_BRACED,
        PackTile<
            cb_out,
            Dst::D0,
            PackTilePolicy::PerTileReserveAndPush,
            PackTileIndexMode::FirstTile,
            PackTileReconfig::None>{});
}
