// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// DestReuseBinary validation kernel.
// Pattern: DEST = mul(x, x) using CopyTile -> Mul (via DestReuseBinary DEST_TO_SRCB).
// First, CopyTile loads cb_in into Dst::D0. Then DestReuseBinary(MUL, DEST_TO_SRCB) reads cb_in into srca,
// reads Dst::D0 into srcb, multiplies, writes back to Dst::D0. Output = x*x = square.
//
// However for simpler validation: golden = x * x. We test DEST_TO_SRCA (CB → srcb, DEST → srca).

#include <cstdint>

#include "api/compute/common.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

#ifndef DEST_REUSE_OP
#define DEST_REUSE_OP Mul
#endif

#ifndef DEST_REUSE_TYPE
#define DEST_REUSE_TYPE DEST_TO_SRCA
#endif

void kernel_main() {
    using namespace compute_kernel_lib;

    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    constexpr BinaryFpuOp op = BinaryFpuOp::DEST_REUSE_OP;
    constexpr DestReuseType reuse = DestReuseType::DEST_REUSE_TYPE;

    const uint32_t per_core_block_count = get_compile_time_arg_val(0);
    const uint32_t per_core_block_dim = get_compile_time_arg_val(1);
    const uint32_t num_tiles = per_core_block_count * per_core_block_dim;

    // D5/D8: caller-side BIG init at the top of MAIN().
    eltwise_chain_with_init(
        num_tiles,
        CopyTile<cb_in, Dst::D0, CopyTilePolicy::WaitNoPop, CbIndexMode::FirstTile, CopyTileReconfig::None>{},
        DestReuseBinary<
            cb_in,
            op,
            reuse,
            Dst::D0,
            Dst::D0,
            DestReuseReconfig::None,
            CopyTilePolicy::NoWaitPop,
            CbIndexMode::FirstTile>{},
        PackTile<
            cb_out,
            Dst::D0,
            PackTilePolicy::PerTileReserveAndPush,
            PackTileIndexMode::FirstTile,
            PackTileReconfig::None>{});
}
