// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    uint32_t per_core_tile_cnt = get_arg_val<uint32_t>(0);

    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_out = tt::CBIndex::c_16;

    unary_op_init_common(cb_in, cb_out);

    // Per-tile copy cb_in -> cb_out. Original used unary_op_init_common +
    // copy_tile_init at boot, then plain copy_tile / pack_tile per iter —
    // no per-iter reconfig, so CopyTileReconfig::None + PackTileReconfig::None.
    compute_kernel_lib::eltwise_chain(
        per_core_tile_cnt,
        compute_kernel_lib::CopyTile<
            cb_in,
            compute_kernel_lib::Dst::D0,
            compute_kernel_lib::Streaming,
            compute_kernel_lib::OperandKind::Scalar,
            compute_kernel_lib::CopyTileReconfig::None>{},
        compute_kernel_lib::PackTile<
            cb_out,
            compute_kernel_lib::Dst::D0,
            compute_kernel_lib::OutStreaming,
            compute_kernel_lib::OperandKind::Scalar,
            compute_kernel_lib::PackTileReconfig::None>{});
}
