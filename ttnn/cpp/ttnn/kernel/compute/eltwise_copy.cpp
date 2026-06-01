// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

void kernel_main() {
    constexpr uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);
    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_out = tt::CBIndex::c_16;

    unary_op_init_common(cb_in, cb_out);

    // Per-tile copy cb_in -> cb_out via a single chain over all tiles.
    // Original used unary_op_init_common + copy_tile_init at boot (no per-iter
    // reconfig) — CopyTileReconfig::None + PackTileReconfig::None preserve that.
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
            compute_kernel_lib::PackTileReconfig::None>{});
}
