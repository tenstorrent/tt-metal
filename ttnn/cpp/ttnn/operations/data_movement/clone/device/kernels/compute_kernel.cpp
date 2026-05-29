// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t src_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t dst_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t num_tiles = get_compile_time_arg_val(2);

    unary_op_init_common(src_cb_id, dst_cb_id);

    // Per-tile copy from src_cb -> dst_cb. Chain does wait/pop on src,
    // reserve/push on dst. Original used unary_op_init_common + copy_tile +
    // pack_tile loop with NO _with_dt reconfigs (boot-time format only) —
    // CopyTileReconfig::None and PackTileReconfig::None match that.
    compute_kernel_lib::eltwise_chain(
        num_tiles,
        compute_kernel_lib::CopyTile<
            src_cb_id,
            compute_kernel_lib::Dst::D0,
            compute_kernel_lib::Streaming,
            compute_kernel_lib::OperandKind::Scalar,
            compute_kernel_lib::CopyTileReconfig::None>{},
        compute_kernel_lib::PackTile<
            dst_cb_id,
            compute_kernel_lib::Dst::D0,
            compute_kernel_lib::OutStreaming,
            compute_kernel_lib::OperandKind::Scalar,
            compute_kernel_lib::PackTileReconfig::None>{});
}
