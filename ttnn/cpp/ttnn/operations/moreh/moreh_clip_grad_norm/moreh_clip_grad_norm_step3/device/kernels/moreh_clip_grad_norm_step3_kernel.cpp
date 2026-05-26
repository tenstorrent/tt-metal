// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    int i{0};
    const auto num_tiles = get_arg_val<uint32_t>(i++);

    constexpr uint32_t cb_x = 0;                  // input
    constexpr uint32_t cb_clip_coef_clamped = 1;  // clip_coef_clamped (held scalar)
    constexpr uint32_t cb_y = 16;                 // output
    CircularBuffer cb_clip_coef_clamped_obj(cb_clip_coef_clamped);

    constexpr uint32_t onetile = 1;

    compute_kernel_hw_startup(cb_x, cb_clip_coef_clamped, cb_y);

    // cb_clip_coef_clamped lifecycle: CallerManaged + Scalar — single tile
    // waited once outside the chain loop, never popped per iter (held scalar),
    // popped once at end of kernel. External wait pairs with CallerManaged per
    // the documented rule.
    cb_clip_coef_clamped_obj.wait_front(onetile);

    // cb_y = cb_x × cb_clip_coef_clamped[scalar bcast]
    // Reconfig: original used `mul_tiles_bcast_scalar_init_short` (no `_with_dt`)
    // and plain `pack_tile` — no per-iter reconfigs of either srca/srcb or pack.
    // Chain emits BinaryDataFormatReconfig::None + PackTileReconfig::None to
    // match (initial state set by compute_kernel_hw_startup).
    compute_kernel_lib::eltwise_chain(
        num_tiles,
        compute_kernel_lib::BinaryFpu<
            cb_x,
            cb_clip_coef_clamped,
            compute_kernel_lib::BinaryFpuOp::Mul,
            compute_kernel_lib::BroadcastDim::Scalar,
            compute_kernel_lib::BinaryDataFormatReconfig::None,
            compute_kernel_lib::Streaming,      // cb_x: per-iter wait+pop
            compute_kernel_lib::CallerManaged,  // cb_clip_coef_clamped: external wait
            compute_kernel_lib::OperandKind::Scalar,
            compute_kernel_lib::Dst::D0,
            compute_kernel_lib::OperandKind::Scalar>{},
        compute_kernel_lib::PackTile<
            cb_y,
            compute_kernel_lib::Dst::D0,
            compute_kernel_lib::OutStreaming,
            compute_kernel_lib::OperandKind::Scalar,
            compute_kernel_lib::PackTileReconfig::None>{});

    cb_clip_coef_clamped_obj.pop_front(onetile);
}
