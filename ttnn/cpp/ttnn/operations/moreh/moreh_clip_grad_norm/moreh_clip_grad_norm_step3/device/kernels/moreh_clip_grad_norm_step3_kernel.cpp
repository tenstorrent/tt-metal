// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    int i{0};
    const auto num_tiles = get_arg_val<uint32_t>(i++);

    constexpr uint32_t cb_x = 0;                  // input
    constexpr uint32_t cb_clip_coef_clamped = 1;  // clip_coef_clamped (held scalar)
    constexpr uint32_t cb_y = 16;                 // output

    compute_kernel_hw_startup(cb_x, cb_clip_coef_clamped, cb_y);

    // cb_y = cb_x × cb_clip_coef_clamped[scalar bcast].
    // Lifecycles:
    //   cb_x — InputLifecycle::Streaming + Scalar OperandKind (per-iter wait+pop of one tile).
    //   cb_clip_coef_clamped — InputLifecycle::Bulk + Scalar. `window_1d<Scalar>` collapses
    //     the InputLifecycle::Bulk window to 1 tile (the old "InputLifecycle::Bulk + Scalar over-waits"
    //     gotcha was fixed in 14a5a61e462), so the chain emits exactly
    //     `cb_wait_front(1)` at the head and `cb_pop_front(1)` at the tail —
    //     no external wait/pop needed.
    // Reconfig: original used `mul_tiles_bcast_scalar_init_short` (no
    // `_with_dt`) and plain `pack_tile` — no per-iter format reconfig.
    // Chain matches with `BinaryDataFormatReconfig::None` + `PackTileReconfig::None`.
    compute_kernel_lib::mul<
        cb_x,
        cb_clip_coef_clamped,
        cb_y,
        compute_kernel_lib::BroadcastDim::Scalar,
        compute_kernel_lib::BinaryDataFormatReconfig::None,
        compute_kernel_lib::OperandKind::Scalar,
        compute_kernel_lib::InputLifecycle::Streaming,
        compute_kernel_lib::InputLifecycle::Bulk,
        compute_kernel_lib::OperandKind::Scalar,
        compute_kernel_lib::OutputLifecycle::Streaming,
        compute_kernel_lib::PackTileReconfig::None>(num_tiles);
}
