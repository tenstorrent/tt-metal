// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t src_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t dst_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t num_tiles = get_compile_time_arg_val(2);

    // Standard hw-config big init only: the chain's CopyTile emits copy_tile_init
    // (the datacopy MOP) unconditionally, so unary_op_init_common's datacopy init was
    // redundant. compute_kernel_hw_startup does the unpack/math/pack hw_configure +
    // pack init; the chain supplies the copy MOP.
    compute_kernel_hw_startup(src_cb_id, dst_cb_id);

    // Per-tile copy from src_cb -> dst_cb. Chain does wait/pop on src, reserve/push
    // on dst. No per-iter reconfig (boot-time format only) —
    // CopyTileReconfig::None and PackTileReconfig::None match that.
    compute_kernel_lib::copy<
        src_cb_id,
        dst_cb_id,
        compute_kernel_lib::InputLifecycle::Streaming,
        compute_kernel_lib::OutputLifecycle::Streaming,
        compute_kernel_lib::CopyTileReconfig::None,
        compute_kernel_lib::PackTileReconfig::None>(num_tiles);
}
