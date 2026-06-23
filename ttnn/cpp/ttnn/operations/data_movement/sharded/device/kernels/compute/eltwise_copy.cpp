// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    uint32_t per_core_tile_cnt = get_arg_val<uint32_t>(0);

    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_out = tt::CBIndex::c_16;

    // Standard hw-config big init only: the chain's CopyTile emits copy_tile_init
    // (the datacopy MOP) unconditionally, so unary_op_init_common's datacopy init was
    // redundant. compute_kernel_hw_startup does the unpack/math/pack hw_configure +
    // pack init; the chain supplies the copy MOP.
    compute_kernel_hw_startup(cb_in, cb_out);

    // Per-tile copy cb_in -> cb_out. No per-iter reconfig (boot-time format only),
    // so CopyTileReconfig::None + PackTileReconfig::None.
    compute_kernel_lib::copy<
        cb_in,
        cb_out,
        compute_kernel_lib::InputLifecycle::Streaming,
        compute_kernel_lib::OutputLifecycle::Streaming,
        compute_kernel_lib::CopyTileReconfig::None,
        compute_kernel_lib::PackTileReconfig::None>(per_core_tile_cnt);
}
