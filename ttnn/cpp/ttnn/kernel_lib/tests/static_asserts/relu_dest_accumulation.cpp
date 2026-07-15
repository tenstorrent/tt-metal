// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Negative compile test: packer ReLU on a DEST-accumulation chain. The packer-ReLU set/reset live
// only on the ordinary (non-DEST-accumulation) walk; a DEST-accumulation chain routes through a
// different loop that would silently drop the activation, so the combination is forbidden.
// MUST fail to compile with "packer ReLU combined with DEST accumulation".

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

void kernel_main() {
    constexpr uint32_t cb_local = tt::CBIndex::c_0;
    constexpr uint32_t cb_remote = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t n = get_compile_time_arg_val(0);

    compute_kernel_hw_startup(cb_local, cb_remote, cb_out);
    using namespace compute_kernel_lib;
    eltwise_chain(
        EltwiseShape::grid(1, n, 1),
        BinaryFpu<
            cb_local,
            cb_remote,
            BinaryFpuOp::Add,
            BroadcastDim::None,
            InputLifecycle::Bulk,
            InputLifecycle::Bulk,
            BinaryDataFormatReconfig::Input,
            Dst::D0,
            OperandKind::Block,
            OperandKind::Block,
            TileOffset::Unset,
            TileOffset::Unset,
            DestAccumulation::Enabled>{},
        PackTile<
            cb_out,
            OutputLifecycle::DestAccumulation,
            PackTileReconfig::Output,
            Dst::D0,
            TileOffset::Unset,
            PackTileL1Accumulation::Disabled,
            PackRelu::Zero>{});
}
