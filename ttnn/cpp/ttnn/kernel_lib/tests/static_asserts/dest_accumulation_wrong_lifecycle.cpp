// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Negative compile test: an accumulating BinaryFpu requires a DEST-accumulation output lifecycle.

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

void kernel_main() {
    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t n = get_compile_time_arg_val(0);

    compute_kernel_hw_startup(cb_in, cb_out);
    using namespace compute_kernel_lib;
    eltwise_chain(
        EltwiseShape::tiles(n),
        BinaryFpu<
            cb_in,
            cb_in,
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
        PackTile<cb_out, OutputLifecycle::Streaming>{});
}
