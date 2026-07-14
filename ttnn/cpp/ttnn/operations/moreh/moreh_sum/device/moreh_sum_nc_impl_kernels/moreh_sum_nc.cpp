// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/kernel/compute/moreh_common.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

void kernel_main() {
    // compile-time args
    constexpr uint32_t num_output_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t num_input_tiles = get_compile_time_arg_val(1);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out0 = tt::CBIndex::c_16;

    binary_op_init_common(cb_in0, cb_in1, cb_out0);

    using namespace compute_kernel_lib;
#if defined FP32_DEST_ACC_EN
    constexpr auto input_reconfig = BinaryDataFormatReconfig::Input;
    constexpr auto pack_reconfig = PackTileReconfig::Output;
#else
    constexpr auto input_reconfig = BinaryDataFormatReconfig::None;
    constexpr auto pack_reconfig = PackTileReconfig::None;
#endif
    using Accumulate = BinaryFpu<
        cb_in0,
        cb_in1,
        BinaryFpuOp::Add,
        BroadcastDim::None,
        InputLifecycle::Streaming,
        InputLifecycle::Bulk,
        input_reconfig,
        Dst::D0,
        OperandKind::Scalar,
        OperandKind::Scalar,
        TileOffset::Unset,
        TileOffset::Unset,
        DestAccumulation::Enabled>;
    using Pack = PackTile<cb_out0, OutputLifecycle::DestAccumulation, pack_reconfig, Dst::D0>;

    eltwise_chain(EltwiseShape::grid(num_output_tiles, num_input_tiles), Accumulate{}, Pack{});
}
