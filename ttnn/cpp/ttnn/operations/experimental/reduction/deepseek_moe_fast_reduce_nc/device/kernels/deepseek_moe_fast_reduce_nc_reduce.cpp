// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

constexpr uint32_t num_output_tiles_to_process = get_compile_time_arg_val(0);
constexpr uint32_t reduction_dim_size = get_compile_time_arg_val(1);
constexpr uint32_t input_granularity = get_compile_time_arg_val(2);
constexpr uint32_t compute_input_cb_id_0 = get_compile_time_arg_val(3);
constexpr uint32_t compute_input_cb_id_1 = get_compile_time_arg_val(4);
constexpr uint32_t compute_output_cb_id = get_compile_time_arg_val(5);

void kernel_main() {
    binary_op_init_common(compute_input_cb_id_0, compute_input_cb_id_1, compute_output_cb_id);

    using namespace compute_kernel_lib;
    using Accumulate = BinaryFpu<
        compute_input_cb_id_0,
        compute_input_cb_id_1,
        BinaryFpuOp::Add,
        BroadcastDim::None,
        InputLifecycle::Chunked,
        InputLifecycle::Bulk,
        BinaryDataFormatReconfig::Input,
        Dst::D0,
        OperandKind::Block,
        OperandKind::Scalar,
        TileOffset::Unset,
        TileOffset::Unset,
        DestAccumulation::Enabled>;
    using Pack = PackTile<compute_output_cb_id, OutputLifecycle::DestAccumulation, PackTileReconfig::Output, Dst::D0>;

    eltwise_chain(
        EltwiseShape::grid(num_output_tiles_to_process, reduction_dim_size, input_granularity), Accumulate{}, Pack{});
}
