// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fused compute kernel: multiply-accumulate using hardware MAC.
//
// For each output tile we iterate over reduction_dim_size experts and sum:
//   dst0 += act_tile[e] * score_col[e]   (hardware MAC with col-broadcast)
//
// Each score tile has expert scores in column 0 (broadcast to all columns).
// The reader pre-loads all reduction_dim_size score tiles before signalling compute_scores.
// Score tiles are kept resident and accessed by index throughout the loop.
//
// The eltwise-chain helper owns the input/output lifecycle and configures the
// broadcast multiply with acc_to_dest enabled. Each output row uses D0 as its
// sticky accumulator, while block_size controls the transient parallel lanes.

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

using namespace ckernel;

constexpr uint32_t num_output_tiles = get_compile_time_arg_val(0);
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
        BinaryFpuOp::Mul,
        BroadcastDim::Col,
        InputLifecycle::Chunked,
        InputLifecycle::Bulk,
        BinaryDataFormatReconfig::Input,
        Dst::D0,
        OperandKind::Block,
        OperandKind::Row,
        TileOffset::Unset,
        TileOffset::Unset,
        DestAccumulation::Enabled>;
    using Pack = PackTile<compute_output_cb_id, OutputLifecycle::DestAccumulation, PackTileReconfig::Output, Dst::D0>;

    eltwise_chain(EltwiseShape::grid(num_output_tiles, reduction_dim_size, input_granularity), Accumulate{}, Pack{});
}
