// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_b = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t n = get_compile_time_arg_val(0);
    constexpr uint32_t block_size = get_compile_time_arg_val(1);
    constexpr bool caller_managed = get_compile_time_arg_val(2) != 0;
    constexpr uint32_t num_outputs = get_compile_time_arg_val(3);
    constexpr bool whole_shape = get_compile_time_arg_val(4) != 0;
    static_assert(n > 0);
    static_assert(block_size > 0);
    static_assert(num_outputs > 0);

    compute_kernel_hw_startup(cb_a, cb_b, cb_out);

    using namespace compute_kernel_lib;
    using PerRowAccumulate = BinaryFpu<
        input(cb_a, WaitPolicy::Upfront, PopPolicy::AtEnd, OperandKind::Block),
        input(cb_b, WaitPolicy::Upfront, PopPolicy::AtEnd, OperandKind::Block),
        BinaryFpuOp::Add,
        BroadcastDim::None,
        Dst::D0,
        DestAccumulation::PerRow>;
    using PerRowManagedPack = PackTile<output(
        cb_out,
        ReservePolicy::PerOuter,
        PushPolicy::PerOuter,
        DataFormatReconfig::Enabled,
        PackRelu::Disabled,
        L1Accumulation::Disabled,
        DestAccumulation::PerRow)>;
    using PerRowCallerManagedPack = PackTile<output(
        cb_out,
        ReservePolicy::None,
        PushPolicy::None,
        DataFormatReconfig::Enabled,
        PackRelu::Disabled,
        L1Accumulation::Disabled,
        DestAccumulation::PerRow)>;

    using WholeShapeAccumulate = BinaryFpu<
        input(cb_a, WaitPolicy::Upfront, PopPolicy::AtEnd, OperandKind::Block),
        input(cb_b, WaitPolicy::Upfront, PopPolicy::AtEnd, OperandKind::Block),
        BinaryFpuOp::Add,
        BroadcastDim::None,
        Dst::D0,
        DestAccumulation::WholeShape>;
    using WholeShapeManagedPack = PackTile<output(
        cb_out,
        ReservePolicy::PerOuter,
        PushPolicy::PerOuter,
        DataFormatReconfig::Enabled,
        PackRelu::Disabled,
        L1Accumulation::Disabled,
        DestAccumulation::WholeShape)>;
    using WholeShapeCallerManagedPack = PackTile<output(
        cb_out,
        ReservePolicy::None,
        PushPolicy::None,
        DataFormatReconfig::Enabled,
        PackRelu::Disabled,
        L1Accumulation::Disabled,
        DestAccumulation::WholeShape)>;

    using ManagedChain = EltwiseChain<PerRowAccumulate, PerRowManagedPack>;
    using CallerManagedChain = EltwiseChain<PerRowAccumulate, PerRowCallerManagedPack>;
    static_assert(chain_transient_lane_width_v<ManagedChain> == 0);
    static_assert(chain_transient_lane_width_v<CallerManagedChain> == 0);
    static_assert(chain_max_block_v<ManagedChain> == ~uint32_t{0});
    static_assert(chain_max_block_v<CallerManagedChain> == ~uint32_t{0});

    CircularBuffer output_buffer(cb_out);
    if constexpr (whole_shape) {
        if constexpr (caller_managed) {
            output_buffer.reserve_back(1);
            eltwise_chain(
                EltwiseShape::grid(num_outputs, n, block_size), WholeShapeAccumulate{}, WholeShapeCallerManagedPack{});
            output_buffer.push_back(1);
        } else {
            eltwise_chain(
                EltwiseShape::grid(num_outputs, n, block_size), WholeShapeAccumulate{}, WholeShapeManagedPack{});
        }
    } else {
        if constexpr (caller_managed) {
            output_buffer.reserve_back(num_outputs);
            eltwise_chain(
                EltwiseShape::grid(num_outputs, n, block_size), PerRowAccumulate{}, PerRowCallerManagedPack{});
            output_buffer.push_back(num_outputs);
        } else {
            eltwise_chain(EltwiseShape::grid(num_outputs, n, block_size), PerRowAccumulate{}, PerRowManagedPack{});
        }
    }
}
