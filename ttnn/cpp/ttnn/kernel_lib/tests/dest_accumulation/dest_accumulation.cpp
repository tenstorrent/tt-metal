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
    static_assert(n > 0);
    static_assert(block_size > 0);
    static_assert(num_outputs > 0);

    compute_kernel_hw_startup(cb_a, cb_b, cb_out);

    using namespace compute_kernel_lib;
    using Accumulate = BinaryFpu<
        input(cb_a, InputLifecycle::Bulk, OperandKind::Block),
        input(cb_b, InputLifecycle::Bulk, OperandKind::Block),
        BinaryFpuOp::Add,
        BroadcastDim::None,
        Dst::D0,
        DestAccumulation::Enabled>;
    using ManagedPack = PackTile<output(
        cb_out,
        OutputLifecycle::DestAccumulation,
        DataFormatReconfig::Enabled,
        PackRelu::Disabled,
        L1Accumulation::Disabled,
        DestAccumulation::Enabled)>;
    using CallerManagedPack = PackTile<output(
        cb_out,
        OutputLifecycle::CallerManaged,
        DataFormatReconfig::Enabled,
        PackRelu::Disabled,
        L1Accumulation::Disabled,
        DestAccumulation::Enabled)>;

    using ManagedChain = EltwiseChain<Accumulate, ManagedPack>;
    using CallerManagedChain = EltwiseChain<Accumulate, CallerManagedPack>;
    static_assert(chain_transient_lane_width_v<ManagedChain> == 0);
    static_assert(chain_transient_lane_width_v<CallerManagedChain> == 0);
    static_assert(chain_max_block_v<ManagedChain> == ~uint32_t{0});
    static_assert(chain_max_block_v<CallerManagedChain> == ~uint32_t{0});

    CircularBuffer output_buffer(cb_out);
    if constexpr (caller_managed) {
        output_buffer.reserve_back(num_outputs);
        eltwise_chain(EltwiseShape::grid(num_outputs, n, block_size), Accumulate{}, CallerManagedPack{});
        output_buffer.push_back(num_outputs);
    } else {
        eltwise_chain(EltwiseShape::grid(num_outputs, n, block_size), Accumulate{}, ManagedPack{});
    }
}
