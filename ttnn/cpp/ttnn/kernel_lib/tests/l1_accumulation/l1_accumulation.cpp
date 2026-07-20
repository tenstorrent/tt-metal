// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_acc = tt::CBIndex::c_15;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t n = get_compile_time_arg_val(0);
    constexpr bool caller_managed = get_compile_time_arg_val(1) != 0;
    static_assert(n > 1);

    compute_kernel_hw_startup(cb_in, cb_acc);

    using namespace compute_kernel_lib;
    CircularBuffer accumulator(cb_acc);

    using ManagedPack = PackTile<
        cb_acc,
        output(
            OutputLifecycle::L1Accumulation,
            DataFormatReconfig::Disabled,
            PackRelu::Disabled,
            L1Accumulation::SeedFirst)>;
    using CallerManagedPack = PackTile<
        cb_acc,
        output(
            OutputLifecycle::CallerManaged,
            DataFormatReconfig::Disabled,
            PackRelu::Disabled,
            L1Accumulation::SeedFirst)>;

    if constexpr (caller_managed) {
        accumulator.reserve_back(1);
        eltwise_chain(
            EltwiseShape::tiles(n),
            CopyTile<cb_in, Dst::D0, input(InputLifecycle::Streaming, DataFormatReconfig::Disabled)>{},
            CallerManagedPack{});
        accumulator.push_back(1);
    } else {
        eltwise_chain(
            EltwiseShape::tiles(n),
            CopyTile<cb_in, Dst::D0, input(InputLifecycle::Streaming, DataFormatReconfig::Disabled)>{},
            ManagedPack{});
    }

    eltwise_chain(EltwiseShape::single(), CopyTile<cb_acc>{}, PackTile<cb_out>{});
}
