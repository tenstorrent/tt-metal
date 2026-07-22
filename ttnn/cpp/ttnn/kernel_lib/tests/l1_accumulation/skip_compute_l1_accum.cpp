// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Skip-compute twin of l1_accumulation.cpp — proves the CKL_ELTWISE_CHAIN_SKIP_COMPUTE knob also
// covers the L1-accumulation (seed-first) walk. Byte-identical to the Run fixture except the macro,
// so both chains emit only the CB lifecycle + tile_regs window and elide all init + reconfig +
// compute (copy, the seed-first L1-accum mode flips, and the pack). No hang; garbage output.

// Skip profiling twin: default the build macro to 1 (overridable by a -D on the build).
#ifndef CKL_ELTWISE_CHAIN_SKIP_COMPUTE
#define CKL_ELTWISE_CHAIN_SKIP_COMPUTE 1
#endif

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

    using L1ManagedPack = PackTile<output(
        cb_acc,
        OutputLifecycle::L1Accumulation,
        DataFormatReconfig::Disabled,
        PackRelu::Disabled,
        L1Accumulation::SeedFirst)>;
    using L1CallerManagedPack = PackTile<output(
        cb_acc,
        OutputLifecycle::CallerManaged,
        DataFormatReconfig::Disabled,
        PackRelu::Disabled,
        L1Accumulation::SeedFirst)>;

    if constexpr (caller_managed) {
        accumulator.reserve_back(1);
        eltwise_chain(
            EltwiseShape::tiles(n),
            CopyTile<input(cb_in, InputLifecycle::Streaming, DataFormatReconfig::Disabled), Dst::D0>{},
            L1CallerManagedPack{});
        accumulator.push_back(1);
    } else {
        eltwise_chain(
            EltwiseShape::tiles(n),
            CopyTile<input(cb_in, InputLifecycle::Streaming, DataFormatReconfig::Disabled), Dst::D0>{},
            L1ManagedPack{});
    }

    eltwise_chain(EltwiseShape::single(), CopyTile<input(cb_acc), Dst::D0>{}, PackTile<output(cb_out)>{});
}
