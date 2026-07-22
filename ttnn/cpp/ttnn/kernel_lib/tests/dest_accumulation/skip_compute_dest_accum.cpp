// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Skip-compute twin of dest_accumulation.cpp — proves the CKL_ELTWISE_CHAIN_SKIP_COMPUTE knob also
// covers the DEST-accumulation walk. Byte-identical to the Run fixture except that this kernel opts
// the macro in to 1 before the include, so the chain emits ONLY the CB lifecycle (input wait/pop,
// output reserve/push) plus the tile_regs dst-sync window and elides all init + reconfig + the
// accumulating add + pack. The CB counts are unchanged, so the reduction kernel does NOT hang — it
// just packs uninitialized sticky-D0 tiles.

// Skip profiling twin: default the build macro to 1 (overridable by a -D on the build).
#ifndef CKL_ELTWISE_CHAIN_SKIP_COMPUTE
#define CKL_ELTWISE_CHAIN_SKIP_COMPUTE 1
#endif

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_local = tt::CBIndex::c_0;
    constexpr uint32_t cb_remote = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    constexpr uint32_t n = get_compile_time_arg_val(0);
    constexpr uint32_t block_size = get_compile_time_arg_val(1);
    constexpr bool caller_managed = get_compile_time_arg_val(2) != 0;
    constexpr uint32_t num_outputs = get_compile_time_arg_val(3);
    static_assert(n > 0);
    static_assert(block_size > 0);
    static_assert(num_outputs > 0);

    compute_kernel_hw_startup(cb_local, cb_remote, cb_out);

    using namespace compute_kernel_lib;
    using Accumulate = BinaryFpu<
        input(cb_local, InputLifecycle::Bulk, OperandKind::Block),
        input(cb_remote, InputLifecycle::Bulk, OperandKind::Block),
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

    CircularBuffer output(cb_out);
    if constexpr (caller_managed) {
        output.reserve_back(num_outputs);
        eltwise_chain(EltwiseShape::grid(num_outputs, n, block_size), Accumulate{}, CallerManagedPack{});
        output.push_back(num_outputs);
    } else {
        eltwise_chain(EltwiseShape::grid(num_outputs, n, block_size), Accumulate{}, ManagedPack{});
    }
}
