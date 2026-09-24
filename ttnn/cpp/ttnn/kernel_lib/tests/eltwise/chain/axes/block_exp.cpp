// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Block-capable exp(x): a Bulk + Block reader stages the window upfront so the chain processes
// block_size tiles per inner iter across DEST lanes. block_size is a compile-time arg.
//
// Blocking is a loop-structure optimization — it must NOT change the per-tile result, so exp(x) is
// identical across block_size; larger blocks should cut loop/DEST-sync overhead (the perf signal).

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"

void kernel_main() {
    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    constexpr uint32_t n = get_compile_time_arg_val(0);
    constexpr uint32_t blk = get_compile_time_arg_val(1);

    compute_kernel_hw_startup(cb_in, cb_out);

    using namespace compute_kernel_lib;
    using SafeBlockedInput =
        CopyTile<input(cb_in, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Block), Dst::D0>;
    using UnsafeBlockedInput = CopyTile<input(cb_in, WaitPolicy::Upfront, PopPolicy::PerTile), Dst::D0>;
    using SafeBlockedOutput = PackTile<output(cb_out, ReservePolicy::Upfront, PushPolicy::AtEnd)>;
    using UnsafeBlockedOutput = PackTile<output(cb_out, ReservePolicy::Upfront, PushPolicy::PerTile)>;
    static_assert(chain_supports_block_v<EltwiseChain<SafeBlockedInput, Exp<>, SafeBlockedOutput>>);
    static_assert(!chain_supports_block_v<EltwiseChain<UnsafeBlockedInput, Exp<>, SafeBlockedOutput>>);
    static_assert(!chain_supports_block_v<EltwiseChain<SafeBlockedInput, Exp<>, UnsafeBlockedOutput>>);
    eltwise_chain(
        IterationShape::tiles(n).block_size(blk),
        CopyTile<input(cb_in, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Block), Dst::D0>{},
        Exp<>{},
        PackTile<output(cb_out, ReservePolicy::Upfront, PushPolicy::AtEnd)>{});
}
