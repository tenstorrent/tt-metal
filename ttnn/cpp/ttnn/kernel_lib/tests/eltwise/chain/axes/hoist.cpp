// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Equivalent exp(x) chains under three setup placements:
//   0: one multi-tile call, so the chain hoists uniform setup;
//   1: one single-tile call per tile, so each call emits setup;
//   2: raw LLK setup followed by InitReconfigOwner::Caller.
//
// CT args: [n, mode].

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"

void kernel_main() {
    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t n = get_compile_time_arg_val(0);
    constexpr uint32_t mode = get_compile_time_arg_val(1);
    static_assert(mode < 3);

    DataflowBuffer cb_in_obj(cb_in);
    DataflowBuffer cb_out_obj(cb_out);

    using namespace compute_kernel_lib;
    compute_kernel_hw_startup(cb_in, cb_out);

    if constexpr (mode == 0) {
        eltwise_chain(
            IterationShape::tiles(n), CopyTile<input(cb_in)>{cb_in_obj}, Exp<>{}, PackTile<output(cb_out)>{cb_out_obj});
    } else if constexpr (mode == 1) {
        for (uint32_t i = 0; i < n; ++i) {
            eltwise_chain(
                IterationShape::one_tile(),
                CopyTile<input(cb_in)>{cb_in_obj},
                Exp<>{},
                PackTile<output(cb_out)>{cb_out_obj});
        }
    } else {
        copy_init(cb_in);
        exp_tile_init();
        eltwise_chain<InitReconfigOwner::Caller>(
            IterationShape::tiles(n),
            CopyTile<input(cb_in, WaitPolicy::PerTile, PopPolicy::PerTile, DataFormatReconfig::Disabled)>{cb_in_obj},
            Exp<>{},
            PackTile<output(cb_out, ReservePolicy::PerTile, PushPolicy::PerTile, DataFormatReconfig::Disabled)>{
                cb_out_obj});
    }
}
