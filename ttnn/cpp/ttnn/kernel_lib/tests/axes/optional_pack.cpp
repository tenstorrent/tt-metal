// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// OptionalChainElement gating a PackTile fan-out. CopyTile -> PackTile(cb_out0) ->
// OptionalChainElement<ON, PackTile(cb_out1)>. ON: DEST packed to both cb_out0 and cb_out1 (distinct
// CBs, no collision). OFF: the optional element remains as an inert chain position and behaves as
// if absent — so OFF compiles and writes only cb_out0 without participating in pack collision checks.

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_optional.hpp"

void kernel_main() {
    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_out0 = tt::CBIndex::c_16;
    constexpr uint32_t cb_out1 = tt::CBIndex::c_17;
    constexpr uint32_t n = get_compile_time_arg_val(0);
    constexpr uint32_t cond = get_compile_time_arg_val(1);
    constexpr bool ON = (cond != 0);

    compute_kernel_hw_startup(cb_in, cb_out0);

    using namespace compute_kernel_lib;
    eltwise_chain(
        EltwiseShape::tiles(n),
        CopyTile<input(cb_in)>{},
        PackTile<output(cb_out0)>{},
        OptionalChainElement<ON, PackTile<output(cb_out1)>>{});
}
