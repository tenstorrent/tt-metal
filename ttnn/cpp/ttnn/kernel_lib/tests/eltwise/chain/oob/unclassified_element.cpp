// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Negative classification probe: a plausible-looking element that forgot to inherit a kind tag
// (CB reader / CB writer / DEST-only). Without the exhaustive-classification static_assert it
// would compile and flow through every chain stage as a silently inert position; the driving
// pytest asserts the chain instead FAILS to compile with "every element must carry exactly one
// kind tag".

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"

namespace {

struct RogueElement {
    static void init() {}
    void exec(uint32_t /*i*/, uint32_t /*ht*/, uint32_t /*wt*/, uint32_t /*slot_offset*/) const {}
};

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    constexpr uint32_t total_tiles = get_compile_time_arg_val(0);

    compute_kernel_hw_startup(cb_in, cb_out);

    compute_kernel_lib::eltwise_chain(
        compute_kernel_lib::IterationShape::tiles(total_tiles),
        compute_kernel_lib::CopyTile<compute_kernel_lib::input(cb_in)>{},
        RogueElement{},
        compute_kernel_lib::PackTile<compute_kernel_lib::output(cb_out)>{});
}
