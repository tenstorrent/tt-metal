// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Negative wrapper probes. The wrappers own a fixed number of CopyTile loads,
// so their SFPU operation must consume the matching number of DEST operands.

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"

namespace compute_kernel_lib {

struct TestUnary : UnaryOp<TestUnary, Dst::D0> {
    static ALWI void init() {}
    static ALWI void exec_impl(uint32_t) {}
};

struct TestBinary : BinaryOp<TestBinary, Dst::D0, Dst::D1, Dst::D0> {
    static ALWI void init() {}
    static ALWI void exec_impl(uint32_t) {}
};

}  // namespace compute_kernel_lib

void kernel_main() {
    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_b = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t total_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t mode = get_compile_time_arg_val(1);
    static_assert(mode < 2);

    compute_kernel_hw_startup(cb_a, cb_b, cb_out);

    using namespace compute_kernel_lib;
    if constexpr (mode == 0) {
        unary<TestBinary, input(cb_a), output(cb_out)>(IterationShape::tiles(total_tiles));
    } else {
        binary_sfpu<TestUnary, input(cb_a), input(cb_b), output(cb_out)>(IterationShape::tiles(total_tiles));
    }
}
