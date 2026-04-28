// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_helpers.hpp"

// y = (a + b) * c via add(..., eltwise_chain(CopyTile<c, D1>, SfpuMul<D0,D1,D0>)).
// Drives clashes_with_fpu reinit: the PostOp's CopyTile clobbers unpack MOP,
// next iteration's binary_exec re-issues binary_short_init.

void kernel_main() {
    constexpr uint32_t num_tiles = get_compile_time_arg_val(0);

    constexpr uint32_t cb_a = 0;
    constexpr uint32_t cb_b = 1;
    constexpr uint32_t cb_c = 2;
    constexpr uint32_t cb_out = 16;

    using namespace compute_kernel_lib::eltwise;

    binary_op_init_common(cb_a, cb_b, cb_out);

    add(cb_a,
        cb_b,
        cb_out,
        BinaryInputBlockShape::of(1, num_tiles),
        eltwise_chain(CopyTile<cb_c, Dst::D1>{}, SfpuMul<Dst::D0, Dst::D1, Dst::D0>{}));
}
