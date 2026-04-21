// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Test: y = (a + b) * c via binary_op(ADD) with a PostOp chain that itself
// loads the third operand (cb_c) into DEST[D1] and multiplies via SfpuMul.
//
// Exercises:
// - Load as a PostOp chain element (not just as a standalone binary input).
// - FPU clash path: the Load calls copy_tile_to_dst_init_short(cb_c) which
//   clobbers the unpack MOP into A-only mode. binary_op must re-run
//   binary_init before each subsequent tile's binary_exec, restoring AB mode.
//   This is the regression surface for the `clashes_with_fpu_v` trait.
// - SfpuMul on DEST slots D0 + D1 -> D0, after binary_exec filled D0 and
//   the PostOp Load filled D1.

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/sfpu_helpers.hpp"

void kernel_main() {
    constexpr uint32_t num_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t cb_a = static_cast<uint32_t>(tt::CBIndex::c_0);
    constexpr uint32_t cb_b = static_cast<uint32_t>(tt::CBIndex::c_1);
    constexpr uint32_t cb_c = static_cast<uint32_t>(tt::CBIndex::c_2);
    constexpr uint32_t cb_output = static_cast<uint32_t>(tt::CBIndex::c_16);

    binary_op_init_common(cb_a, cb_b, cb_output);

    using namespace compute_kernel_lib;
    add(cb_a,
        cb_b,
        cb_output,
        BinaryInputBlockShape::of(1, num_tiles),
        sfpu_chain(
            // WaitAndPop: cb_c streams per tile alongside cb_a / cb_b.
            Load<cb_c, Dst::D1, LoadPolicy::WaitAndPop>{},
            SfpuMul<Dst::D0, Dst::D1, Dst::D0>{}));
}
