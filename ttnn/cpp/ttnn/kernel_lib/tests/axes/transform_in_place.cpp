// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// transform_in_place convenience helper: out = in * 2 + 1, applied IN PLACE.
//
// Exercises the surface that `transform_in_place` adds over `unary<Op, ...>`:
//   - in-place CB (CopyTile and PackTile share the same buffer),
//   - more than one SFPU op in the chain (MulUnary then AddUnary), and
//   - runtime-scalar op objects passed as constructed chain elements
//     (the fp32-encoded scalar lives in each op's ctor argument).
//
// The reader streams the input into cb_in; we copy it into a scratch CB sized for
// the whole window so the in-place transform has no concurrent producer on its FIFO,
// transform it in place, then copy the result out for the writer.

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_scalar.hpp"

void kernel_main() {
    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_scratch = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t n = get_compile_time_arg_val(0);

    // fp32 scalars encoded as uint32 (binop_with_scalar param semantics): 2.0f, 1.0f.
    constexpr uint32_t kMul2 = 0x40000000u;
    constexpr uint32_t kAdd1 = 0x3F800000u;

    compute_kernel_hw_startup(cb_in, cb_out);

    using namespace compute_kernel_lib;

    // Drain the streamed input into a window-sized scratch CB (no producer left on it).
    copy<cb_in, cb_scratch>(n);

    // In-place finalizer on the scratch CB: scratch = scratch * 2 + 1.
    transform_in_place<cb_scratch>(n, MulUnary<>{kMul2}, AddUnary<>{kAdd1});

    // Drain the transformed scratch to the output CB for the writer.
    copy<cb_scratch, cb_out>(n);
}
