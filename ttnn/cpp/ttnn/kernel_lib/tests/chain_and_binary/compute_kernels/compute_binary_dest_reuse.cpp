// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Test: y = (a - b) * scale via binary_op(SUB) with a DestReuseMul PostOp chain.
// `scale` is a single tile waited upfront in CB c_2 and reused across all tiles.
//
// Exercises:
// - DestReuseOp as an sfpu_chain element (LoadTag-derived).
// - Chain-only PostOp dispatch (binary_op static_asserts NoOp || SfpuChain).
// - clashes_with_fpu -> per-tile binary_init reinit path. After DestReuseMul
//   runs binary_dest_reuse_tiles_init(cb_scale) the unpack MOP is in dest-reuse
//   mode; binary_op must restore AB mode before the next tile's binary_exec.
//   If the reinit is missed, the second tile's subtract produces garbage.

#include "api/compute/common_globals.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/cb_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t num_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_b = tt::CBIndex::c_1;
    constexpr uint32_t cb_scale = tt::CBIndex::c_2;
    constexpr uint32_t cb_output = tt::CBIndex::c_16;

    binary_op_init_common(cb_a, cb_b, cb_output);

    // Scale is persistent: one tile, waited here, popped at the end.
    cb_wait_front(cb_scale, 1);

    using namespace compute_kernel_lib;
    sub(cb_a, cb_b, cb_output, BinaryInputBlockShape::of(1, num_tiles), sfpu_chain(DestReuseMul<cb_scale>{}));

    cb_pop_front(cb_scale, 1);
}
}  // namespace NAMESPACE
