// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// DEST multi-slot independence: two FPU adds write independent DEST slots in one tile_regs window,
// then an SFPU DEST+DEST add combines them.
//   D0 = A + B ; D1 = C + E (must NOT disturb D0) ; D0 = D0 + D1 ; pack D0  ->  out = A+B+C+E
// If the slots aliased, the result would collapse to (A+B) or (C+E). Runs under both fp32_dest_acc
// settings (halved-capacity path: half-sync limit 8 vs 4 — D0,D1 fit in both).

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu_basic.hpp"

void kernel_main() {
    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_b = tt::CBIndex::c_1;
    constexpr uint32_t cb_c = tt::CBIndex::c_2;
    constexpr uint32_t cb_e = tt::CBIndex::c_3;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t n = get_compile_time_arg_val(0);

    compute_kernel_hw_startup(cb_a, cb_b, cb_out);

    using namespace compute_kernel_lib;
    eltwise_chain(
        EltwiseShape::tiles(n),
        BinaryFpu<input(cb_a), input(cb_b)>{},
        BinaryFpu<input(cb_c), input(cb_e), BinaryFpuOp::Add, BroadcastDim::None, Dst::D1>{},
        AddBinary<Dst::D0, Dst::D1, Dst::D0>{},
        PackTile<output(cb_out)>{});
}
