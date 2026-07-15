// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Helper chain out=gelu(A)*B for the block_size perf test (mirrors the moe_compute GELU path).
// Gelu's init loads an SFPU LUT (expensive), and {Gelu, MulBinary} is non-uniform, so the LUT init
// is emitted per block-iteration (not hoisted) — block_size loads it once per blk tiles instead of
// per tile. 2 DEST slots/tile (D0=gelu(A), D1=B) => block<=4 half-sync. Bulk lifecycle, matching
// raw_gelu_noblock.cpp so only block_size differs. CT args: [n, blk].

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_activations.hpp"  // Gelu
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu.hpp"  // MulBinary

void kernel_main() {
    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_b = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    constexpr uint32_t n = get_compile_time_arg_val(0);
    constexpr uint32_t blk = get_compile_time_arg_val(1);

    compute_kernel_hw_startup(cb_a, cb_out);

    using namespace compute_kernel_lib;
    eltwise_chain(
        EltwiseShape::tiles(n, blk),
        CopyTile<cb_a, Dst::D0, InputLifecycle::Bulk, CopyTileReconfig::Input, OperandKind::Block>{},
        CopyTile<cb_b, Dst::D1, InputLifecycle::Bulk, CopyTileReconfig::Input, OperandKind::Block>{},
        Gelu<Dst::D0>{},                         // D0 = gelu(A)  — LUT init, emitted per block-iteration
        MulBinary<Dst::D0, Dst::D1, Dst::D0>{},  // D0 = gelu(A) * B  (clobbers the gelu LUT)
        PackTile<cb_out, OutputLifecycle::Bulk, PackTileReconfig::Output>{});
}
