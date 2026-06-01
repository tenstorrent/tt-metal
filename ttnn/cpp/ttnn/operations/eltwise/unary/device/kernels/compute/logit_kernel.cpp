// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"         // Log
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_scalar.hpp"       // Clamp, RsubUnary
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu.hpp"  // DivBinary
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    const uint32_t packed_scalar1 = get_arg_val<uint32_t>(1);
    const uint32_t packed_scalar2 = get_arg_val<uint32_t>(2);

    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_tmp0 = tt::CBIndex::c_1;
    constexpr auto cb_output = tt::CBIndex::c_2;

    init_sfpu(cb_input, cb_output);

    // Logit(x) = log(x / (1 - x)).
    //
    // Stage 1: cb_input -> [Clamp(s1, s2)] -> cb_tmp0
    //   Clamp is gated on #ifdef CLAMP — when undefined, the chain is just
    //   CopyTile + PackTile (an in-place pre-clip pass that matches the
    //   original copy + conditional clamp + pack to cb_tmp0).
    //
    // Stage 2: cb_tmp0 (held + popped at end) ->
    //   CopyTile<cb_tmp0, D0 HeldStream> + CopyTile<cb_tmp0, D1 NoWaitPop>
    //   RsubUnary<D0>{1.0f bits} -> D0 = 1 - cb_tmp0
    //   DivBinary<D1, D0, D0>    -> D0 = cb_tmp0 / (1 - cb_tmp0)
    //   Log<D0>                  -> D0 = log(D0)
    //   PackTile<cb_output, D0>
    //
    // Reconfig matches original init_sfpu + copy_tile_init at boot —
    // CopyTileReconfig::None + PackTileReconfig::None.
    compute_kernel_lib::eltwise_chain(
        num_tiles,
        compute_kernel_lib::CopyTile<
            cb_input,
            compute_kernel_lib::Dst::D0,
            compute_kernel_lib::Streaming,
            compute_kernel_lib::OperandKind::Scalar,
            compute_kernel_lib::CopyTileReconfig::None>{},
#ifdef CLAMP
        compute_kernel_lib::Clamp<compute_kernel_lib::Dst::D0>{packed_scalar1, packed_scalar2},
#endif
        compute_kernel_lib::PackTile<
            cb_tmp0,
            compute_kernel_lib::Dst::D0,
            compute_kernel_lib::OutStreaming,
            compute_kernel_lib::PackTileReconfig::None>{});

    compute_kernel_lib::eltwise_chain(
        num_tiles,
        compute_kernel_lib::CopyTile<
            cb_tmp0,
            compute_kernel_lib::Dst::D0,
            compute_kernel_lib::HeldStream,
            compute_kernel_lib::OperandKind::Scalar,
            compute_kernel_lib::CopyTileReconfig::None>{},
        compute_kernel_lib::CopyTile<
            cb_tmp0,
            compute_kernel_lib::Dst::D1,
            compute_kernel_lib::NoWaitPop,
            compute_kernel_lib::OperandKind::Scalar,
            compute_kernel_lib::CopyTileReconfig::None>{},
        compute_kernel_lib::RsubUnary<compute_kernel_lib::Dst::D0>{0x3F800000u},  // 1.0 - x
        compute_kernel_lib::
            DivBinary<compute_kernel_lib::Dst::D1, compute_kernel_lib::Dst::D0, compute_kernel_lib::Dst::D0>{},
        compute_kernel_lib::Log<compute_kernel_lib::Approx::Exact, compute_kernel_lib::Dst::D0>{},
        compute_kernel_lib::PackTile<
            cb_output,
            compute_kernel_lib::Dst::D0,
            compute_kernel_lib::OutStreaming,
            compute_kernel_lib::PackTileReconfig::None>{});
}
