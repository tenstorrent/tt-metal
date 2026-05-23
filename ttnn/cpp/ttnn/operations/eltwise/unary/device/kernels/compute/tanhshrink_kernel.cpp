// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_activations.hpp"  // Tanh
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu.hpp"  // SubBinary
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_output = tt::CBIndex::c_2;

    binary_op_init_common(cb_input, cb_input, cb_output);

    // Tanhshrink: x - tanh(x).
    //
    // FLOAT32 path: cb_input copied twice (D1 = tanh(cb_input), D0 = cb_input)
    //   then SubBinary D0 - D1 -> D0. Mirrors mul_binary_tile-style two-DEST
    //   pattern. cb_input HeldStream on first CopyTile (wait, no pop), then
    //   NoWaitPop on second (pops cb_input).
    //
    // FLOAT path: CopyTile(cb_input -> D0) + Tanh<D0> + DestReuseBinary
    //   ELWSUB DEST_TO_SRCB (srca = cb_input, srcb = DEST = tanh(cb_input),
    //   result = cb_input - tanh(cb_input) -> D0).
    //
    // Reconfig matches original init_sfpu + copy_tile_init at boot, no
    // _with_dt mid-kernel — CopyTileReconfig::None / PackTileReconfig::None.
#ifdef INP_FLOAT32
    compute_kernel_lib::eltwise_chain(
        num_tiles,
        compute_kernel_lib::CopyTile<
            cb_input,
            compute_kernel_lib::Dst::D1,
            compute_kernel_lib::HeldStream,
            compute_kernel_lib::OperandKind::Scalar,
            compute_kernel_lib::CopyTileReconfig::None>{},
        compute_kernel_lib::Tanh<compute_kernel_lib::Dst::D1>{},
        compute_kernel_lib::CopyTile<
            cb_input,
            compute_kernel_lib::Dst::D0,
            compute_kernel_lib::NoWaitPop,
            compute_kernel_lib::OperandKind::Scalar,
            compute_kernel_lib::CopyTileReconfig::None>{},
        compute_kernel_lib::
            SubBinary<compute_kernel_lib::Dst::D0, compute_kernel_lib::Dst::D1, compute_kernel_lib::Dst::D0>{},
        compute_kernel_lib::PackTile<
            cb_output,
            compute_kernel_lib::Dst::D0,
            compute_kernel_lib::OutStreaming,
            compute_kernel_lib::OperandKind::Scalar,
            compute_kernel_lib::PackTileReconfig::None>{});
#endif
#ifdef INP_FLOAT
    compute_kernel_lib::eltwise_chain(
        num_tiles,
        compute_kernel_lib::CopyTile<
            cb_input,
            compute_kernel_lib::Dst::D0,
            compute_kernel_lib::HeldStream,
            compute_kernel_lib::OperandKind::Scalar,
            compute_kernel_lib::CopyTileReconfig::None>{},
        compute_kernel_lib::Tanh<compute_kernel_lib::Dst::D0>{},
        compute_kernel_lib::DestReuseBinary<
            cb_input,
            compute_kernel_lib::BinaryFpuOp::Sub,
            compute_kernel_lib::DestReuseType::DEST_TO_SRCB,
            compute_kernel_lib::Dst::D0,
            compute_kernel_lib::Dst::D0,
            compute_kernel_lib::DestReuseReconfig::Input,
            compute_kernel_lib::Streaming,
            compute_kernel_lib::OperandKind::Scalar>{},
        compute_kernel_lib::PackTile<
            cb_output,
            compute_kernel_lib::Dst::D0,
            compute_kernel_lib::OutStreaming,
            compute_kernel_lib::OperandKind::Scalar,
            compute_kernel_lib::PackTileReconfig::None>{});
#endif
}
