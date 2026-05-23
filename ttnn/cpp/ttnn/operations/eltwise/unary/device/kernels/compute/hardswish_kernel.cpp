// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_activations.hpp"  // Hardsigmoid
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu.hpp"  // MulBinary
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_output = tt::CBIndex::c_2;

    binary_op_init_common(cb_input, cb_input, cb_output);

    // Hardswish: x * hardsigmoid(x). Same shape as tanhshrink (27bcccc7fea)
    // with Tanh -> Hardsigmoid and Sub -> Mul.
    //
    // FLOAT32: CopyTile(cb_input -> D0 HeldStream) + Hardsigmoid<D0>
    //          + CopyTile(cb_input -> D1 NoWaitPop) + MulBinary<D0, D1, D0>
    //          + PackTile.
    // FLOAT:   CopyTile(cb_input -> D0 HeldStream) + Hardsigmoid<D0>
    //          + DestReuseBinary<cb_input, Mul, DEST_TO_SRCA> + PackTile.
    //          srca = DEST = hardsigmoid(x), srcb = cb_input,
    //          result = hardsigmoid(x) * cb_input.
#ifdef INP_FLOAT32
    compute_kernel_lib::eltwise_chain(
        num_tiles,
        compute_kernel_lib::CopyTile<
            cb_input,
            compute_kernel_lib::Dst::D0,
            compute_kernel_lib::HeldStream,
            compute_kernel_lib::OperandKind::Scalar,
            compute_kernel_lib::CopyTileReconfig::None>{},
        compute_kernel_lib::Hardsigmoid<compute_kernel_lib::Dst::D0>{},
        compute_kernel_lib::CopyTile<
            cb_input,
            compute_kernel_lib::Dst::D1,
            compute_kernel_lib::NoWaitPop,
            compute_kernel_lib::OperandKind::Scalar,
            compute_kernel_lib::CopyTileReconfig::None>{},
        compute_kernel_lib::
            MulBinary<compute_kernel_lib::Dst::D0, compute_kernel_lib::Dst::D1, compute_kernel_lib::Dst::D0>{},
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
        compute_kernel_lib::Hardsigmoid<compute_kernel_lib::Dst::D0>{},
        compute_kernel_lib::DestReuseBinary<
            cb_input,
            compute_kernel_lib::BinaryFpuOp::Mul,
            compute_kernel_lib::DestReuseType::DEST_TO_SRCA,
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
