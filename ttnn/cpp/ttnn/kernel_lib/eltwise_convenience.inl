// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace compute_kernel_lib {

template <InputSpec AInput, InputSpec BInput, OutputSpec Output, BroadcastDim Bcast>
ALWI void add(EltwiseShape shape) {
    eltwise_chain(
        shape,
        BinaryFpu<AInput, BInput, BinaryFpuOp::Add, Bcast, Dst::D0, Output.dest_accumulation>{},
        PackTile<Output>{});
}

template <InputSpec AInput, InputSpec BInput, OutputSpec Output, BroadcastDim Bcast>
ALWI void sub(EltwiseShape shape) {
    eltwise_chain(
        shape,
        BinaryFpu<AInput, BInput, BinaryFpuOp::Sub, Bcast, Dst::D0, Output.dest_accumulation>{},
        PackTile<Output>{});
}

template <InputSpec AInput, InputSpec BInput, OutputSpec Output, BroadcastDim Bcast>
ALWI void mul(EltwiseShape shape) {
    eltwise_chain(
        shape,
        BinaryFpu<AInput, BInput, BinaryFpuOp::Mul, Bcast, Dst::D0, Output.dest_accumulation>{},
        PackTile<Output>{});
}

template <InputSpec Input, OutputSpec Output>
ALWI void square(EltwiseShape shape) {
    eltwise_chain(
        shape,
        BinaryFpu<Input, Input, BinaryFpuOp::Mul, BroadcastDim::None, Dst::D0, Output.dest_accumulation>{},
        PackTile<Output>{});
}

template <class SfpuOp, InputSpec Input, OutputSpec Output>
ALWI void unary(EltwiseShape shape) {
    static_assert(is_dest_only_op_v<SfpuOp>, "unary<SfpuOp, ...>: SfpuOp must be a DEST-only SFPU element");
    eltwise_chain(shape, CopyTile<Input>{}, SfpuOp{}, PackTile<Output>{});
}

template <class SfpuBinOp, InputSpec AInput, InputSpec BInput, OutputSpec Output>
ALWI void binary_sfpu(EltwiseShape shape) {
    static_assert(is_dest_only_op_v<SfpuBinOp>, "binary_sfpu<Op, ...>: Op must be a DEST-only SFPU binary element");
    eltwise_chain(shape, CopyTile<AInput>{}, CopyTile<BInput, Dst::D1>{}, SfpuBinOp{}, PackTile<Output>{});
}

template <InputSpec Input, OutputSpec Output>
ALWI void copy(EltwiseShape shape) {
    eltwise_chain(shape, CopyTile<Input>{}, PackTile<Output>{});
}

}  // namespace compute_kernel_lib
