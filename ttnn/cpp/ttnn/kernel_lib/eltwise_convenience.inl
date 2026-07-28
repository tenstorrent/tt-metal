// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace compute_kernel_lib {

template <InputSpec AInput, InputSpec BInput, OutputSpec Output, BroadcastDim Bcast, EltwiseShapeKind Kind>
ALWI void add(TypedEltwiseShape<Kind> shape) {
    eltwise_chain(
        shape,
        BinaryFpu<AInput, BInput, BinaryFpuOp::Add, Bcast, Dst::D0, Output.dest_accumulation>{},
        PackTile<Output>{});
}

template <InputSpec AInput, InputSpec BInput, OutputSpec Output, BroadcastDim Bcast, EltwiseShapeKind Kind>
ALWI void sub(TypedEltwiseShape<Kind> shape) {
    eltwise_chain(
        shape,
        BinaryFpu<AInput, BInput, BinaryFpuOp::Sub, Bcast, Dst::D0, Output.dest_accumulation>{},
        PackTile<Output>{});
}

template <InputSpec AInput, InputSpec BInput, OutputSpec Output, BroadcastDim Bcast, EltwiseShapeKind Kind>
ALWI void mul(TypedEltwiseShape<Kind> shape) {
    eltwise_chain(
        shape,
        BinaryFpu<AInput, BInput, BinaryFpuOp::Mul, Bcast, Dst::D0, Output.dest_accumulation>{},
        PackTile<Output>{});
}

template <InputSpec Input, OutputSpec Output, EltwiseShapeKind Kind>
ALWI void square(TypedEltwiseShape<Kind> shape) {
    eltwise_chain(
        shape,
        BinaryFpu<Input, Input, BinaryFpuOp::Mul, BroadcastDim::None, Dst::D0, Output.dest_accumulation>{},
        PackTile<Output>{});
}

template <class SfpuOp, InputSpec Input, OutputSpec Output, EltwiseShapeKind Kind>
ALWI void unary(TypedEltwiseShape<Kind> shape) {
    static_assert(is_dest_only_op_v<SfpuOp>, "unary<SfpuOp, ...>: SfpuOp must be a DEST-only SFPU element");
    eltwise_chain(shape, CopyTile<Input>{}, SfpuOp{}, PackTile<Output>{});
}

template <class SfpuBinOp, InputSpec AInput, InputSpec BInput, OutputSpec Output, EltwiseShapeKind Kind>
ALWI void binary_sfpu(TypedEltwiseShape<Kind> shape) {
    static_assert(is_dest_only_op_v<SfpuBinOp>, "binary_sfpu<Op, ...>: Op must be a DEST-only SFPU binary element");
    eltwise_chain(shape, CopyTile<AInput>{}, CopyTile<BInput, Dst::D1>{}, SfpuBinOp{}, PackTile<Output>{});
}

template <InputSpec Input, OutputSpec Output, EltwiseShapeKind Kind>
ALWI void copy(TypedEltwiseShape<Kind> shape) {
    eltwise_chain(shape, CopyTile<Input>{}, PackTile<Output>{});
}

}  // namespace compute_kernel_lib
