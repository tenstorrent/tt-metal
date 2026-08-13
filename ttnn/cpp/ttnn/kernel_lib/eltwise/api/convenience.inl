// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace compute_kernel_lib {

template <InputSpec AInput, auto BInput, OutputSpec Output, IterationShapeKind Kind>
ALWI void add(TypedIterationShape<Kind> shape) {
    eltwise_chain(
        shape, BinaryFpu<BinaryFpuOp::Add, AInput, BInput, Dst::D0, Output.dest_accumulation>{}, PackTile<Output>{});
}

template <InputSpec AInput, auto BInput, OutputSpec Output, IterationShapeKind Kind>
ALWI void sub(TypedIterationShape<Kind> shape) {
    eltwise_chain(
        shape, BinaryFpu<BinaryFpuOp::Sub, AInput, BInput, Dst::D0, Output.dest_accumulation>{}, PackTile<Output>{});
}

template <InputSpec AInput, auto BInput, OutputSpec Output, IterationShapeKind Kind>
ALWI void mul(TypedIterationShape<Kind> shape) {
    eltwise_chain(
        shape, BinaryFpu<BinaryFpuOp::Mul, AInput, BInput, Dst::D0, Output.dest_accumulation>{}, PackTile<Output>{});
}

template <InputSpec Input, OutputSpec Output, IterationShapeKind Kind>
ALWI void square(TypedIterationShape<Kind> shape) {
    eltwise_chain(
        shape, BinaryFpu<BinaryFpuOp::Mul, Input, Input, Dst::D0, Output.dest_accumulation>{}, PackTile<Output>{});
}

constexpr RowOutputSpec row_output(uint32_t cb_id, DataFormatReconfig reconfig, PackRelu relu) noexcept {
    return {cb_id, reconfig, relu};
}

template <InputSpec Input, RowOutputSpec RowOutput>
ALWI void sum_of_squares(TypedIterationShape<IterationShapeKind::Grid> shape) {
    constexpr auto output_spec = output(
        RowOutput.cb_id,
        ReservePolicy::PerOuter,
        PushPolicy::PerOuter,
        RowOutput.reconfig,
        RowOutput.relu,
        L1Accumulation::Disabled,
        DestAccumulation::PerRow);
    square<Input, output_spec>(shape);
}

template <class SfpuOp, InputSpec Input, OutputSpec Output, IterationShapeKind Kind>
ALWI void unary(TypedIterationShape<Kind> shape) {
    static_assert(is_dest_only_op_v<SfpuOp>, "unary<SfpuOp, ...>: SfpuOp must be a DEST-only SFPU element");
    eltwise_chain(shape, CopyTile<Input>{}, SfpuOp{}, PackTile<Output>{});
}

template <InputSpec Input, OutputSpec Output, IterationShapeKind Kind>
ALWI void typecast(TypedIterationShape<Kind> shape) {
    constexpr auto in_df = dfb_l1_format<Input.cb_id>();
    constexpr auto out_df = dfb_l1_format<Output.cb_id>();
    unary<Typecast<in_df, out_df>, Input, Output>(shape);
}

template <class SfpuBinOp, InputSpec AInput, InputSpec BInput, OutputSpec Output, IterationShapeKind Kind>
ALWI void binary_sfpu(TypedIterationShape<Kind> shape) {
    static_assert(is_dest_only_op_v<SfpuBinOp>, "binary_sfpu<Op, ...>: Op must be a DEST-only SFPU binary element");
    eltwise_chain(shape, CopyTile<AInput>{}, CopyTile<BInput, Dst::D1>{}, SfpuBinOp{}, PackTile<Output>{});
}

template <InputSpec Input, OutputSpec Output, IterationShapeKind Kind>
ALWI void copy(TypedIterationShape<Kind> shape) {
    eltwise_chain(shape, CopyTile<Input>{}, PackTile<Output>{});
}

}  // namespace compute_kernel_lib
