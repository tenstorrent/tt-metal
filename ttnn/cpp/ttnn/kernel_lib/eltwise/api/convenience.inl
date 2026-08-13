// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace compute_kernel_lib {

template <InputSpec AInput, BroadcastInputSpec BInput, OutputSpec Output>
ALWI void add(IterationShape shape) {
    eltwise_chain(
        shape, BinaryFpu<BinaryFpuOp::Add, AInput, BInput, Dst::D0, Output.dest_accumulation>{}, PackTile<Output>{});
}

template <InputSpec AInput, BroadcastInputSpec BInput, OutputSpec Output>
ALWI void sub(IterationShape shape) {
    eltwise_chain(
        shape, BinaryFpu<BinaryFpuOp::Sub, AInput, BInput, Dst::D0, Output.dest_accumulation>{}, PackTile<Output>{});
}

template <InputSpec AInput, BroadcastInputSpec BInput, OutputSpec Output>
ALWI void mul(IterationShape shape) {
    eltwise_chain(
        shape, BinaryFpu<BinaryFpuOp::Mul, AInput, BInput, Dst::D0, Output.dest_accumulation>{}, PackTile<Output>{});
}

template <InputSpec Input, OutputSpec Output>
ALWI void square(IterationShape shape) {
    eltwise_chain(
        shape, BinaryFpu<BinaryFpuOp::Mul, Input, Input, Dst::D0, Output.dest_accumulation>{}, PackTile<Output>{});
}

constexpr RowOutputSpec row_output(uint32_t cb_id, DataFormatReconfig reconfig, PackRelu relu) noexcept {
    return {cb_id, reconfig, relu};
}

template <InputSpec Input, RowOutputSpec RowOutput>
ALWI void sum_of_squares(IterationShape shape) {
    constexpr auto output_spec = output(
        RowOutput.cb_id,
        ReservePolicy::PerOuter,
        PushPolicy::PerOuter,
        RowOutput.reconfig,
        TileAddressing::Direct,
        DestAccumulation::PerRow,
        L1Accumulation::Disabled,
        RowOutput.relu);
    square<Input, output_spec>(shape);
}

template <class SfpuOp, InputSpec Input, OutputSpec Output>
ALWI void unary(IterationShape shape) {
    static_assert(
        is_unary_op_v<SfpuOp> && is_sfpu_op_v<SfpuOp>,
        "unary<SfpuOp, ...>: SfpuOp must be a unary DEST operation");
    eltwise_chain(shape, CopyTile<Input>{}, SfpuOp{}, PackTile<Output>{});
}

template <InputSpec Input, OutputSpec Output>
ALWI void typecast(IterationShape shape) {
    constexpr auto in_df = dfb_l1_format<Input.cb_id>();
    constexpr auto out_df = dfb_l1_format<Output.cb_id>();
    unary<Typecast<in_df, out_df>, Input, Output>(shape);
}

template <class SfpuBinOp, InputSpec AInput, InputSpec BInput, OutputSpec Output>
ALWI void binary_sfpu(IterationShape shape) {
    static_assert(
        is_binary_op_v<SfpuBinOp> && is_sfpu_op_v<SfpuBinOp>,
        "binary_sfpu<Op, ...>: Op must be a binary DEST operation");
    eltwise_chain(shape, CopyTile<AInput>{}, CopyTile<BInput, Dst::D1>{}, SfpuBinOp{}, PackTile<Output>{});
}

template <InputSpec Input, OutputSpec Output>
ALWI void copy(IterationShape shape) {
    eltwise_chain(shape, CopyTile<Input>{}, PackTile<Output>{});
}

}  // namespace compute_kernel_lib
