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

template <class SfpuOp, InputSpec Input, OutputSpec Output>
ALWI void unary(IterationShape shape) {
    static_assert(is_dest_only_op_v<SfpuOp>, "unary<SfpuOp, ...>: SfpuOp must be a DEST-only SFPU element");
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
    static_assert(is_dest_only_op_v<SfpuBinOp>, "binary_sfpu<Op, ...>: Op must be a DEST-only SFPU binary element");
    eltwise_chain(shape, CopyTile<AInput>{}, CopyTile<BInput, Dst::D1>{}, SfpuBinOp{}, PackTile<Output>{});
}

template <InputSpec Input, OutputSpec Output>
ALWI void copy(IterationShape shape) {
    eltwise_chain(shape, CopyTile<Input>{}, PackTile<Output>{});
}

}  // namespace compute_kernel_lib
