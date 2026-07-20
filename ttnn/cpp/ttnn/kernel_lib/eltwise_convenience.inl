// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace compute_kernel_lib {

template <
    uint32_t CbA,
    uint32_t CbB,
    uint32_t CbOut,
    BroadcastDim Bcast,
    InputSpec AInput,
    InputSpec BInput,
    OutputSpec Output>
ALWI void add(EltwiseShape shape) {
    eltwise_chain(shape, BinaryFpu<CbA, CbB, BinaryFpuOp::Add, Bcast, AInput, BInput>{}, PackTile<CbOut, Output>{});
}

template <
    uint32_t CbA,
    uint32_t CbB,
    uint32_t CbOut,
    BroadcastDim Bcast,
    InputSpec AInput,
    InputSpec BInput,
    OutputSpec Output>
ALWI void sub(EltwiseShape shape) {
    eltwise_chain(shape, BinaryFpu<CbA, CbB, BinaryFpuOp::Sub, Bcast, AInput, BInput>{}, PackTile<CbOut, Output>{});
}

template <
    uint32_t CbA,
    uint32_t CbB,
    uint32_t CbOut,
    BroadcastDim Bcast,
    InputSpec AInput,
    InputSpec BInput,
    OutputSpec Output>
ALWI void mul(EltwiseShape shape) {
    eltwise_chain(shape, BinaryFpu<CbA, CbB, BinaryFpuOp::Mul, Bcast, AInput, BInput>{}, PackTile<CbOut, Output>{});
}

template <uint32_t CbIn, uint32_t CbOut, InputSpec Input, OutputSpec Output>
ALWI void square(EltwiseShape shape) {
    eltwise_chain(
        shape, BinaryFpu<CbIn, CbIn, BinaryFpuOp::Mul, BroadcastDim::None, Input, Input>{}, PackTile<CbOut, Output>{});
}

template <class SfpuOp, uint32_t CbIn, uint32_t CbOut, InputSpec Input, OutputSpec Output>
ALWI void unary(EltwiseShape shape) {
    static_assert(is_dest_only_op_v<SfpuOp>, "unary<SfpuOp, ...>: SfpuOp must be a DEST-only SFPU element");
    eltwise_chain(shape, CopyTile<CbIn, Dst::D0, Input>{}, SfpuOp{}, PackTile<CbOut, Output>{});
}

template <
    class SfpuBinOp,
    uint32_t CbA,
    uint32_t CbB,
    uint32_t CbOut,
    InputSpec AInput,
    InputSpec BInput,
    OutputSpec Output>
ALWI void binary_sfpu(EltwiseShape shape) {
    static_assert(is_dest_only_op_v<SfpuBinOp>, "binary_sfpu<Op, ...>: Op must be a DEST-only SFPU binary element");
    eltwise_chain(
        shape,
        CopyTile<CbA, Dst::D0, AInput>{},
        CopyTile<CbB, Dst::D1, BInput>{},
        SfpuBinOp{},
        PackTile<CbOut, Output>{});
}

template <uint32_t CbIn, uint32_t CbOut, InputSpec Input, OutputSpec Output>
ALWI void copy(EltwiseShape shape) {
    eltwise_chain(shape, CopyTile<CbIn, Dst::D0, Input>{}, PackTile<CbOut, Output>{});
}

}  // namespace compute_kernel_lib
