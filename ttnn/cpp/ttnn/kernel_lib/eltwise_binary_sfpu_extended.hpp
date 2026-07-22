// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file eltwise_binary_sfpu_extended.hpp
 * @brief Extended DEST-DEST SFPU binary chain elements.
 */

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

namespace compute_kernel_lib {

template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct RemainderBinary;
template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct RemainderInt32Binary;
template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct FmodBinary;
template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct FmodInt32Binary;
template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct PowerBinary;
template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct RsubBinary;
template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct GcdBinary;
template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct LcmBinary;
template <DataFormat DF, Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct LeftShiftBinary;
template <DataFormat DF, Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct RightShiftBinary;
template <DataFormat DF, Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct LogicalRightShiftBinary;
template <DataFormat DF, Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct BitwiseAndBinary;
template <DataFormat DF, Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct BitwiseOrBinary;
template <DataFormat DF, Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct BitwiseXorBinary;
template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct XlogyBinary;
template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct Atan2Binary;
template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct LtBinary;
template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct GtBinary;
template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct LeBinary;
template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct GeBinary;
template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct EqBinary;
template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct NeBinary;
template <DataFormat DF, Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct LtIntBinary;
template <DataFormat DF, Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct GtIntBinary;
template <DataFormat DF, Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct LeIntBinary;
template <DataFormat DF, Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct GeIntBinary;

}  // namespace compute_kernel_lib

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu_extended.inl"
