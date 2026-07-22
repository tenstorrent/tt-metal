// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file eltwise_binary_sfpu_int.hpp
 * @brief Integer arithmetic DEST-DEST SFPU binary chain elements.
 */

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

namespace compute_kernel_lib {

template <DataFormat DF, Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct AddIntBinary;
template <DataFormat DF, Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct SubIntBinary;
template <DataFormat DF, Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct MulIntBinary;
template <DataFormat DF, Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct RsubIntBinary;
template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct DivInt32Binary;
template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct DivInt32FloorBinary;
template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct DivInt32TruncBinary;

}  // namespace compute_kernel_lib

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu_int.inl"
