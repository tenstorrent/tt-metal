// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file basic.hpp
 * @brief Basic floating-point DEST-DEST SFPU binary chain elements.
 */

#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"

namespace compute_kernel_lib {

template <
    Dst In0 = Dst::D0,
    Dst In1 = Dst::D1,
    Dst Out = Dst::D0,
    ckernel::DstRoundingMode Rounding = ckernel::DstRoundingMode::Default>
struct AddBinary;
template <
    Dst In0 = Dst::D0,
    Dst In1 = Dst::D1,
    Dst Out = Dst::D0,
    ckernel::DstRoundingMode Rounding = ckernel::DstRoundingMode::Default>
struct SubBinary;
template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct MulBinary;
template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct DivBinary;

}  // namespace compute_kernel_lib

#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/basic.inl"
