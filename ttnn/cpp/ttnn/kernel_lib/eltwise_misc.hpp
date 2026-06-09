// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file eltwise_misc.hpp
 * @brief Misc / utility SFPU op structs — Identity, Negative, Typecast, Sign, Abs, Square.
 */

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

namespace compute_kernel_lib {

template <Dst Slot = Dst::D0>
struct Identity;

template <Dst Slot = Dst::D0>
struct Negative;

template <Dst Slot = Dst::D0>
struct Abs;

template <Dst Slot = Dst::D0>
struct Sign;

template <Dst Slot = Dst::D0>
struct Square;

// CopyDest — copy a tile's values from one DEST slot to another (no defaults).
template <Dst In, Dst Out>
struct CopyDest;

// Typecast — compile-time in/out dtype encoded as numeric IDs.
template <uint32_t InDF, uint32_t OutDF, Dst Slot = Dst::D0>
struct Typecast;

// Mask / MaskPosInf.
template <DataFormat DF = DataFormat::Float16_b, Dst DataSlot = Dst::D0>
struct Mask;

template <Dst DataSlot = Dst::D0>
struct MaskPosInf;

}  // namespace compute_kernel_lib

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.inl"
