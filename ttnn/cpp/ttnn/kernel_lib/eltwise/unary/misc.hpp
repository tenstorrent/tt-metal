// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file misc.hpp
 * @brief Misc / utility SFPU op structs — Identity, Negative, Typecast, Sign, Abs, Square.
 */

#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"

namespace compute_kernel_lib {

// identity_tile LLK is BH/WH-only (ckernel_sfpu_identity.h not ported to Quasar).
#ifndef ARCH_QUASAR
template <Dst Slot = Dst::D0>
struct Identity;
#endif

template <Dst Slot = Dst::D0>
struct Negative;

// abs_tile / sign_tile are declared for every arch in compute_kernel_api.h; the Quasar SFPU LLKs
// (ckernel_sfpu_abs.h / ckernel_sfpu_sign.h) landed with #58209.
template <Dst Slot = Dst::D0>
struct Abs;

template <Dst Slot = Dst::D0>
struct Sign;

template <Dst Slot = Dst::D0>
struct Square;

// CopyDest — copy a tile's values from one DEST slot to another. The LLK data
// format is explicit because format-agnostic copying is deprecated.
// copy_dest_values LLK is BH/WH-only (ckernel_sfpu_copy_dest_values.h not ported to Quasar).
#ifndef ARCH_QUASAR
template <Dst In, Dst Out, DataFormat DF>
struct CopyDest;
#endif

// Typecast — compile-time in/out dtype encoded as numeric IDs.
template <uint32_t InDF, uint32_t OutDF, Dst Slot = Dst::D0>
struct Typecast;

// Mask / MaskPosInf. The mask_tile LLK is available on BH/WH and, since #58209, on Quasar
// (ckernel_sfpu_mask.h in the Quasar llk_sfpu directory).
template <DataFormat DF = DataFormat::Float16_b, Dst DataSlot = Dst::D0>
struct Mask;

template <Dst DataSlot = Dst::D0>
struct MaskPosInf;

}  // namespace compute_kernel_lib

#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/misc.inl"
