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

// abs_tile / sign_tile are Blackhole/Wormhole-only in compute_kernel_api.h (no Quasar SFPU impl),
// so these ops are not available on Quasar.
#ifndef ARCH_QUASAR
template <Dst Slot = Dst::D0>
struct Abs;

template <Dst Slot = Dst::D0>
struct Sign;
#endif

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

// Mask / MaskPosInf. mask_tile LLK is BH/WH-only (ckernel_sfpu_mask.h not ported to Quasar).
#ifndef ARCH_QUASAR
template <DataFormat DF = DataFormat::Float16_b, Dst DataSlot = Dst::D0>
struct Mask;

template <Dst DataSlot = Dst::D0>
struct MaskPosInf;
#endif

}  // namespace compute_kernel_lib

#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/misc.inl"
