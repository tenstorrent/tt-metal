// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Implementation detail of eltwise_misc.hpp — full op-struct definitions live here. The public
// header forward-declares these structs and includes this file at its tail.

#include "api/compute/eltwise_unary/identity.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/compute/mask.h"
#include "api/compute/copy_dest_values.h"    // CopyDest (DST -> DST)
#include "api/compute/compute_kernel_api.h"  // sign_tile, abs_tile, square_tile fallbacks

namespace compute_kernel_lib {

template <Dst Slot>
struct Identity : UnaryOp<Identity<Slot>, Slot> {
    static ALWI void init() { identity_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) { identity_tile(to_u32(Slot) + slot_offset); }
};

template <Dst Slot>
struct Negative : UnaryOp<Negative<Slot>, Slot> {
    static ALWI void init() { negative_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) { negative_tile(to_u32(Slot) + slot_offset); }
};

template <Dst Slot>
struct Abs : UnaryOp<Abs<Slot>, Slot> {
    static ALWI void init() { abs_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) { abs_tile(to_u32(Slot) + slot_offset); }
};

template <Dst Slot>
struct Sign : UnaryOp<Sign<Slot>, Slot> {
    static ALWI void init() { sign_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) { sign_tile(to_u32(Slot) + slot_offset); }
};

template <Dst Slot>
struct Square : UnaryOp<Square<Slot>, Slot> {
    static ALWI void init() { square_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) { square_tile(to_u32(Slot) + slot_offset); }
};

// CopyDest — copy a tile's values from one DEST slot to another (copy_dest_values).
// Two slots (In -> Out), no CB. Uses the un-templated copy_dest_values(in, out)
// overload, which the build permits via -Wno-error=deprecated-declarations.
template <Dst In, Dst Out>
struct CopyDest : DestOnlyTag {
    static constexpr uint32_t lane_width = (to_u32(In) > to_u32(Out) ? to_u32(In) : to_u32(Out)) + 1;
    static ALWI void init() { copy_dest_values_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        copy_dest_values(to_u32(In) + slot_offset, to_u32(Out) + slot_offset);
    }
    ALWI void exec(uint32_t /*i*/, uint32_t slot_offset) const { exec_impl(slot_offset); }
};

// Typecast — compile-time in/out dtype encoded as numeric IDs (uint32_t form expected by LLK).
template <uint32_t InDF, uint32_t OutDF, Dst Slot>
struct Typecast : UnaryOp<Typecast<InDF, OutDF, Slot>, Slot> {
    static ALWI void init() { typecast_tile_init<InDF, OutDF>(); }
    static ALWI void exec_impl(uint32_t slot_offset) { typecast_tile<InDF, OutDF>(to_u32(Slot) + slot_offset); }
};

// Mask — bakes the fixed `mask_tile` LLK contract (mask lives at DataSlot+1) into
// the type. Caller pre-loads data into DataSlot and mask into DataSlot+1; the
// op writes the masked result back into DataSlot. Compile-time `static_assert` rejects
// `DataSlot == D_LAST` (no slot for the mask tile).
//
// LLK supports `DataFormat::Float16`, `DataFormat::Float16_b`, and `DataFormat::Int32`.
template <DataFormat DF, Dst DataSlot>
struct Mask : BinaryOp<Mask<DF, DataSlot>, DataSlot, static_cast<Dst>(to_u32(DataSlot) + 1), DataSlot> {
    static_assert(
        to_u32(DataSlot) + 1 < DEST_AUTO_LIMIT,
        "Mask: DataSlot + 1 exceeds DEST_AUTO_LIMIT (mask tile lives at DataSlot + 1).");
    static ALWI void init() { mask_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        mask_tile(to_u32(DataSlot) + slot_offset, to_u32(DataSlot) + 1 + slot_offset, DF);
    }
};

// MaskPosInf — same LLK contract as Mask (data at DataSlot, mask at DataSlot+1) but
// masks each element with +inf instead of 0. Shares mask_tile_init.
template <Dst DataSlot>
struct MaskPosInf : BinaryOp<MaskPosInf<DataSlot>, DataSlot, static_cast<Dst>(to_u32(DataSlot) + 1), DataSlot> {
    static_assert(
        to_u32(DataSlot) + 1 < DEST_AUTO_LIMIT,
        "MaskPosInf: DataSlot + 1 exceeds DEST_AUTO_LIMIT (mask tile lives at DataSlot + 1).");
    static ALWI void init() { mask_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        mask_posinf_tile(to_u32(DataSlot) + slot_offset, to_u32(DataSlot) + 1 + slot_offset);
    }
};

}  // namespace compute_kernel_lib
