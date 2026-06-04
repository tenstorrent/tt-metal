// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Implementation detail of eltwise_fill.hpp — full op-struct definitions live here. The public
// header forward-declares these structs and includes this file at its tail.

#include "api/compute/eltwise_unary/fill.h"

namespace compute_kernel_lib {

template <Dst DstSlot>
struct FillScalar : FillTileTag, UnaryOp<FillScalar<DstSlot>, DstSlot> {
    float value;
    constexpr explicit FillScalar(float v) noexcept : value(v) {}
    constexpr FillScalar() noexcept : value(0.0f) {}

    static ALWI void init() { fill_tile_init(); }
    ALWI void exec(uint32_t /*i*/, uint32_t slot_offset) const { fill_tile(to_u32(DstSlot) + slot_offset, value); }
};

template <DataFormat DF, Dst DstSlot>
struct FillInt : FillTileTag, UnaryOp<FillInt<DF, DstSlot>, DstSlot> {
    uint32_t value;
    constexpr explicit FillInt(uint32_t v) noexcept : value(v) {}
    constexpr FillInt() noexcept : value(0) {}

    // fill_tile_int shares the same init as fill_tile (no separate `_int_init`).
    static ALWI void init() { fill_tile_init(); }
    ALWI void exec(uint32_t /*i*/, uint32_t slot_offset) const {
        fill_tile_int<DF>(to_u32(DstSlot) + slot_offset, value);
    }
};

template <Dst DstSlot>
struct FillBitcast : FillTileTag, UnaryOp<FillBitcast<DstSlot>, DstSlot> {
    uint32_t bits;
    constexpr explicit FillBitcast(uint32_t b) noexcept : bits(b) {}
    constexpr FillBitcast() noexcept : bits(0) {}

    static ALWI void init() { fill_tile_init(); }
    ALWI void exec(uint32_t /*i*/, uint32_t slot_offset) const {
        fill_tile_bitcast(to_u32(DstSlot) + slot_offset, bits);
    }
};

}  // namespace compute_kernel_lib
