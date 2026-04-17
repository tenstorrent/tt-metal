// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace compute_kernel_lib {
using namespace ckernel;

template <uint32_t in_dtype, uint32_t out_dtype, Dst Slot>
ALWI void Typecast<in_dtype, out_dtype, Slot>::init() const { typecast_tile_init<in_dtype, out_dtype>(); }
template <uint32_t in_dtype, uint32_t out_dtype, Dst Slot>
ALWI void Typecast<in_dtype, out_dtype, Slot>::call(uint32_t d0) const { typecast_tile<in_dtype, out_dtype>(d0); }

template <Dst Slot> ALWI void Identity<Slot>::init() const { identity_tile_init(); }
template <Dst Slot> ALWI void Identity<Slot>::call(uint32_t d0) const { identity_tile(d0); }
// NOTE: Bitwise/shift op implementations excluded — see sfpu_misc.hpp for rationale.

template <Dst Slot> ALWI void FillTile<Slot>::init() const { fill_tile_init(); }
template <Dst Slot> ALWI void FillTile<Slot>::call(uint32_t d0) const { fill_tile(d0, fill_val); }
template <Dst Slot> ALWI void FillTileBitcast<Slot>::init() const { fill_tile_init(); }
template <Dst Slot> ALWI void FillTileBitcast<Slot>::call(uint32_t d0) const { fill_tile_bitcast(d0, param0); }
template <Dst Slot> ALWI void RandTile<Slot>::init() const {}
template <Dst Slot> ALWI void RandTile<Slot>::call(uint32_t d0) const { rand_tile(d0, from, scale); }

}  // namespace compute_kernel_lib
