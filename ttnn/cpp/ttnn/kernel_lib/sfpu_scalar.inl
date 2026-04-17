// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace compute_kernel_lib {
using namespace ckernel;

template <Dst Slot> ALWI void AddScalar<Slot>::init() const { binop_with_scalar_tile_init(); }
template <Dst Slot> ALWI void AddScalar<Slot>::call(uint32_t d0) const { add_unary_tile(d0, scalar); }
template <Dst Slot> ALWI void SubScalar<Slot>::init() const { binop_with_scalar_tile_init(); }
template <Dst Slot> ALWI void SubScalar<Slot>::call(uint32_t d0) const { sub_unary_tile(d0, scalar); }
template <Dst Slot> ALWI void MulScalar<Slot>::init() const { binop_with_scalar_tile_init(); }
template <Dst Slot> ALWI void MulScalar<Slot>::call(uint32_t d0) const { mul_unary_tile(d0, scalar); }
template <Dst Slot> ALWI void DivScalar<Slot>::init() const { binop_with_scalar_tile_init(); }
template <Dst Slot> ALWI void DivScalar<Slot>::call(uint32_t d0) const { div_unary_tile(d0, scalar); }
template <Dst Slot> ALWI void RsubScalar<Slot>::init() const { binop_with_scalar_tile_init(); }
template <Dst Slot> ALWI void RsubScalar<Slot>::call(uint32_t d0) const { rsub_unary_tile(d0, scalar); }

template <Dst Slot> ALWI void Rsub<Slot>::init() const { rsub_tile_init(); }
template <Dst Slot> ALWI void Rsub<Slot>::call(uint32_t d0) const { rsub_tile(d0, param0); }

template <RoundingMode rounding_mode, Dst Slot>
ALWI void Rdiv<rounding_mode, Slot>::init() const { rdiv_tile_init(); }
template <RoundingMode rounding_mode, Dst Slot>
ALWI void Rdiv<rounding_mode, Slot>::call(uint32_t d0) const { rdiv_tile<rounding_mode>(d0, value); }

template <Dst Slot>
ALWI void Fmod<Slot>::init() const { fmod_tile_init(param0, param1); }
template <Dst Slot>
ALWI void Fmod<Slot>::call(uint32_t d0) const { fmod_tile(d0, param0, param1); }

template <Dst Slot>
ALWI void Remainder<Slot>::init() const { remainder_tile_init(param0, param1); }
template <Dst Slot>
ALWI void Remainder<Slot>::call(uint32_t d0) const { remainder_tile(d0, param0, param1); }

template <Dst Slot>
ALWI void Dropout<Slot>::init() const {}
template <Dst Slot>
ALWI void Dropout<Slot>::call(uint32_t d0) const { dropout_tile(d0, probability, scale_factor); }

}  // namespace compute_kernel_lib
