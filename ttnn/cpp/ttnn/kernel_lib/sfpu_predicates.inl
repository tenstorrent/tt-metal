// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace compute_kernel_lib {
using namespace ckernel;

template <Dst Slot> ALWI void Isinf<Slot>::init() const { isinf_tile_init(); }
template <Dst Slot> ALWI void Isinf<Slot>::call(uint32_t d0) const { isinf_tile(d0); }
template <Dst Slot> ALWI void Isposinf<Slot>::init() const { isposinf_tile_init(); }
template <Dst Slot> ALWI void Isposinf<Slot>::call(uint32_t d0) const { isposinf_tile(d0); }
template <Dst Slot> ALWI void Isneginf<Slot>::init() const { isneginf_tile_init(); }
template <Dst Slot> ALWI void Isneginf<Slot>::call(uint32_t d0) const { isneginf_tile(d0); }
template <Dst Slot> ALWI void Isnan<Slot>::init() const { isnan_tile_init(); }
template <Dst Slot> ALWI void Isnan<Slot>::call(uint32_t d0) const { isnan_tile(d0); }
template <Dst Slot> ALWI void Isfinite<Slot>::init() const { isfinite_tile_init(); }
template <Dst Slot> ALWI void Isfinite<Slot>::call(uint32_t d0) const { isfinite_tile(d0); }

template <DataFormat df, Dst Slot>
ALWI void LogicalNot<df, Slot>::init() const { logical_not_tile_init(); }
template <DataFormat df, Dst Slot>
ALWI void LogicalNot<df, Slot>::call(uint32_t d0) const { logical_not_tile<df>(d0); }

template <Dst Slot> ALWI void Gtz<Slot>::init() const { gtz_tile_init(); }
template <Dst Slot> ALWI void Gtz<Slot>::call(uint32_t d0) const { gtz_tile(d0); }
template <Dst Slot> ALWI void Ltz<Slot>::init() const { ltz_tile_init(); }
template <Dst Slot> ALWI void Ltz<Slot>::call(uint32_t d0) const { ltz_tile(d0); }
template <Dst Slot> ALWI void Lez<Slot>::init() const { lez_tile_init(); }
template <Dst Slot> ALWI void Lez<Slot>::call(uint32_t d0) const { lez_tile(d0); }
template <Dst Slot> ALWI void Gez<Slot>::init() const { gez_tile_init(); }
template <Dst Slot> ALWI void Gez<Slot>::call(uint32_t d0) const { gez_tile(d0); }
template <Dst Slot> ALWI void Eqz<Slot>::init() const { eqz_tile_init(); }
template <Dst Slot> ALWI void Eqz<Slot>::call(uint32_t d0) const { eqz_tile(d0); }
template <Dst Slot> ALWI void Nez<Slot>::init() const { nez_tile_init(); }
template <Dst Slot> ALWI void Nez<Slot>::call(uint32_t d0) const { nez_tile(d0); }

template <Dst Slot> ALWI void UnaryEq<Slot>::init() const { unary_eq_tile_init(); }
template <Dst Slot> ALWI void UnaryEq<Slot>::call(uint32_t d0) const { unary_eq_tile(d0, param0); }
template <Dst Slot> ALWI void UnaryNe<Slot>::init() const { unary_ne_tile_init(); }
template <Dst Slot> ALWI void UnaryNe<Slot>::call(uint32_t d0) const { unary_ne_tile(d0, param0); }
template <Dst Slot> ALWI void UnaryGt<Slot>::init() const { unary_gt_tile_init(); }
template <Dst Slot> ALWI void UnaryGt<Slot>::call(uint32_t d0) const { unary_gt_tile(d0, param0); }
template <Dst Slot> ALWI void UnaryGe<Slot>::init() const { unary_ge_tile_init(); }
template <Dst Slot> ALWI void UnaryGe<Slot>::call(uint32_t d0) const { unary_ge_tile(d0, param0); }
template <Dst Slot> ALWI void UnaryLt<Slot>::init() const { unary_lt_tile_init(); }
template <Dst Slot> ALWI void UnaryLt<Slot>::call(uint32_t d0) const { unary_lt_tile(d0, param0); }
template <Dst Slot> ALWI void UnaryLe<Slot>::init() const { unary_le_tile_init(); }
template <Dst Slot> ALWI void UnaryLe<Slot>::call(uint32_t d0) const { unary_le_tile(d0, param0); }

// Aliases
template <uint32_t ICB, SfpuBatching B, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_isinf(uint32_t ocb, uint32_t num_tiles) { sfpu_op<ICB, B, P, O, R>(ocb, num_tiles, Isinf<>{}); }
template <uint32_t ICB, SfpuBatching B, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_isnan(uint32_t ocb, uint32_t num_tiles) { sfpu_op<ICB, B, P, O, R>(ocb, num_tiles, Isnan<>{}); }
template <uint32_t ICB, SfpuBatching B, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_isfinite(uint32_t ocb, uint32_t num_tiles) { sfpu_op<ICB, B, P, O, R>(ocb, num_tiles, Isfinite<>{}); }
template <uint32_t ICB, SfpuBatching B, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_gtz(uint32_t ocb, uint32_t num_tiles) { sfpu_op<ICB, B, P, O, R>(ocb, num_tiles, Gtz<>{}); }
template <uint32_t ICB, SfpuBatching B, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_ltz(uint32_t ocb, uint32_t num_tiles) { sfpu_op<ICB, B, P, O, R>(ocb, num_tiles, Ltz<>{}); }
template <uint32_t ICB, SfpuBatching B, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_lez(uint32_t ocb, uint32_t num_tiles) { sfpu_op<ICB, B, P, O, R>(ocb, num_tiles, Lez<>{}); }
template <uint32_t ICB, SfpuBatching B, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_gez(uint32_t ocb, uint32_t num_tiles) { sfpu_op<ICB, B, P, O, R>(ocb, num_tiles, Gez<>{}); }
template <uint32_t ICB, SfpuBatching B, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_eqz(uint32_t ocb, uint32_t num_tiles) { sfpu_op<ICB, B, P, O, R>(ocb, num_tiles, Eqz<>{}); }
template <uint32_t ICB, SfpuBatching B, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_nez(uint32_t ocb, uint32_t num_tiles) { sfpu_op<ICB, B, P, O, R>(ocb, num_tiles, Nez<>{}); }

}  // namespace compute_kernel_lib
