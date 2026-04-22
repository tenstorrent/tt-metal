// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace compute_kernel_lib {
using namespace ckernel;

template <Dst Slot> ALWI void Sin<Slot>::init() const { sin_tile_init(); }
template <Dst Slot> ALWI void Sin<Slot>::call(uint32_t d0) const { sin_tile(d0); }
template <Dst Slot> ALWI void Cos<Slot>::init() const { cos_tile_init(); }
template <Dst Slot> ALWI void Cos<Slot>::call(uint32_t d0) const { cos_tile(d0); }
template <Dst Slot> ALWI void Tan<Slot>::init() const { tan_tile_init(); }
template <Dst Slot> ALWI void Tan<Slot>::call(uint32_t d0) const { tan_tile(d0); }
template <Dst Slot> ALWI void Asin<Slot>::init() const { asin_tile_init(); }
template <Dst Slot> ALWI void Asin<Slot>::call(uint32_t d0) const { asin_tile(d0); }
template <Dst Slot> ALWI void Acos<Slot>::init() const { acos_tile_init(); }
template <Dst Slot> ALWI void Acos<Slot>::call(uint32_t d0) const { acos_tile(d0); }
template <Dst Slot> ALWI void Atan<Slot>::init() const { atan_tile_init(); }
template <Dst Slot> ALWI void Atan<Slot>::call(uint32_t d0) const { atan_tile(d0); }
template <Dst Slot> ALWI void Sinh<Slot>::init() const { sinh_tile_init(); }
template <Dst Slot> ALWI void Sinh<Slot>::call(uint32_t d0) const { sinh_tile(d0); }
template <Dst Slot> ALWI void Cosh<Slot>::init() const { cosh_tile_init(); }
template <Dst Slot> ALWI void Cosh<Slot>::call(uint32_t d0) const { cosh_tile(d0); }
template <Dst Slot> ALWI void Asinh<Slot>::init() const { asinh_tile_init(); }
template <Dst Slot> ALWI void Asinh<Slot>::call(uint32_t d0) const { asinh_tile(d0); }
template <Dst Slot> ALWI void Acosh<Slot>::init() const { acosh_tile_init(); }
template <Dst Slot> ALWI void Acosh<Slot>::call(uint32_t d0) const { acosh_tile(d0); }
template <Dst Slot> ALWI void Atanh<Slot>::init() const { atanh_tile_init(); }
template <Dst Slot> ALWI void Atanh<Slot>::call(uint32_t d0) const { atanh_tile(d0); }

// Aliases
template <uint32_t ICB, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_sin(uint32_t ocb, uint32_t num_tiles) { sfpu_op<ICB, O, R>(ocb, num_tiles, Sin<>{}); }
template <uint32_t ICB, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_cos(uint32_t ocb, uint32_t num_tiles) { sfpu_op<ICB, O, R>(ocb, num_tiles, Cos<>{}); }
template <uint32_t ICB, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_tan(uint32_t ocb, uint32_t num_tiles) { sfpu_op<ICB, O, R>(ocb, num_tiles, Tan<>{}); }
template <uint32_t ICB, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_asin(uint32_t ocb, uint32_t num_tiles) { sfpu_op<ICB, O, R>(ocb, num_tiles, Asin<>{}); }
template <uint32_t ICB, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_acos(uint32_t ocb, uint32_t num_tiles) { sfpu_op<ICB, O, R>(ocb, num_tiles, Acos<>{}); }
template <uint32_t ICB, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_atan(uint32_t ocb, uint32_t num_tiles) { sfpu_op<ICB, O, R>(ocb, num_tiles, Atan<>{}); }
template <uint32_t ICB, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_sinh(uint32_t ocb, uint32_t num_tiles) { sfpu_op<ICB, O, R>(ocb, num_tiles, Sinh<>{}); }
template <uint32_t ICB, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_cosh(uint32_t ocb, uint32_t num_tiles) { sfpu_op<ICB, O, R>(ocb, num_tiles, Cosh<>{}); }
template <uint32_t ICB, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_asinh(uint32_t ocb, uint32_t num_tiles) { sfpu_op<ICB, O, R>(ocb, num_tiles, Asinh<>{}); }
template <uint32_t ICB, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_acosh(uint32_t ocb, uint32_t num_tiles) { sfpu_op<ICB, O, R>(ocb, num_tiles, Acosh<>{}); }
template <uint32_t ICB, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_atanh(uint32_t ocb, uint32_t num_tiles) { sfpu_op<ICB, O, R>(ocb, num_tiles, Atanh<>{}); }

}  // namespace compute_kernel_lib
