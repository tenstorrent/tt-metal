// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace compute_kernel_lib {
using namespace ckernel;

template <Approx approx, Dst Slot>
ALWI void Erf<approx, Slot>::init() const { erf_tile_init<static_cast<bool>(approx)>(); }
template <Approx approx, Dst Slot>
ALWI void Erf<approx, Slot>::call(uint32_t d0) const { erf_tile<static_cast<bool>(approx)>(d0); }

template <Approx approx, Dst Slot>
ALWI void Erfc<approx, Slot>::init() const { erfc_tile_init(); }
template <Approx approx, Dst Slot>
ALWI void Erfc<approx, Slot>::call(uint32_t d0) const { erfc_tile(d0); }

template <Dst Slot> ALWI void Erfinv<Slot>::init() const { erfinv_tile_init(); }
template <Dst Slot> ALWI void Erfinv<Slot>::call(uint32_t d0) const { erfinv_tile(d0); }
template <Dst Slot> ALWI void I0<Slot>::init() const { i0_tile_init(); }
template <Dst Slot> ALWI void I0<Slot>::call(uint32_t d0) const { i0_tile(d0); }
template <Dst Slot> ALWI void I1<Slot>::init() const { i1_tile_init(); }
template <Dst Slot> ALWI void I1<Slot>::call(uint32_t d0) const { i1_tile(d0); }
template <Dst Slot> ALWI void Lgamma<Slot>::init() const { lgamma_stirling_tile_init(); }
template <Dst Slot> ALWI void Lgamma<Slot>::call(uint32_t d0) const { lgamma_stirling_tile(d0); }

// Aliases
template <uint32_t ICB, SfpuOutputPolicy O, SfpuDataFormatReconfig R, SfpuBatching B>
ALWI void sfpu_erf(uint32_t ocb, uint32_t num_tiles) { sfpu_op<ICB, O, R, B>(ocb, num_tiles, Erf<>{}); }
template <uint32_t ICB, SfpuOutputPolicy O, SfpuDataFormatReconfig R, SfpuBatching B>
ALWI void sfpu_erfc(uint32_t ocb, uint32_t num_tiles) { sfpu_op<ICB, O, R, B>(ocb, num_tiles, Erfc<>{}); }
template <uint32_t ICB, SfpuOutputPolicy O, SfpuDataFormatReconfig R, SfpuBatching B>
ALWI void sfpu_erfinv(uint32_t ocb, uint32_t num_tiles) { sfpu_op<ICB, O, R, B>(ocb, num_tiles, Erfinv<>{}); }
template <uint32_t ICB, SfpuOutputPolicy O, SfpuDataFormatReconfig R, SfpuBatching B>
ALWI void sfpu_i0(uint32_t ocb, uint32_t num_tiles) { sfpu_op<ICB, O, R, B>(ocb, num_tiles, I0<>{}); }
template <uint32_t ICB, SfpuOutputPolicy O, SfpuDataFormatReconfig R, SfpuBatching B>
ALWI void sfpu_i1(uint32_t ocb, uint32_t num_tiles) { sfpu_op<ICB, O, R, B>(ocb, num_tiles, I1<>{}); }
template <uint32_t ICB, SfpuOutputPolicy O, SfpuDataFormatReconfig R, SfpuBatching B>
ALWI void sfpu_lgamma(uint32_t ocb, uint32_t num_tiles) { sfpu_op<ICB, O, R, B>(ocb, num_tiles, Lgamma<>{}); }

}  // namespace compute_kernel_lib
