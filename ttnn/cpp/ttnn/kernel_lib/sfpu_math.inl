// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace compute_kernel_lib {
using namespace ckernel;

template <Approx approx, Approx fast, Dst Slot>
ALWI void Exp<approx, fast, Slot>::init() const { exp_tile_init<static_cast<bool>(approx), static_cast<bool>(fast)>(); }
template <Approx approx, Approx fast, Dst Slot>
ALWI void Exp<approx, fast, Slot>::call(uint32_t d0) const { exp_tile<static_cast<bool>(approx), static_cast<bool>(fast)>(d0); }

template <Approx approx, Dst Slot>
ALWI void Log<approx, Slot>::init() const { log_tile_init<static_cast<bool>(approx)>(); }
template <Approx approx, Dst Slot>
ALWI void Log<approx, Slot>::call(uint32_t d0) const { log_tile<static_cast<bool>(approx)>(d0); }

template <Dst Slot>
ALWI void LogWithBase<Slot>::init() const { log_with_base_tile_init(); }
template <Dst Slot>
ALWI void LogWithBase<Slot>::call(uint32_t d0) const { log_with_base_tile(d0, base_scale); }

template <Approx approx, Dst Slot>
ALWI void Log1p<approx, Slot>::init() const { log1p_tile_init<static_cast<bool>(approx)>(); }
template <Approx approx, Dst Slot>
ALWI void Log1p<approx, Slot>::call(uint32_t d0) const { log1p_tile<static_cast<bool>(approx)>(d0); }

template <Approx approx, Dst Slot>
ALWI void Sqrt<approx, Slot>::init() const { sqrt_tile_init(); }
template <Approx approx, Dst Slot>
ALWI void Sqrt<approx, Slot>::call(uint32_t d0) const { sqrt_tile<static_cast<bool>(approx)>(d0); }

template <Legacy legacy, Approx approx, Dst Slot>
ALWI void Rsqrt<legacy, approx, Slot>::init() const { rsqrt_tile_init<static_cast<bool>(legacy)>(); }
template <Legacy legacy, Approx approx, Dst Slot>
ALWI void Rsqrt<legacy, approx, Slot>::call(uint32_t d0) const { rsqrt_tile<static_cast<bool>(legacy), static_cast<bool>(approx)>(d0); }

template <Dst Slot>
ALWI void Cbrt<Slot>::init() const { cbrt_tile_init(); }
template <Dst Slot>
ALWI void Cbrt<Slot>::call(uint32_t d0) const { cbrt_tile(d0); }

template <Legacy legacy, Dst Slot>
ALWI void Recip<legacy, Slot>::init() const { recip_tile_init<static_cast<bool>(legacy)>(); }
template <Legacy legacy, Dst Slot>
ALWI void Recip<legacy, Slot>::call(uint32_t d0) const { recip_tile<static_cast<bool>(legacy)>(d0); }

template <Dst Slot>
ALWI void Abs<Slot>::init() const { abs_tile_init(); }
template <Dst Slot>
ALWI void Abs<Slot>::call(uint32_t d0) const { abs_tile(d0); }

template <Dst Slot>
ALWI void Neg<Slot>::init() const { negative_tile_init(); }
template <Dst Slot>
ALWI void Neg<Slot>::call(uint32_t d0) const { negative_tile(d0); }

template <Dst Slot>
ALWI void Square<Slot>::init() const { square_tile_init(); }
template <Dst Slot>
ALWI void Square<Slot>::call(uint32_t d0) const { square_tile(d0); }

template <Dst Slot>
ALWI void Sign<Slot>::init() const { sign_tile_init(); }
template <Dst Slot>
ALWI void Sign<Slot>::call(uint32_t d0) const { sign_tile(d0); }

template <Dst Slot>
ALWI void Signbit<Slot>::init() const { signbit_tile_init(); }
template <Dst Slot>
ALWI void Signbit<Slot>::call(uint32_t d0) const { signbit_tile(d0); }

template <Dst Slot>
ALWI void Exp2<Slot>::init() const { exp2_tile_init(); }
template <Dst Slot>
ALWI void Exp2<Slot>::call(uint32_t d0) const { exp2_tile(d0); }

template <Approx approx, Dst Slot>
ALWI void Expm1<approx, Slot>::init() const { expm1_tile_init<static_cast<bool>(approx)>(); }
template <Approx approx, Dst Slot>
ALWI void Expm1<approx, Slot>::call(uint32_t d0) const { expm1_tile<static_cast<bool>(approx)>(d0); }

template <Dst Slot>
ALWI void Power<Slot>::init() const { power_tile_init(); }
template <Dst Slot>
ALWI void Power<Slot>::call(uint32_t d0) const { power_tile(d0, exponent); }

template <Dst Slot>
ALWI void PowerIterative<Slot>::init() const { power_tile_init(); }
template <Dst Slot>
ALWI void PowerIterative<Slot>::call(uint32_t d0) const { power_tile(d0, int_exponent); }

template <Dst Slot>
ALWI void Rpow<Slot>::init() const { rpow_tile_init(); }
template <Dst Slot>
ALWI void Rpow<Slot>::call(uint32_t d0) const { rpow_tile(d0, base_val); }

template <Dst Slot>
ALWI void FillScalar<Slot>::init() const { fill_tile_init(); }
template <Dst Slot>
ALWI void FillScalar<Slot>::call(uint32_t d0) const { fill_tile(d0, value); }

template <uint32_t Bits, Dst Slot>
ALWI void FillConst<Bits, Slot>::init() const { fill_tile_init(); }
template <uint32_t Bits, Dst Slot>
ALWI void FillConst<Bits, Slot>::call(uint32_t d0) const { fill_tile_bitcast(d0, Bits); }

template <Approx approx, Dst Slot>
ALWI void TanhDerivative<approx, Slot>::init() const { tanh_derivative_tile_init<static_cast<bool>(approx)>(); }
template <Approx approx, Dst Slot>
ALWI void TanhDerivative<approx, Slot>::call(uint32_t d0) const { tanh_derivative_tile<static_cast<bool>(approx)>(d0); }

// Aliases
template <uint32_t ICB, SfpuOutputPolicy O, SfpuDataFormatReconfig R, SfpuBatching B>
ALWI void sfpu_exp(uint32_t ocb, uint32_t num_tiles) { sfpu_op<ICB, O, R, B>(ocb, num_tiles, Exp<>{}); }
template <uint32_t ICB, SfpuOutputPolicy O, SfpuDataFormatReconfig R, SfpuBatching B>
ALWI void sfpu_log(uint32_t ocb, uint32_t num_tiles) { sfpu_op<ICB, O, R, B>(ocb, num_tiles, Log<>{}); }
template <uint32_t ICB, SfpuOutputPolicy O, SfpuDataFormatReconfig R, SfpuBatching B>
ALWI void sfpu_log1p(uint32_t ocb, uint32_t num_tiles) { sfpu_op<ICB, O, R, B>(ocb, num_tiles, Log1p<>{}); }
template <uint32_t ICB, SfpuOutputPolicy O, SfpuDataFormatReconfig R, SfpuBatching B>
ALWI void sfpu_sqrt(uint32_t ocb, uint32_t num_tiles) { sfpu_op<ICB, O, R, B>(ocb, num_tiles, Sqrt<>{}); }
template <uint32_t ICB, SfpuOutputPolicy O, SfpuDataFormatReconfig R, SfpuBatching B>
ALWI void sfpu_rsqrt(uint32_t ocb, uint32_t num_tiles) { sfpu_op<ICB, O, R, B>(ocb, num_tiles, Rsqrt<>{}); }
template <uint32_t ICB, SfpuOutputPolicy O, SfpuDataFormatReconfig R, SfpuBatching B>
ALWI void sfpu_recip(uint32_t ocb, uint32_t num_tiles) { sfpu_op<ICB, O, R, B>(ocb, num_tiles, Recip<>{}); }
template <uint32_t ICB, SfpuOutputPolicy O, SfpuDataFormatReconfig R, SfpuBatching B>
ALWI void sfpu_abs(uint32_t ocb, uint32_t num_tiles) { sfpu_op<ICB, O, R, B>(ocb, num_tiles, Abs<>{}); }
template <uint32_t ICB, SfpuOutputPolicy O, SfpuDataFormatReconfig R, SfpuBatching B>
ALWI void sfpu_neg(uint32_t ocb, uint32_t num_tiles) { sfpu_op<ICB, O, R, B>(ocb, num_tiles, Neg<>{}); }

}  // namespace compute_kernel_lib
