// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace compute_kernel_lib {
using namespace ckernel;

template <Dst Slot> ALWI void Floor<Slot>::init() const { rounding_op_tile_init(); }
template <Dst Slot> ALWI void Floor<Slot>::call(uint32_t d0) const { floor_tile(d0); }
template <Dst Slot> ALWI void Ceil<Slot>::init() const { rounding_op_tile_init(); }
template <Dst Slot> ALWI void Ceil<Slot>::call(uint32_t d0) const { ceil_tile(d0); }
template <Dst Slot> ALWI void Trunc<Slot>::init() const { rounding_op_tile_init(); }
template <Dst Slot> ALWI void Trunc<Slot>::call(uint32_t d0) const { trunc_tile(d0); }
template <Dst Slot> ALWI void Round<Slot>::init() const { rounding_op_tile_init(); }
template <Dst Slot> ALWI void Round<Slot>::call(uint32_t d0) const { round_tile(d0, decimals); }
template <Dst Slot> ALWI void Frac<Slot>::init() const { rounding_op_tile_init(); }
template <Dst Slot> ALWI void Frac<Slot>::call(uint32_t d0) const { frac_tile(d0); }
template <Dst Slot> ALWI void StochasticRound<Slot>::init() const { rounding_op_tile_init(); }
template <Dst Slot> ALWI void StochasticRound<Slot>::call(uint32_t d0) const { stochastic_round_tile(d0); }

// Aliases
template <uint32_t ICB, SfpuBatching B, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_floor(uint32_t ocb, uint32_t num_tiles) { sfpu_op<ICB, B, P, O, R>(ocb, num_tiles, Floor<>{}); }
template <uint32_t ICB, SfpuBatching B, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_ceil(uint32_t ocb, uint32_t num_tiles) { sfpu_op<ICB, B, P, O, R>(ocb, num_tiles, Ceil<>{}); }
template <uint32_t ICB, SfpuBatching B, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_trunc(uint32_t ocb, uint32_t num_tiles) { sfpu_op<ICB, B, P, O, R>(ocb, num_tiles, Trunc<>{}); }
template <uint32_t ICB, SfpuBatching B, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_frac(uint32_t ocb, uint32_t num_tiles) { sfpu_op<ICB, B, P, O, R>(ocb, num_tiles, Frac<>{}); }

}  // namespace compute_kernel_lib
