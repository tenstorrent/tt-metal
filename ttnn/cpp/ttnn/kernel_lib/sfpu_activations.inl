// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace compute_kernel_lib {
using namespace ckernel;

template <Approx approx, Dst Slot>
ALWI void Sigmoid<approx, Slot>::init() const { sigmoid_tile_init<static_cast<bool>(approx)>(); }
template <Approx approx, Dst Slot>
ALWI void Sigmoid<approx, Slot>::call(uint32_t d0) const { sigmoid_tile<(int)VectorMode::RC, static_cast<bool>(approx)>(d0); }

template <Approx approx, Dst Slot>
ALWI void Tanh<approx, Slot>::init() const { tanh_tile_init<static_cast<bool>(approx)>(); }
template <Approx approx, Dst Slot>
ALWI void Tanh<approx, Slot>::call(uint32_t d0) const { tanh_tile<static_cast<bool>(approx)>(d0); }

template <Approx approx, Dst Slot>
ALWI void Gelu<approx, Slot>::init() const { gelu_tile_init<static_cast<bool>(approx)>(); }
template <Approx approx, Dst Slot>
ALWI void Gelu<approx, Slot>::call(uint32_t d0) const { gelu_tile<static_cast<bool>(approx)>(d0); }

template <Dst Slot>
ALWI void Silu<Slot>::init() const { silu_tile_init(); }
template <Dst Slot>
ALWI void Silu<Slot>::call(uint32_t d0) const { silu_tile(d0); }

template <Dst Slot>
ALWI void Relu<Slot>::init() const { relu_tile_init(); }
template <Dst Slot>
ALWI void Relu<Slot>::call(uint32_t d0) const { relu_tile(d0); }

template <Dst Slot>
ALWI void Hardmish<Slot>::init() const { hardmish_tile_init(); }
template <Dst Slot>
ALWI void Hardmish<Slot>::call(uint32_t d0) const { hardmish_tile(d0); }

template <Dst Slot>
ALWI void Hardsigmoid<Slot>::init() const { hardsigmoid_tile_init(); }
template <Dst Slot>
ALWI void Hardsigmoid<Slot>::call(uint32_t d0) const { hardsigmoid_tile(d0); }

template <Dst Slot>
ALWI void Hardtanh<Slot>::init() const { hardtanh_tile_init(); }
template <Dst Slot>
ALWI void Hardtanh<Slot>::call(uint32_t d0) const { hardtanh_tile(d0, param_min, param_max); }

template <Dst Slot>
ALWI void Softsign<Slot>::init() const { softsign_tile_init(); }
template <Dst Slot>
ALWI void Softsign<Slot>::call(uint32_t d0) const { softsign_tile(d0); }

template <Dst Slot>
ALWI void Softplus<Slot>::init() const { softplus_tile_init(); }
template <Dst Slot>
ALWI void Softplus<Slot>::call(uint32_t d0) const { softplus_tile(d0, beta, beta_recip, threshold); }

template <Dst Slot>
ALWI void Xielu<Slot>::init() const { xielu_tile_init(); }
template <Dst Slot>
ALWI void Xielu<Slot>::call(uint32_t d0) const { xielu_tile(d0, alpha_p, alpha_n); }

template <Dst Slot> ALWI void Elu<Slot>::init() const { elu_tile_init(); }
template <Dst Slot> ALWI void Elu<Slot>::call(uint32_t d0) const { elu_tile(d0, alpha); }
template <Dst Slot> ALWI void Selu<Slot>::init() const { selu_tile_init(); }
template <Dst Slot> ALWI void Selu<Slot>::call(uint32_t d0) const { selu_tile(d0, scale, alpha); }
template <Dst Slot> ALWI void Celu<Slot>::init() const { celu_tile_init(); }
template <Dst Slot> ALWI void Celu<Slot>::call(uint32_t d0) const { celu_tile(d0, alpha, alpha_recip); }
template <Dst Slot> ALWI void Softshrink<Slot>::init() const { softshrink_tile_init(); }
template <Dst Slot> ALWI void Softshrink<Slot>::call(uint32_t d0) const { softshrink_tile(d0, lambda); }
template <Dst Slot> ALWI void Clamp<Slot>::init() const { clamp_tile_init(); }
template <Dst Slot> ALWI void Clamp<Slot>::call(uint32_t d0) const { clamp_tile(d0, param_min, param_max); }
template <Dst Slot> ALWI void Threshold<Slot>::init() const { threshold_tile_init(); }
template <Dst Slot> ALWI void Threshold<Slot>::call(uint32_t d0) const { threshold_tile(d0, threshold, value); }
template <Dst Slot> ALWI void Prelu<Slot>::init() const { prelu_tile_init(); }
template <Dst Slot> ALWI void Prelu<Slot>::call(uint32_t d0) const { prelu_tile(d0, weight); }

// Aliases
template <uint32_t ICB, SfpuBatching B, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_sigmoid(uint32_t ocb, uint32_t num_tiles) { sfpu_op<ICB, B, P, O, R>(ocb, num_tiles, Sigmoid<>{}); }
template <uint32_t ICB, SfpuBatching B, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_tanh(uint32_t ocb, uint32_t num_tiles) { sfpu_op<ICB, B, P, O, R>(ocb, num_tiles, Tanh<>{}); }
template <uint32_t ICB, SfpuBatching B, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_gelu(uint32_t ocb, uint32_t num_tiles) { sfpu_op<ICB, B, P, O, R>(ocb, num_tiles, Gelu<>{}); }
template <uint32_t ICB, SfpuBatching B, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_silu(uint32_t ocb, uint32_t num_tiles) { sfpu_op<ICB, B, P, O, R>(ocb, num_tiles, Silu<>{}); }
template <uint32_t ICB, SfpuBatching B, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_relu(uint32_t ocb, uint32_t num_tiles) { sfpu_op<ICB, B, P, O, R>(ocb, num_tiles, Relu<>{}); }

}  // namespace compute_kernel_lib
