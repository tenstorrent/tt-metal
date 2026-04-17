// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace compute_kernel_lib {
using namespace ckernel;

template <Dst In0, Dst In1, Dst Out>
ALWI void SfpuAdd<In0, In1, Out>::init() const { add_binary_tile_init(); }
template <Dst In0, Dst In1, Dst Out>
ALWI void SfpuAdd<In0, In1, Out>::call(uint32_t a, uint32_t b, uint32_t c) const { add_binary_tile(a, b, c); }

template <Dst In0, Dst In1, Dst Out>
ALWI void SfpuSub<In0, In1, Out>::init() const { sub_binary_tile_init(); }
template <Dst In0, Dst In1, Dst Out>
ALWI void SfpuSub<In0, In1, Out>::call(uint32_t a, uint32_t b, uint32_t c) const { sub_binary_tile(a, b, c); }

template <Dst In0, Dst In1, Dst Out>
ALWI void SfpuMul<In0, In1, Out>::init() const { mul_binary_tile_init(); }
template <Dst In0, Dst In1, Dst Out>
ALWI void SfpuMul<In0, In1, Out>::call(uint32_t a, uint32_t b, uint32_t c) const { mul_binary_tile(a, b, c); }

template <Dst In0, Dst In1, Dst Out>
ALWI void SfpuDiv<In0, In1, Out>::init() const { div_binary_tile_init(); }
template <Dst In0, Dst In1, Dst Out>
ALWI void SfpuDiv<In0, In1, Out>::call(uint32_t a, uint32_t b, uint32_t c) const { div_binary_tile(a, b, c); }

template <Dst In0, Dst In1, Dst Out>
ALWI void SfpuRsub<In0, In1, Out>::init() const { rsub_binary_tile_init(); }
template <Dst In0, Dst In1, Dst Out>
ALWI void SfpuRsub<In0, In1, Out>::call(uint32_t a, uint32_t b, uint32_t c) const { rsub_binary_tile(a, b, c); }

template <Dst In0, Dst In1, Dst Out>
ALWI void SfpuPow<In0, In1, Out>::init() const { power_binary_tile_init(); }
template <Dst In0, Dst In1, Dst Out>
ALWI void SfpuPow<In0, In1, Out>::call(uint32_t a, uint32_t b, uint32_t c) const { power_binary_tile(a, b, c); }

template <Dst In0, Dst In1, Dst Out>
ALWI void SfpuEq<In0, In1, Out>::init() const { eq_binary_tile_init(); }
template <Dst In0, Dst In1, Dst Out>
ALWI void SfpuEq<In0, In1, Out>::call(uint32_t a, uint32_t b, uint32_t c) const { eq_binary_tile(a, b, c); }

template <Dst In0, Dst In1, Dst Out>
ALWI void SfpuMax<In0, In1, Out>::init() const { binary_max_tile_init(); }
template <Dst In0, Dst In1, Dst Out>
ALWI void SfpuMax<In0, In1, Out>::call(uint32_t a, uint32_t b, uint32_t c) const { binary_max_tile(a, b, c); }

template <Dst In0, Dst In1, Dst Out>
ALWI void SfpuMin<In0, In1, Out>::init() const { binary_min_tile_init(); }
template <Dst In0, Dst In1, Dst Out>
ALWI void SfpuMin<In0, In1, Out>::call(uint32_t a, uint32_t b, uint32_t c) const { binary_min_tile(a, b, c); }

}  // namespace compute_kernel_lib
