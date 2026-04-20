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

// --- Mask (in-place zero-mask): data at DataSlot, mask at DataSlot+1 ---
template <DataFormat DF, Dst DataSlot>
ALWI void Mask<DF, DataSlot>::init() const { mask_tile_init(); }
template <DataFormat DF, Dst DataSlot>
ALWI void Mask<DF, DataSlot>::call(uint32_t data, uint32_t mask, uint32_t /*out = data*/) const {
    // The LLK ignores `mask` and reads from `data + 1`; we still pass it for
    // parity with the raw API and for future-proofing (see mask.h TODO).
    mask_tile(data, mask, DF);
}

// --- MaskPosInf (in-place +inf-mask): data at DataSlot, mask at DataSlot+1 ---
template <Dst DataSlot>
ALWI void MaskPosInf<DataSlot>::init() const { mask_tile_init(); }
template <Dst DataSlot>
ALWI void MaskPosInf<DataSlot>::call(uint32_t data, uint32_t mask, uint32_t /*out = data*/) const {
    mask_posinf_tile(data, mask);
}

}  // namespace compute_kernel_lib
