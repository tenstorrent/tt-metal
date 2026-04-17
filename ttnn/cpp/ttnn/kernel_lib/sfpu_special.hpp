// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "ttnn/cpp/ttnn/kernel_lib/sfpu_chain.hpp"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/erf_erfc.h"
#include "api/compute/eltwise_unary/erfinv.h"
#include "api/compute/eltwise_unary/i0.h"
#include "api/compute/eltwise_unary/i1.h"
#include "api/compute/eltwise_unary/lgamma.h"

namespace compute_kernel_lib {

// --- Error / Special Functions ---

template <Approx approx = Approx::Fast, Dst Slot = Dst::D0>
struct Erf : UnaryOp<Erf<approx, Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Approx approx = Approx::Fast, Dst Slot = Dst::D0>
struct Erfc : UnaryOp<Erfc<approx, Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Erfinv : UnaryOp<Erfinv<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct I0 : UnaryOp<I0<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct I1 : UnaryOp<I1<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Lgamma : UnaryOp<Lgamma<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

// --- Special function aliases ---
template <
    uint32_t ICB,
    SfpuBatching B = SfpuBatching::Auto,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_erf(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuBatching B = SfpuBatching::Auto,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_erfc(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuBatching B = SfpuBatching::Auto,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_erfinv(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuBatching B = SfpuBatching::Auto,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_i0(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuBatching B = SfpuBatching::Auto,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_i1(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuBatching B = SfpuBatching::Auto,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_lgamma(uint32_t ocb, uint32_t num_tiles);

}  // namespace compute_kernel_lib

#include "sfpu_special.inl"
