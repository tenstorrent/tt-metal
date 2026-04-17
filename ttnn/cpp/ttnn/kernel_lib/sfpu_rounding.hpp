// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "ttnn/cpp/ttnn/kernel_lib/sfpu_chain.hpp"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rounding.h"

namespace compute_kernel_lib {

// --- Rounding ---

template <Dst Slot = Dst::D0>
struct Floor : UnaryOp<Floor<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Ceil : UnaryOp<Ceil<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Trunc : UnaryOp<Trunc<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Round : UnaryOp<Round<Slot>, Slot> {
    int32_t decimals;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Frac : UnaryOp<Frac<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct StochasticRound : UnaryOp<StochasticRound<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

// --- Rounding aliases ---
template <
    uint32_t ICB,
    SfpuBatching B = SfpuBatching::Auto,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_floor(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuBatching B = SfpuBatching::Auto,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_ceil(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuBatching B = SfpuBatching::Auto,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_trunc(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuBatching B = SfpuBatching::Auto,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_frac(uint32_t ocb, uint32_t num_tiles);

}  // namespace compute_kernel_lib

#include "sfpu_rounding.inl"
