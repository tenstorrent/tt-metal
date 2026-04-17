// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "ttnn/cpp/ttnn/kernel_lib/sfpu_chain.hpp"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/isinf_isnan.h"
#include "api/compute/eltwise_unary/logical_not.h"
#include "api/compute/eltwise_unary/comp.h"

namespace compute_kernel_lib {

// --- Predicates & Comparisons ---

template <Dst Slot = Dst::D0>
struct Isinf : UnaryOp<Isinf<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Isposinf : UnaryOp<Isposinf<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Isneginf : UnaryOp<Isneginf<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Isnan : UnaryOp<Isnan<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Isfinite : UnaryOp<Isfinite<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <DataFormat df = DataFormat::Float16_b, Dst Slot = Dst::D0>
struct LogicalNot : UnaryOp<LogicalNot<df, Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Gtz : UnaryOp<Gtz<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Ltz : UnaryOp<Ltz<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Lez : UnaryOp<Lez<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Gez : UnaryOp<Gez<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Eqz : UnaryOp<Eqz<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Nez : UnaryOp<Nez<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct UnaryEq : UnaryOp<UnaryEq<Slot>, Slot> {
    uint32_t param0;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct UnaryNe : UnaryOp<UnaryNe<Slot>, Slot> {
    uint32_t param0;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct UnaryGt : UnaryOp<UnaryGt<Slot>, Slot> {
    uint32_t param0;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct UnaryGe : UnaryOp<UnaryGe<Slot>, Slot> {
    uint32_t param0;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct UnaryLt : UnaryOp<UnaryLt<Slot>, Slot> {
    uint32_t param0;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct UnaryLe : UnaryOp<UnaryLe<Slot>, Slot> {
    uint32_t param0;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

// --- Predicate aliases ---
template <
    uint32_t ICB,
    SfpuBatching B = SfpuBatching::Auto,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_isinf(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuBatching B = SfpuBatching::Auto,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_isnan(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuBatching B = SfpuBatching::Auto,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_isfinite(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuBatching B = SfpuBatching::Auto,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_gtz(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuBatching B = SfpuBatching::Auto,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_ltz(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuBatching B = SfpuBatching::Auto,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_lez(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuBatching B = SfpuBatching::Auto,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_gez(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuBatching B = SfpuBatching::Auto,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_eqz(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuBatching B = SfpuBatching::Auto,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_nez(uint32_t ocb, uint32_t num_tiles);

}  // namespace compute_kernel_lib

#include "sfpu_predicates.inl"
