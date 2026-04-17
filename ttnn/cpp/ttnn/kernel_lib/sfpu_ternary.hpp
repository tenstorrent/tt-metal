// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "ttnn/cpp/ttnn/kernel_lib/sfpu_chain.hpp"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/where.h"
#include "api/compute/eltwise_unary/lerp.h"
#include "api/compute/eltwise_unary/addcmul.h"
#include "api/compute/eltwise_unary/addcdiv.h"

namespace compute_kernel_lib {

// --- Ternary SFPU Ops ---

template <
    DataFormat df = DataFormat::Float16_b,
    Dst In0 = Dst::D0,
    Dst In1 = Dst::D1,
    Dst In2 = Dst::D2,
    Dst Out = Dst::D0>
struct Where : TernaryOp<Where<df, In0, In1, In2, Out>, In0, In1, In2, Out> {
    ALWI void init() const;
    ALWI void call(uint32_t a, uint32_t b, uint32_t c, uint32_t d) const;
};

template <
    DataFormat df = DataFormat::Float16_b,
    Dst In0 = Dst::D0,
    Dst In1 = Dst::D1,
    Dst In2 = Dst::D2,
    Dst Out = Dst::D0>
struct Lerp : TernaryOp<Lerp<df, In0, In1, In2, Out>, In0, In1, In2, Out> {
    ALWI void init() const;
    ALWI void call(uint32_t a, uint32_t b, uint32_t c, uint32_t d) const;
};

template <
    DataFormat df = DataFormat::Float16_b,
    Dst In0 = Dst::D0,
    Dst In1 = Dst::D1,
    Dst In2 = Dst::D2,
    Dst Out = Dst::D0>
struct Addcmul : TernaryOp<Addcmul<df, In0, In1, In2, Out>, In0, In1, In2, Out> {
    uint32_t value;
    ALWI void init() const;
    ALWI void call(uint32_t a, uint32_t b, uint32_t c, uint32_t d) const;
};

template <
    DataFormat df = DataFormat::Float16_b,
    Dst In0 = Dst::D0,
    Dst In1 = Dst::D1,
    Dst In2 = Dst::D2,
    Dst Out = Dst::D0>
struct Addcdiv : TernaryOp<Addcdiv<df, In0, In1, In2, Out>, In0, In1, In2, Out> {
    uint32_t value;
    ALWI void init() const;
    ALWI void call(uint32_t a, uint32_t b, uint32_t c, uint32_t d) const;
};

}  // namespace compute_kernel_lib

#include "sfpu_ternary.inl"
