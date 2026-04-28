// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/where.h"
#include "api/compute/eltwise_unary/lerp.h"
#include "api/compute/eltwise_unary/addcmul.h"
#include "api/compute/eltwise_unary/addcdiv.h"

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

/**
 * @file eltwise_ternary.hpp
 * @brief Tier 2 ternary chain elements: Where, Lerp, Addcmul, Addcdiv.
 *
 * Each consumes three DEST slots + writes one. `DataFormat` template arg
 * is required by the underlying LLK ternary tile fns.
 */

namespace compute_kernel_lib::eltwise {

using namespace ckernel;

template <
    DataFormat DF = DataFormat::Float16_b,
    Dst In0 = Dst::D0,
    Dst In1 = Dst::D1,
    Dst In2 = Dst::D2,
    Dst Out = Dst::D0>
struct Where : TernaryOp<Where<DF, In0, In1, In2, Out>, In0, In1, In2, Out> {
    ALWI void init() const { where_tile_init(); }
    ALWI void call(uint32_t a, uint32_t b, uint32_t c, uint32_t out_) const { where_tile<DF>(a, b, c, out_); }
};

template <
    DataFormat DF = DataFormat::Float16_b,
    Dst In0 = Dst::D0,
    Dst In1 = Dst::D1,
    Dst In2 = Dst::D2,
    Dst Out = Dst::D0>
struct Lerp : TernaryOp<Lerp<DF, In0, In1, In2, Out>, In0, In1, In2, Out> {
    ALWI void init() const { lerp_tile_init(); }
    ALWI void call(uint32_t a, uint32_t b, uint32_t c, uint32_t out_) const { lerp_tile<DF>(a, b, c, out_); }
};

template <
    DataFormat DF = DataFormat::Float16_b,
    Dst In0 = Dst::D0,
    Dst In1 = Dst::D1,
    Dst In2 = Dst::D2,
    Dst Out = Dst::D0>
struct Addcmul : TernaryOp<Addcmul<DF, In0, In1, In2, Out>, In0, In1, In2, Out> {
    uint32_t value;
    ALWI void init() const { addcmul_tile_init(); }
    ALWI void call(uint32_t a, uint32_t b, uint32_t c, uint32_t out_) const { addcmul_tile<DF>(a, b, c, out_, value); }
};

template <
    DataFormat DF = DataFormat::Float16_b,
    Dst In0 = Dst::D0,
    Dst In1 = Dst::D1,
    Dst In2 = Dst::D2,
    Dst Out = Dst::D0>
struct Addcdiv : TernaryOp<Addcdiv<DF, In0, In1, In2, Out>, In0, In1, In2, Out> {
    uint32_t value;
    ALWI void init() const { addcdiv_tile_init(); }
    ALWI void call(uint32_t a, uint32_t b, uint32_t c, uint32_t out_) const { addcdiv_tile<DF>(a, b, c, out_, value); }
};

}  // namespace compute_kernel_lib::eltwise
