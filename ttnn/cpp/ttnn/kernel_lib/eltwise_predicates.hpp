// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/comp.h"
#include "api/compute/eltwise_unary/isinf_isnan.h"
#include "api/compute/eltwise_unary/logical_not.h"

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

/**
 * @file eltwise_predicates.hpp
 * @brief Tier 1 predicates: Eqz, Gtz.
 */

namespace compute_kernel_lib::eltwise {

using namespace ckernel;

template <Dst Slot = Dst::D0>
struct Eqz : UnaryOp<Eqz<Slot>, Slot> {
    ALWI void init() const { eqz_tile_init(); }
    ALWI void call(uint32_t d) const { eqz_tile(d); }
};

template <Dst Slot = Dst::D0>
struct Gtz : UnaryOp<Gtz<Slot>, Slot> {
    ALWI void init() const { gtz_tile_init(); }
    ALWI void call(uint32_t d) const { gtz_tile(d); }
};

template <Dst Slot = Dst::D0>
struct Ltz : UnaryOp<Ltz<Slot>, Slot> {
    ALWI void init() const { ltz_tile_init(); }
    ALWI void call(uint32_t d) const { ltz_tile(d); }
};

template <Dst Slot = Dst::D0>
struct Lez : UnaryOp<Lez<Slot>, Slot> {
    ALWI void init() const { lez_tile_init(); }
    ALWI void call(uint32_t d) const { lez_tile(d); }
};

template <Dst Slot = Dst::D0>
struct Gez : UnaryOp<Gez<Slot>, Slot> {
    ALWI void init() const { gez_tile_init(); }
    ALWI void call(uint32_t d) const { gez_tile(d); }
};

template <Dst Slot = Dst::D0>
struct Nez : UnaryOp<Nez<Slot>, Slot> {
    ALWI void init() const { nez_tile_init(); }
    ALWI void call(uint32_t d) const { nez_tile(d); }
};

// ---- isinf / isnan family ----

template <Dst Slot = Dst::D0>
struct Isinf : UnaryOp<Isinf<Slot>, Slot> {
    ALWI void init() const { isinf_tile_init(); }
    ALWI void call(uint32_t d) const { isinf_tile(d); }
};

template <Dst Slot = Dst::D0>
struct Isposinf : UnaryOp<Isposinf<Slot>, Slot> {
    ALWI void init() const { isposinf_tile_init(); }
    ALWI void call(uint32_t d) const { isposinf_tile(d); }
};

template <Dst Slot = Dst::D0>
struct Isneginf : UnaryOp<Isneginf<Slot>, Slot> {
    ALWI void init() const { isneginf_tile_init(); }
    ALWI void call(uint32_t d) const { isneginf_tile(d); }
};

template <Dst Slot = Dst::D0>
struct Isnan : UnaryOp<Isnan<Slot>, Slot> {
    ALWI void init() const { isnan_tile_init(); }
    ALWI void call(uint32_t d) const { isnan_tile(d); }
};

template <Dst Slot = Dst::D0>
struct Isfinite : UnaryOp<Isfinite<Slot>, Slot> {
    ALWI void init() const { isfinite_tile_init(); }
    ALWI void call(uint32_t d) const { isfinite_tile(d); }
};

template <DataFormat DF = DataFormat::Float16_b, Dst Slot = Dst::D0>
struct LogicalNot : UnaryOp<LogicalNot<DF, Slot>, Slot> {
    ALWI void init() const { logical_not_tile_init(); }
    ALWI void call(uint32_t d) const { logical_not_tile<DF>(d); }
};

// ---- unary comparison vs scalar ----

template <Dst Slot = Dst::D0>
struct UnaryEq : UnaryOp<UnaryEq<Slot>, Slot> {
    uint32_t param0;
    ALWI void init() const { unary_eq_tile_init(); }
    ALWI void call(uint32_t d) const { unary_eq_tile(d, param0); }
};

template <Dst Slot = Dst::D0>
struct UnaryNe : UnaryOp<UnaryNe<Slot>, Slot> {
    uint32_t param0;
    ALWI void init() const { unary_ne_tile_init(); }
    ALWI void call(uint32_t d) const { unary_ne_tile(d, param0); }
};

template <Dst Slot = Dst::D0>
struct UnaryGt : UnaryOp<UnaryGt<Slot>, Slot> {
    uint32_t param0;
    ALWI void init() const { unary_gt_tile_init(); }
    ALWI void call(uint32_t d) const { unary_gt_tile(d, param0); }
};

template <Dst Slot = Dst::D0>
struct UnaryGe : UnaryOp<UnaryGe<Slot>, Slot> {
    uint32_t param0;
    ALWI void init() const { unary_ge_tile_init(); }
    ALWI void call(uint32_t d) const { unary_ge_tile(d, param0); }
};

template <Dst Slot = Dst::D0>
struct UnaryLt : UnaryOp<UnaryLt<Slot>, Slot> {
    uint32_t param0;
    ALWI void init() const { unary_lt_tile_init(); }
    ALWI void call(uint32_t d) const { unary_lt_tile(d, param0); }
};

template <Dst Slot = Dst::D0>
struct UnaryLe : UnaryOp<UnaryLe<Slot>, Slot> {
    uint32_t param0;
    ALWI void init() const { unary_le_tile_init(); }
    ALWI void call(uint32_t d) const { unary_le_tile(d, param0); }
};

}  // namespace compute_kernel_lib::eltwise
