// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/rounding.h"

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

/**
 * @file eltwise_rounding.hpp
 * @brief Tier 2 rounding: Floor, Ceil, Trunc, Round, Frac, StochasticRound.
 *
 * All five share the same `rounding_op_tile_init()` (programs the rounding
 * SFPU op) and dispatch to their per-op `*_tile`. `Round` carries a runtime
 * `decimals` int.
 */

namespace compute_kernel_lib::eltwise {

using namespace ckernel;

template <Dst Slot = Dst::D0>
struct Floor : UnaryOp<Floor<Slot>, Slot> {
    ALWI void init() const { rounding_op_tile_init(); }
    ALWI void call(uint32_t d) const { floor_tile(d); }
};

template <Dst Slot = Dst::D0>
struct Ceil : UnaryOp<Ceil<Slot>, Slot> {
    ALWI void init() const { rounding_op_tile_init(); }
    ALWI void call(uint32_t d) const { ceil_tile(d); }
};

template <Dst Slot = Dst::D0>
struct Trunc : UnaryOp<Trunc<Slot>, Slot> {
    ALWI void init() const { rounding_op_tile_init(); }
    ALWI void call(uint32_t d) const { trunc_tile(d); }
};

template <Dst Slot = Dst::D0>
struct Round : UnaryOp<Round<Slot>, Slot> {
    int32_t decimals = 0;
    ALWI void init() const { rounding_op_tile_init(); }
    ALWI void call(uint32_t d) const { round_tile(d, decimals); }
};

template <Dst Slot = Dst::D0>
struct Frac : UnaryOp<Frac<Slot>, Slot> {
    ALWI void init() const { rounding_op_tile_init(); }
    ALWI void call(uint32_t d) const { frac_tile(d); }
};

template <Dst Slot = Dst::D0>
struct StochasticRound : UnaryOp<StochasticRound<Slot>, Slot> {
    ALWI void init() const { rounding_op_tile_init(); }
    ALWI void call(uint32_t d) const { stochastic_round_tile(d); }
};

}  // namespace compute_kernel_lib::eltwise
