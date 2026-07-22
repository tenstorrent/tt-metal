// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file eltwise_bcast.hpp
 * @brief Unary broadcast chain element and convenience entry point.
 */

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

namespace compute_kernel_lib {

namespace detail {

constexpr uint32_t unary_bcast_config_bits(BroadcastDim dim, InputSpec input_spec, Dst dst) noexcept;

template <uint32_t Cb, uint32_t ConfigBits>
struct UnaryBcastImpl;

}  // namespace detail

template <BroadcastDim Dim, InputSpec Input, Dst DstSlot = Dst::D0>
using UnaryBcast = detail::UnaryBcastImpl<Input.cb_id, detail::unary_bcast_config_bits(Dim, Input, DstSlot)>;

template <BroadcastDim Dim, InputSpec Input, OutputSpec Output>
ALWI void unary_bcast(EltwiseShape shape);

}  // namespace compute_kernel_lib

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_bcast.inl"
