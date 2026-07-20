// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file eltwise_bcast.hpp
 * @brief Unary broadcast chain element and convenience entry point.
 */

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

namespace compute_kernel_lib {

template <BroadcastDim Dim, uint32_t Cb, InputSpec Input = input(), Dst DstSlot = Dst::D0>
struct UnaryBcast;

template <BroadcastDim Dim, uint32_t CbIn, uint32_t CbOut, InputSpec Input = input(), OutputSpec Output = output()>
ALWI void unary_bcast(EltwiseShape shape);

}  // namespace compute_kernel_lib

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_bcast.inl"
