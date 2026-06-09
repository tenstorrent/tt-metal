// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

namespace compute_kernel_lib {

template <Dst Slot = Dst::D0>
struct Floor;
template <Dst Slot = Dst::D0>
struct Ceil;
template <Dst Slot = Dst::D0>
struct Trunc;
template <Dst Slot = Dst::D0>
struct Frac;
// Round — runtime decimals.
template <Dst Slot = Dst::D0>
struct Round;

}  // namespace compute_kernel_lib

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_rounding.inl"
