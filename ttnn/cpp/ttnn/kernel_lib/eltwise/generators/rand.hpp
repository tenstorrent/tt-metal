// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file rand.hpp
 * @brief Rand chain elements — RandTile and RandKeyedTile.
 */

#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"

namespace compute_kernel_lib {

template <Dst DstSlot>
struct RandTile;

template <Dst DstSlot>
struct RandKeyedTile;

}  // namespace compute_kernel_lib

#include "ttnn/cpp/ttnn/kernel_lib/eltwise/generators/rand.inl"
