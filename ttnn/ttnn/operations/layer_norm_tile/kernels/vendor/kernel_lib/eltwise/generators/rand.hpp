// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file rand.hpp
 * @brief Rand chain element — RandTile.
 */

#include "../api/chain.hpp"

namespace compute_kernel_lib {

template <Dst DstSlot>
struct RandTile;

}  // namespace compute_kernel_lib

#include "rand.inl"
