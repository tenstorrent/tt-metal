// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file eltwise_fill.hpp
 * @brief Fill chain elements — FillScalar, FillInt, FillBitcast.
 *
 * These elements write a constant into a DEST slot. They derive `FillTileTag` (rooted in
 * `DestOnlyTag`) so trait sweeps that look at CB consumers / CB producers correctly skip them.
 * Each overrides `exec(uint32_t)` directly to capture the runtime constant.
 */

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

namespace compute_kernel_lib {

template <Dst DstSlot>
struct FillScalar;
template <DataFormat DF, Dst DstSlot>
struct FillInt;
template <Dst DstSlot>
struct FillBitcast;

}  // namespace compute_kernel_lib

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_fill.inl"
