// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// The entries of the table compact_runtime_args (hybrid_half_merge.cpp) writes and hybrid_expand_runtime_args
// (hybrid_rt_expand.hpp) reads, one per original runtime-argument index: where the value for this core lives. The
// top three bits are the kind, the rest a slot -- the core's own runtime argument for kHybridRtPerCore, a common
// argument for every other kind.

#pragma once

#include <cstdint>

inline constexpr uint32_t kHybridRtKindShift = 29;
inline constexpr uint32_t kHybridRtSlotMask = (1u << kHybridRtKindShift) - 1;

// The core's own runtime argument.
inline constexpr uint32_t kHybridRtPerCore = 0;
// A value every core shares.
inline constexpr uint32_t kHybridRtShared = 1;
// A table of one value per logical row of cores, indexed by the core's logical y.
inline constexpr uint32_t kHybridRtPerRow = 2;
// A table of one value per logical column of cores, indexed by the core's logical x.
inline constexpr uint32_t kHybridRtPerColumn = 3;
// Two values a, b: a + b * the core's logical y, in uint32 arithmetic.
inline constexpr uint32_t kHybridRtLinearInRow = 4;
// Two values a, b: a + b * the core's logical x, in uint32 arithmetic.
inline constexpr uint32_t kHybridRtLinearInColumn = 5;

inline constexpr uint32_t hybrid_rt_entry(uint32_t kind, uint32_t slot) { return (kind << kHybridRtKindShift) | slot; }
