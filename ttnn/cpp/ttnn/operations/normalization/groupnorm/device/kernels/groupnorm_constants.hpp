// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Pitch (in bytes) between consecutive per-core slots in the cb_ex_external
// circular buffer used by the legacy mcast group-norm path. Each slot holds
// one core's partial-reduction scalar (datum_size_bytes wide) gathered by the
// sender from every core in the mcast group; the downstream `reduce_tile` SUM
// consumer treats every byte of the reserved tiles as part of the reduction,
// so all gap bytes inside a slot (positions [datum_size_bytes, slot_pitch))
// and any trailing bytes in the last reserved tile must stay zero.
//
// To ensure consistency:
//   * Any code that strides between per-core slots, or sizes cb_ex_external
//     to hold N slots, should use this constant (rather than e.g. a literal 16).
//   * Any producer writing into a slot should static_assert
//     `datum_size_bytes <= cb_ex_external_slot_pitch_bytes` so a too-wide
//     datum fails the build instead of silently corrupting the reduction.
//   * Raising this value to support a wider datum (e.g. fp32) requires
//     auditing every consumer of the constant, since SRAM footprint and tile
//     packing change in lockstep.
inline constexpr std::uint32_t cb_ex_external_slot_pitch_bytes = 16;
