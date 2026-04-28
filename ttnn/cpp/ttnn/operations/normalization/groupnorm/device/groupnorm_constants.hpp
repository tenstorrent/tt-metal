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
// This value must match exactly across:
//   * the reader kernels' per-slot stride
//     (`l1_write_addr_external += cb_ex_external_slot_pitch_bytes`):
//       device/kernels/dataflow/reader_mcast_sender_unary_gn.cpp
//       device/kernels/dataflow/reader_mcast_sender_unary_sharded_gn_v2.cpp
//   * the compute kernel's cb_ex_external_tiles_required sizing:
//       device/kernels/compute/groupnorm.cpp
//   * the host program factory's cb_ex_external CB size (legacy path only;
//     the welford path requests a single-tile cb_ex_external as a placeholder
//     and never reads/writes it):
//       device/groupnorm_mcast_program_factory.cpp
//
// Each producer also has a
// `static_assert(datum_size_bytes <= cb_ex_external_slot_pitch_bytes)` so a
// wider datum will fail the build instead of silently corrupting reductions.
// Raising this above 16 to support a wider datum (e.g. fp32) requires
// updating all of the above call sites together.
inline constexpr std::uint32_t cb_ex_external_slot_pitch_bytes = 16;
