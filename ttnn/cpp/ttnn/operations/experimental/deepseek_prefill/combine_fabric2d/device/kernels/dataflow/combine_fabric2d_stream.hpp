// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The stream encoding, shared by the host and the kernels: the share a stream carries is decoded from it on
// both sides, so it has one definition.

#include <cstdint>

namespace cmbf2d {

// A stream is one routing plane travelled in one direction along the ring axis. Every chip runs one
// reader+sender pair per stream, and a stream keeps its identity across chips: the pair on the next chip
// with the same id continues in the same direction on the same plane.
using StreamId = uint32_t;

constexpr StreamId make_stream_id(uint32_t link_idx, bool is_cw) { return link_idx * 2 + (is_cw ? 0u : 1u); }
constexpr bool stream_is_cw(StreamId stream) { return stream % 2 == 0; }
constexpr uint32_t stream_link(StreamId stream) { return stream / 2; }
constexpr uint32_t stream_count(uint32_t num_links) { return num_links * 2; }

// The dispatch-group index `hops` chips from `dg_index` in the direction `stream` travels; negative `hops`
// goes upstream. |hops| must not exceed the ring extent.
constexpr uint32_t ring_step(StreamId stream, uint32_t dg_index, int32_t hops, uint32_t ring_extent) {
    const int32_t step = stream_is_cw(stream) ? hops : -hops;
    return static_cast<uint32_t>(static_cast<int32_t>(dg_index + ring_extent) + step) % ring_extent;
}

}  // namespace cmbf2d
