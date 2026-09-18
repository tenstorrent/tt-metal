// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
namespace kda_chronology {
namespace selection {
constexpr uint32_t record_width = 8;  // One aligned 32-byte UINT32 record.
constexpr uint32_t history_rows = 3;  // Four-tap convolution retains three preceding tokens.
constexpr uint32_t slice_rank = 4;
constexpr uint32_t outgoing_history = 0;
constexpr uint32_t predecessor_history = outgoing_history + 1;
constexpr uint32_t final_history = predecessor_history + 1;
constexpr uint32_t local_entry_state = final_history + 1;
constexpr uint32_t final_state = local_entry_state + 2;
constexpr uint32_t affine_transforms = final_state + 2;
constexpr uint32_t affine_transform(uint32_t step) { return affine_transforms + 2 * step; }
constexpr uint32_t record_count(uint32_t sp_size) { return affine_transform(sp_size); }
}  // namespace selection

struct Topology {
    uint32_t first_rank;
    uint32_t head_rows;
    uint32_t local_split;
    uint32_t rank;
    uint32_t final_owner;
    uint32_t split;
    uint32_t local_rows;
    uint32_t reserved;
    uint32_t group_chunks(uint32_t groups) const { return local_rows / 32 / groups; }
    uint32_t split_group(uint32_t groups) const { return head_rows / 32 / group_chunks(groups); }
    uint32_t split_in_group(uint32_t groups) const { return head_rows / 32 % group_chunks(groups); }
    uint32_t reset_chunk(uint32_t group, uint32_t groups) const {
        return local_split && group == split_group(groups) ? split_in_group(groups) : 0;
    }
    uint32_t reset_group(uint32_t groups) const {
        return local_split ? split_group(groups) + uint32_t(split_in_group(groups) != 0) - 1 : groups;
    }
    uint32_t head_groups(uint32_t groups) const {
        return local_split ? split_group(groups) + uint32_t(split_in_group(groups) != 0) : groups;
    }
};
inline Topology load(const volatile uint32_t* words) {
    return {words[0], words[1], words[2], words[3], words[4], words[5], words[6], words[7]};
}
inline void store(volatile uint32_t* words, const Topology& topology) {
    words[0] = topology.first_rank;
    words[1] = topology.head_rows;
    words[2] = topology.local_split;
    words[3] = topology.rank;
    words[4] = topology.final_owner;
    words[5] = topology.split;
    words[6] = topology.local_rows;
    words[7] = 0;
}
template <typename Buffer>
inline Topology receive(Buffer& buffer) {
    buffer.wait_front(1);
    Topology result{
        buffer.read_tile_value(0, 0),
        buffer.read_tile_value(0, 1),
        buffer.read_tile_value(0, 2),
        buffer.read_tile_value(0, 3),
        buffer.read_tile_value(0, 4),
        buffer.read_tile_value(0, 5),
        buffer.read_tile_value(0, 6),
        buffer.read_tile_value(0, 7)};
    buffer.pop_front(1);
    return result;
}
inline Topology derive(uint32_t actual_start, uint32_t rank, uint32_t partitions, uint32_t rows) {
    const uint32_t first_rank = (actual_start / rows) % partitions;
    const bool split = partitions > 1 && actual_start % rows != 0;
    return {
        first_rank,
        rows - actual_start % rows,
        uint32_t(split && rank == first_rank),
        rank,
        split ? first_rank : (first_rank + partitions - 1) % partitions,
        uint32_t(split),
        rows,
        0};
}
}  // namespace kda_chronology
