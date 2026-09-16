// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
namespace kda_chronology {
struct Topology {
    uint32_t boundary;
    uint32_t head_rows;
    uint32_t local_split;
    uint32_t rank;
    uint32_t final_owner;
    uint32_t split;
    uint32_t local_rows;
    uint32_t reserved;
    uint32_t group_chunks(uint32_t groups) const { return local_rows / 32 / groups; }
    uint32_t wrap_group(uint32_t groups) const { return head_rows / 32 / group_chunks(groups); }
    uint32_t split_in_group(uint32_t groups) const { return head_rows / 32 % group_chunks(groups); }
    uint32_t reset_chunk(uint32_t group, uint32_t groups) const {
        return local_split && group == wrap_group(groups) ? split_in_group(groups) : 0;
    }
    uint32_t reset_group(uint32_t groups) const {
        return local_split ? wrap_group(groups) + uint32_t(split_in_group(groups) != 0) - 1 : groups;
    }
    uint32_t head_groups(uint32_t groups) const {
        return local_split ? wrap_group(groups) + uint32_t(split_in_group(groups) != 0) : groups;
    }
};
inline Topology load(const volatile uint32_t* words) {
    return {words[0], words[1], words[2], words[3], words[4], words[5], words[6], words[7]};
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
inline Topology derive(uint32_t start, uint32_t rank, uint32_t partitions, uint32_t rows) {
    const uint32_t boundary = (start / rows) % partitions;
    const bool split = partitions > 1 && start % rows != 0;
    return {
        boundary,
        rows - start % rows,
        uint32_t(split && rank == boundary),
        rank,
        split ? boundary : (boundary + partitions - 1) % partitions,
        uint32_t(split),
        rows,
        0};
}
}  // namespace kda_chronology
