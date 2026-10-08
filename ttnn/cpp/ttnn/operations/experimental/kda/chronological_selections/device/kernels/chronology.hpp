// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <tt-metalium/constants.hpp>
namespace kda_chronology {
namespace selection {
constexpr uint32_t record_width = 8;  // One aligned 32-byte UINT32 record.
constexpr uint32_t history_rows = 3;  // Four-tap convolution retains three preceding tokens.
constexpr uint32_t slice_rank = 4;
constexpr uint32_t outgoing_history = 0;
constexpr uint32_t predecessor_history = outgoing_history + 1;
constexpr uint32_t final_history = predecessor_history + 1;
constexpr uint32_t final_state = final_history + 1;
constexpr uint32_t local_final_history = final_state + 2;
// Packed history records for the sequence-parallel convolution exchange. The first selects the outgoing then the
// local final history from the projection, so one gather carries both as six rows per rank; the second selects the
// predecessor's outgoing then the final owner's local final history from that gathered table.
constexpr uint32_t packed_history_rows = 2 * history_rows;
constexpr uint32_t outgoing_and_local_final_history = local_final_history + 1;
constexpr uint32_t predecessor_and_final_history = outgoing_and_local_final_history + 1;
constexpr uint32_t record_count = predecessor_and_final_history + 1;
}  // namespace selection

struct Topology {
    uint32_t first_rank;
    uint32_t head_rows;
    uint32_t local_split;  // This rank physically holds head and tail segments, even if the tail is padded.
    uint32_t rank;
    uint32_t final_owner;
    uint32_t split;  // The global interval includes a valid separated tail; select its final state.
    uint32_t local_rows;
    uint32_t valid_rows;
    uint32_t active_groups(uint32_t groups) const {
        const uint32_t rows_per_group = local_rows / groups;
        return (valid_rows + rows_per_group - 1) / rows_per_group;
    }
    uint32_t valid_chunks(uint32_t group, uint32_t groups) const {
        const uint32_t chunks = group_chunks(groups);
        const uint32_t begin = group * chunks;
        const uint32_t end = valid_rows / tt::constants::TILE_HEIGHT;
        return end <= begin ? 0 : (end - begin < chunks ? end - begin : chunks);
    }
    bool has_valid_tail() const { return local_split && valid_rows > head_rows; }
    uint32_t group_chunks(uint32_t groups) const { return local_rows / tt::constants::TILE_HEIGHT / groups; }
    uint32_t split_group(uint32_t groups) const {
        return head_rows / tt::constants::TILE_HEIGHT / group_chunks(groups);
    }
    uint32_t split_in_group(uint32_t groups) const {
        return head_rows / tt::constants::TILE_HEIGHT % group_chunks(groups);
    }
    uint32_t reset_chunk(uint32_t group, uint32_t groups) const {
        return has_valid_tail() && group == split_group(groups) ? split_in_group(groups) : 0;
    }
    uint32_t reset_group(uint32_t groups) const {
        return has_valid_tail() ? split_group(groups) + uint32_t(split_in_group(groups) != 0) - 1 : groups;
    }
    uint32_t head_groups(uint32_t groups) const {
        const uint32_t rows = local_split && head_rows < valid_rows ? head_rows : valid_rows;
        const uint32_t group_rows = local_rows / groups;
        return (rows + group_rows - 1) / group_rows;
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
    words[7] = topology.valid_rows;
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
// The caller supplies a nonnegative, TILE_HEIGHT-aligned absolute position as UINT32.
// Values may exceed the mesh span; the modulo below maps them to physical partitions.
// Value validation is intentionally omitted: actual_start is read on device and can
// change on every trace replay, so a host check at capture would not validate replay.
// The caller must preserve this contract on every update; misalignment is unchecked.
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
        rows};
}
// Clip chronological segments without changing their physical placement.
inline Topology derive_interval(uint32_t start, uint32_t end, uint32_t rank, uint32_t partitions, uint32_t rows) {
    auto t = derive(start, rank, partitions, rows);
    const uint32_t length = end - start;
    if (partitions == 1) {
        t.valid_rows = length;
        t.head_rows = rows;
        return t;
    }
    const uint32_t step = (rank + partitions - t.first_rank) % partitions;
    const uint32_t tail_begin = t.head_rows + (partitions - 1) * rows;
    const uint32_t begin = step == 0 ? 0 : t.head_rows + (step - 1) * rows;
    if (step == 0) {
        t.valid_rows = length < t.head_rows ? length : t.head_rows;
        if (length > tail_begin) {
            t.valid_rows += length - tail_begin;
        }
    } else {
        t.valid_rows = length <= begin ? 0 : (length - begin < rows ? length - begin : rows);
    }
    const uint32_t last_step = length <= t.head_rows ? 0 : 1 + (length - t.head_rows - 1) / rows;
    t.final_owner = (t.first_rank + last_step) % partitions;
    // This flag chooses a final state after a separated tail, not merely a physical split.
    t.split = length > tail_begin;
    return t;
}
// Row indices of a history selection record, as chronological_selections stores them; returns their count. The
// outgoing and local final histories index this rank's projected rows; the predecessor and final histories index
// the rank-major gathered table (three rows per rank, or six for the packed selections).
inline uint32_t selection_history_rows(
    const Topology& topology, uint32_t record, uint32_t sp_size, uint32_t local_rows, uint32_t* rows) {
    using namespace selection;
    const uint32_t outgoing_base = (topology.local_split ? topology.head_rows : local_rows) - history_rows;
    const uint32_t history_end = topology.valid_rows == 0 ? history_rows : topology.valid_rows;
    const uint32_t local_final_base = history_end - history_rows;
    const uint32_t predecessor_rank = (topology.rank + sp_size - 1) % sp_size;
    uint32_t first = 0;
    uint32_t second = 0;
    uint32_t count = history_rows;
    if (record == outgoing_history) {
        first = outgoing_base;
    } else if (record == predecessor_history) {
        first = predecessor_rank * history_rows;
    } else if (record == final_history) {
        first = topology.final_owner * history_rows;
    } else if (record == local_final_history) {
        first = local_final_base;
    } else if (record == outgoing_and_local_final_history) {
        first = outgoing_base;
        second = local_final_base;
        count = packed_history_rows;
    } else {  // predecessor_and_final_history
        first = predecessor_rank * packed_history_rows;
        second = topology.final_owner * packed_history_rows + history_rows;
        count = packed_history_rows;
    }
    for (uint32_t i = 0; i < history_rows; ++i) {
        rows[i] = first + i;
        if (count == packed_history_rows) {
            rows[history_rows + i] = second + i;
        }
    }
    return count;
}
}  // namespace kda_chronology
