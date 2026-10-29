// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Sizes, layouts and schedule arithmetic shared by the expert rows host sources and kernels. Terms:
// - job: up to row_tiles x 32 routed rows of one local expert, computed in one pass over its weights.
// - expert group: the cores that compute one job, one or more reader cores per DRAM bank (shard of the weights).
// - column group: one block-column of W0/W1 (2 intermediate columns, 4 tiles per K row). Output group: 4 W2 output
//   tiles.
// - weight-stationary: jobs of several row tiles (row_tiles > 1). x moves in chunks of chunk_tiles hidden tiles; each
//   unit (one chunk of one group's weights) stays in cb_w for all of the job's row tiles; partials spill to L1 between
//   chunks; slots are handed back by credits; the root deals jobs by predicted time.
// - pass: the gate/up or the down projection of one job; the stream runs gate/up(0), gate/up(1), down(0), ...
#pragma once

#include <stdint.h>

#include "../moe_ring_common.h"

namespace expert_rows {

constexpr uint32_t u16_bytes = sizeof(uint16_t);
constexpr uint32_t bf16_bytes = 2;

__attribute__((always_inline)) constexpr uint32_t div_up(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

__attribute__((always_inline)) constexpr uint32_t align_up(uint32_t bytes, uint32_t alignment) {
    return div_up(bytes, alignment) * alignment;
}

// Rows of a row tile; a job of row_tiles row tiles holds at most tile_rows * row_tiles rows of one expert.
constexpr uint32_t tile_rows = 32;
// A tile is tile_rows x tile_rows values in four faces of face_rows x face_columns: rows 0..15 in faces 0 (columns
// 0..15) and 1 (columns 16..31), rows 16..31 in faces 2 and 3.
constexpr uint32_t face_rows = 16;
constexpr uint32_t face_columns = 16;

__attribute__((always_inline)) constexpr uint32_t row_tiles_of(uint32_t rows) { return div_up(rows, tile_rows); }

// Tiles per K row of a W0/W1 column group (W0 j, W1 j, W0 j + 1, W1 j + 1) and of a W2 output group. The last group of
// a ring position can be half wide: a half block-column (W0 j, W1 j), or the last output group, two tiles wide.
constexpr uint32_t group_tiles = moe_ring::W0_W1_BLOCK_TILES_W;
constexpr uint32_t half_group_tiles = moe_ring::W0_W1_HALF_BLOCK_TILES_W;
static_assert(group_tiles == moe_ring::W2_TILES_PER_A2A_ITER_W, "column and output groups are equally wide");
static_assert(half_group_tiles == moe_ring::W2_HALF_A2A_ITER_TILES_W, "half column and output groups too");

// A full column group's intermediate columns, one activated tile each.
constexpr uint32_t group_columns = 2;

// `count` groups of group_tiles tiles per K row; the last is half_group_tiles wide when half_last.
struct GroupRun {
    uint32_t count;
    bool half_last;
    __attribute__((always_inline)) constexpr uint32_t width(uint32_t group) const {
        return half_last && group + 1 == count ? half_group_tiles : group_tiles;
    }
    // tiles per K row of the whole run
    constexpr uint32_t tiles() const { return count * group_tiles - (half_last ? group_tiles - half_group_tiles : 0); }
};

// CB blocks of block_tiles tiles that a run of `width` tiles per K row over `k_rows` K rows fills.
__attribute__((always_inline)) constexpr uint32_t run_blocks(uint32_t width, uint32_t k_rows, uint32_t block_tiles) {
    return div_up(width * k_rows, block_tiles);
}

// Hidden tiles per x chunk of a job of one row tile: what an x slot of row_tiles chunks of chunk_tiles holds, at most
// all of them.
constexpr uint32_t single_chunk_tiles(uint32_t chunk_tiles, uint32_t row_tiles, uint32_t hidden_tiles) {
    return chunk_tiles * row_tiles < hidden_tiles ? chunk_tiles * row_tiles : hidden_tiles;
}

// NoC transfers between L1s move whole 16-byte words.
constexpr uint32_t l1_word_bytes = 16;
// Every part of the routing scratch and of the routing table block starts at a multiple of 64 bytes, the DRAM
// alignment, so that each part of the table block can be read from DRAM on its own.
constexpr uint32_t part_alignment = 64;

// The routing table block (rows_reader.cpp writes it to DRAM, place.cpp and rows_combine.cpp read it), byte offsets
// from its start, each part at a multiple of part_alignment: a head (u32 routed rows), u16 per-expert row counts and
// first rows, each routed row's token and k, then the same entries in token order: expert, token, score and k (a token
// that names one expert at several k has one entry per k).
struct TableLayout {
    uint32_t row_counts, first_rows, row_tokens, row_k, entry_experts, entry_tokens, entry_scores, entry_k, bytes;
};

constexpr TableLayout table_layout(uint32_t tokens, uint32_t top_k, uint32_t local_experts) {
    const uint32_t expert_part = align_up(local_experts * u16_bytes, part_alignment);
    const uint32_t entry_part = align_up(tokens * top_k * u16_bytes, part_alignment);
    TableLayout layout{};
    layout.row_counts = part_alignment;  // after the head
    layout.first_rows = layout.row_counts + expert_part;
    layout.row_tokens = layout.first_rows + expert_part;
    layout.row_k = layout.row_tokens + entry_part;
    layout.entry_experts = layout.row_k + entry_part;
    layout.entry_tokens = layout.entry_experts + entry_part;
    layout.entry_scores = layout.entry_tokens + entry_part;
    layout.entry_k = layout.entry_scores + entry_part;
    layout.bytes = layout.entry_k + entry_part;
    return layout;
}

// The routing scratch's ctl block: this header, then with weight-stationary jobs u16 the first job of each expert group
// and the job count.
struct CtlHeader {
    uint32_t table_jobs, active_experts, routed_rows, unused;
};
constexpr uint32_t ctl_header_bytes = sizeof(CtlHeader);
static_assert(ctl_header_bytes == 16, "the ctl header is one 16-byte L1 word");
constexpr uint32_t ctl_bytes(uint32_t expert_groups) { return ctl_header_bytes + u16_bytes * (expert_groups + 1); }

// An entry area of the routing scratch: u32 entry count, then u16 expert, token, score and k of each entry; a core
// routes at most div_up(tokens, program_cores) tokens, one entry per k.
constexpr uint32_t entry_capacity(uint32_t tokens, uint32_t top_k, uint32_t program_cores) {
    return div_up(tokens, program_cores) * top_k;
}
constexpr uint32_t entry_area_bytes(uint32_t capacity) { return align_up(4 + 4 * u16_bytes * capacity, l1_word_bytes); }

// One job of the routing table: rows [first_row, first_row + rows) of one local expert.
struct JobRecord {
    uint16_t expert;
    uint16_t first_row;
    uint16_t rows;
};
constexpr uint32_t job_record_bytes = sizeof(JobRecord);
static_assert(job_record_bytes == 3 * u16_bytes, "job records are packed u16 triples");

// This expert group's `count` jobs in the routing table, from its first, table job `first`. With one row tile per job
// the group's job j is table job group + expert_groups j; weight-stationary, the root lists each group's jobs together.
template <bool weight_stationary, uint32_t expert_groups, typename JobRecords>
struct GroupJobs {
    JobRecords table;
    uint32_t group, first, count;
    __attribute__((always_inline)) uint32_t at(uint32_t job) const {
        return weight_stationary ? first + job : group + expert_groups * job;
    }
    __attribute__((always_inline)) uint32_t expert(uint32_t job) const { return table[at(job)].expert; }
    __attribute__((always_inline)) uint32_t first_row(uint32_t job) const { return table[at(job)].first_row; }
    __attribute__((always_inline)) uint32_t rows(uint32_t job) const { return table[at(job)].rows; }
    // a job of several row tiles holds each weight unit for all of them
    __attribute__((always_inline)) bool held(uint32_t job) const { return weight_stationary && rows(job) > tile_rows; }
};

// The group's jobs from the routing scratch: its job records, the ctl block's first job of each group
// (weight-stationary) and the table's job count.
template <bool weight_stationary, uint32_t expert_groups, typename JobRecords, typename FirstJobs>
__attribute__((always_inline)) inline GroupJobs<weight_stationary, expert_groups, JobRecords> group_jobs(
    JobRecords table, FirstJobs group_first_job, uint32_t group, uint32_t table_jobs) {
    if (weight_stationary) {
        const uint32_t first = group_first_job[group];
        return {table, group, first, uint32_t(group_first_job[group + 1]) - first};
    }
    return {table, group, group, table_jobs > group ? div_up(table_jobs - group, expert_groups) : 0};
}

// CB blocks of a one-row-tile pass before its padding (rows_reader.cpp streams them, rows_compute.cpp takes them).
// Gate/up: each column group's run over each x chunk of single_chunk_tiles hidden tiles, the bias row ending the last.
constexpr uint32_t gate_up_pass_blocks(
    GroupRun columns, uint32_t hidden_tiles, uint32_t single_chunk_tiles, uint32_t bias_rows, uint32_t block_tiles) {
    const uint32_t chunks = div_up(hidden_tiles, single_chunk_tiles);
    uint32_t blocks = 0;
    for (uint32_t column_group = 0; column_group < columns.count; ++column_group) {
        for (uint32_t chunk = 0; chunk < chunks; ++chunk) {
            const bool last = chunk + 1 == chunks;
            const uint32_t k_rows = last ? hidden_tiles - chunk * single_chunk_tiles + bias_rows : single_chunk_tiles;
            blocks += run_blocks(columns.width(column_group), k_rows, block_tiles);
        }
    }
    return blocks;
}

// Down: each output group's run over all W2 rows.
constexpr uint32_t down_pass_blocks(
    GroupRun outputs, uint32_t intermediate_tiles, uint32_t bias_rows, uint32_t block_tiles) {
    uint32_t blocks = 0;
    for (uint32_t output_group = 0; output_group < outputs.count; ++output_group) {
        blocks += run_blocks(outputs.width(output_group), intermediate_tiles + bias_rows, block_tiles);
    }
    return blocks;
}

// Blocks that pad a one-row-tile pass of `blocks` blocks to whole units of unit_blocks (weight-stationary jobs).
constexpr uint32_t unit_padding(uint32_t blocks, uint32_t unit_blocks) {
    return (unit_blocks - blocks % unit_blocks) % unit_blocks;
}

// Activation rows place.cpp keeps in flight before a write barrier.
constexpr uint32_t activation_buffers = 8;

}  // namespace expert_rows
