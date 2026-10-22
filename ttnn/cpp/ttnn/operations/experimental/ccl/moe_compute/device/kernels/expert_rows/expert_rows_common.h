// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Sizes and layouts shared by moe_expert_rows.cpp and the expert rows kernels.
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

// The routing scratch's ctl block: u32 job count, active experts and routed rows in a 16-byte header, then with several
// row tiles per job u16 the first job of each expert group and the job count.
constexpr uint32_t ctl_header_bytes = 16;
constexpr uint32_t ctl_bytes(uint32_t expert_groups) { return ctl_header_bytes + u16_bytes * (expert_groups + 1); }

// An entry area of the routing scratch: u32 entry count, then u16 expert, token, score and k of each entry.
constexpr uint32_t entry_area_bytes(uint32_t capacity) { return align_up(4 + 4 * u16_bytes * capacity, l1_word_bytes); }

// One job of the routing table: rows [first_row, first_row + rows) of one local expert.
struct JobRecord {
    uint16_t expert;
    uint16_t first_row;
    uint16_t rows;
};
constexpr uint32_t job_record_bytes = sizeof(JobRecord);
static_assert(job_record_bytes == 3 * u16_bytes, "job records are packed u16 triples");

// This expert group's jobs in the routing table, from its first, table job `first`. With one row tile per job the
// group's job j is table job group + expert_groups j; with several (dealt) the root lists each group's jobs together.
template <bool dealt, uint32_t expert_groups, typename JobRecords>
struct GroupJobs {
    JobRecords table;
    uint32_t group, first;
    __attribute__((always_inline)) uint32_t at(uint32_t job) const {
        return dealt ? first + job : group + expert_groups * job;
    }
    __attribute__((always_inline)) uint32_t expert(uint32_t job) const { return table[at(job)].expert; }
    __attribute__((always_inline)) uint32_t first_row(uint32_t job) const { return table[at(job)].first_row; }
    __attribute__((always_inline)) uint32_t rows(uint32_t job) const { return table[at(job)].rows; }
    // a dealt job of several row tiles holds each weight unit for all of them
    __attribute__((always_inline)) bool held(uint32_t job) const { return dealt && rows(job) > tile_rows; }
};

// The group's jobs from the routing scratch: its job records and the ctl block's first job of each group.
template <bool dealt, uint32_t expert_groups, typename JobRecords, typename FirstJobs>
__attribute__((always_inline)) inline GroupJobs<dealt, expert_groups, JobRecords> group_jobs(
    JobRecords table, FirstJobs group_first_job, uint32_t group) {
    return {table, group, dealt ? uint32_t(group_first_job[group]) : group};
}

// Activation rows place.cpp keeps in flight before a write barrier.
constexpr uint32_t activation_buffers = 8;

// Width shards the W2 output columns of one feeder (ring position) span at most (feed.cpp; the host checks every
// feeder).
constexpr uint32_t max_slice_shards = 3;

}  // namespace expert_rows
