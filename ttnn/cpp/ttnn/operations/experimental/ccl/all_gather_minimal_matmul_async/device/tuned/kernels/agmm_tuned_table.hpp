// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// agmm_tuned's per-device table: what differs between the devices of the mesh, so that one program
// serves them all. A [1, 1, rows, kTableRowWords] UINT32 row-major DRAM tensor holding each device's own rows; a kernel
// reads the row it needs (one page) when it starts.
//
//   row 0                   kRank: this device's rank along the gathered axis
//   row 1 + k (sender k)    kNumDests, then per destination (up to kMaxDests) its fabric chip, mesh, slice counter
//                           address, partner core (x | y << 16) and hops; kNumConnectionWords and the sender's fabric
//                           connection arguments.
namespace agmm_tuned_table {

constexpr uint32_t kTableRowWords = 64;  // 256 B pages
constexpr uint32_t kMaxDests = 2;

constexpr uint32_t kRank = 0;

constexpr uint32_t kNumDests = 0;
constexpr uint32_t kDestWords = 5;
constexpr uint32_t kDestChip = 0;
constexpr uint32_t kDestMesh = 1;
constexpr uint32_t kDestCounter = 2;
constexpr uint32_t kDestPartner = 3;
constexpr uint32_t kDestHops = 4;
constexpr uint32_t dest_word(uint32_t dest, uint32_t field) { return 1 + dest * kDestWords + field; }
constexpr uint32_t kNumConnectionWords = 1 + kMaxDests * kDestWords;
constexpr uint32_t kConnectionWords = kNumConnectionWords + 1;
constexpr uint32_t kMaxConnectionWords = kTableRowWords - kConnectionWords;

// The K slices in the order the matmul takes them: this rank's first, then the others nearest first along the ring
// (rank + 1, rank - 1, rank + 2, ...). `order` gets `ranks` entries.
inline void slice_order(uint32_t rank, uint32_t ranks, uint32_t* order) {
    uint32_t n = 0;
    order[n++] = rank;
    for (uint32_t d = 1; n < ranks; ++d) {
        order[n++] = (rank + d) % ranks;
        const uint32_t behind = (rank + ranks - d % ranks) % ranks;
        if (n < ranks && behind != order[n - 1]) {
            order[n++] = behind;
        }
    }
}

}  // namespace agmm_tuned_table
