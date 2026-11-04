// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// The stage batches of a core's work items, shared by the prep reader, writer and compute: they must agree exactly, or
// the kernels deadlock on each other's buffers.
#pragma once

#include <cstdint>

namespace kda_prep {

// A work item is valid when its chunk is below the rank's valid chunk count.
inline bool is_valid(uint32_t work_item, uint32_t num_chunks, uint32_t valid_chunks) {
    return work_item % num_chunks < valid_chunks;
}

inline uint32_t count_valid(uint32_t start, uint32_t count, uint32_t num_chunks, uint32_t valid_chunks) {
    uint32_t valid = 0;
    for (uint32_t index = 0; index < count; ++index) {
        if (is_valid(start + index, num_chunks, valid_chunks)) {
            ++valid;
        }
    }
    return valid;
}

// The valid items run in batches of batch_items; the items past the last whole batch run one at a time.
template <uint32_t batch_items>
inline uint32_t batch_size(uint32_t remaining) {
    return remaining >= batch_items ? batch_items : 1;
}

// The valid work items of [start, start + count), batch by batch.
template <uint32_t batch_items>
struct Batches {
    uint32_t start;
    uint32_t num_chunks;
    uint32_t valid_chunks;
    uint32_t remaining;
    uint32_t next_index = 0;
    uint32_t size = 0;
    uint32_t items[batch_items];

    Batches(uint32_t start, uint32_t count, uint32_t num_chunks, uint32_t valid_chunks) :
        start(start),
        num_chunks(num_chunks),
        valid_chunks(valid_chunks),
        remaining(count_valid(start, count, num_chunks, valid_chunks)) {}

    // Fill items and size with the next batch; false when none remain.
    bool next() {
        if (remaining == 0) {
            return false;
        }
        size = batch_size<batch_items>(remaining);
        for (uint32_t item = 0; item < size; ++next_index) {
            if (is_valid(start + next_index, num_chunks, valid_chunks)) {
                items[item++] = start + next_index;
            }
        }
        remaining -= size;
        return true;
    }
};

}  // namespace kda_prep
