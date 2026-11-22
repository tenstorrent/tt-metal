// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::sparse_kv_push {

struct SparsePushPlanParams {
    uint32_t stripe_chunk = 0;
    uint32_t max_rows = 0;
    uint32_t page_bytes = 0;  // the push's cache row page; sets the packet shape the compact slots follow
    // The chunk start row from the host: a runtime arg, not hashed. Rows of blocks before it are the prefix part.
    std::optional<uint32_t> chunk_start;
    // Dense mode: with the chunk start below dense_below, the indices are rewritten to rows of the dense gather into
    // the compact buffer (chip c's rows from c * dense_slot_rows on) and the table lists no rows to push. 0: off.
    // A compile-time arg, so hashed.
    uint32_t dense_below = 0;
    // A second dense threshold, above dense_below (0: off), for a reader whose buffer holds the dense gather below it:
    // a third output gets the indices rewritten to that dense layout for chunk starts below dense_below_2, and the same
    // indices as the second output otherwise. Lets a layer with a higher cutoff share the plan of the layers below it.
    uint32_t dense_below_2 = 0;
    // The table as one slot per owner (sparse_kv_push's inbox layout: the header, then that owner's block bitmaps,
    // contiguous), for a push that exchanges the tables itself, instead of one bitmap in block order. Hashed.
    bool owner_major = false;
};

struct SparsePushPlanInputs {
    Tensor indices;  // uint32 row-major [1, 1, rows, topk] per device, 0xFFFFFFFF = masked
    // The trace-safe chunk start: a 1-element uint32 tensor holding the row, read on device.
    std::optional<Tensor> chunk_start_tensor;
};

// A chip's slot in dense mode, as high_bw_all_gather with gathered_active_below lays it out: the largest extent the
// gather copies (a chunk starting just below dense_below, plus that chunk, in whole slabs), per chip.
inline uint32_t dense_slot_rows(uint32_t stripe_chunk, uint32_t dense_below, uint32_t num_devices) {
    const uint64_t slab = uint64_t{stripe_chunk} * num_devices;
    return static_cast<uint32_t>((dense_below - 1 + slab + slab - 1) / slab * stripe_chunk);
}

inline uint32_t dense_slot_rows(const SparsePushPlanParams& attrs, uint32_t num_devices) {
    return dense_slot_rows(attrs.stripe_chunk, attrs.dense_below, num_devices);
}

}  // namespace ttnn::operations::experimental::sparse_kv_push
