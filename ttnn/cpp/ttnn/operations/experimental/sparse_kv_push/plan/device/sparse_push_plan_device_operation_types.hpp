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
};

struct SparsePushPlanInputs {
    Tensor indices;  // uint32 row-major [1, 1, rows, topk] per device, 0xFFFFFFFF = masked
    // The trace-safe chunk start: a 1-element uint32 tensor holding the row, read on device.
    std::optional<Tensor> chunk_start_tensor;
};

// A chip's slot in dense mode, as high_bw_all_gather with gathered_active_below lays it out: the largest extent the
// gather copies (a chunk starting just below dense_below, plus that chunk, in whole slabs), per chip.
inline uint32_t dense_slot_rows(const SparsePushPlanParams& attrs, uint32_t num_devices) {
    const uint64_t slab = uint64_t{attrs.stripe_chunk} * num_devices;
    return static_cast<uint32_t>((attrs.dense_below - 1 + slab + slab - 1) / slab * attrs.stripe_chunk);
}

}  // namespace ttnn::operations::experimental::sparse_kv_push
