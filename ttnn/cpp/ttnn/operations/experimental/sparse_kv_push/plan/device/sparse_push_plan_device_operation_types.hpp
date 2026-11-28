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
    // The table as one slot per owner (sparse_kv_push's inbox layout: the header, then that owner's block bitmaps,
    // contiguous), for a push that exchanges the tables itself, instead of one bitmap in block order. Hashed.
    bool owner_major = false;
    // A further output for sparse_kv_dispatch's compact landing (owner-major): page 0 the owners'
    // compact-buffer run bases, then one page a block: each bitmap word's rank within its owner's run, then the words.
    // The landing page of row g is compact_page(run_base[owner], rank[word] + popcount(bits below g)). Hashed.
    bool rank_table = false;
};

// sparse_push_plan's rank table page: max(2 x words a block, kMaxDevices) uint32.
inline uint32_t rank_table_page_words(uint32_t stripe_chunk) { return std::max(2 * stripe_chunk / 32, 32u); }

struct SparsePushPlanInputs {
    Tensor indices;  // uint32 row-major [1, 1, rows, topk] per device, 0xFFFFFFFF = masked
    // The trace-safe chunk start: a 1-element uint32 tensor holding the row, read on device.
    std::optional<Tensor> chunk_start_tensor;
};

}  // namespace ttnn::operations::experimental::sparse_kv_push
