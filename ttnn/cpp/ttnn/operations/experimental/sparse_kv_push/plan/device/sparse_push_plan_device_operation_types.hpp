// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include <tt-metalium/core_coord.hpp>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::sparse_kv_push {

struct SparsePushPlanParams {
    uint32_t stripe_chunk = 0;
    uint32_t max_rows = 0;
    uint32_t page_bytes = 0;  // the push's cache row page; sets the packet shape the compact slots follow
    // The chunk start row from the host: a runtime arg, not hashed. Rows of blocks before it are the prefix part.
    std::optional<uint32_t> chunk_start;
    // The cores the plan runs on (one rectangle); the whole worker grid if unset.
    std::optional<tt::tt_metal::CoreRange> core_range;
    // Blocks before the chunk start's block that go to the rest part instead of the prefix.
    uint32_t rest_lookback_blocks = 0;
};

struct SparsePushPlanInputs {
    Tensor indices;  // uint32 row-major [1, 1, rows, topk] per device, 0xFFFFFFFF = masked
    // The trace-safe chunk start: a 1-element uint32 tensor holding the row, read on device.
    std::optional<Tensor> chunk_start_tensor;
};

}  // namespace ttnn::operations::experimental::sparse_kv_push
