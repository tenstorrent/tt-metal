// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::sparse_kv_push {

struct SparsePushPlanParams {
    uint32_t num_devices = 0;
    uint32_t stripe_chunk = 0;
    uint32_t max_rows = 0;
    uint32_t page_bytes = 0;  // the push's cache row page; sets the packet shape the compact slots follow
};

struct SparsePushPlanInputs {
    Tensor indices;  // uint32 row-major [1, 1, rows, topk] per device, 0xFFFFFFFF = masked
    // 1-element uint32 chunk start row: rows before it are the prefix part.
    std::optional<Tensor> chunk_start;
};

}  // namespace ttnn::operations::experimental::sparse_kv_push
