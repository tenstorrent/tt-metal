// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/core_coord.hpp>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct IndexedFillParams {
    tt::tt_metal::MemoryConfig output_mem_config;
    int64_t dim;
    // Worker grid chosen for this op: defaults to all worker cores; may be restricted by
    // sharded output, sharded input native path, or explicit memory_config.
    CoreRangeSet worker_grid;
};

struct IndexedFillInputs {
    Tensor batch_id;
    Tensor input_tensor_a;
    Tensor input_tensor_b;
};

}  // namespace ttnn::prim
