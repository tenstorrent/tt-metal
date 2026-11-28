// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct GetRowsParams {
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct GetRowsInputs {
    // The [H, W] row-major table the rows are gathered from.
    Tensor input_tensor;
    // The [N] uint32/int32 row index vector. One output row per index, in index order.
    Tensor indices_tensor;
    std::optional<Tensor> optional_output_tensor;
};

}  // namespace ttnn::prim
