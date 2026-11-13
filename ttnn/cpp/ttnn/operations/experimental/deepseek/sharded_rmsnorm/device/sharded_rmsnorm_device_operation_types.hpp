// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::deepseek::sharded_rmsnorm {

struct ShardedRmsNormParams {
    float epsilon = 1e-6f;
    MemoryConfig output_mem_config;
    Layout output_layout = Layout::TILE;
};

struct ShardedRmsNormInputs {
    const Tensor& input_tensor;
    const Tensor& weight;
};

using ShardedRmsNormTensorReturn = Tensor;

}  // namespace ttnn::operations::experimental::deepseek::sharded_rmsnorm
