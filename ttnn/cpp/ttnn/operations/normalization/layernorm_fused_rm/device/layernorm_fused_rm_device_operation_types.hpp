// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::normalization::layernorm_fused_rm {

struct LayernormFusedRmParams {
    const float epsilon;
    const tt::tt_metal::MemoryConfig output_mem_config;
};

struct LayernormFusedRmInputs {
    const Tensor& input;
    const Tensor& gamma;
    const Tensor& beta;
};

}  // namespace ttnn::operations::normalization::layernorm_fused_rm
