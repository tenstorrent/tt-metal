// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::normalization::layer_norm_w_rm {

struct LayerNormWRmParams {
    const float epsilon;
    const tt::tt_metal::MemoryConfig output_mem_config;
};

struct LayerNormWRmInputs {
    const Tensor& input;
    const Tensor& gamma;
    const Tensor& beta;
};

}  // namespace ttnn::operations::normalization::layer_norm_w_rm