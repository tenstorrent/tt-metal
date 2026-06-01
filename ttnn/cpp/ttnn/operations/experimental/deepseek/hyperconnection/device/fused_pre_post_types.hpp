// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::deepseek::hyperconnection {

struct FusedPrePostParams {
    float pre_scale;
    float post_scale;
    float eps;
    MemoryConfig output_mem_config;
};

struct FusedPrePostInputs {
    const Tensor& pre_w;
    const Tensor& post_w;
    const Tensor& pre_bias;
    const Tensor& post_bias;
    const Tensor& hidden_streams;
};

using FusedPrePostSpecReturn = std::array<TensorSpec, 2>;
using FusedPrePostTensorReturn = std::array<Tensor, 2>;

}  // namespace ttnn::operations::experimental::deepseek::hyperconnection
