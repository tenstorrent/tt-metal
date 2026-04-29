// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::gpt_oss_swiglu {

struct operation_attributes_t {
    float alpha;        // sigmoid scale (1.702 for GPT-OSS)
    float clamp_limit;  // clamp limit (7.0 for GPT-OSS)
    tt::tt_metal::MemoryConfig output_memory_config;
};

struct tensor_args_t {
    const Tensor& gate_tensor;
    const Tensor& up_tensor;
};

using tensor_return_value_t = Tensor;
using spec_return_value_t = TensorSpec;

}  // namespace ttnn::operations::experimental::gpt_oss_swiglu
