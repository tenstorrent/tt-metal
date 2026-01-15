// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::transformer::qkv_heads_falcon7b {

struct QkvHeadsFalcon7bParams {
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct QkvHeadsFalcon7bInputs {
    Tensor input;
};

// Return types using named structs for Q, K, V heads
struct spec_return_value_t {
    TensorSpec q;
    TensorSpec k;
    TensorSpec v;
};

struct tensor_return_value_t {
    Tensor q;
    Tensor k;
    Tensor v;
};

}  // namespace ttnn::operations::experimental::transformer::qkv_heads_falcon7b
