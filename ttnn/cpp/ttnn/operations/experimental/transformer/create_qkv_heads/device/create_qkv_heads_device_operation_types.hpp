// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/memory_config/memory_config.hpp"

namespace ttnn::experimental::prim {

struct CreateQKVHeadsParams {
    const uint32_t num_q_heads;
    const uint32_t num_kv_heads;
    const uint32_t head_dim;
    const bool transpose_k_heads;
    const tt::tt_metal::MemoryConfig output_mem_config;
};

struct CreateQKVHeadsInputs {
    const Tensor input;
    const std::optional<std::tuple<Tensor, Tensor, Tensor>> preallocated_outputs;
};

using CreateQKVHeadsResultSpec = std::tuple<TensorSpec, TensorSpec, TensorSpec>;
using CreateQKVHeadsResult = std::tuple<Tensor, Tensor, Tensor>;

}  // namespace ttnn::experimental::prim
