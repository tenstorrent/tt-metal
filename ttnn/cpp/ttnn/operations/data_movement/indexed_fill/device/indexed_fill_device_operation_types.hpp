// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include <tuple>

namespace ttnn::prim {

struct IndexedFillParams {
    tt::tt_metal::MemoryConfig output_mem_config;
    int64_t dim;

    static constexpr auto attribute_names = std::forward_as_tuple("output_mem_config", "dim");
    auto attribute_values() const { return std::forward_as_tuple(output_mem_config, dim); }
};

struct IndexedFillInputs {
    Tensor batch_id;
    Tensor input_tensor_a;
    Tensor input_tensor_b;

    static constexpr auto attribute_names = std::forward_as_tuple("batch_id", "input_tensor_a", "input_tensor_b");
    auto attribute_values() const { return std::forward_as_tuple(batch_id, input_tensor_a, input_tensor_b); }
};

}  // namespace ttnn::prim
