// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::prim {

struct InterleavedToShardedParams {
    tt::tt_metal::MemoryConfig output_mem_config;
    tt::tt_metal::DataType output_dtype{tt::tt_metal::DataType::INVALID};
    bool keep_l1_aligned{};

    static constexpr auto attribute_names =
        std::forward_as_tuple("output_mem_config", "output_dtype", "keep_l1_aligned");
    auto attribute_values() const {
        return std::make_tuple(std::cref(output_mem_config), output_dtype, keep_l1_aligned);
    }
};

struct InterleavedToShardedInputs {
    tt::tt_metal::Tensor input_tensor;
    std::optional<tt::tt_metal::Tensor> output_tensor;

    InterleavedToShardedInputs() = default;
    InterleavedToShardedInputs(
        tt::tt_metal::Tensor input_tensor_in, std::optional<tt::tt_metal::Tensor> output_tensor_in = std::nullopt) :
        input_tensor(std::move(input_tensor_in)), output_tensor(std::move(output_tensor_in)) {}

    static constexpr auto attribute_names = std::forward_as_tuple(
        "input_tensor_dtype", "input_tensor_memory_config", "input_tensor_layout", "input_tensor_padded_shape");
    auto attribute_values() const {
        return std::make_tuple(
            input_tensor.dtype(),
            std::cref(input_tensor.memory_config()),
            input_tensor.layout(),
            std::cref(input_tensor.padded_shape()));
    }
};

}  // namespace ttnn::prim
