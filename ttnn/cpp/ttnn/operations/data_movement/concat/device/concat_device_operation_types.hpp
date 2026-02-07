// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include <tuple>

namespace ttnn::prim {

struct ConcatParams {
    uint32_t dim;
    unsigned int groups;
    tt::tt_metal::MemoryConfig output_mem_config;

    static constexpr auto attribute_names = std::forward_as_tuple("dim", "output_mem_config");
    auto attribute_values() const { return std::forward_as_tuple(dim, output_mem_config); }
};

struct ConcatInputs {
    std::vector<Tensor> input_tensors;

    static constexpr auto attribute_names = std::forward_as_tuple("input_tensors");
    auto attribute_values() const { return std::forward_as_tuple(input_tensors); }
};

}  // namespace ttnn::prim
