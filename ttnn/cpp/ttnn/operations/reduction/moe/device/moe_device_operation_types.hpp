// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include "ttnn/tensor/tensor.hpp"
#include <tuple>

namespace ttnn::prim {

struct MoeParams {
    uint16_t k{};
    tt::tt_metal::MemoryConfig output_memory_config;

    static constexpr auto attribute_names = std::forward_as_tuple("k", "output_memory_config");
    auto attribute_values() const { return std::forward_as_tuple(k, output_memory_config); }
};

struct MoeInputs {
    Tensor input;
    Tensor expert_mask;
    Tensor topk_mask;
    std::optional<Tensor> preallocated_output;

    static constexpr auto attribute_names =
        std::forward_as_tuple("input", "expert_mask", "topk_mask", "preallocated_output");
    auto attribute_values() const { return std::forward_as_tuple(input, expert_mask, topk_mask, preallocated_output); }
};

}  // namespace ttnn::prim
