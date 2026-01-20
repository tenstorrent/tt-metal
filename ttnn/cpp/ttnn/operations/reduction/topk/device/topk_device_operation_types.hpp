// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <tuple>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct TopkParams {
    uint32_t k{};
    int8_t dim{};
    bool largest{};
    bool sorted{};
    tt::tt_metal::MemoryConfig output_memory_config;
    tt::tt_metal::CoreRangeSet sub_core_grids;
};

struct TopkInputs {
    Tensor input;
    std::optional<Tensor> indices;
    std::optional<std::tuple<Tensor, Tensor>> preallocated_outputs;
};

}  // namespace ttnn::prim
