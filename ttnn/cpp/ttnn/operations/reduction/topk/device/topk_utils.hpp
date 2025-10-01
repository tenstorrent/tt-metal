// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations {
namespace reduction {
namespace topk_utils {

uint32_t largest_power_of_two(uint32_t x);

struct TopKCoreConfig {
    uint16_t num_cores = 0;
    uint16_t split_size = 0;
    uint16_t rem = 0;
    uint16_t final_input_size = 0;
    uint16_t selected_x = 0;
    uint16_t selected_y = 0;
};

std::optional<TopKCoreConfig> find_topk_core_config(
    uint32_t width,
    uint32_t min_dim,
    uint32_t max_dim,
    uint32_t k,
    const tt::tt_metal::CoreRange& core_range,
    uint32_t l1_size,
    uint32_t value_tile_size,
    uint32_t index_tile_size);

bool verify_multi_core_cost(
    const std::vector<ttnn::Tensor>& input_tensors,
    uint32_t width,
    uint32_t min_dim,
    uint32_t max_dim,
    uint32_t k,
    const tt::tt_metal::CoreRange& core_range,
    uint32_t l1_size,
    uint32_t value_tile_size,
    uint32_t index_tile_size);

bool verify_single_core_cost(const std::vector<ttnn::Tensor>& input_tensors, uint32_t k, bool uint16_output);

}  // namespace topk_utils
}  // namespace reduction
}  // namespace operations
}  // namespace ttnn
