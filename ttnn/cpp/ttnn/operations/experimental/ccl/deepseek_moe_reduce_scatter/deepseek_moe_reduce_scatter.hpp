// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"

namespace ttnn::experimental {

ttnn::Tensor deepseek_moe_reduce_scatter(
    const std::vector<ttnn::Tensor>& input_tensors,
    const tt::tt_metal::MemoryConfig& output_memory_config,
    int32_t dim,
    uint32_t num_links = 4,
    tt::tt_fabric::Topology topology = tt::tt_fabric::Topology::Ring,
    std::optional<uint32_t> cluster_axis = std::nullopt);

}  // namespace ttnn::experimental
