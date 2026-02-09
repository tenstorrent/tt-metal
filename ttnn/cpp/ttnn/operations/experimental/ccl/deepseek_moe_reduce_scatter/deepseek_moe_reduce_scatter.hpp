// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "ttnn/decorators.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"

namespace ttnn {
namespace operations::experimental::ccl {

struct ExecuteDeepseekMoEReduceScatter {
    static ttnn::Tensor invoke(
        const std::vector<ttnn::Tensor>& input_tensors,
        const tt::tt_metal::MemoryConfig& output_memory_config,
        int32_t dim,
        uint32_t num_links = 4,
        tt::tt_fabric::Topology topology = tt::tt_fabric::Topology::Ring,
        std::optional<uint32_t> cluster_axis = std::nullopt);
};

}  // namespace operations::experimental::ccl

namespace experimental {

constexpr auto deepseek_moe_reduce_scatter = ttnn::register_operation<
    "ttnn::experimental::deepseek_moe_reduce_scatter",
    ttnn::operations::experimental::ccl::ExecuteDeepseekMoEReduceScatter>();

}  // namespace experimental
}  // namespace ttnn
