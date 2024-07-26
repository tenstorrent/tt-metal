// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/ccl/reduce_scatter/device/reduce_scatter_op.hpp"
#include "ttnn/multi_device.hpp"

namespace ttnn {
namespace operations {
namespace ccl {

struct ExecuteReduceScatter {
    static std::vector<ttnn::Tensor> operator()(
        const std::vector<ttnn::Tensor>& input_tensors,
        const uint32_t scatter_dim,
        ReduceOpMath math_op,
        const uint32_t num_links = 1,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt) {
        MemoryConfig out_memory_config = memory_config.value_or(input_tensors.at(0).memory_config());
        return ttnn::operations::ccl::reduce_scatter(input_tensors, scatter_dim, math_op, num_links, out_memory_config);
    }
};

}  // namespace ccl
}  // namespace operations

constexpr auto reduce_scatter =
    ttnn::register_operation<"ttnn::reduce_scatter", ttnn::operations::ccl::ExecuteReduceScatter>();

}  // namespace ttnn
