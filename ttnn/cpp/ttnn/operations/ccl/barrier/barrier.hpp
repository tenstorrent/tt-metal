// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations::ccl {

struct BarrierOperation {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring);
    static std::vector<ttnn::Tensor> invoke(
        const std::vector<ttnn::Tensor>& input_tensors,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring);
};
}  // namespace operations::ccl
constexpr auto barrier = ttnn::register_operation<"ttnn::barrier", ttnn::operations::ccl::BarrierOperation>();
}  // namespace ttnn
