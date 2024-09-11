// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/ccl/ccl_fabric.hpp"

namespace ttnn {
namespace operations {
namespace ccl {

struct ExecuteLineAllGather {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const uint32_t dim,
        const uint32_t num_links = 1,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        const ttnn::ccl::OpFabricMode op_fabric_mode = ttnn::ccl::OpFabricMode::TEMPORARY_EDM);

    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const uint32_t dim,
        const uint32_t cluster_axis,
        const MeshDevice& mesh_device,
        const uint32_t num_links = 1,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        const ttnn::ccl::OpFabricMode op_fabric_mode = ttnn::ccl::OpFabricMode::TEMPORARY_EDM
        );
};

}  // namespace ccl
}  // namespace operations

constexpr auto line_all_gather = ttnn::register_operation<"ttnn::line_all_gather", ttnn::operations::ccl::ExecuteLineAllGather>();

}  // namespace ttnn
