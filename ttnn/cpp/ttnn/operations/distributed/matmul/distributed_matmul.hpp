// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include <tt-metalium/mesh_device.hpp>

namespace ttnn {
namespace operations::distributed {

struct ExecuteDistributedMatmul {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor_a,
        const ttnn::Tensor& input_tensor_b,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        std::optional<ttnn::DataType> dtype = std::nullopt);
};

}  // namespace operations::distributed

namespace distributed {
constexpr auto matmul =
    ttnn::register_operation<"ttnn::distributed::matmul", ttnn::operations::distributed::ExecuteDistributedMatmul>();
}  // namespace distributed

}  // namespace ttnn
