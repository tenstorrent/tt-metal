// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stddef.h>
#include <stdint.h>
#include <optional>

#include "cpp/ttnn/operations/ccl/ccl_host_types.hpp"
#include <tt-metalium/fabric_edm_types.hpp>
#include "ttnn/decorators.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"

namespace ttnn {
namespace operations {
namespace reduction {
enum class ReduceType;
}  // namespace reduction

namespace experimental {
namespace ccl {

struct ExecuteAllReduce {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        ttnn::operations::reduction::ReduceType math_op,
        const uint32_t num_links = 1,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring,
        const std::optional<size_t> num_workers = std::nullopt,
        const std::optional<size_t> num_buffers_per_channel = std::nullopt);
};

}  // namespace ccl
}  // namespace experimental
}  // namespace operations

namespace experimental {
constexpr auto all_reduce =
    ttnn::register_operation<"ttnn::experimental::all_reduce", ttnn::operations::experimental::ccl::ExecuteAllReduce>();

}  // namespace experimental

}  // namespace ttnn
