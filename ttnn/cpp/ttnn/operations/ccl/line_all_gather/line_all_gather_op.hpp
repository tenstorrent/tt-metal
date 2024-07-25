// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/ccl/line_all_gather/device/line_all_gather_op.hpp"
#include "ttnn/multi_device.hpp"

namespace ttnn {
namespace operations {
namespace ccl {

struct ExecuteLineAllGather {
    static ttnn::Tensor operator()(
        const ttnn::Tensor& input_tensor,
        const uint32_t dim,
        const uint32_t num_links = 1,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt) {
        return ttnn::operations::ccl::line_all_gather(input_tensor, dim, num_links, memory_config);
    }
};

}  // namespace ccl
}  // namespace operations

constexpr auto line_all_gather = ttnn::register_operation<"ttnn::line_all_gather", ttnn::operations::ccl::ExecuteLineAllGather>();

}  // namespace ttnn
