// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ttnn/decorators.hpp>
#include <ttnn/tensor/tensor.hpp>

namespace ttnn::operations::experimental::cnn {

struct ExecuteConvertToHWC {
    static ttnn::Tensor invoke(
        QueueId queue_id,
        const Tensor& a,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<DataType>& dtype = std::nullopt);
};

}  // namespace ttnn::operations::experimental::cnn

namespace ttnn::experimental {

constexpr auto convert_to_hwc = ttnn::register_operation<
    "ttnn::experimental::convert_to_hwc",
    ttnn::operations::experimental::cnn::ExecuteConvertToHWC>();

}  // namespace ttnn::experimental
