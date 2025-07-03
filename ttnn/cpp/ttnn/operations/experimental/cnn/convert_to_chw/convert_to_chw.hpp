// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ttnn/decorators.hpp>
#include <ttnn/tensor/tensor.hpp>

namespace ttnn::operations::experimental::cnn {

struct ExecuteConvertToCHW {
    static ttnn::Tensor invoke(
        QueueId queue_id,
        const Tensor& a,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<DataType>& dtype = std::nullopt);
};

}  // namespace ttnn::operations::experimental::cnn

namespace ttnn::experimental {

constexpr auto convert_to_chw = ttnn::register_operation<
    "ttnn::experimental::convert_to_chw",
    ttnn::operations::experimental::cnn::ExecuteConvertToCHW>();

}  // namespace ttnn::experimental
