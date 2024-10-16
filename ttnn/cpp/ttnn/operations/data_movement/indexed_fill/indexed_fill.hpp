// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations {
namespace data_movement {

struct IndexedFillOperation {
    static ttnn::Tensor invoke(uint8_t queue_id,
                               const ttnn::Tensor& batch_id,
                               const ttnn::Tensor& input_tensor_a,
                               const ttnn::Tensor& input_tensor_b,
                               const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
                               int64_t dim = 0);

    static ttnn::Tensor invoke(const ttnn::Tensor& batch_id,
                               const ttnn::Tensor& input_tensor_a,
                               const ttnn::Tensor& input_tensor_b,
                               const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
                               int64_t dim = 0);
};

}  // namespace data_movement
}  // namespace operations

constexpr auto indexed_fill =
    ttnn::register_operation<"ttnn::indexed_fill", ttnn::operations::data_movement::IndexedFillOperation>();

}  // namespace ttnn
