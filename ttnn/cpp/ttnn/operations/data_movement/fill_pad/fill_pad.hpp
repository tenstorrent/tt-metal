// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/common/queue_id.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"

namespace ttnn {
namespace operations {
namespace data_movement {

struct FillPadOperation {
    static ttnn::Tensor invoke(
        QueueId queue_id,
        const ttnn::Tensor& input_tensor,
        float fill_value,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt);
};

}  // namespace data_movement
}  // namespace operations

constexpr auto fill_implicit_tile_padding =
    ttnn::register_operation<"ttnn::fill_implicit_tile_padding", ttnn::operations::data_movement::FillPadOperation>();
}  // namespace ttnn
