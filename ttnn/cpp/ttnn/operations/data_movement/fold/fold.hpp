// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <variant>

#include "ttnn/core.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"

#include "device/fold_device_op.hpp"

namespace ttnn {
namespace operations::data_movement {

struct FoldOperation {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        uint32_t stride_h,
        uint32_t stride_w,
        bool use_transpose_as_fold = false,
        const std::optional<const ttnn::Shape>& output_shape = std::nullopt,
        std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>, std::array<uint32_t, 6>> padding =
            std::array<uint32_t, 2>{0, 0},
        const std::optional<CoreRangeSet>& core_grid = std::nullopt,
        const std::optional<MemoryConfig>& override_memory_config = std::nullopt);
};

}  // namespace operations::data_movement

constexpr auto fold = register_operation<"ttnn::fold", operations::data_movement::FoldOperation>();

}  // namespace ttnn
