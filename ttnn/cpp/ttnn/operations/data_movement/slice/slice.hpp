// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations {
namespace data_movement {

struct SliceOperation {
    static ttnn::Tensor invoke(
        uint8_t queue_id,
        const ttnn::Tensor& input_tensor,
        tt::tt_metal::LegacyShape output_tensor_start,
        tt::tt_metal::LegacyShape output_tensor_end,
        const std::optional<MemoryConfig>& memory_config_arg);

    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        tt::tt_metal::LegacyShape output_tensor_start,
        tt::tt_metal::LegacyShape output_tensor_end,
        const std::optional<MemoryConfig>& memory_config_arg);

    static ttnn::Tensor invoke(
        uint8_t queue_id,
        const ttnn::Tensor& input_tensor,
        tt::tt_metal::Array1D output_tensor_start,
        tt::tt_metal::Array1D output_tensor_end,
        const std::optional<MemoryConfig>& memory_config_arg);

    static ttnn::Tensor invoke(
        uint8_t queue_id,
        const ttnn::Tensor& input_tensor,
        tt::tt_metal::Array4D output_tensor_start,
        tt::tt_metal::Array4D output_tensor_end,
        const std::optional<MemoryConfig>& memory_config_arg);

    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        tt::tt_metal::Array4D output_tensor_start,
        tt::tt_metal::Array4D output_tensor_end,
        const std::optional<MemoryConfig>& memory_config_arg);

    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        tt::tt_metal::Array4D output_tensor_start,
        tt::tt_metal::Array4D output_tensor_end);
};

}  // namespace data_movement
}  // namespace operations

constexpr auto slice =
    ttnn::register_operation_with_auto_launch_op<"ttnn::slice", ttnn::operations::data_movement::SliceOperation>();

}  // namespace ttnn
