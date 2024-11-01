// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <optional>
#include <unordered_set>

#include "conv2d_utils.hpp"
#include "ttnn/core.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/operations/matmul/device/matmul_op.hpp"
#include "ttnn/types.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "tt_metal/common/math.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/operations/conv/conv2d/device/conv2d_op.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"
#include "ttnn/operations/sliding_window/halo/halo.hpp"

namespace ttnn {

namespace operations::conv {
namespace conv2d {

using OutputHeight = uint32_t;
using OutputWidth = uint32_t;
using Result = std::tuple<ttnn::Tensor, OutputHeight, OutputWidth, ttnn::Tensor, std::optional<ttnn::Tensor>>;

template <typename T>
Result conv2d(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    T * device,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t batch_size,
    uint32_t input_height,
    uint32_t input_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::array<uint32_t, 2> padding,
    std::array<uint32_t, 2> dilation,
    uint32_t groups,
    std::optional<const ttnn::Tensor> bias_tensor = std::nullopt,
    const std::optional<const Conv2dConfig>& conv_config_ = std::nullopt,
    const std::optional<const MemoryConfig>& memory_config = std::nullopt);

struct Conv2dOperation{
    static Result invoke(
        uint8_t queue_id,
        const ttnn::Tensor& input_tensor,
        const ttnn::Tensor& weight_tensor,
        Device * device,
        uint32_t in_channels,
        uint32_t out_channels,
        uint32_t batch_size,
        uint32_t input_height,
        uint32_t input_width,
        std::array<uint32_t, 2> kernel_size,
        std::array<uint32_t, 2> stride,
        std::array<uint32_t, 2> padding,
        std::array<uint32_t, 2> dilation,
        uint32_t groups,
        std::optional<const ttnn::Tensor> bias_tensor = std::nullopt,
        const std::optional<const Conv2dConfig>& conv_config_ = std::nullopt,
        const std::optional<const MemoryConfig>& memory_config = std::nullopt);

    static Result invoke(
        uint8_t queue_id,
        const ttnn::Tensor& input_tensor,
        const ttnn::Tensor& weight_tensor,
        MeshDevice * device,
        uint32_t in_channels,
        uint32_t out_channels,
        uint32_t batch_size,
        uint32_t input_height,
        uint32_t input_width,
        std::array<uint32_t, 2> kernel_size,
        std::array<uint32_t, 2> stride,
        std::array<uint32_t, 2> padding,
        std::array<uint32_t, 2> dilation,
        uint32_t groups,
        std::optional<const ttnn::Tensor> bias_tensor = std::nullopt,
        const std::optional<const Conv2dConfig>& conv_config_ = std::nullopt,
        const std::optional<const MemoryConfig>& memory_config = std::nullopt);
};
}  // namespace conv2d
}  // namespace operations::conv
}  // namespace ttnn

namespace ttnn{
    constexpr auto conv2d = ttnn::register_operation<"ttnn::conv2d", operations::conv::conv2d::Conv2dOperation>();
}
