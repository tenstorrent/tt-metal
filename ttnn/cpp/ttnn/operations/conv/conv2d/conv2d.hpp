// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <optional>
#include <stdexcept>
#include <tuple>
#include <variant>

#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/conv/conv2d/device/conv2d_op.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/layout/layout.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/conv/conv_types.hpp"
#include "ttnn/operations/conv/conv2d/conv2d_utils.hpp"
namespace ttnn {

namespace operations::conv {
namespace conv2d {

using OutputHeight = uint32_t;
using OutputWidth = uint32_t;
using Result = std::tuple<ttnn::Tensor, OutputHeight, OutputWidth, ttnn::Tensor, std::optional<ttnn::Tensor>>;
using ResultWithOptions = std::variant<
    ttnn::Tensor,
    std::tuple<ttnn::Tensor, std::tuple<OutputHeight, OutputWidth>>,
    std::tuple<ttnn::Tensor, std::tuple<ttnn::Tensor, std::optional<ttnn::Tensor>>>,
    std::tuple<
        ttnn::Tensor,
        std::tuple<OutputHeight, OutputWidth>,
        std::tuple<ttnn::Tensor, std::optional<ttnn::Tensor>>>>;

Result conv2d_L1(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    MeshDevice* device,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t batch_size,
    uint32_t input_height,
    uint32_t input_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>> padding,
    std::array<uint32_t, 2> dilation,
    uint32_t groups,
    const std::optional<const DataType>& dtype = std::nullopt,
    const std::optional<const ttnn::Tensor>& bias_tensor = std::nullopt,
    const std::optional<const Conv2dConfig>& conv_config_ = std::nullopt,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_ = std::nullopt,
    const std::optional<const MemoryConfig>& memory_config = std::nullopt);

Result conv2d_DRAM(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    MeshDevice* device,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t batch_size,
    uint32_t input_height,
    uint32_t input_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>> padding,
    std::array<uint32_t, 2> dilation,
    uint32_t groups,
    const std::optional<const DataType>& dtype = std::nullopt,
    const std::optional<const ttnn::Tensor>& bias_tensor = std::nullopt,
    const std::optional<const Conv2dConfig>& conv_config_ = std::nullopt,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_ = std::nullopt,
    const std::optional<const MemoryConfig>& memory_config_ = std::nullopt,
    const std::optional<const Conv2dSliceConfig>& dram_slice_config_ = std::nullopt);

ResultWithOptions conv2d(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    MeshDevice* device,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t batch_size,
    uint32_t input_height,
    uint32_t input_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>> padding,
    std::array<uint32_t, 2> dilation,
    uint32_t groups,
    const std::optional<const DataType>& dtype = std::nullopt,
    const std::optional<const ttnn::Tensor>& bias_tensor = std::nullopt,
    const std::optional<const Conv2dConfig>& conv_config_ = std::nullopt,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_ = std::nullopt,
    const std::optional<const MemoryConfig>& memory_config_ = std::nullopt,
    const std::optional<const Conv2dSliceConfig>& dram_slice_config_ = std::nullopt,
    bool return_output_dim = false,
    bool return_weights_and_bias = false);

struct Conv2dOperation {
    static ResultWithOptions invoke(
        const ttnn::Tensor& input_tensor,
        const ttnn::Tensor& weight_tensor,
        MeshDevice* device,
        uint32_t in_channels,
        uint32_t out_channels,
        uint32_t batch_size,
        uint32_t input_height,
        uint32_t input_width,
        std::array<uint32_t, 2> kernel_size,
        std::array<uint32_t, 2> stride = std::array<uint32_t, 2>{1, 1},
        std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>> padding = std::array<uint32_t, 2>{0, 0},
        std::array<uint32_t, 2> dilation = std::array<uint32_t, 2>{1, 1},
        uint32_t groups = 1,
        const std::optional<const DataType>& dtype = std::nullopt,
        const std::optional<const ttnn::Tensor>& bias_tensor = std::nullopt,
        const std::optional<const Conv2dConfig>& conv_config_ = std::nullopt,
        const std::optional<const DeviceComputeKernelConfig>& compute_config_ = std::nullopt,
        const std::optional<const MemoryConfig>& memory_config_ = std::nullopt,
        const std::optional<const Conv2dSliceConfig>& dram_slice_config_ = std::nullopt,
        bool return_output_dim = false,
        bool return_weights_and_bias = false);
};

class Conv2dSliceAttr : public ttnn::operations::op_slicing::OpSliceAttr {
    using OptionalRefTensor = std::optional<std::reference_wrapper<ttnn::Tensor>>;
    using RefTensor = std::reference_wrapper<ttnn::Tensor>;

    uint32_t batch_size;
    IOShape input_shape;
    uint32_t input_channels;
    uint32_t output_channels;
    std::array<uint32_t, 2> kernel_size;
    std::array<uint32_t, 2> stride;
    std::array<uint32_t, 4> padding_n4;
    std::array<uint32_t, 2> dilation;
    uint32_t groups;
    Layout input_layout;
    DataType input_dtype;
    DataType output_dtype;
    Tensor& weight_tensor;
    OptionalRefTensor bias_tensor;
    Conv2dConfig conv_config;
    DeviceComputeKernelConfig compute_config;
    MeshDevice* device;

public:
    Conv2dSliceAttr(
        uint32_t batch_size,
        IOShape input_shape,
        uint32_t input_channels,
        uint32_t output_channels,
        std::array<uint32_t, 2> kernel_size,
        std::array<uint32_t, 2> stride,
        std::array<uint32_t, 4> padding_n4,
        std::array<uint32_t, 2> dilation,
        uint32_t groups,
        Layout input_layout,
        DataType input_dtype,
        DataType output_dtype,
        Tensor& weight_tensor,
        OptionalRefTensor bias_tensor,
        Conv2dConfig& conv_config,
        DeviceComputeKernelConfig& compute_config,
        MeshDevice* device);
    std::tuple<IOShape, IOShape> get_input_slice(IOShape output_slice_start, IOShape output_slice_end) override;
    uint32_t get_L1_usage() override;
    tt::tt_metal::MemoryConfig get_input_memory_config(IOShape output_slice_start, IOShape output_slice_end) override;
    ttnn::Tensor run_L1_op(
        const ttnn::Tensor& sliced_input_tensor, IOShape output_slice_start, IOShape output_slice_end) override;
    std::string name() override;
};

}  // namespace conv2d
}  // namespace operations::conv
}  // namespace ttnn

namespace ttnn {
constexpr auto conv2d = ttnn::register_operation<"ttnn::conv2d", operations::conv::conv2d::Conv2dOperation>();
}
