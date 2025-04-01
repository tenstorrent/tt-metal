// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/constants.hpp>
#include "cpp/pybind11/decorators.hpp"

#include "conv1d_pybind.hpp"
#include "conv1d.hpp"
#include "ttnn/types.hpp"

namespace py = pybind11;

namespace ttnn {
namespace operations::conv {
namespace conv1d {

void py_bind_conv1d(py::module& module) {
    bind_registered_operation(
        module,
        ttnn::conv1d,
        R"doc(
        Conv 1D
        +-------------------+-------------------------------+-----------------------------+-------------+----------+
        | Argument          | Description                   | Data type                   | Valid range | Required |
        +===================+===============================+=============================+=============+==========+
        | input_tensor      | Input activations tensor      | Tensor                      |             | Yes      |
        | weight_tensor     | Weight tensor                 | Tensor                      |             | Yes      |
        | device            | Device                        | Device                      |             | Yes      |
        | in_channels       | Input channels                | uint32_t                    |             | Yes      |
        | out_channels      | Output channels               | uint32_t                    |             | Yes      |
        | batch_size        | Batch size                    | uint32_t                    |             | Yes      |
        | input_length      | Input length                  | uint32_t                    |             | Yes      |
        | kernel_size       | Kernel size                   | uint32_t                    |             | Yes      |
        | stride            | Stride                        | uint32_t                    |             | Yes      |
        | padding           | Padding                       | uint32_t                    |             | Yes      |
        | dilation          | Dilation                      | uint32_t                    |             | No       |
        | groups            | Groups                        | uint32_t                    |             | No       |
        | bias_tensor       | Bias tensor                   | Tensor                      |             | No       |
        | conv_config       | Conv config                   | Conv1dConfig                |             | No       |
        | compute_config    | Compute config                | DeviceComputeKernelConfig   |             | No       |
        | memory_config     | Memory config                 | MemoryConfig                |             | No       |
        +-------------------+-------------------------------+-----------------------------+------------------------+

        )doc",
        ttnn::pybind_overload_t{
            [](const decltype(ttnn::conv1d)& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::Tensor& weight_tensor,
               ttnn::IDevice* device,
               uint32_t in_channels,
               uint32_t out_channels,
               uint32_t batch_size,
               uint32_t input_length,
               uint32_t kernel_size,
               uint32_t stride,
               std::variant<std::array<uint32_t, 2>, uint32_t> padding,
               uint32_t dilation,
               uint32_t groups,
               std::optional<const ttnn::Tensor> bias_tensor,
               const std::optional<const Conv1dConfig>& conv_config,
               const std::optional<const DeviceComputeKernelConfig>& compute_config,
               const std::optional<const MemoryConfig>& memory_config,
               QueueId queue_id) -> Result {
                return self(
                    queue_id,
                    input_tensor,
                    weight_tensor,
                    device,
                    in_channels,
                    out_channels,
                    batch_size,
                    input_length,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    groups,
                    bias_tensor,
                    conv_config,
                    compute_config,
                    memory_config);
            },
            py::kw_only(),
            py::arg("input_tensor"),
            py::arg("weight_tensor"),
            py::arg("device"),
            py::arg("in_channels"),
            py::arg("out_channels"),
            py::arg("batch_size"),
            py::arg("input_length"),
            py::arg("kernel_size"),
            py::arg("stride"),
            py::arg("padding"),
            py::arg("dilation") = 1,
            py::arg("groups") = 1,
            py::arg("bias_tensor") = std::nullopt,
            py::arg("conv_config") = std::nullopt,
            py::arg("compute_config") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId},

        ttnn::pybind_overload_t{
            [](const decltype(ttnn::conv1d)& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::Tensor& weight_tensor,
               ttnn::MeshDevice* device,
               uint32_t in_channels,
               uint32_t out_channels,
               uint32_t batch_size,
               uint32_t input_length,
               uint32_t kernel_size,
               uint32_t stride,
               std::variant<std::array<uint32_t, 2>, uint32_t> padding,
               uint32_t dilation,
               uint32_t groups,
               std::optional<const ttnn::Tensor> bias_tensor,
               const std::optional<const Conv1dConfig>& conv_config,
               const std::optional<const DeviceComputeKernelConfig>& compute_config,
               const std::optional<const MemoryConfig>& memory_config,
               QueueId queue_id) -> Result {
                return self(
                    queue_id,
                    input_tensor,
                    weight_tensor,
                    device,
                    in_channels,
                    out_channels,
                    batch_size,
                    input_length,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    groups,
                    bias_tensor,
                    conv_config,
                    compute_config,
                    memory_config);
            },
            py::kw_only(),
            py::arg("input_tensor"),
            py::arg("weight_tensor"),
            py::arg("device"),
            py::arg("in_channels"),
            py::arg("out_channels"),
            py::arg("batch_size"),
            py::arg("input_length"),
            py::arg("kernel_size"),
            py::arg("stride"),
            py::arg("padding"),
            py::arg("dilation") = 1,
            py::arg("groups") = 1,
            py::arg("bias_tensor") = std::nullopt,
            py::arg("conv_config") = std::nullopt,
            py::arg("compute_config") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId});
}
}  // namespace conv1d
}  // namespace operations::conv
}  // namespace ttnn
