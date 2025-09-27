// cpp
// File: `ttnn/cpp/ttnn/operations/experimental/conv3d/conv3d_pybind.cpp`

// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "conv3d_pybind.hpp"

#include <optional>
#include <fmt/format.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "conv3d.hpp"
#include "ttnn-pybind/decorators.hpp"
#include "ttnn/types.hpp"
#include <tt-metalium/constants.hpp>

namespace ttnn::operations::experimental::conv3d::detail {

void py_bind_conv3d(py::module& module) {
    bind_registered_operation(
        module,
        ttnn::experimental::conv3d,
        R"doc(
        Applies a 3D convolution over an input signal composed of several input planes.

        Expects Input Tensor in [N, D, H, W, C] format.
        Expects Weight Tensor in [1, 1, kD * kH * kW * C_in, C_out] format.
        Expects Bias Tensor in [1, 1, 1, 32, C_out] format.
        Input must be in row major interleaved format.
        Output will be in row major interleaved format.

        This API aligns with PyTorch's signature by accepting kernel_size, stride, padding,
        padding_mode, groups, dtype and out_channels as direct keyword arguments.

        :param ttnn.Tensor input_tensor: Input tensor.
        :param ttnn.Tensor weight_tensor: Weight tensor.
        :param ttnn.Tensor bias_tensor: Optional bias tensor.
        :param int out_channels: Number of output channels.
        :param tuple kernel_size: Kernel size (kD, kH, kW).
        :param tuple stride: Stride (dT, dH, dW).
        :param tuple padding: Padding (pT, pH, pW).
        :param str padding_mode: Padding mode (e.g. 'zeros').
        :param int groups: Number of groups.
        :param ttnn.DataType dtype: Dtype for output/compute.
        :param ttnn.Conv3dConfig config: Optional low-level tuning config.
        :param ttnn.MemoryConfig memory_config: Optional memory config.
        :param ttnn.DeviceComputeKernelConfig compute_kernel_config: Optional compute kernel config.
        :param queue_id: Queue ID for the Conv3D operation.
        )doc",
        ttnn::pybind_overload_t{
            [](const decltype(ttnn::experimental::conv3d)& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::Tensor& weight_tensor,
               const std::optional<ttnn::Tensor>& bias_tensor,
               uint32_t out_channels,
               const std::array<uint32_t, 3>& kernel_size,
               const std::array<uint32_t, 3>& stride,
               const std::array<uint32_t, 3>& padding,
               const std::string& padding_mode,
               uint32_t groups,
               DataType dtype,
               const Conv3dConfig& config,
               const std::optional<const MemoryConfig>& memory_config,
               const std::optional<DeviceComputeKernelConfig>& compute_kernel_config,
               const QueueId& queue_id) {
                return self(
                    queue_id,
                    input_tensor,
                    weight_tensor,
                    bias_tensor,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    padding_mode,
                    groups,
                    dtype,
                    config,
                    memory_config,
                    compute_kernel_config);
            },
            py::kw_only(),
            py::arg("input_tensor"),
            py::arg("weight_tensor"),
            py::arg("bias_tensor") = std::nullopt,
            py::arg("out_channels"),
            py::arg("kernel_size"),
            py::arg("stride") = std::array<uint32_t, 3>{1, 1, 1},
            py::arg("padding") = std::array<uint32_t, 3>{0, 0, 0},
            py::arg("padding_mode") = "zeros",
            py::arg("groups") = 1,
            py::arg("dtype") = DataType::BFLOAT16,
            py::arg("config") = Conv3dConfig(),
            py::arg("memory_config") = std::nullopt,
            py::arg("compute_kernel_config") = std::nullopt,
            py::arg("queue_id") = 0});

    auto py_conv3d_config = py::class_<Conv3dConfig>(
        module,
        "Conv3dConfig",
        R"doc(
            Low-level configuration for Conv3D (tuning / blocking). Kernel/stride/padding/etc.
            are exposed on the op-level and should not be provided here in normal usage.
        )doc")
        .def(py::init<>())
        .def(
            py::init<
                DataType,
                DataType,
                Layout,
                uint32_t,
                uint32_t,
                uint32_t,
                uint32_t,
                uint32_t,
                CoreCoord>(),
            py::kw_only(),
            py::arg("dtype") = DataType::BFLOAT16,
            py::arg("weights_dtype") = DataType::BFLOAT16,
            py::arg("output_layout") = Layout::ROW_MAJOR,
            py::arg("T_out_block") = 1,
            py::arg("W_out_block") = 1,
            py::arg("H_out_block") = 1,
            py::arg("C_out_block") = 1,
            py::arg("C_in_block") = 1,
            py::arg("core_coord") = CoreCoord(),
            py::arg("compute_with_storage_grid_size") = uint32_t{0}
        )
        .def_readwrite("dtype", &Conv3dConfig::dtype)
        .def_readwrite("weights_dtype", &Conv3dConfig::weights_dtype)
        .def_readwrite("output_layout", &Conv3dConfig::output_layout)
        .def_readwrite("T_out_block", &Conv3dConfig::T_out_block)
        .def_readwrite("W_out_block", &Conv3dConfig::W_out_block)
        .def_readwrite("H_out_block", &Conv3dConfig::H_out_block)
        .def_readwrite("C_out_block", &Conv3dConfig::C_out_block)
        .def_readwrite("C_in_block", &Conv3dConfig::C_in_block)
        .def_readwrite("core_coord", &Conv3dConfig::core_coord)
        .def_readwrite("compute_with_storage_grid_size", &Conv3dConfig::compute_with_storage_grid_size)
        .def("__repr__", [](const Conv3dConfig& config) { return fmt::format("{}", config); });
}

}  // namespace ttnn::operations::experimental::conv3d::detail