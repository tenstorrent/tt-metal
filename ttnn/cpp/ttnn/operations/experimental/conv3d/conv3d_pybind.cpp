// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
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
        Applies a 3D convolution over an input signal composed of several input planes. \
        Expects Input Tensor in [N, D, H, W, C] format.  \
        Expects Weight Tensor in [1, 1, kD * kH * kW * C_in, C_out] format. \
        Expects Bias Tensor in [1, 1, 1, 32, C_out] format. \
        Input must be in row major interleaved format. \
        Output will be in row major interleaved format.

        :param ttnn.Tensor input_tensor: Input tensor.
        :param ttnn.Tensor weight_tensor: Weight tensor.
        :param ttnn.Tensor bias_tensor: Bias tensor.
        :param ttnn.Conv3dConfig config: Configuration for the Conv3D operation.
        :param ttnn.MemoryConfig memory_config: Memory configuration for the output of the Conv3D operation.
        :param ttnn.DeviceComputeKernelConfig compute_kernel_config: Compute kernel configuration for the Conv3D operation.
        :param queue_id: Queue ID for the Conv3D operation.

        :return: Output tensor after applying the Conv3D operation.
        :rtype: ttnn.Tensor
        )doc",
        ttnn::pybind_overload_t{
            [](const decltype(ttnn::experimental::conv3d)& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::Tensor& weight_tensor,
               const std::optional<ttnn::Tensor>& bias_tensor,
               const Conv3dConfig& config,
               const std::optional<const MemoryConfig>& memory_config,
               const std::optional<DeviceComputeKernelConfig>& compute_kernel_config,
               const QueueId& queue_id) {
                return self(
                    queue_id, input_tensor, weight_tensor, bias_tensor, config, memory_config, compute_kernel_config);
            },
            py::kw_only(),
            py::arg("input_tensor"),
            py::arg("weight_tensor"),
            py::arg("bias_tensor") = std::nullopt,
            py::arg("config"),
            py::arg("memory_config") = std::nullopt,
            py::arg("compute_kernel_config") = std::nullopt,
            py::arg("queue_id") = 0});

    auto py_conv3d_config = py::class_<Conv3dConfig>(
                                module,
                                "Conv3dConfig",
                                R"doc(
                            Configuration for the Conv3D operation.
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
                                        uint32_t,
                                        const std::array<uint32_t, 3>&,
                                        const std::array<uint32_t, 3>&,
                                        const std::array<uint32_t, 3>&,
                                        const std::string&,
                                        uint32_t,
                                        CoreCoord>(),
                                    py::kw_only(),
                                    py::arg("dtype") = DataType::BFLOAT16,
                                    py::arg("weights_dtype") = DataType::BFLOAT16,
                                    py::arg("output_layout") = Layout::ROW_MAJOR,
                                    py::arg("T_out_block") = 1,
                                    py::arg("W_out_block") = 1,
                                    py::arg("H_out_block") = 1,
                                    py::arg("C_out_block") = 0,
                                    py::arg("C_in_block") = 0,
                                    py::arg("output_channels"),
                                    py::arg("kernel_size"),
                                    py::arg("stride"),
                                    py::arg("padding"),
                                    py::arg("padding_mode") = "zeros",
                                    py::arg("groups") = 1,
                                    py::arg("compute_with_storage_grid_size") = CoreCoord{1, 1});

    py_conv3d_config.def_readwrite("dtype", &Conv3dConfig::dtype, "");
    py_conv3d_config.def_readwrite("weights_dtype", &Conv3dConfig::weights_dtype, "");
    py_conv3d_config.def_readwrite("output_layout", &Conv3dConfig::output_layout, "");
    py_conv3d_config.def_readwrite("T_out_block", &Conv3dConfig::T_out_block, "");
    py_conv3d_config.def_readwrite("W_out_block", &Conv3dConfig::W_out_block, "");
    py_conv3d_config.def_readwrite("H_out_block", &Conv3dConfig::H_out_block, "");
    py_conv3d_config.def_readwrite("C_out_block", &Conv3dConfig::C_out_block, "");
    py_conv3d_config.def_readwrite("C_in_block", &Conv3dConfig::C_in_block, "");
    py_conv3d_config.def_readwrite("output_channels", &Conv3dConfig::output_channels, "");
    py_conv3d_config.def_readwrite("kernel_size", &Conv3dConfig::kernel_size, "");
    py_conv3d_config.def_readwrite("stride", &Conv3dConfig::stride, "");
    py_conv3d_config.def_readwrite("padding", &Conv3dConfig::padding, "");
    py_conv3d_config.def_readwrite("padding_mode", &Conv3dConfig::padding_mode, "");
    py_conv3d_config.def_readwrite("groups", &Conv3dConfig::groups, "");
    py_conv3d_config.def_readwrite("compute_with_storage_grid_size", &Conv3dConfig::compute_with_storage_grid_size, "");

    py_conv3d_config.def("__repr__", [](const Conv3dConfig& config) { return fmt::format("{}", config); });
}

}  // namespace ttnn::operations::experimental::conv3d::detail
