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

        Args:
            input_tensor (ttnn.Tensor): Input tensor.
            weight_tensor (ttnn.Tensor): Weight tensor.
            config (ttnn.Conv3dConfig): Configuration for the Conv3D operation.

        Keyword Args:
            bias_tensor (ttnn.Tensor, optional): Bias tensor.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the output of the Conv3D operation.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): Compute kernel configuration for the Conv3D operation.

        Returns:
            ttnn.Tensor: Output tensor after applying the Conv3D operation.
        )doc",
        ttnn::pybind_overload_t{
            [](const decltype(ttnn::experimental::conv3d)& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::Tensor& weight_tensor,
               const std::optional<ttnn::Tensor>& bias_tensor,
               const Conv3dConfig& config,
               const tt::tt_metal::DataType& dtype,
               const uint32_t& output_channels,
               const std::array<uint32_t, 3>& kernel_size,
               const std::array<uint32_t, 3>& stride,
               const std::array<uint32_t, 3>& padding,
               const std::array<uint32_t, 3>& dilation,
               const std::string& padding_mode,
               const uint32_t& groups,
               const std::optional<const MemoryConfig>& memory_config,
               const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
                return self(
                    input_tensor,
                    weight_tensor,
                    bias_tensor,
                    config,
                    dtype,
                    output_channels,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    padding_mode,
                    groups,
                    memory_config,
                    compute_kernel_config);
            },
            py::kw_only(),
            py::arg("input_tensor"),
            py::arg("weight_tensor"),
            py::arg("bias_tensor") = std::nullopt,
            py::arg("config"),
            py::arg("dtype"),
            py::arg("output_channels"),
            py::arg("kernel_size"),
            py::arg("stride") = std::array<uint32_t, 3>{1, 1, 1},
            py::arg("padding") = std::array<uint32_t, 3>{0, 0, 0},
            py::arg("dilation") = std::array<uint32_t, 3>{1, 1, 1},
            py::arg("padding_mode") = "zeros",
            py::arg("groups") = 1,
            py::arg("memory_config") = std::nullopt,
            py::arg("compute_kernel_config") = std::nullopt});

    auto py_conv3d_config =
        py::class_<Conv3dConfig>(
            module,
            "Conv3dConfig",
            R"doc(
                            Configuration for the Conv3D operation.
                            )doc")
            .def(py::init<>())
            .def(
                py::init<DataType, Layout, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, CoreCoord>(),
                py::kw_only(),
                py::arg("weights_dtype") = DataType::BFLOAT16,
                py::arg("output_layout") = Layout::ROW_MAJOR,
                py::arg("T_out_block") = 1,
                py::arg("W_out_block") = 1,
                py::arg("H_out_block") = 1,
                py::arg("C_out_block") = 0,
                py::arg("C_in_block") = 0,
                py::arg("compute_with_storage_grid_size") = CoreCoord{1, 1});

    py_conv3d_config.def_readwrite("weights_dtype", &Conv3dConfig::weights_dtype, "");
    py_conv3d_config.def_readwrite("output_layout", &Conv3dConfig::output_layout, "");
    py_conv3d_config.def_readwrite("T_out_block", &Conv3dConfig::T_out_block, "");
    py_conv3d_config.def_readwrite("W_out_block", &Conv3dConfig::W_out_block, "");
    py_conv3d_config.def_readwrite("H_out_block", &Conv3dConfig::H_out_block, "");
    py_conv3d_config.def_readwrite("C_out_block", &Conv3dConfig::C_out_block, "");
    py_conv3d_config.def_readwrite("C_in_block", &Conv3dConfig::C_in_block, "");
    py_conv3d_config.def_readwrite("compute_with_storage_grid_size", &Conv3dConfig::compute_with_storage_grid_size, "");

    py_conv3d_config.def("__repr__", [](const Conv3dConfig& config) { return fmt::format("{}", config); });
}

}  // namespace ttnn::operations::experimental::conv3d::detail
