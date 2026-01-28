// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "conv3d_nanobind.hpp"

#include <array>
#include <cstdint>
#include <optional>

#include <fmt/format.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include "conv3d.hpp"
#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/types.hpp"
#include <tt-metalium/constants.hpp>

namespace ttnn::operations::experimental::conv3d::detail {

void bind_conv3d(nb::module_& mod) {
    bind_registered_operation(
        mod,
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
        ttnn::nanobind_overload_t{
            [](const decltype(ttnn::experimental::conv3d)& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::Tensor& weight_tensor,
               const std::optional<ttnn::Tensor>& bias_tensor,
               const ttnn::experimental::prim::Conv3dConfig& config,
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
            nb::kw_only(),
            nb::arg("input_tensor"),
            nb::arg("weight_tensor"),
            nb::arg("bias_tensor") = nb::none(),
            nb::arg("config"),
            nb::arg("dtype"),
            nb::arg("output_channels"),
            nb::arg("kernel_size"),
            nb::arg("stride") = std::array<uint32_t, 3>{1, 1, 1},
            nb::arg("padding") = std::array<uint32_t, 3>{0, 0, 0},
            nb::arg("dilation") = std::array<uint32_t, 3>{1, 1, 1},
            nb::arg("padding_mode") = "zeros",
            nb::arg("groups") = 1,
            nb::arg("memory_config") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none()});

    auto py_conv3d_config =
        nb::class_<ttnn::experimental::prim::Conv3dConfig>(
            mod,
            "Conv3dConfig",
            R"doc(
                            Configuration for the Conv3D operation.
                            )doc")
            .def(nb::init<>())
            .def(
                nb::init<DataType, Layout, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, CoreCoord>(),
                nb::kw_only(),
                nb::arg("weights_dtype") = DataType::BFLOAT16,
                nb::arg("output_layout") = Layout::ROW_MAJOR,
                nb::arg("T_out_block") = 1,
                nb::arg("W_out_block") = 1,
                nb::arg("H_out_block") = 1,
                nb::arg("C_out_block") = 0,
                nb::arg("C_in_block") = 0,
                nb::arg("compute_with_storage_grid_size") = nb::cast(CoreCoord{1, 1}));

    py_conv3d_config.def_rw("weights_dtype", &ttnn::experimental::prim::Conv3dConfig::weights_dtype, "");
    py_conv3d_config.def_rw("output_layout", &ttnn::experimental::prim::Conv3dConfig::output_layout, "");
    py_conv3d_config.def_rw("T_out_block", &ttnn::experimental::prim::Conv3dConfig::T_out_block, "");
    py_conv3d_config.def_rw("W_out_block", &ttnn::experimental::prim::Conv3dConfig::W_out_block, "");
    py_conv3d_config.def_rw("H_out_block", &ttnn::experimental::prim::Conv3dConfig::H_out_block, "");
    py_conv3d_config.def_rw("C_out_block", &ttnn::experimental::prim::Conv3dConfig::C_out_block, "");
    py_conv3d_config.def_rw("C_in_block", &ttnn::experimental::prim::Conv3dConfig::C_in_block, "");
    py_conv3d_config.def_rw(
        "compute_with_storage_grid_size", &ttnn::experimental::prim::Conv3dConfig::compute_with_storage_grid_size, "");

    py_conv3d_config.def(
        "__repr__", [](const ttnn::experimental::prim::Conv3dConfig& config) { return fmt::format("{}", config); });
}

}  // namespace ttnn::operations::experimental::conv3d::detail
