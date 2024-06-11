// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/operations/depthwise_conv1d.hpp"
#include "ttnn/types.hpp"

namespace py = pybind11;

namespace ttnn {
namespace operations {
namespace depthwise_conv1d {

void py_module(py::module& module) {
    module.def(
        "depthwise_conv1d",
        [](const ttnn::Tensor& input_tensor,
            const ttnn::Tensor& weight_tensor,
            ttnn::Device& device,
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
            std::optional<const DepthwiseConv1dConfig> conv_config_ = std::nullopt) -> std::tuple<ttnn::Tensor, uint32_t, uint32_t, ttnn::Tensor, std::optional<ttnn::Tensor>> {
            return ttnn::operations::depthwise_conv1d::depthwise_conv1d(
                input_tensor, weight_tensor, device, in_channels, out_channels, batch_size, input_height, input_width, kernel_size, stride, padding, dilation,
                    groups, bias_tensor, conv_config_);
        },
        py::kw_only(),
        py::arg("input_tensor"),
        py::arg("weight_tensor"),
        py::arg("device"),
        py::arg("in_channels"),
        py::arg("out_channels"),
        py::arg("batch_size"),
        py::arg("input_height"),
        py::arg("input_width"),
        py::arg("kernel_size"),
        py::arg("stride"),
        py::arg("padding"),
        py::arg("dilation"),
        py::arg("groups"),
        py::arg("bias_tensor") = std::nullopt,
        py::arg("conv_config") = std::nullopt);

    auto py_conv_config = py::class_<DepthwiseConv1dConfig>(module, "DepthwiseConv1dConfig");
    py_conv_config.def(
            py::init<MathFidelity, DataType, DataType, bool, bool, bool, string, uint32_t, bool, bool, uint32_t, bool, bool, bool, std::optional<CoreRangeSet>, bool, Layout>(),
            py::kw_only(),
            py::arg("math_fidelity") = MathFidelity::HiFi4,
            py::arg("dtype") = DataType::BFLOAT16,
            py::arg("weights_dtype") = DataType::BFLOAT16,
            py::arg("math_approx_mode_enabled") = true,
            py::arg("fp32_dest_acc_enabled") = false,
            py::arg("packer_l1_accum_enabled") = false,
            py::arg("activation") = "",
            py::arg("input_channels_alignment") = 32,
            py::arg("deallocate_activation") = false,
            py::arg("reallocate_halo_output") = false,
            py::arg("act_block_h_override") = 0,
            py::arg("reshard_if_not_optimal") = false,
            py::arg("override_sharding_config") = false,
            py::arg("height_sharding") = true,
            py::arg("core_grid") = std::nullopt,
            py::arg("transpose_shards") = true,
            py::arg("output_layout") = Layout::TILE
        );
        py_conv_config.def_readwrite("math_fidelity", &DepthwiseConv1dConfig::math_fidelity);
        py_conv_config.def_readwrite("dtype", &DepthwiseConv1dConfig::dtype);
        py_conv_config.def_readwrite("weights_dtype", &DepthwiseConv1dConfig::weights_dtype);
        py_conv_config.def_readwrite("math_approx_mode_enabled", &DepthwiseConv1dConfig::math_approx_mode_enabled);
        py_conv_config.def_readwrite("fp32_dest_acc_enabled", &DepthwiseConv1dConfig::fp32_dest_acc_enabled);
        py_conv_config.def_readwrite("packer_l1_accum_enabled", &DepthwiseConv1dConfig::packer_l1_accum_enabled);
        py_conv_config.def_readwrite("activation", &DepthwiseConv1dConfig::activation);
        py_conv_config.def_readwrite("input_channels_alignment", &DepthwiseConv1dConfig::input_channels_alignment);
        py_conv_config.def_readwrite("deallocate_activation", &DepthwiseConv1dConfig::deallocate_activation);
        py_conv_config.def_readwrite("reallocate_halo_output", &DepthwiseConv1dConfig::reallocate_halo_output);
        py_conv_config.def_readwrite("act_block_h_override", &DepthwiseConv1dConfig::act_block_h_override);
        py_conv_config.def_readwrite("reshard_if_not_optimal", &DepthwiseConv1dConfig::reshard_if_not_optimal);
        py_conv_config.def_readwrite("override_sharding_config", &DepthwiseConv1dConfig::override_sharding_config);
        py_conv_config.def_readwrite("height_sharding", &DepthwiseConv1dConfig::height_sharding);
        py_conv_config.def_readwrite("core_grid", &DepthwiseConv1dConfig::core_grid);
        py_conv_config.def_readwrite("transpose_shards", &DepthwiseConv1dConfig::transpose_shards);
        py_conv_config.def_readwrite("output_layout", &DepthwiseConv1dConfig::output_layout);
}

}  // namespace depthwise_conv1d
}  // namespace operations
}  // namespace ttnn
