// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/operations/conv2d.hpp"
#include "ttnn/types.hpp"

namespace py = pybind11;

namespace ttnn {
namespace operations {
namespace conv2d {

void py_module(py::module& module) {
    module.def(
        "conv2d",
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
            std::optional<const Conv2dConfig> conv_config_ = std::nullopt) -> std::tuple<ttnn::Tensor, uint32_t, uint32_t, ttnn::Tensor, std::optional<ttnn::Tensor>> {
            return ttnn::operations::conv2d::conv2d(
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

    auto py_conv_config = py::class_<Conv2dConfig>(module, "Conv2dConfig");
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
        py_conv_config.def_readwrite("math_fidelity", &Conv2dConfig::math_fidelity);
        py_conv_config.def_readwrite("dtype", &Conv2dConfig::dtype);
        py_conv_config.def_readwrite("weights_dtype", &Conv2dConfig::weights_dtype);
        py_conv_config.def_readwrite("math_approx_mode_enabled", &Conv2dConfig::math_approx_mode_enabled);
        py_conv_config.def_readwrite("fp32_dest_acc_enabled", &Conv2dConfig::fp32_dest_acc_enabled);
        py_conv_config.def_readwrite("packer_l1_accum_enabled", &Conv2dConfig::packer_l1_accum_enabled);
        py_conv_config.def_readwrite("activation", &Conv2dConfig::activation);
        py_conv_config.def_readwrite("input_channels_alignment", &Conv2dConfig::input_channels_alignment);
        py_conv_config.def_readwrite("deallocate_activation", &Conv2dConfig::deallocate_activation);
        py_conv_config.def_readwrite("reallocate_halo_output", &Conv2dConfig::reallocate_halo_output);
        py_conv_config.def_readwrite("act_block_h_override", &Conv2dConfig::act_block_h_override);
        py_conv_config.def_readwrite("reshard_if_not_optimal", &Conv2dConfig::reshard_if_not_optimal);
        py_conv_config.def_readwrite("override_sharding_config", &Conv2dConfig::override_sharding_config);
        py_conv_config.def_readwrite("height_sharding", &Conv2dConfig::height_sharding);
        py_conv_config.def_readwrite("core_grid", &Conv2dConfig::core_grid);
        py_conv_config.def_readwrite("transpose_shards", &Conv2dConfig::transpose_shards);
        py_conv_config.def_readwrite("output_layout", &Conv2dConfig::output_layout);

    module.def(
        "get_conv_padded_input_shape_and_mem_config",
        [](ttnn::Device& device,
            const ttnn::Tensor& input_tensor,
            const Conv2dConfig& conv_config,
            uint32_t batch_size,
            uint32_t height,
            uint32_t width,
            uint32_t in_channels,
            uint32_t out_channels) -> std::tuple<ttnn::Shape, ttnn::MemoryConfig, bool> {
            return ttnn::operations::conv2d::get_conv_padded_input_shape_and_mem_config(
                device, input_tensor, conv_config, batch_size, height, width, in_channels, out_channels);
        },
        py::kw_only(),
        py::arg("device"),
        py::arg("input_tensor"),
        py::arg("conv_config"),
        py::arg("batch_size"),
        py::arg("height"),
        py::arg("width"),
        py::arg("in_channels"),
        py::arg("out_channels"));
}

}  // namespace conv2d
}  // namespace operations
}  // namespace ttnn
