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
            std::optional<const ConvConfig> conv_config_ = std::nullopt) -> std::tuple<ttnn::Tensor, uint32_t, uint32_t, ttnn::Tensor, std::optional<ttnn::Tensor>> {
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

    py::class_<ConvConfig>(module, "ConvConfig")
        .def(
            py::init<MathFidelity, DataType, DataType, bool, bool, bool, string, uint32_t, bool, bool, uint32_t, bool, bool, bool, CoreRangeSet, bool, Layout>(),
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
            py::arg("core_grid") = CoreRangeSet({CoreRange({})}),
            py::arg("transpose_shards") = true,
            py::arg("output_layout") = Layout::TILE
        );
}

}  // namespace conv2d
}  // namespace operations
}  // namespace ttnn
