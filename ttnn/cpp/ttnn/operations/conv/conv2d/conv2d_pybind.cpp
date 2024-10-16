// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/pybind11/decorators.hpp"

#include "conv2d_pybind.hpp"
#include "conv2d.hpp"

namespace py = pybind11;

namespace ttnn {
namespace operations::conv {
namespace conv2d {

void py_bind_conv2d(py::module& module) {
    bind_registered_operation(
        module,
        ttnn::conv2d,
        R"doc(
        Conv 2D
        +-------------------+-------------------------------+---------------+-------------+----------+
        | Argument          | Description                   | Data type     | Valid range | Required |
        +===================+===============================+===============+=============+==========+
        | input             | Input activations tensor      | Tensor        |             | Yes      |
        | in_n              | Input nbatch                  | Tensor        |             | Yes      |
        | in_h              | Input height                  | Tensor        |             | Yes      |
        | in_w              | Input width                   | Tensor        |             | Yes      |
        | kernel_h          | kernel window height          | uint32_t      |             | Yes      |
        | kernel_w          | kernel window width           | uint32_t      |             | Yes      |
        | stride_h          | stride in height dim          | uint32_t      |             | No       |
        | stride_w          | stride in width dim           | uint32_t      |             | No       |
        | pad_h             | padding in height dim         | uint32_t      |             | No       |
        | pad_w             | padding in width dim          | uint32_t      |             | No       |
        | dilation_h        | kernel dilation in height dim | uint32_t      |             | No       |
        | dilation_w        | kernel dilation in width dim  | uint32_t      |             | No       |
        | memory_config     | Output memory config          | MemoryConfig  |             | No       |
        +-------------------+-------------------------------+---------------+-------------+----------+
        )doc",
        ttnn::pybind_overload_t{
            [](const decltype(ttnn::conv2d)& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::Tensor& weight_tensor,
               ttnn::Device* device,
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
               std::optional<const ttnn::Tensor> bias_tensor,
               std::optional<const Conv2dConfig> conv_config,
               const uint8_t& queue_id)
                -> std::tuple<ttnn::Tensor, uint32_t, uint32_t, ttnn::Tensor, std::optional<ttnn::Tensor>> {
                return self(queue_id,
                            input_tensor,
                            weight_tensor,
                            device,
                            in_channels,
                            out_channels,
                            batch_size,
                            input_height,
                            input_width,
                            kernel_size,
                            stride,
                            padding,
                            dilation,
                            groups,
                            bias_tensor,
                            conv_config);
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
            py::arg("conv_config") = std::nullopt,
            py::arg("queue_id") = 0},

        ttnn::pybind_overload_t{
            [](const decltype(ttnn::conv2d)& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::Tensor& weight_tensor,
               ttnn::MeshDevice* device,
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
               std::optional<const ttnn::Tensor> bias_tensor,
               std::optional<const Conv2dConfig> conv_config,
               const uint8_t& queue_id)
                -> std::tuple<ttnn::Tensor, uint32_t, uint32_t, ttnn::Tensor, std::optional<ttnn::Tensor>> {
                return self(queue_id,
                            input_tensor,
                            weight_tensor,
                            device,
                            in_channels,
                            out_channels,
                            batch_size,
                            input_height,
                            input_width,
                            kernel_size,
                            stride,
                            padding,
                            dilation,
                            groups,
                            bias_tensor,
                            conv_config);
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
            py::arg("conv_config") = std::nullopt,
            py::arg("queue_id") = 0});

    module.def(
        "get_conv_padded_input_shape_and_mem_config",
        [](ttnn::Device* device,
           const ttnn::Tensor& input_tensor,
           const Conv2dConfig& conv_config,
           uint32_t batch_size,
           uint32_t height,
           uint32_t width,
           uint32_t in_channels,
           uint32_t out_channels,
           std::array<uint32_t, 2> kernel_size,
           std::array<uint32_t, 2> stride) -> std::tuple<ttnn::Shape, ttnn::MemoryConfig, bool> {
            return ttnn::operations::conv::conv2d::get_conv_padded_input_shape_and_mem_config<ttnn::Device>(
                device,
                input_tensor,
                conv_config,
                batch_size,
                height,
                width,
                in_channels,
                out_channels,
                kernel_size,
                stride);
        },
        py::kw_only(),
        py::arg("device"),
        py::arg("input_tensor"),
        py::arg("conv_config"),
        py::arg("batch_size"),
        py::arg("height"),
        py::arg("width"),
        py::arg("in_channels"),
        py::arg("out_channels"),
        py::arg("kernel_size"),
        py::arg("stride"));

    module.def(
        "get_conv_padded_input_shape_and_mem_config",
        [](MeshDevice* device,
           const ttnn::Tensor& input_tensor,
           const Conv2dConfig& conv_config,
           uint32_t batch_size,
           uint32_t height,
           uint32_t width,
           uint32_t in_channels,
           uint32_t out_channels,
           std::array<uint32_t, 2> kernel_size,
           std::array<uint32_t, 2> stride) -> std::tuple<ttnn::Shape, ttnn::MemoryConfig, bool> {
            return ttnn::operations::conv::conv2d::get_conv_padded_input_shape_and_mem_config<MeshDevice>(device,
                                                                                                          input_tensor,
                                                                                                          conv_config,
                                                                                                          batch_size,
                                                                                                          height,
                                                                                                          width,
                                                                                                          in_channels,
                                                                                                          out_channels,
                                                                                                          kernel_size,
                                                                                                          stride);
        },
        py::kw_only(),
        py::arg("device"),
        py::arg("input_tensor"),
        py::arg("conv_config"),
        py::arg("batch_size"),
        py::arg("height"),
        py::arg("width"),
        py::arg("in_channels"),
        py::arg("out_channels"),
        py::arg("kernel_size"),
        py::arg("stride"));

    module.def("convert_conv_weight_tensor_to_tiled_layout",
               &ttnn::operations::conv::conv2d::convert_conv_weight_tensor_to_tiled_layout,
               py::arg("conv_weight_tensor").noconvert(),
               py::arg("in1_block_h"),
               py::arg("in1_block_w"),
               py::arg("output_dtype").noconvert() = std::nullopt);

    module.def("convert_conv_weight_tensor_to_special_padding_tiled_layout",
               &ttnn::operations::conv::conv2d::convert_conv_weight_tensor_to_special_padding_tiled_layout,
               py::arg("conv_weight_tensor").noconvert(),
               py::arg("in1_block_h"),
               py::arg("in1_block_w"),
               py::arg("output_dtype").noconvert() = std::nullopt);

    module.def("convert_conv_weight_tensor_to_grouped_layout",
               &ttnn::operations::conv::conv2d::convert_conv_weight_tensor_to_grouped_layout,
               py::arg("conv_weight_tensor").noconvert(),
               py::arg("num_groups"),
               py::arg("output_dtype").noconvert() = std::nullopt);

    auto py_conv_config = py::class_<Conv2dConfig>(module, "Conv2dConfig");
    py_conv_config.def(py::init<MathFidelity,
                                DataType,
                                DataType,
                                bool,
                                bool,
                                bool,
                                string,
                                uint32_t,
                                bool,
                                bool,
                                uint32_t,
                                uint32_t,
                                bool,
                                bool,
                                std::optional<TensorMemoryLayout>,
                                std::optional<CoreRangeSet>,
                                bool,
                                Layout,
                                bool,
                                bool,
                                bool>(),
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
                       py::arg("act_block_w_div") = 1,
                       py::arg("reshard_if_not_optimal") = false,
                       py::arg("override_sharding_config") = false,
                       py::arg("shard_layout") = std::nullopt,
                       py::arg("core_grid") = std::nullopt,
                       py::arg("transpose_shards") = true,
                       py::arg("output_layout") = Layout::TILE,
                       py::arg("enable_act_double_buffer") = false,
                       py::arg("enable_split_reader") = false,
                       py::arg("enable_subblock_padding") = false);
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
    py_conv_config.def_readwrite("act_block_w_div", &Conv2dConfig::act_block_w_div);
    py_conv_config.def_readwrite("reshard_if_not_optimal", &Conv2dConfig::reshard_if_not_optimal);
    py_conv_config.def_readwrite("override_sharding_config", &Conv2dConfig::override_sharding_config);
    py_conv_config.def_readwrite("shard_layout", &Conv2dConfig::shard_layout);
    py_conv_config.def_readwrite("core_grid", &Conv2dConfig::core_grid);
    py_conv_config.def_readwrite("transpose_shards", &Conv2dConfig::transpose_shards);
    py_conv_config.def_readwrite("output_layout", &Conv2dConfig::output_layout);
    py_conv_config.def_readwrite("enable_act_double_buffer", &Conv2dConfig::enable_act_double_buffer);
    py_conv_config.def_readwrite("enable_split_reader", &Conv2dConfig::enable_split_reader);
    py_conv_config.def_readwrite("enable_subblock_padding", &Conv2dConfig::enable_subblock_padding);

    py::class_<OptimizedConvParallelizationConfig>(module, "OptimizedConvParallelizationConfig")
        .def(py::init<CoreCoord, uint32_t, uint32_t, uint32_t, uint32_t>(),
             py::kw_only(),
             py::arg("grid_size"),
             py::arg("num_cores_nhw") = 1,
             py::arg("num_cores_c") = 1,
             py::arg("per_core_out_matrix_height_ntiles").noconvert() = 1,
             py::arg("per_core_out_matrix_width_ntiles").noconvert() = 1)
        .def_property_readonly("grid_size",
                               [](OptimizedConvParallelizationConfig const& c) {
                                   return c.grid_size;
                               })
        .def_property_readonly("num_cores_nhw",
                               [](OptimizedConvParallelizationConfig const& c) {
                                   return c.num_cores_nhw;
                               })
        .def_property_readonly("per_core_out_matrix_height_ntiles",
                               [](OptimizedConvParallelizationConfig const& c) {
                                   return c.per_core_out_matrix_height_ntiles;
                               })
        .def_property_readonly("per_core_out_matrix_width_ntiles", [](OptimizedConvParallelizationConfig const& c) {
            return c.per_core_out_matrix_width_ntiles;
        });

    py::class_<OptimizedConvBlockConfig>(module, "OptimizedConvBlockConfig")
        .def(py::init<uint32_t, uint32_t, uint32_t, uint32_t>(),
             py::kw_only(),
             py::arg("act_block_h_ntiles").noconvert(),
             py::arg("act_block_w_ntiles").noconvert(),
             py::arg("out_subblock_h_ntiles").noconvert(),
             py::arg("out_subblock_w_ntiles").noconvert())
        .def_property_readonly("act_block_h_ntiles",
                               [](OptimizedConvBlockConfig const& c) {
                                   return c.act_block_h_ntiles;
                               })
        .def_property_readonly("act_block_w_ntiles",
                               [](OptimizedConvBlockConfig const& c) {
                                   return c.act_block_w_ntiles;
                               })
        .def_property_readonly("out_subblock_h_ntiles",
                               [](OptimizedConvBlockConfig const& c) {
                                   return c.out_subblock_h_ntiles;
                               })
        .def_property_readonly("out_subblock_w_ntiles", [](OptimizedConvBlockConfig const& c) {
            return c.out_subblock_w_ntiles;
        });
}

}  // namespace conv2d
}  // namespace operations::conv
}  // namespace ttnn
