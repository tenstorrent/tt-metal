// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "conv2d_nanobind.hpp"

#include <array>
#include <optional>
#include <variant>

#include <fmt/format.h>
#include <tt-metalium/constants.hpp>

#include <nanobind/nanobind.h>
#include "cpp/ttnn-nanobind/decorators.hpp"

#include "cpp/ttnn/operations/sliding_window/sliding_window_nanobind.hpp"
#include "conv2d.hpp"
#include "conv2d_utils.hpp"
#include "prepare_conv2d_weights.hpp"
#include "ttnn/types.hpp"


namespace nb = nanobind;

namespace ttnn {
namespace operations::conv {
namespace conv2d {

void bind_conv2d(nb::module_& mod) {
    bind_registered_operation(
        mod,
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
        ttnn::nanobind_overload_t{
            [](const decltype(ttnn::conv2d)& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::Tensor& weight_tensor,
               ttnn::IDevice* device,
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
               std::optional<const ttnn::Tensor> bias_tensor,
               const std::optional<const Conv2dConfig>& conv_config,
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
                    input_height,
                    input_width,
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
            nb::kw_only(),
            nb::arg("input_tensor"),
            nb::arg("weight_tensor"),
            nb::arg("device"),
            nb::arg("in_channels"),
            nb::arg("out_channels"),
            nb::arg("batch_size"),
            nb::arg("input_height"),
            nb::arg("input_width"),
            nb::arg("kernel_size"),
            nb::arg("stride"),
            nb::arg("padding"),
            nb::arg("dilation"),
            nb::arg("groups"),
            nb::arg("bias_tensor") = std::nullopt,
            nb::arg("conv_config") = std::nullopt,
            nb::arg("compute_config") = std::nullopt,
            nb::arg("memory_config") = std::nullopt,
            nb::arg("queue_id") = DefaultQueueId},

        ttnn::nanobind_overload_t{
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
               std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>> padding,
               std::array<uint32_t, 2> dilation,
               uint32_t groups,
               std::optional<const ttnn::Tensor> bias_tensor,
               const std::optional<const Conv2dConfig>& conv_config,
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
                    input_height,
                    input_width,
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
            nb::kw_only(),
            nb::arg("input_tensor"),
            nb::arg("weight_tensor"),
            nb::arg("device"),
            nb::arg("in_channels"),
            nb::arg("out_channels"),
            nb::arg("batch_size"),
            nb::arg("input_height"),
            nb::arg("input_width"),
            nb::arg("kernel_size"),
            nb::arg("stride"),
            nb::arg("padding"),
            nb::arg("dilation"),
            nb::arg("groups"),
            nb::arg("bias_tensor") = std::nullopt,
            nb::arg("conv_config") = std::nullopt,
            nb::arg("compute_config") = std::nullopt,
            nb::arg("memory_config") = std::nullopt,
            nb::arg("queue_id") = DefaultQueueId});

    mod.def(
        "prepare_conv_weights",
        prepare_conv_weights<ttnn::IDevice>,
        nb::kw_only(),
        nb::arg("weight_tensor"),
        nb::arg("input_memory_config"),
        nb::arg("input_tensor_layout"),
        nb::arg("weights_format"),
        nb::arg("in_channels"),
        nb::arg("out_channels"),
        nb::arg("batch_size"),
        nb::arg("input_height"),
        nb::arg("input_width"),
        nb::arg("kernel_size"),
        nb::arg("stride"),
        nb::arg("padding"),
        nb::arg("dilation"),
        nb::arg("has_bias"),
        nb::arg("groups"),
        nb::arg("device"),
        nb::arg("conv_config") = std::nullopt,
        nb::arg("compute_config") = std::nullopt);

    mod.def(
        "prepare_conv_weights",
        prepare_conv_weights<ttnn::MeshDevice>,
        nb::kw_only(),
        nb::arg("weight_tensor"),
        nb::arg("input_memory_config"),
        nb::arg("input_tensor_layout"),
        nb::arg("weights_format"),
        nb::arg("in_channels"),
        nb::arg("out_channels"),
        nb::arg("batch_size"),
        nb::arg("input_height"),
        nb::arg("input_width"),
        nb::arg("kernel_size"),
        nb::arg("stride"),
        nb::arg("padding"),
        nb::arg("dilation"),
        nb::arg("has_bias"),
        nb::arg("groups"),
        nb::arg("device"),
        nb::arg("conv_config") = std::nullopt,
        nb::arg("compute_config") = std::nullopt);

    mod.def(
        "prepare_conv_bias",
        prepare_conv_bias<ttnn::IDevice>,
        nb::kw_only(),
        nb::arg("bias_tensor"),
        nb::arg("input_memory_config"),
        nb::arg("input_tensor_layout"),
        nb::arg("in_channels"),
        nb::arg("out_channels"),
        nb::arg("batch_size"),
        nb::arg("input_height"),
        nb::arg("input_width"),
        nb::arg("kernel_size"),
        nb::arg("stride"),
        nb::arg("padding"),
        nb::arg("dilation"),
        nb::arg("groups"),
        nb::arg("device"),
        nb::arg("conv_config") = std::nullopt,
        nb::arg("compute_config") = std::nullopt);

    mod.def(
        "prepare_conv_bias",
        prepare_conv_bias<ttnn::MeshDevice>,
        nb::kw_only(),
        nb::arg("bias_tensor"),
        nb::arg("input_memory_config"),
        nb::arg("input_tensor_layout"),
        nb::arg("in_channels"),
        nb::arg("out_channels"),
        nb::arg("batch_size"),
        nb::arg("input_height"),
        nb::arg("input_width"),
        nb::arg("kernel_size"),
        nb::arg("stride"),
        nb::arg("padding"),
        nb::arg("dilation"),
        nb::arg("groups"),
        nb::arg("device"),
        nb::arg("conv_config") = std::nullopt,
        nb::arg("compute_config") = std::nullopt);

    mod.def(
        "convert_conv_weight_tensor_to_tiled_layout",
        &convert_conv_weight_tensor_to_tiled_layout,
        nb::arg("conv_weight_tensor").noconvert(),
        nb::arg("in1_block_h"),
        nb::arg("in1_block_w"),
        nb::arg("output_dtype").noconvert() = std::nullopt);

    mod.def(
        "convert_conv_weight_tensor_to_special_padding_tiled_layout",
        &convert_conv_weight_tensor_to_special_padding_tiled_layout,
        nb::arg("conv_weight_tensor").noconvert(),
        nb::arg("in1_block_h"),
        nb::arg("in1_block_w"),
        nb::arg("output_dtype").noconvert() = std::nullopt);

    mod.def(
        "convert_conv_weight_tensor_to_grouped_layout",
        &convert_conv_weight_tensor_to_grouped_layout,
        nb::arg("conv_weight_tensor").noconvert(),
        nb::arg("num_groups"),
        nb::arg("output_dtype").noconvert() = std::nullopt);

    mod.def(
        "determine_parallel_config",
        [](const ttnn::TensorMemoryLayout& shard_layout,
           uint32_t batch_size,
           uint32_t input_channels,
           uint32_t output_height,
           uint32_t output_width,
           uint32_t output_channels,
           const CoreCoord& compute_grid_size,
           tt::tt_metal::ShardOrientation block_shard_orientation,
           bool enable_channels_padding,
           bool is_shard_height_tile_multiple,
           bool is_shard_width_tile_multiple) -> ttnn::operations::sliding_window::ParallelConfig {
            return determine_parallel_config(
                shard_layout,
                batch_size,
                input_channels,
                output_height,
                output_width,
                output_channels,
                compute_grid_size,
                block_shard_orientation,
                enable_channels_padding,
                is_shard_height_tile_multiple,
                is_shard_width_tile_multiple);
        },
        nb::arg("shard_layout"),
        nb::arg("batch_size"),
        nb::arg("input_channels"),
        nb::arg("output_height"),
        nb::arg("output_width"),
        nb::arg("output_channels"),
        nb::arg("compute_grid_size"),
        nb::arg("block_shard_orientation"),
        nb::arg("enable_channels_padding"),
        nb::arg("is_shard_height_tile_multiple") = true,
        nb::arg("is_shard_width_tile_multiple") = true);

    mod.def(
        "create_sharded_memory_config_from_parallel_config",
        &create_sharded_memory_config_from_parallel_config,
        nb::arg("tensor_shape"),
        nb::arg("parallel_config"),
        nb::arg("tile_size"));

    auto py_conv_config = nb::class_<Conv2dConfig>(mod, "Conv2dConfig");
    py_conv_config.def(
        nb::init<
            DataType,
            DataType,
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
            bool,
            bool,
            bool,
            bool,
            bool>(),
        nb::kw_only(),
        nb::arg("dtype") = DataType::BFLOAT16,
        nb::arg("weights_dtype") = DataType::BFLOAT16,
        nb::arg("activation") = "",
        nb::arg("input_channels_alignment") = 32,
        nb::arg("deallocate_activation") = false,
        nb::arg("reallocate_halo_output") = false,
        nb::arg("act_block_h_override") = 0,
        nb::arg("act_block_w_div") = 1,
        nb::arg("reshard_if_not_optimal") = false,
        nb::arg("override_sharding_config") = false,
        nb::arg("shard_layout") = std::nullopt,
        nb::arg("core_grid") = std::nullopt,
        nb::arg("transpose_shards") = true,
        nb::arg("output_layout") = Layout::TILE,
        nb::arg("preprocess_weights_on_device") = false,
        nb::arg("always_preprocess_weights") = false,
        nb::arg("enable_act_double_buffer") = false,
        nb::arg("enable_weights_double_buffer") = false,
        nb::arg("enable_split_reader") = false,
        nb::arg("enable_subblock_padding") = false,
        nb::arg("in_place") = false);
    py_conv_config.def_ro("dtype", &Conv2dConfig::dtype);
    py_conv_config.def_ro("weights_dtype", &Conv2dConfig::weights_dtype);
    py_conv_config.def_ro("activation", &Conv2dConfig::activation);
    py_conv_config.def_ro("input_channels_alignment", &Conv2dConfig::input_channels_alignment);
    py_conv_config.def_ro("deallocate_activation", &Conv2dConfig::deallocate_activation);
    py_conv_config.def_ro("reallocate_halo_output", &Conv2dConfig::reallocate_halo_output);
    py_conv_config.def_ro("act_block_h_override", &Conv2dConfig::act_block_h_override);
    py_conv_config.def_ro("act_block_w_div", &Conv2dConfig::act_block_w_div);
    py_conv_config.def_ro("reshard_if_not_optimal", &Conv2dConfig::reshard_if_not_optimal);
    py_conv_config.def_ro("override_sharding_config", &Conv2dConfig::override_sharding_config);
    py_conv_config.def_ro("shard_layout", &Conv2dConfig::shard_layout);
    py_conv_config.def_ro("core_grid", &Conv2dConfig::core_grid);
    py_conv_config.def_ro("transpose_shards", &Conv2dConfig::transpose_shards);
    py_conv_config.def_ro("output_layout", &Conv2dConfig::output_layout);
    py_conv_config.def_ro("preprocess_weights_on_device", &Conv2dConfig::preprocess_weights_on_device);
    py_conv_config.def_ro("always_preprocess_weights", &Conv2dConfig::always_preprocess_weights);
    py_conv_config.def_ro("enable_act_double_buffer", &Conv2dConfig::enable_act_double_buffer);
    py_conv_config.def_ro("enable_weights_double_buffer", &Conv2dConfig::enable_weights_double_buffer);
    py_conv_config.def_ro("enable_split_reader", &Conv2dConfig::enable_split_reader);
    py_conv_config.def_ro("enable_subblock_padding", &Conv2dConfig::enable_subblock_padding);
    py_conv_config.def_ro("in_place", &Conv2dConfig::in_place);

    py_conv_config.def("__repr__", [](const Conv2dConfig& config) { return fmt::format("{}", config); });

    nb::class_<OptimizedConvParallelizationConfig>(mod, "OptimizedConvParallelizationConfig")
        .def(
            nb::init<CoreCoord, uint32_t, uint32_t, uint32_t, uint32_t>(),
            nb::kw_only(),
            nb::arg("grid_size"),
            nb::arg("num_cores_nhw") = 1,
            nb::arg("num_cores_c") = 1,
            nb::arg("per_core_out_matrix_height_ntiles").noconvert(),
            nb::arg("per_core_out_matrix_width_ntiles").noconvert())
        .def_prop_ro("grid_size", [](const OptimizedConvParallelizationConfig& c) { return c.grid_size; })
        .def_prop_ro(
            "num_cores_nhw", [](const OptimizedConvParallelizationConfig& c) { return c.num_cores_nhw; })
        .def_prop_ro(
            "per_core_out_matrix_height_ntiles",
            [](const OptimizedConvParallelizationConfig& c) { return c.per_core_out_matrix_height_ntile; })
        .def_prop_ro("per_core_out_matrix_width_ntiles", [](const OptimizedConvParallelizationConfig& c) {
            return c.per_core_out_matrix_width_ntile;
        });

    nb::class_<OptimizedConvBlockConfig>(mod, "OptimizedConvBlockConfig")
        .def(
            nb::init<uint32_t, uint32_t, uint32_t, uint32_t>(),
            nb::kw_only(),
            nb::arg("act_block_h_ntiles").noconvert(),
            nb::arg("act_block_w_ntiles").noconvert(),
            nb::arg("out_subblock_h_ntiles").noconvert(),
            nb::arg("out_subblock_w_ntiles").noconvert())
        .def_prop_ro(
            "act_block_h_ntiles", [](const OptimizedConvBlockConfig& c) { return c.act_block_h_ntiles; })
        .def_prop_ro(
            "act_block_w_ntiles", [](const OptimizedConvBlockConfig& c) { return c.act_block_w_ntiles; })
        .def_prop_ro(
            "out_subblock_h_ntiles", [](const OptimizedConvBlockConfig& c) { return c.out_subblock_h_ntiles; })
        .def_prop_ro(
            "out_subblock_w_ntiles", [](const OptimizedConvBlockConfig& c) { return c.out_subblock_w_ntiles; });
}

}  // namespace conv2d
}  // namespace operations::conv
}  // namespace ttnn
