// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/constants.hpp>
#include "cpp/pybind11/decorators.hpp"

#include "conv2d_pybind.hpp"
#include "cpp/ttnn/operations/sliding_window/sliding_window_pybind.hpp"
#include "conv2d.hpp"
#include "conv2d_utils.hpp"
#include "prepare_conv2d_weights.hpp"
#include "ttnn/types.hpp"

namespace py = pybind11;

namespace ttnn {
namespace operations::conv {
namespace conv2d {

void py_bind_conv2d(py::module& module) {
    bind_registered_operation(
        module,
        ttnn::conv2d,
        R"doc(
            Applies a 2D convolution over an input signal composed of several input planes.

            Args:
                input_tensor (ttnn.Tensor): the input tensor.
                weight_tensor (ttnn.Tensor): the weight tensor.
                device (ttnn.IDevice): the device to use.
                in_channels (int): number of input channels.
                out_channels (int): number of output channels.
                batch_size (int): batch size.
                input_height (int): height of the input tensor.
                input_width (int): width of the input tensor.
                kernel_size (tuple[int, int]): size of the convolving kernel.
                stride (tuple[int, int]): stride of the cross-correlation.
                padding (tuple[int, int] or tuple[int, int, int, int]): zero-padding added to both sides of the input. [pad_height, pad_width] or [pad_top, pad_bottom, pad_left, pad_right].
                dilation (tuple[int, int]): spacing between kernel elements.
                groups (int): number of blocked connections from input channels to output channels.
                bias_tensor (ttnn.Tensor, optional): optional bias tensor. Default: None
                conv_config (ttnn.Conv2dConfig, optional): configuration for convolution. Default: None
                compute_config (ttnn.DeviceComputeKernelConfig, optional): configuration for compute kernel. Default: None
                memory_config (ttnn.MemoryConfig, optional): configuration for memory. Default: None
                queue_id (QueueId): queue id for operation. Default: DefaultQueueId
            Returns:
                ttnn.Tensor: the output tensor.

        )doc",
        ttnn::pybind_overload_t{
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
            py::arg("compute_config") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId},

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
            py::arg("compute_config") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId});

    module.def(
        "prepare_conv_weights",
        prepare_conv_weights<ttnn::IDevice>,
        py::kw_only(),
        py::arg("weight_tensor"),
        py::arg("input_memory_config"),
        py::arg("input_tensor_layout"),
        py::arg("weights_format"),
        py::arg("in_channels"),
        py::arg("out_channels"),
        py::arg("batch_size"),
        py::arg("input_height"),
        py::arg("input_width"),
        py::arg("kernel_size"),
        py::arg("stride"),
        py::arg("padding"),
        py::arg("dilation"),
        py::arg("has_bias"),
        py::arg("groups"),
        py::arg("device"),
        py::arg("conv_config") = std::nullopt,
        py::arg("compute_config") = std::nullopt);

    module.def(
        "prepare_conv_weights",
        prepare_conv_weights<ttnn::MeshDevice>,
        py::kw_only(),
        py::arg("weight_tensor"),
        py::arg("input_memory_config"),
        py::arg("input_tensor_layout"),
        py::arg("weights_format"),
        py::arg("in_channels"),
        py::arg("out_channels"),
        py::arg("batch_size"),
        py::arg("input_height"),
        py::arg("input_width"),
        py::arg("kernel_size"),
        py::arg("stride"),
        py::arg("padding"),
        py::arg("dilation"),
        py::arg("has_bias"),
        py::arg("groups"),
        py::arg("device"),
        py::arg("conv_config") = std::nullopt,
        py::arg("compute_config") = std::nullopt);

    module.def(
        "prepare_conv_bias",
        prepare_conv_bias<ttnn::IDevice>,
        py::kw_only(),
        py::arg("bias_tensor"),
        py::arg("input_memory_config"),
        py::arg("input_tensor_layout"),
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
        py::arg("device"),
        py::arg("conv_config") = std::nullopt,
        py::arg("compute_config") = std::nullopt);

    module.def(
        "prepare_conv_bias",
        prepare_conv_bias<ttnn::MeshDevice>,
        py::kw_only(),
        py::arg("bias_tensor"),
        py::arg("input_memory_config"),
        py::arg("input_tensor_layout"),
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
        py::arg("device"),
        py::arg("conv_config") = std::nullopt,
        py::arg("compute_config") = std::nullopt);

    module.def(
        "convert_conv_weight_tensor_to_tiled_layout",
        &convert_conv_weight_tensor_to_tiled_layout,
        py::arg("conv_weight_tensor").noconvert(),
        py::arg("in1_block_h"),
        py::arg("in1_block_w"),
        py::arg("output_dtype").noconvert() = std::nullopt);

    module.def(
        "convert_conv_weight_tensor_to_special_padding_tiled_layout",
        &convert_conv_weight_tensor_to_special_padding_tiled_layout,
        py::arg("conv_weight_tensor").noconvert(),
        py::arg("in1_block_h"),
        py::arg("in1_block_w"),
        py::arg("output_dtype").noconvert() = std::nullopt);

    module.def(
        "convert_conv_weight_tensor_to_grouped_layout",
        &convert_conv_weight_tensor_to_grouped_layout,
        py::arg("conv_weight_tensor").noconvert(),
        py::arg("num_groups"),
        py::arg("output_dtype").noconvert() = std::nullopt);

    module.def(
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
        py::arg("shard_layout"),
        py::arg("batch_size"),
        py::arg("input_channels"),
        py::arg("output_height"),
        py::arg("output_width"),
        py::arg("output_channels"),
        py::arg("compute_grid_size"),
        py::arg("block_shard_orientation"),
        py::arg("enable_channels_padding"),
        py::arg("is_shard_height_tile_multiple") = true,
        py::arg("is_shard_width_tile_multiple") = true);

    module.def(
        "create_sharded_memory_config_from_parallel_config",
        &create_sharded_memory_config_from_parallel_config,
        py::arg("tensor_shape"),
        py::arg("parallel_config"),
        py::arg("tile_size"));

    auto py_conv_config = py::class_<Conv2dConfig>(module, "Conv2dConfig");
    py_conv_config.def(
        py::init<
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
        py::kw_only(),
        py::arg("dtype") = DataType::BFLOAT16,
        py::arg("weights_dtype") = DataType::BFLOAT16,
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
        py::arg("preprocess_weights_on_device") = false,
        py::arg("always_preprocess_weights") = false,
        py::arg("enable_act_double_buffer") = false,
        py::arg("enable_weights_double_buffer") = false,
        py::arg("enable_split_reader") = false,
        py::arg("enable_subblock_padding") = false,
        py::arg("in_place") = false,
        R"doc(
        Conv2D configuration
        +-------------------------------+-------------------------------+---------------+-------------+----------+
        | Argument                      | Description                   | Data type     | Valid range | Required |
        +===============================+===============================+===============+=============+==========+
        | dtype                         | Activation data type          | DataType      |             | No       |
        +-------------------------------+-------------------------------+---------------+-------------+----------+
        | weights_dtype                 | Weights data type             | DataType      |             | No       |
        +-------------------------------+-------------------------------+---------------+-------------+----------+
        | activation                    | Activation function           | string        |             | No       |
        +-------------------------------+-------------------------------+---------------+-------------+----------+
        | input_channels_alignment      | Input channels alignment      | uint32_t      |             | No       |
        +-------------------------------+-------------------------------+---------------+-------------+----------+
        | deallocate_activation         | Deallocate activation tensor  | bool          |             | No       |
        +-------------------------------+-------------------------------+---------------+-------------+----------+
        | reallocate_halo_output       | Reallocate halo output tensor | bool          |             | No       |
        +-------------------------------+-------------------------------+---------------+-------------+----------+
        | act_block_h_override          | Activation block height       | uint32_t      |             | No       |
        +-------------------------------+-------------------------------+---------------+-------------+----------+
        | act_block_w_div               | Activation block width        | uint32_t      |             | No       |
        +-------------------------------+-------------------------------+---------------+-------------+----------+
        | reshard_if_not_optimal       | Reshard if not optimal        | bool          |             | No       |
        +-------------------------------+-------------------------------+---------------+-------------+----------+
        | override_sharding_config      | Override sharding config      | bool          |             | No       |
        +-------------------------------+-------------------------------+---------------+-------------+----------+
        | shard_layout                  | Shard layout                  | TensorLayout  |             | No       |
        +-------------------------------+-------------------------------+---------------+-------------+----------+
        | core_grid                     | Core grid                     | CoreRangeSet  |             | No       |
        +-------------------------------+-------------------------------+---------------+-------------+----------+
        | transpose_shards              | Transpose shards              | bool          |             | No       |
        +-------------------------------+-------------------------------+---------------+-------------+----------+
        | output_layout                 | Output layout                 | Layout        |             | No       |
        +-------------------------------+-------------------------------+---------------+-------------+----------+
        | preprocess_weights_on_device  | Preprocess weights on device  | bool          |             | No       |
        +-------------------------------+-------------------------------+---------------+-------------+----------+
        | always_preprocess_weights     | Always preprocess weights     | bool          |             | No       |
        +-------------------------------+-------------------------------+---------------+-------------+----------+
        | enable_act_double_buffer      | Enable activation double buff | bool          |             | No       |
        +-------------------------------+-------------------------------+---------------+-------------+----------+
        | enable_weights_double_buffer  | Enable weights double buffer  | bool          |             | No       |
        +-------------------------------+-------------------------------+---------------+-------------+----------+
        | enable_split_reader           | Enable split reader           | bool          |             | No       |
        +-------------------------------+-------------------------------+---------------+-------------+----------+
        | enable_subblock_padding       | Enable subblock padding       | bool          |             | No       |
        +-------------------------------+-------------------------------+---------------+-------------+----------+
        | in_place                      | In place                      | bool          |             | No       |
        +-------------------------------+-------------------------------+---------------+-------------+----------+
        )doc");
    py_conv_config.def_readwrite("dtype", &Conv2dConfig::dtype);
    py_conv_config.def_readwrite("weights_dtype", &Conv2dConfig::weights_dtype);
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
    py_conv_config.def_readwrite("preprocess_weights_on_device", &Conv2dConfig::preprocess_weights_on_device);
    py_conv_config.def_readwrite("always_preprocess_weights", &Conv2dConfig::always_preprocess_weights);
    py_conv_config.def_readwrite("enable_act_double_buffer", &Conv2dConfig::enable_act_double_buffer);
    py_conv_config.def_readwrite("enable_weights_double_buffer", &Conv2dConfig::enable_weights_double_buffer);
    py_conv_config.def_readwrite("enable_split_reader", &Conv2dConfig::enable_split_reader);
    py_conv_config.def_readwrite("enable_subblock_padding", &Conv2dConfig::enable_subblock_padding);
    py_conv_config.def_readwrite("in_place", &Conv2dConfig::in_place);

    py_conv_config.def("__repr__", [](const Conv2dConfig& config) { return fmt::format("{}", config); });

    py::class_<OptimizedConvParallelizationConfig>(module, "OptimizedConvParallelizationConfig")
        .def(
            py::init<CoreCoord, uint32_t, uint32_t, uint32_t, uint32_t>(),
            py::kw_only(),
            py::arg("grid_size"),
            py::arg("num_cores_nhw") = 1,
            py::arg("num_cores_c") = 1,
            py::arg("per_core_out_matrix_height_ntiles").noconvert(),
            py::arg("per_core_out_matrix_width_ntiles").noconvert())
        .def_property_readonly("grid_size", [](const OptimizedConvParallelizationConfig& c) { return c.grid_size; })
        .def_property_readonly(
            "num_cores_nhw", [](const OptimizedConvParallelizationConfig& c) { return c.num_cores_nhw; })
        .def_property_readonly(
            "per_core_out_matrix_height_ntiles",
            [](const OptimizedConvParallelizationConfig& c) { return c.per_core_out_matrix_height_ntile; })
        .def_property_readonly("per_core_out_matrix_width_ntiles", [](const OptimizedConvParallelizationConfig& c) {
            return c.per_core_out_matrix_width_ntile;
        });

    py::class_<OptimizedConvBlockConfig>(module, "OptimizedConvBlockConfig")
        .def(
            py::init<uint32_t, uint32_t, uint32_t, uint32_t>(),
            py::kw_only(),
            py::arg("act_block_h_ntiles").noconvert(),
            py::arg("act_block_w_ntiles").noconvert(),
            py::arg("out_subblock_h_ntiles").noconvert(),
            py::arg("out_subblock_w_ntiles").noconvert())
        .def_property_readonly(
            "act_block_h_ntiles", [](const OptimizedConvBlockConfig& c) { return c.act_block_h_ntiles; })
        .def_property_readonly(
            "act_block_w_ntiles", [](const OptimizedConvBlockConfig& c) { return c.act_block_w_ntiles; })
        .def_property_readonly(
            "out_subblock_h_ntiles", [](const OptimizedConvBlockConfig& c) { return c.out_subblock_h_ntiles; })
        .def_property_readonly(
            "out_subblock_w_ntiles", [](const OptimizedConvBlockConfig& c) { return c.out_subblock_w_ntiles; });
}

}  // namespace conv2d
}  // namespace operations::conv
}  // namespace ttnn
