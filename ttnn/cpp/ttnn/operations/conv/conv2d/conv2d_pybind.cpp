// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "conv2d_pybind.hpp"

#include <array>
#include <cstdint>
#include <variant>
#include <optional>
#include <string>

#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
#include "ttnn/operations/conv/conv2d/conv2d.hpp"
#include "ttnn/operations/conv/conv2d/conv2d_utils.hpp"
#include "ttnn/operations/conv/conv2d/device/conv2d_op.hpp"
#include "ttnn/operations/conv/conv2d/prepare_conv2d_weights.hpp"
#include "ttnn/operations/sliding_window/sliding_window_pybind.hpp"
#include "ttnn/types.hpp"
#include <tt-metalium/constants.hpp>

namespace ttnn::operations::conv::conv2d {

void py_bind_conv2d(py::module& module) {
    bind_registered_operation(
        module,
        ttnn::conv2d,
        R"doc(
        Applies a 2D convolution over an input signal composed of several input planes.

        For more information, refer to `this tech report. <https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/CNNs/ttcnn.md>`_

        :param ttnn.Tensor input_tensor:  The input tensor. This must be in the format [N, H, W, C]. It can be on host or device.
        :param ttnn.Tensor weight_tensor: The weight tensor. The weights can be passed in the same format as PyTorch, [out_channels, in_channels, kernel_height, kernel_width]. The op w
        :param ttnn.Tensor, None bias_tensor:   Optional bias tensor. Default: None
        :param ttnn.IDevice device:  The device to use.
        :param int in_channels:  Number of input channels.
        :param int out_channels:  Number of output channels.
        :param int batch_size:  Batch size.
        :param int input_height:  Height of the input tensor.
        :param int input_width:  Width of the input tensor.
        :param tuple[int, int] kernel_size: Size of the convolving kernel.
        :param tuple[int, int] stride: Stride of the cross-correlation.
        :param tuple[int, int] or tuple[int, int, int, int]) padding: Zero-padding added to both sides of the input. [pad_height, pad_width] or [pad_top, pad_bottom, pad_left, pad_right].
        :param tuple[int, int] dilation: Spacing between kernel elements.
        :param int groups:  Number of blocked connections from input channels to output channels.
        :param ttnn.DataType, None dtype:  The data type of the output tensor. Default: None (inferred from input tensor).
        :param ttnn.Conv2dConfig, None conv_config: Configuration for convolution. Default: None
        :param ttnn.DeviceComputeKernelConfig, None compute_config: Configuration for compute kernel. Default: None
        :param ttnn.MemoryConfig, None memory_config: Output Tensor's Memory Configuration. Default: None
        :param bool return_output_dim:  If true, the op also returns the height and width of the output tensor in [N, H, W, C] format,
        :param bool return_weights_and_bias:  If true, the op also returns the preprocessed weight and bias on device .

        :return: The output tensor, output height and width, and the preprocessed weights and bias.

        :rtype: [ttnn.Tensor]: The output tensor, when return_output_dim = False and return_weights_and_bias = False
        :rtype: [ttnn.Tensor, Tuple[int, int]]: The output tensor, and it's height and width, if return_output_dim = True
        :rtype: [ttnn.Tensor, Tuple[ttnn.Tensor, ttnn.Tensor]]: The output tensor, and it's height and width, if return_weights_and_bias = True
        :rtype: [ttnn.Tensor, Tuple[int, int], Tuple[ttnn.Tensor, ttnn.Tensor]]: The output tensor, and it's height and width, if return_output_dim = True and return_weights_and_bias = True
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
               const std::optional<const DataType>& dtype,
               std::optional<const ttnn::Tensor> bias_tensor,
               const std::optional<const Conv2dConfig>& conv_config,
               const std::optional<const DeviceComputeKernelConfig>& compute_config,
               const std::optional<const MemoryConfig>& memory_config,
               const std::optional<const Conv2dSliceConfig>& slice_config_,
               bool return_output_dim,
               bool return_weights_and_bias,
               QueueId queue_id) -> ResultWithOptions {
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
                    dtype,
                    bias_tensor,
                    conv_config,
                    compute_config,
                    memory_config,
                    slice_config_,
                    return_output_dim,
                    return_weights_and_bias);
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
            py::arg("stride") = std::array<uint32_t, 2>{1, 1},
            py::arg("padding") = std::array<uint32_t, 2>{0, 0},
            py::arg("dilation") = std::array<uint32_t, 2>{1, 1},
            py::arg("groups") = 1,
            py::arg("dtype") = std::nullopt,
            py::arg("bias_tensor") = std::nullopt,
            py::arg("conv_config") = std::nullopt,
            py::arg("compute_config") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("slice_config") = std::nullopt,
            py::arg("return_output_dim") = false,
            py::arg("return_weights_and_bias") = false,
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
               const std::optional<const DataType>& dtype,
               std::optional<const ttnn::Tensor> bias_tensor,
               const std::optional<const Conv2dConfig>& conv_config,
               const std::optional<const DeviceComputeKernelConfig>& compute_config,
               const std::optional<const MemoryConfig>& memory_config,
               const std::optional<const Conv2dSliceConfig>& slice_config_,
               bool return_output_dim,
               bool return_weights_and_bias,
               QueueId queue_id) -> ResultWithOptions {
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
                    dtype,
                    bias_tensor,
                    conv_config,
                    compute_config,
                    memory_config,
                    slice_config_,
                    return_output_dim,
                    return_weights_and_bias);
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
            py::arg("stride") = std::array<uint32_t, 2>{1, 1},
            py::arg("padding") = std::array<uint32_t, 2>{0, 0},
            py::arg("dilation") = std::array<uint32_t, 2>{1, 1},
            py::arg("groups") = 1,
            py::arg("dtype") = std::nullopt,
            py::arg("bias_tensor") = std::nullopt,
            py::arg("conv_config") = std::nullopt,
            py::arg("compute_config") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("slice_config") = std::nullopt,
            py::arg("return_output_dim") = false,
            py::arg("return_weights_and_bias") = false,
            py::arg("queue_id") = DefaultQueueId});

    module.def(
        "prepare_conv_weights",
        prepare_conv_weights<ttnn::IDevice>,
        py::kw_only(),
        py::arg("weight_tensor"),
        py::arg("input_memory_config"),
        py::arg("input_layout"),
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
        py::arg("input_dtype"),
        py::arg("output_dtype") = std::nullopt,
        py::arg("conv_config") = std::nullopt,
        py::arg("compute_config") = std::nullopt,
        py::arg("slice_config") = std::nullopt);

    module.def(
        "prepare_conv_weights",
        prepare_conv_weights<ttnn::MeshDevice>,
        py::kw_only(),
        py::arg("weight_tensor"),
        py::arg("input_memory_config"),
        py::arg("input_layout"),
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
        py::arg("input_dtype"),
        py::arg("output_dtype") = std::nullopt,
        py::arg("conv_config") = std::nullopt,
        py::arg("compute_config") = std::nullopt,
        py::arg("slice_config") = std::nullopt);

    module.def(
        "prepare_conv_bias",
        prepare_conv_bias<ttnn::IDevice>,
        py::kw_only(),
        py::arg("bias_tensor"),
        py::arg("input_memory_config"),
        py::arg("input_layout"),
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
        py::arg("input_dtype"),
        py::arg("output_dtype") = std::nullopt,
        py::arg("conv_config") = std::nullopt,
        py::arg("compute_config") = std::nullopt);

    module.def(
        "prepare_conv_bias",
        prepare_conv_bias<ttnn::MeshDevice>,
        py::kw_only(),
        py::arg("bias_tensor"),
        py::arg("input_memory_config"),
        py::arg("input_layout"),
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
        py::arg("input_dtype"),
        py::arg("output_dtype") = std::nullopt,
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

    auto py_conv_slice_config = py::class_<Conv2dSliceConfig>(
        module,
        "Conv2dSliceConfig",
        R"doc(
        | Conv2dSliceConfig is a structure that is used to configure how the input & output tensors of Conv2D are sliced when they are placed in DRAM. \
        | Conv2D only supports inputs in L1. If the input tensor or output tensor are too large to fit into L1, then the Conv2d_DRAM version can be used. \
        | It slices the input & output into slices and applies the Conv2D op on each slice. \
        | Conv2dSliceConfig determines how this slicing happens.
        )doc");
    py_conv_slice_config.def(
        py::init<Conv2dSliceConfig::SliceType, uint32_t>(),
        py::kw_only(),
        py::arg("slice_type"),
        py::arg("num_slices"));
    py_conv_slice_config.def_readwrite(
        "slice_type",
        &Conv2dSliceConfig::slice_type,
        R"doc(
        | The type of slice to be used. Can be either SliceHeight or SliceWidth. When the tensor is in [N, H, W, C] format, then it can slice either along the height or width dimension.
        | Slicing along the width is preferable as it reduces the size of the output of the Halo operation.
        | Use SliceHeight only when the height dimension is much larger than the width dimension.
        )doc");
    py_conv_slice_config.def_readwrite(
        "num_slices",
        &Conv2dSliceConfig::num_slices,
        R"doc(
        | The number of slices that the input & output tensors are divided into.
        | The output tensor is divided into num_slices slices along the slice_type dimension.
        | The corresponding input tensor needed to calculate that output is determined and sliced.
        | If the size of the slice dimension is not divisible by num_slices, then the last slice will be smaller than the rest.
        )doc");
    py::enum_<Conv2dSliceConfig::SliceType>(py_conv_slice_config, "SliceTypeEnum")
        .value("SliceHeight", Conv2dSliceConfig::SliceType::HEIGHT)
        .value("SliceWidth", Conv2dSliceConfig::SliceType::WIDTH);

    auto py_conv_config = py::class_<Conv2dConfig>(
        module,
        "Conv2dConfig",
        R"doc(
        Conv2DConfig is a structure that contains all the Tenstorrent device specific & implementation specific flags for the :func:`ttnn.conv1d`, :func:`ttnn.conv2d` and :func:`ttnn.conv_transpose2d` ops
        )doc");
    py_conv_config.def(
        py::init<
            std::optional<DataType>,
            std::string,
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
            bool>(),
        py::kw_only(),
        py::arg("weights_dtype") = std::nullopt,
        py::arg("activation") = "",
        py::arg("deallocate_activation") = false,
        py::arg("reallocate_halo_output") = true,
        py::arg("act_block_h_override") = 0,
        py::arg("act_block_w_div") = 1,
        py::arg("reshard_if_not_optimal") = false,
        py::arg("override_sharding_config") = false,
        py::arg("shard_layout") = std::nullopt,
        py::arg("core_grid") = std::nullopt,
        py::arg("transpose_shards") = false,
        py::arg("output_layout") = Layout::TILE,
        py::arg("enable_act_double_buffer") = false,
        py::arg("enable_weights_double_buffer") = false,
        py::arg("enable_split_reader") = false,
        py::arg("enable_subblock_padding") = false,
        py::arg("in_place") = false,
        py::arg("enable_kernel_stride_folding") = false);
    py_conv_config.def_readwrite("weights_dtype", &Conv2dConfig::weights_dtype, R"doc(
        Optional argument which specifies the data type of the preprocessed weights & bias tensor if the Conv2D op is responsible for preparing the weights.
        Supports ttnn.bfloat16 and ttnn.bfloat8_b.
        If unspecified, the preprocessed weights will be in the same format as the input weights.
        If ttnn.bfloat8_b is selected, then the weights should be passed in as ttnn.bfloat16 or ttnn.float32 in row major format.
    )doc");
    py_conv_config.def_readwrite(
        "activation",
        &Conv2dConfig::activation,
        R"doc(A string that selects the fused activation function to be applied on the output.
        Empty string means no activation function.
        Supported activation function strings are:
        relu, silu, mish, sigmoid, sigmoid_approx, tanh, log, softplus, gelu, sqrt
    )doc");
    py_conv_config.def_readwrite("deallocate_activation", &Conv2dConfig::deallocate_activation, R"doc(
        Boolean that indicates whether the activation tensor should be deallocated after the conv op is done.
        If true, the activation tensor will be deallocated after the halo micro-op is done.
        Should not be used if the input to the conv op is used by another op.
        )doc");
    py_conv_config.def_readwrite("reallocate_halo_output", &Conv2dConfig::reallocate_halo_output, R"doc(
        reallocate_halo_output is a boolean that indicates whether the halo output tensor should be moved to reduce memory fragmentation, before the conv micro-op is called.
        This is ideally used with deallocate_activation = true, when facing OOM issues in the conv micro-op.

    )doc");
    py_conv_config.def_readwrite("act_block_h_override", &Conv2dConfig::act_block_h_override, R"doc(
            Controls the size of the activation block height.

            The activation matrix is created from the input tensor, and is matrix multiplied with the weights tensor to generate the output tensor.
            The activation block is the chunk of the activation matrix that is available in L1 Memory, as the activation matrix gets divided among cores, and also can be further subdivided within a core.
            If set to 0, the the maximum possible size for the activation block is used, which is equal to output_matrix_height_per_core.
            This leads to large temporary Circular Buffers when the output matrix height is large, leading to OOM.

            This flag specifies the height of the activation block to act_block_h_override. This must be a multiple of 32, and must evenly divide the maximum possible size of the activation block.
        )doc");
    py_conv_config.def_readwrite("act_block_w_div", &Conv2dConfig::act_block_w_div, R"doc(
            Reduces the width of the activation block to reduce Circular Buffer sizes and prevent OOM. Valid only for Width Sharded Conv2d.
            This is only useful when the input channels is greater than 32 * num_cores. For n150, thats 32 * 64 =  2048.
            This is a divisor of the activation block width.
            A value of 1 means no reduction, and a value of 2 means the activation block width is halved.
        )doc");
    py_conv_config.def_readwrite("reshard_if_not_optimal", &Conv2dConfig::reshard_if_not_optimal, R"doc(
        This flag is used to determine if the input tensor should be resharded if the input tensor current shard config is not optimal.
        This flag is used only if the input tensor is already sharded. If it is not sharded, the input tensor will anyway be sharded to the optimal config.

        If this flag is false, the conv op will try to execute the op with the current shard config.
        It is recommended to set this flag to true if the input dimensions of the previous conv op and the current op are significantly different, either due to differences in the input vs output channels, or large stride / kernel size / dilation.
        )doc");
    py_conv_config.def_readwrite("override_sharding_config", &Conv2dConfig::override_sharding_config, R"doc(
        Boolean flag that allows the core grid for the conv op to be specified.
        If true, then core_grid must also be specified.
        )doc");
    py_conv_config.def_readwrite("shard_layout", &Conv2dConfig::shard_layout, R"doc(
        Optional argument that determines the TensorMemoryLayout to be used for the input and output tensor.
        If this is not specified, the op will try to determine the optimal layout based on it's own heuristics.
        Can be either :class:`ttnn.TensorMemoryLayout.HEIGHT_SHARDED`, :class:`ttnn.TensorMemoryLayout.BLOCK_SHARDED` or :class:`ttnn.TensorMemoryLayout.WIDTH_SHARDED`.
        )doc");
    py_conv_config.def_readwrite("core_grid", &Conv2dConfig::core_grid, R"doc(
        Core Grid to be used for sharding the input tensor.
        This flag is only used when override_sharding_config is set to true. )doc");

    py_conv_config.def_readwrite("transpose_shards", &Conv2dConfig::transpose_shards, R"doc(
        Determines if the Shard Orientation should be Row Major or Column Major.
        If true, the shard orientation is Row Major. If false, the shard orientation is Column Major.
        This is useful for Block Sharded Conv2D when the device core grid is not a square.
        )doc");
    py_conv_config.def_readwrite("output_layout", &Conv2dConfig::output_layout, R"doc(
        The layout of the output tensor. Can be either :class:`ttnn.Layout.TILE` or :class:`ttnn.Layout.ROW_MAJOR`.
        Conv2D expects it's input to be in :class:`ttnn.Layout.ROW_MAJOR` format.
        If the input is in :class:`ttnn.Layout.TILE` format, the halo micro-op will convert it to :class:`ttnn.Layout.ROW_MAJOR` format.
        So if the next op is a conv op, it is recommended to set this to :class:`ttnn.Layout.ROW_MAJOR`.
        )doc");
    py_conv_config.def_readwrite("enable_act_double_buffer", &Conv2dConfig::enable_act_double_buffer, R"doc(
            Doubles the size of the Activation Circular Buffer to allow for double buffering, preventing stalls of the activation reader kernel.
            This improves performance, but increases memory usage.
    )doc");
    py_conv_config.def_readwrite("enable_weights_double_buffer", &Conv2dConfig::enable_weights_double_buffer, R"doc(
            Doubles the size of the Weights Circular Buffer to allow for double buffering, preventing stalls of the weights reader kernel.
            This improves performance, but increases the memory usage of the weights tensor.
        )doc");
    py_conv_config.def_readwrite("enable_split_reader", &Conv2dConfig::enable_split_reader, R"doc(
            This uses both the reader & writer cores to carry out the activation reader operation.
            This is useful when the input tensor is large, and the activation reader is a bottleneck.
            This is only supported for Height Sharded Conv2D.
        )doc");
    py_conv_config.def_readwrite("enable_subblock_padding", &Conv2dConfig::enable_subblock_padding);
    py_conv_config.def_readwrite("in_place", &Conv2dConfig::in_place, R"doc(
            Enables support for in_place halo.
            This re-uses the input tensor as the output for halo, overwriting the input tensor.
            This can be used if the input tensor is not used by any other op after the conv op.
        )doc");

    py_conv_config.def_readwrite("enable_kernel_stride_folding", &Conv2dConfig::enable_kernel_stride_folding, R"doc(
        ===================== EXPERIMENTAL FEATURE ======================

        Enables tensor folding optimization when strides match kernel dimensions.

        This feature is under development and may change without notice.
        Use with caution in production environments (Issue: #22378).

        When enabled, this optimization reshapes tensors as follows:

        * Input tensor (NHWC format):
          - From: (N, H, W, IC)
          - To: (N, H/stride[0], W/stride[1], IC * kernel[0] * kernel[1])

        * Weight tensor:
          - From: (OC, IC, kernel[0], kernel[1])
          - To: (1, 1, IC * kernel[0] * kernel[1], OC)

        Note: This optimization is currently only applied when all of the following conditions are met:
        1. The stride dimensions exactly match the kernel dimensions (stride[0] == kernel[0] and stride[1] == kernel[1])
        2. The input tensor is stored in DRAM memory

        ===============================================================
        )doc");

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

}  // namespace ttnn::operations::conv::conv2d
