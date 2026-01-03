// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "conv2d_nanobind.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn-nanobind/export_enum.hpp"
#include "ttnn/operations/conv/conv2d/conv2d.hpp"
#include "ttnn/operations/conv/conv2d/conv2d_utils.hpp"
#include "ttnn/operations/conv/conv2d/device/conv2d_device_operation_types.hpp"
#include "ttnn/operations/conv/conv2d/prepare_conv2d_weights.hpp"
#include "ttnn/operations/sliding_window/sliding_window_nanobind.hpp"
#include "ttnn/types.hpp"
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"

namespace ttnn::operations::conv::conv2d {

void bind_conv2d(nb::module_& mod) {
    bind_registered_operation(
        mod,
        ttnn::conv2d,
        R"doc(
        Applies a 2D convolution over an input signal composed of several input planes.

        Performs a 2D convolution between the input tensor and weight tensor. A 2D kernel (weights tensor) traverses the image (4D input tensor) and a dot product is computed over the overlapping region. For more information, refer to `CNNs on Tenstorrent Architectures <https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/CNNs/ttcnn.md>`_ tech report.

        Args:
            input_tensor (ttnn.Tensor): The input tensor in [N, H, W, C] format. The tensor can be on either the host or the device.
            weight_tensor (ttnn.Tensor): The convolution weights, typically in [out_channels, in_channels // groups, kernel_height, kernel_width] format.
            device (ttnn.MeshDevice): This is a Tenstorrent-specific parameter. The device which will run the operation.
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels produced by the convolution.
            batch_size (int): The batch size of the input tensor.
            input_height (int): This is a Tenstorrent-specific parameter. The height of the input tensor.
            input_width (int): This is a Tenstorrent-specific parameter. The width of the input tensor.
            kernel_size (tuple[int, int]): The size of the convolving kernel.
            stride (tuple[int, int]): The stride of the convolution. Default: (1, 1).
            padding (tuple[int, int] or tuple[int, int, int, int]): Zero-padding added to both sides of the input. Default: (0, 0). [pad_height, pad_width] or [pad_top, pad_bottom, pad_left, pad_right].
            dilation (tuple[int, int]): The spacing between kernel elements. Default: (1, 1).
            groups (int): Number of blocked connections from input channels to output channels. Default: 1.

        Keyword Args:
            dtype (ttnn.DataType, optional): The data type of the output tensor. If not provided, it is inferred from the input tensor.
            bias_tensor (ttnn.Tensor, optional): The bias tensor to be added. Default: None.
            conv_config (ttnn.Conv2dConfig, optional): Configuration for convolution. Default: None.
            compute_config (ttnn.DeviceComputeKernelConfig, optional): Configuration for compute kernel. Default: None
            memory_config (ttnn.MemoryConfig, optional): Output Tensor's Memory Configuration. Default: None.
            slice_config (ttnn.Conv2dSliceConfig, optional): Configuration for slicing input & output tensors in DRAM. If set to None and input is in DRAM, DRAM slicing is automatically enabled. Default: None.
            return_output_dim (bool, optional): If true, the op also returns the height and width of the output tensor in [N, H, W, C] format. Default: False
            return_weights_and_bias (bool, optional): If true, the op also returns the preprocessed weight and bias on device. Default: False

        Returns:
            The output tensor, output height and width, and the preprocessed weights and bias.

            - ttnn.Tensor: Default. The output tensor, when return_output_dim = False and return_weights_and_bias = False
            - tuple[ttnn.Tensor, tuple[int, int]]: The output tensor, and its height and width, if return_output_dim = True
            - tuple[ttnn.Tensor, tuple[ttnn.Tensor, ttnn.Tensor]]: The output tensor, and its height and width, if return_weights_and_bias = True
            - tuple[ttnn.Tensor, tuple[int, int], tuple[ttnn.Tensor, ttnn.Tensor]]: The output tensor, and its height and width, if return_output_dim = True and return_weights_and_bias = True

        Note:
            The `input_tensor` supports the following data type and layout:

            .. list-table:: input_tensor
                :header-rows: 1

                * - dtype
                  - layout
                * - FLOAT32
                  - ROW_MAJOR, TILE
                * - BFLOAT16
                  - ROW_MAJOR, TILE
                * - BFLOAT8_B
                  - TILE

            The `output_tensor` supports the following data type and layout:

            .. list-table:: output_tensor
                :header-rows: 1

                * - dtype
                  - layout
                * - FLOAT32
                  - ROW_MAJOR, TILE
                * - BFLOAT16
                  - ROW_MAJOR, TILE
                * - BFLOAT8_B
                  - TILE

            The `weights_tensor` on the host, supports the following data type and layout:

            .. list-table:: weights_tensor (host)
                :header-rows: 1

                * - dtype
                  - layout
                * - FLOAT32
                  - ROW_MAJOR
                * - BFLOAT16
                  - ROW_MAJOR

            The `weights_tensor` prepared on device, supports the following data type and layout:

            .. list-table:: weights_tensor (prepared on device)
                :header-rows: 1

                * - dtype
                  - layout
                * - FLOAT32
                  - TILE
                * - BFLOAT16
                  - TILE
                * - BFLOAT8_B
                  - TILE

            The `bias_tensor` on the host, supports the following data type and layout:

            .. list-table:: bias_tensor (host)
                :header-rows: 1

                * - dtype
                  - layout
                * - FLOAT32
                  - ROW_MAJOR
                * - BFLOAT16
                  - ROW_MAJOR

            The `bias_tensor` prepared on device, supports the following data type and layout:

            .. list-table:: bias_tensor (prepared on device)
                :header-rows: 1

                * - dtype
                  - layout
                * - FLOAT32
                  - TILE
                * - BFLOAT16
                  - TILE
                * - BFLOAT8_B
                  - TILE
        )doc",
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
               const std::optional<const DataType>& dtype,
               std::optional<const ttnn::Tensor> bias_tensor,
               const std::optional<const Conv2dConfig>& conv_config,
               const std::optional<const DeviceComputeKernelConfig>& compute_config,
               const std::optional<const MemoryConfig>& memory_config,
               const std::optional<const Conv2dSliceConfig>& slice_config_,
               bool return_output_dim,
               bool return_weights_and_bias) -> ResultWithOptions {
                return self(
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
            nb::arg("stride") = nb::cast(std::array<uint32_t, 2>{1, 1}),
            nb::arg("padding") = nb::cast(std::array<uint32_t, 2>{0, 0}),
            nb::arg("dilation") = nb::cast(std::array<uint32_t, 2>{1, 1}),
            nb::arg("groups") = 1,
            nb::arg("dtype") = nb::none(),
            nb::arg("bias_tensor") = nb::none(),
            nb::arg("conv_config") = nb::none(),
            nb::arg("compute_config") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("slice_config") = nb::none(),
            nb::arg("return_output_dim") = false,
            nb::arg("return_weights_and_bias") = false});

    mod.def(
        "prepare_conv_weights",
        prepare_conv_weights,
        nb::kw_only(),
        nb::arg("weight_tensor"),
        nb::arg("input_memory_config"),
        nb::arg("input_layout"),
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
        nb::arg("input_dtype"),
        nb::arg("output_dtype") = nb::none(),
        nb::arg("conv_config") = nb::none(),
        nb::arg("compute_config") = nb::none(),
        nb::arg("slice_config") = nb::none());

    mod.def(
        "prepare_conv_bias",
        prepare_conv_bias,
        nb::kw_only(),
        nb::arg("bias_tensor"),
        nb::arg("input_memory_config"),
        nb::arg("input_layout"),
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
        nb::arg("input_dtype"),
        nb::arg("output_dtype") = nb::none(),
        nb::arg("conv_config") = nb::none(),
        nb::arg("compute_config") = nb::none(),
        nb::arg("slice_config") = nb::none());

    auto py_conv_config = nb::class_<Conv2dConfig>(
        mod,
        "Conv2dConfig",
        R"doc(
        Conv2DConfig is a structure that contains all the Tenstorrent device specific & implementation specific flags for the :func:`ttnn.conv1d`, :func:`ttnn.conv2d` and :func:`ttnn.conv_transpose2d` ops
        )doc");

    py_conv_config.def(
        nb::init<
            std::optional<DataType>,
            std::optional<ttnn::operations::unary::UnaryWithParam>,
            bool,
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
            std::optional<bool>,
            bool,
            std::optional<bool>,
            bool>(),
        nb::kw_only(),
        nb::arg("weights_dtype") = nb::none(),
        nb::arg("activation") = nb::none(),
        nb::arg("deallocate_activation") = false,
        nb::arg("reallocate_halo_output") = true,
        nb::arg("config_tensors_in_dram") = false,
        nb::arg("act_block_h_override") = 0,
        nb::arg("act_block_w_div") = 1,
        nb::arg("reshard_if_not_optimal") = false,
        nb::arg("override_sharding_config") = false,
        nb::arg("shard_layout") = nb::none(),
        nb::arg("core_grid") = nb::none(),
        nb::arg("transpose_shards") = false,
        nb::arg("output_layout") = nb::cast(Layout::TILE),
        nb::arg("enable_act_double_buffer") = false,
        nb::arg("enable_weights_double_buffer") = false,
        nb::arg("full_inner_dim") = false,
        nb::arg("enable_kernel_stride_folding") = nb::none(),
        nb::arg("enable_activation_reuse") = false,
        nb::arg("force_split_reader") = nb::none(),
        nb::arg("override_output_sharding_config") = false);

    py_conv_config.def_rw("weights_dtype", &Conv2dConfig::weights_dtype, R"doc(
        Optional argument which specifies the data type of the preprocessed weights & bias tensor if the Conv2D op is responsible for preparing the weights.
        Supports ttnn.bfloat16 and ttnn.bfloat8_b.
        If unspecified, the preprocessed weights will be in the same format as the input weights.
        If ttnn.bfloat8_b is selected, then the weights should be passed in as ttnn.bfloat16 or ttnn.float32 in row major format.
    )doc");

    py_conv_config.def_rw(
        "activation",
        &Conv2dConfig::activation,
        R"doc(Fused activation function to be applied on the output.
        None means no activation function.
        Use ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU) for ReLU activation.
        Supported activation functions include:
        RELU, SILU, GELU, SIGMOID, TANH, etc.
    )doc");

    py_conv_config.def_rw("deallocate_activation", &Conv2dConfig::deallocate_activation, R"doc(
        Boolean that indicates whether the activation tensor should be deallocated after the conv op is done.
        If true, the activation tensor will be deallocated after the halo micro-op is done.
        Should not be used if the input to the conv op is used by another op.
        Has no effect if input tensor is in DRAM.
        )doc");

    py_conv_config.def_rw("reallocate_halo_output", &Conv2dConfig::reallocate_halo_output, R"doc(
        reallocate_halo_output is a boolean that indicates whether the halo output tensor should be moved to reduce memory fragmentation, before the conv micro-op is called.
        This is ideally used with deallocate_activation = true, when facing OOM issues in the conv micro-op.
    )doc");

    py_conv_config.def_rw("config_tensors_in_dram", &Conv2dConfig::config_tensors_in_dram, R"doc(
        Boolean that determines where config tensors should be stored. Setting it to true stores them in DRAM. False stores them in L1_SMALL.
        Config tensors are used by Conv2D, Pooling and other 2D ops to store how data should be loaded, instead of computing on device RISC-cores.
    )doc");

    py_conv_config.def_rw("act_block_h_override", &Conv2dConfig::act_block_h_override, R"doc(
            Controls the size of the activation block height.

            The activation matrix is created from the input tensor, and is matrix multiplied with the weights tensor to generate the output tensor.
            The activation block is the chunk of the activation matrix that is available in L1 Memory, as the activation matrix gets divided among cores, and also can be further subdivided within a core.
            If set to 0, the the maximum possible size for the activation block is used, which is equal to output_matrix_height_per_core.
            This leads to large temporary Circular Buffers when the output matrix height is large, leading to OOM.

            This flag specifies the height of the activation block to act_block_h_override. This must be a multiple of 32, and must evenly divide the maximum possible size of the activation block.
        )doc");

    py_conv_config.def_rw("act_block_w_div", &Conv2dConfig::act_block_w_div, R"doc(
            Reduces the width of the activation block to reduce Circular Buffer sizes and prevent OOM. Valid only for Width Sharded Conv2d.
            This is only useful when the input channels is greater than 32 * num_cores. For n150, thats 32 * 64 =  2048.
            This is a divisor of the activation block width.
            A value of 1 means no reduction, and a value of 2 means the activation block width is halved.
        )doc");

    py_conv_config.def_rw("reshard_if_not_optimal", &Conv2dConfig::reshard_if_not_optimal, R"doc(
        This flag is used to determine if the input tensor should be resharded if the input tensor current shard config is not optimal.
        This flag is used only if the input tensor is already sharded. If it is not sharded, the input tensor will anyway be sharded to the optimal config.

        If this flag is false, the conv op will try to execute the op with the current shard config.
        It is recommended to set this flag to true if the input dimensions of the previous conv op and the current op are significantly different, either due to differences in the input vs output channels, or large stride / kernel size / dilation.
        )doc");

    py_conv_config.def_rw("override_sharding_config", &Conv2dConfig::override_sharding_config, R"doc(
        Boolean flag that allows the core grid for the conv op to be specified.
        If true, then core_grid must also be specified.
        )doc");

    // TODO_NANOBIND: May need to change to prop_rw and add explicit none setter arg
    py_conv_config.def_rw(
        "shard_layout",
        &Conv2dConfig::shard_layout,
        R"doc(
        Optional argument that determines the TensorMemoryLayout to be used for the input and output tensor.
        If this is not specified, the op will try to determine the optimal layout based on it's own heuristics.
        Can be either :class:`ttnn.TensorMemoryLayout.HEIGHT_SHARDED`, :class:`ttnn.TensorMemoryLayout.BLOCK_SHARDED` or :class:`ttnn.TensorMemoryLayout.WIDTH_SHARDED`.
        )doc");

    py_conv_config.def_rw("core_grid", &Conv2dConfig::core_grid, R"doc(
        Core Grid to be used for sharding the input tensor.
        This flag is only used when override_sharding_config or override_output_sharding_config is set to true. )doc");

    py_conv_config.def_rw("transpose_shards", &Conv2dConfig::transpose_shards, R"doc(
        Determines if the Shard Orientation should be Row Major or Column Major.
        If true, the shard orientation is Row Major. If false, the shard orientation is Column Major.
        This is useful for Block Sharded Conv2D when the device core grid is not a square.
        )doc");

    py_conv_config.def_rw("output_layout", &Conv2dConfig::output_layout, R"doc(
        The layout of the output tensor. Can be either :class:`ttnn.Layout.TILE` or :class:`ttnn.Layout.ROW_MAJOR`.
        Conv2D expects it's input to be in :class:`ttnn.Layout.ROW_MAJOR` format.
        If the input is in :class:`ttnn.Layout.TILE` format, the halo micro-op will convert it to :class:`ttnn.Layout.ROW_MAJOR` format.
        So if the next op is a conv op, it is recommended to set this to :class:`ttnn.Layout.ROW_MAJOR`.
        )doc");

    py_conv_config.def_rw("enable_act_double_buffer", &Conv2dConfig::enable_act_double_buffer, R"doc(
            Doubles the size of the Activation Circular Buffer to allow for double buffering, preventing stalls of the activation reader kernel.
            This improves performance, but increases memory usage.
    )doc");

    py_conv_config.def_rw("enable_weights_double_buffer", &Conv2dConfig::enable_weights_double_buffer, R"doc(
            Doubles the size of the Weights Circular Buffer to allow for double buffering, preventing stalls of the weights reader kernel.
            This improves performance, but increases the memory usage of the weights tensor.
        )doc");

    py_conv_config.def_rw("full_inner_dim", &Conv2dConfig::full_inner_dim, R"doc(
            Applies only to block sharded layout.
            By default inner dim of activation matrix will be sliced by kernel_h.
            If L1 constraints allowed it we can use full inner dim.
            This will increase perf, but it will take more L1 space.
        )doc");

    py_conv_config.def_rw("enable_kernel_stride_folding", &Conv2dConfig::enable_kernel_stride_folding, R"doc(
        ===================== EXPERIMENTAL FEATURE ======================

        Enables tensor folding optimization that transforms convolution operations by reshaping tensors
        and adjusting stride patterns for improved computational efficiency.

        Args:
            enable_kernel_stride_folding (Optional[bool]):
                - None (default): Automatic enablement based on optimal conditions
                - True: Force enable the optimization
                - False: Disable the optimization

        Behavior:
        When enabled, this optimization reshapes tensors as follows:

        * Input tensor (NHWC format):
          - From: (N, H, W, IC)
          - To: (N, H / stride[0], W / stride[1], IC * stride[0] * stride[1])

        * Weight tensor:
          - From: (OC, IC, kernel[0], kernel[1])
          - To: (1, 1, IC * (kernel[0] + pad_h) * (kernel[1] + pad_w), OC)
          where pad_h = kernel[0] % stride[0] and pad_w = kernel[1] % stride[1]

        * Stride: Becomes (1, 1) after folding

        Automatic Enablement:
        When set to None, automatically enabled when ALL conditions are met (transforms conv2d into Fold + MatMul):
        1. Stride equals kernel size in both dimensions (stride == kernel_size)
        2. Stride is greater than 1 in at least one dimension
        3. No dilation applied (dilation == [1, 1])
        4. Input height and width (after padding) are divisible by respective stride values
        5. Input tensor memory: DRAM (all types except bfloat8_b) OR L1 Height-sharded (all types)

        Manual Enablement:
        Particularly beneficial for unaligned input channels (e.g., small channel counts like 3 RGB channels).

        Requirements when forcing enable_kernel_stride_folding=True:
        - Stride ≤ kernel size in both dimensions
        - Input tensor supports folding (DRAM except bfloat8_b, or L1 Height-sharded)
        - Input dimensions after padding are divisible by stride values

        Example:
        For small channel counts (like 3 RGB channels) with stride=2x2, kernel=7x7:
        - Transforms 3 channels → 12 channels, stride 2x2 → 1x1
        - Reduces required padding for alignment (3→12 uses alignment more efficiently)
        - Kernel size reduces to kernel/stride (e.g., 7x7 kernel → 4x4 kernel with padding)

        Note: The weight tensor padding is applied implicitly and not passed via the padding argument.

        ===============================================================
        )doc");
    py_conv_config.def_rw("enable_activation_reuse", &Conv2dConfig::enable_activation_reuse, R"doc(
        ===================== EXPERIMENTAL FEATURE ======================

        Enables reusing data between consecutive image rows.
        It can be enabled for height sharding only and boosts image2column performance,
        so its meant to be used for reader-bound convolutions.

        ===============================================================
    )doc");

    py_conv_config.def_rw("force_split_reader", &Conv2dConfig::force_split_reader, R"doc(
        ===================== EXPERIMENTAL FEATURE ======================

        This uses both the reader & writer cores to carry out the activation reader operation.
        This is useful when the input tensor is large, and the activation reader is a bottleneck.
        This is only supported for Height Sharded Conv2D.
        Setting this overrides the split reader heuristic.

        ===============================================================
    )doc");

    py_conv_config.def_rw("override_output_sharding_config", &Conv2dConfig::override_output_sharding_config, R"doc(
        ===================== EXPERIMENTAL FEATURE ======================

        override_output_sharding_config enables the user to specify the memory config of the output tensor
        This impacts the core grid that executes matmul part of conv2d
        Feature is currently supported only for BLOCK_SHARDED layout, without DRAM slicing
        Additionally, NHW number of cores must match between input and output tensors

        ===============================================================
    )doc");

    py_conv_config.def("__repr__", [](const Conv2dConfig& config) { return fmt::format("{}", config); });
}

}  // namespace ttnn::operations::conv::conv2d
