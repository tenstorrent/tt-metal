// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/pool/generic/generic_pools_nanobind.hpp"
#include "ttnn/operations/pool/generic/generic_pools.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <variant>

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::pool {

void bind_max_pool2d_operation(nb::module_& mod) {
    bind_registered_operation(
        mod,
        ttnn::max_pool2d,
        R"doc(
        Applies a max pool convolution to the input tensor. The resulting output Tensor will contain the maximum
        value for each channel within a kernel window. The input tensor is expected to be in [NHW, C] format and
        should be on the device. Height, width and block sharding schemes are supported.

        Args:
            input_tensor_a (ttnn.Tensor): the tensor to be convolved.
            batch_size (int): the number of batches (N in a [N, C, H, W] shaped tensor).
            input_h (int): the height of the input tensor (H in a [N, C, H, W] shaped tensor).
            input_w (int): the width of the input tensor (W in a [N, C, H, W] shaped tensor).
            channels (int): the number of channels (C in a [N, C, H, W] shaped tensor).
            kernel_size (List of [int]): the (h, w) size of the kernel window.
            stride (List of [int]): the (h, w) stride of the kernel window.
            padding (List of [int]): the (h, w) padding of the input tensor.
            dilation (List of [int]): the (h, w) dilation of the kernel window.
            ceil_mode (bool): whether to use ceil mode for the output shape. Defaults to `False`.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): the memory configuration for the output tensor. Defaults to `None`.
            applied_shard_scheme (ttnn.TensorMemoryLayout, optional): the sharding scheme to apply to a non-pre-sharded input tensor. Defaults to `None`, which should be used with pre-sharded input tensors.
            in_place (bool, optional): whether to perform the halo operation in place. Defaults to `False`.
            deallocate_input (bool, optional): whether to deallocate the input tensor after the operation. Defaults to `False`.
            reallocate_halo_output (bool, optional): whether to reallocate the halo output tensor after the operation, ideally used with deallocate_activation = true. Defaults to `True`.
            return_indices (bool, optional): whether to return both values and indices. When True, returns a tuple (values, indices). Defaults to `False`.
            dtype (ttnn.DataType, optional): the data format for the output tensor. Defaults to `ttnn.bfloat16`.
            output_layout (ttnn.Layout, optional): the layout for the output tensor. Defaults to `ttnn.ROW_MAJOR_LAYOUT`.

        Returns:
            ttnn.Tensor or tuple[ttnn.Tensor, ttnn.Tensor]: the max pool convolved output tensor, or a tuple of (values, indices) if return_indices is True.

        Example:
            >>> import ttnn
            >>> import torch
            >>> device = ttnn.CreateDevice(0, l1_small_size=8192)
            >>> kernel_h, kernel_w = 2, 2
            >>> stride_h, stride_w = 1, 1
            >>> pad_h, pad_w = 0, 0
            >>> dilation_h, dilation_w = 1, 1
            >>> nchw_shape = (4, 256, 40, 40)
            >>> in_N, in_C, in_H, in_W = nchw_shape
            >>> input_shape = (1, 1, in_N * in_H * in_W, in_C)
            >>> input = torch.randn(nchw_shape, dtype=torch.bfloat16)
            >>> input_perm = torch.permute(input, (0, 2, 3, 1)) # this op expects a [N, H, W, C] format
            >>> input_reshape = input_perm.reshape(input_shape)
            >>> tt_input= ttnn.from_torch(input_reshape, ttnn.bfloat16)
            >>> tt_input_dev = ttnn.to_device(tt_input, device)
            >>> tt_output = ttnn.max_pool2d(
                                input_tensor=tt_input_dev,
                                batch_size=in_N,
                                input_h=in_H,
                                input_w=in_W,
                                channels=in_C,
                                kernel_size=[kernel_h, kernel_w],
                                stride=[stride_h, stride_w],
                                padding=[pad_h, pad_w],
                                dilation=[dilation_h, dilation_w],
                                ceil_mode=False,
                                memory_config=None,
                                applied_shard_scheme=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                                in_place_halo=False,
                                deallocate_input=False,
                                reallocate_halo_output=True,
                                dtype=ttnn.bfloat16,
                                output_layout=ttnn.ROW_MAJOR_LAYOUT,
                            )

        )doc",
        ttnn::nanobind_overload_t{
            [](const decltype(ttnn::max_pool2d)& self,
               const ttnn::Tensor& input_tensor,
               uint32_t batch_size,
               uint32_t input_h,
               uint32_t input_w,
               uint32_t channels,
               std::array<uint32_t, 2> kernel_size,
               std::array<uint32_t, 2> stride,
               std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>> padding,
               std::array<uint32_t, 2> dilation,
               bool ceil_mode,
               const std::optional<const MemoryConfig>& memory_config,
               const std::optional<const ttnn::TensorMemoryLayout> applied_shard_scheme,
               bool in_place_halo,
               bool deallocate_input,
               bool reallocate_halo_output,
               bool return_indices,
               const DataType dtype,
               const Layout output_layout) -> nb::object {
                auto result = self(
                    input_tensor,
                    batch_size,
                    input_h,
                    input_w,
                    channels,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    ceil_mode,
                    memory_config,
                    applied_shard_scheme,
                    in_place_halo,
                    deallocate_input,
                    reallocate_halo_output,
                    return_indices,
                    dtype,
                    output_layout);

                // Handle variant return type
                if (std::holds_alternative<MaxPoolWithIndicesResult>(result)) {
                    auto mpwi_result = std::get<MaxPoolWithIndicesResult>(result);
                    return nb::make_tuple(mpwi_result.output, mpwi_result.indices);
                } else {
                    return nb::cast(std::get<ttnn::Tensor>(result));
                }
            },
            nb::arg("input_tensor"),
            nb::arg("batch_size"),
            nb::arg("input_h"),
            nb::arg("input_w"),
            nb::arg("channels"),
            nb::arg("kernel_size"),
            nb::arg("stride"),
            nb::arg("padding"),
            nb::arg("dilation"),
            nb::arg("ceil_mode") = false,
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("applied_shard_scheme") = nb::none(),
            nb::arg("in_place_halo") = false,
            nb::arg("deallocate_input") = false,
            nb::arg("reallocate_halo_output") = true,
            nb::arg("return_indices") = false,
            nb::arg("dtype") = DataType::BFLOAT16,
            nb::arg("output_layout") = Layout::ROW_MAJOR});
}

void bind_avg_pool2d_operation(nb::module_& mod) {
    bind_registered_operation(
        mod,
        ttnn::avg_pool2d,
        R"doc(
        Applies an average pool convolution to the input tensor. The resulting output Tensor will contain the average
        value for each channel within a kernel window. The input tensor is expected to be in [NHW, C] format and
        should be on the device. Height, width and block sharding schemes are supported.

        Args:
            input_tensor_a (ttnn.Tensor): the tensor to be convolved.
            batch_size (int): the number of batches (N in a [N, C, H, W] shaped tensor).
            input_h (int): the height of the input tensor (H in a [N, C, H, W] shaped tensor).
            input_w (int): the width of the input tensor (W in a [N, C, H, W] shaped tensor).
            channels (int): the number of channels (C in a [N, C, H, W] shaped tensor).
            kernel_size (List of [int]): the (h, w) size of the kernel window.
            stride (List of [int]): the (h, w) stride of the kernel window.
            padding (List of [int]): the (h, w) padding of the input tensor.
            ceil_mode (bool): When True, uses 'ceiling' function instead of 'floor' function in the formula to compute output shape. Default: False.
            count_include_pad (bool): When True, includes zero-padding in the avg calculation. Default: True.
            divisor_override (int, optional): If specified, it will be used as a divisor, otherwise size of the pooling region will be used. Default: None. Not currently supported in ttnn.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): the memory configuration for the output tensor. Defaults to `None`.
            applied_shard_scheme (ttnn.TensorMemoryLayout, optional): the sharding scheme to apply to a non-pre-sharded input tensor. Defaults to `None`, which should be used with pre-sharded input tensors.
            in_place (bool, optional): whether to perform the halo operation in place. Defaults to `False`.
            deallocate_input (bool, optional): whether to deallocate the input tensor after the operation. Defaults to `False`.
            reallocate_halo_output (bool, optional): whether to reallocate the halo output tensor after the operation, ideally used with deallocate_activation = true. Defaults to `True`.
            dtype (ttnn.DataType, optional): the data format for the output tensor. Defaults to `ttnn.bfloat16`.
            output_layout (ttnn.Layout, optional): the layout for the output tensor. Defaults to `ttnn.ROW_MAJOR_LAYOUT`.

        Returns:
            ttnn.Tensor: the average pool convolved output tensor.

        Example:
            >>> import ttnn
            >>> import torch
            >>> device = ttnn.open_device(device_id=0, l1_small_size=8192)
            >>> kernel_h, kernel_w = 2, 2
            >>> stride_h, stride_w = 1, 1
            >>> pad_h, pad_w = 0, 0
            >>> nchw_shape = (4, 256, 40, 40)
            >>> in_N, in_C, in_H, in_W = nchw_shape
            >>> input_shape = (1, 1, in_N * in_H * in_W, in_C)
            >>> input = torch.randn(nchw_shape, dtype=torch.bfloat16)
            >>> input_perm = torch.permute(input, (0, 2, 3, 1)) # this op expects a [N, H, W, C] format
            >>> input_reshape = input_perm.reshape(input_shape) # this op expects [1, 1, NHW, C]
            >>> tt_input = ttnn.from_torch(input_reshape, device=device)
            >>> tt_output = ttnn.avg_pool2d(
                            input_tensor=tt_input,
                            batch_size=in_N,
                            input_h=in_H,
                            input_w=in_W,
                            channels=in_C,
                            kernel_size=[kernel_h, kernel_w],
                            stride=[stride_h, stride_w],
                            padding=[pad_h, pad_w],
                            dilation=[dilation_h, dilation_w],
                            ceil_mode=False,
                            count_include_pad=True,
                            divisor_override=None,
                            memory_config=None,
                            applied_shard_scheme=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                            in_place_halo=False,
                            deallocate_input=False,
                            reallocate_halo_output=True,
                            dtype=ttnn.bfloat16,
                            output_layout=ttnn.ROW_MAJOR_LAYOUT,
                        )
        )doc",
        ttnn::nanobind_overload_t{
            [](const decltype(ttnn::avg_pool2d)& self,
               const ttnn::Tensor& input_tensor,
               uint32_t batch_size,
               uint32_t input_h,
               uint32_t input_w,
               uint32_t channels,
               std::array<uint32_t, 2> kernel_size,
               std::array<uint32_t, 2> stride,
               std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>> padding,
               bool ceil_mode,
               bool count_include_pad,
               std::optional<int32_t> divisor_override,
               const std::optional<const MemoryConfig>& memory_config,
               const std::optional<const ttnn::TensorMemoryLayout> applied_shard_scheme,
               bool in_place_halo,
               bool deallocate_input,
               bool reallocate_halo_output,
               const DataType dtype,
               const Layout output_layout) -> ttnn::Tensor {
                return self(
                    input_tensor,
                    batch_size,
                    input_h,
                    input_w,
                    channels,
                    kernel_size,
                    stride,
                    padding,
                    ceil_mode,
                    count_include_pad,
                    divisor_override,
                    memory_config,
                    applied_shard_scheme,
                    in_place_halo,
                    deallocate_input,
                    reallocate_halo_output,
                    dtype,
                    output_layout);
            },
            nb::arg("input_tensor"),
            nb::arg("batch_size"),
            nb::arg("input_h"),
            nb::arg("input_w"),
            nb::arg("channels"),
            nb::arg("kernel_size"),
            nb::arg("stride"),
            nb::arg("padding"),
            nb::arg("ceil_mode") = false,
            nb::arg("count_include_pad") = true,
            nb::arg("divisor_override") = nb::none(),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("applied_shard_scheme") = nb::none(),
            nb::arg("in_place_halo") = false,
            nb::arg("deallocate_input") = false,
            nb::arg("reallocate_halo_output") = true,
            nb::arg("dtype") = DataType::BFLOAT16,
            nb::arg("output_layout") = Layout::ROW_MAJOR});
}

void py_module(nb::module_& mod) {
    bind_max_pool2d_operation(mod);
    bind_avg_pool2d_operation(mod);
}

}  // namespace ttnn::operations::pool
