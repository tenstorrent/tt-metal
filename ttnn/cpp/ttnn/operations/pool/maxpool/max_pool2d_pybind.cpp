// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/pool/maxpool/max_pool2d_pybind.hpp"
#include "ttnn/operations/pool/maxpool/max_pool2d.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
#include "ttnn/types.hpp"


namespace ttnn::operations::pool {

void bind_max_pool2d_operation(py::module& module) {
    bind_registered_operation(
        module,
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

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): the memory configuration for the output tensor. Defaults to `None`.
            applied_shard_scheme (ttnn.TensorMemoryLayout, optional): the sharding scheme to apply to a non-pre-sharded input tensor. Defaults to `None`, which should be used with pre-sharded input tensors.
            queue_id (int, optional): the queue id to use for the operation. Defaults to `0`.

        Returns:
            ttnn.Tensor: the max pool convolved output tensor.

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
                                memory_config=None,
                                applied_shard_scheme=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                            )

        )doc",
        ttnn::pybind_overload_t{
            [](const decltype(ttnn::max_pool2d)& self, const ttnn::Tensor& input_tensor,
                uint32_t batch_size,
                uint32_t input_h,
                uint32_t input_w,
                uint32_t channels,
                std::array<uint32_t, 2> kernel_size,
                std::array<uint32_t, 2> stride,
                std::array<uint32_t, 2> padding,
                std::array<uint32_t, 2> dilation,
                const std::optional<const MemoryConfig> memory_config,
                const std::optional<const ttnn::TensorMemoryLayout> applied_shard_scheme,
                const uint8_t& queue_id)
                -> ttnn::Tensor { return self(queue_id,
                                              input_tensor,
                                              batch_size,
                                              input_h,
                                              input_w,
                                              channels,
                                              kernel_size,
                                              stride,
                                              padding,
                                              dilation,
                                              memory_config,
                                              applied_shard_scheme); },
                py::arg("input_tensor"),
                py::arg("batch_size"),
                py::arg("input_h"),
                py::arg("input_w"),
                py::arg("channels"),
                py::arg("kernel_size"),
                py::arg("stride"),
                py::arg("padding"),
                py::arg("dilation"),
                py::kw_only(),
                py::arg("memory_config") = std::nullopt,
                py::arg("applied_shard_scheme") = std::nullopt,
                py::arg("queue_id") = 0});
}

void py_module(py::module& module) {
    bind_max_pool2d_operation(module);
}

}  // namespace ttnn::operations::pool
