// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/adaptive_pool/adaptive_pools_pybind.hpp"
#include "ttnn/operations/experimental/adaptive_pool/adaptive_pools.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <array>
#include <variant>
#include <cstdint>

#include "ttnn-pybind/decorators.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::adaptive_pool {
namespace py = pybind11;

void bind_adaptive_avg_pool2d_operation(py::module& module) {
    bind_registered_operation(
        module,
        ttnn::adaptive_avg_pool2d,
        R"doc(
        Applies experimental adaptive average pooling to the input tensor. Unlike regular pooling, adaptive pooling
        automatically calculates the kernel size and stride to produce the desired output size.
        The input tensor is expected to be in [NHW, C] format and should be on the device.

        Args:
            input_tensor (ttnn.Tensor): the tensor to be pooled.
            batch_size (int): the number of batches (N in a [N, C, H, W] shaped tensor).
            input_h (int): the height of the input tensor (H in a [N, C, H, W] shaped tensor).
            input_w (int): the width of the input tensor (W in a [N, C, H, W] shaped tensor).
            channels (int): the number of channels (C in a [N, C, H, W] shaped tensor).
            output_size (List of [int]): the target (h, w) size of the output tensor.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): the memory configuration for the output tensor. Defaults to `None`.
            applied_shard_scheme (ttnn.TensorMemoryLayout, optional): the sharding scheme to apply to a non-pre-sharded input tensor. Defaults to `None`.
            in_place_halo (bool, optional): whether to perform the halo operation in place. Defaults to `False`.
            deallocate_input (bool, optional): whether to deallocate the input tensor. Defaults to `False`.
            reallocate_output (bool, optional): whether to reallocate the output tensor. Defaults to `True`.
            queue_id (int, optional): the queue id to use for the operation. Defaults to `0`.

        Returns:
            ttnn.Tensor: the experimental adaptive average pooled output tensor.

        Example:
            >>> import ttnn
            >>> import torch
            >>> device = ttnn.open_device(device_id=0, l1_small_size=8192)
            >>> nchw_shape = (1, 256, 64, 64)
            >>> in_N, in_C, in_H, in_W = nchw_shape
            >>> input_shape = (1, 1, in_N * in_H * in_W, in_C)
            >>> input = torch.randn(nchw_shape, dtype=torch.bfloat16)
            >>> input_perm = torch.permute(input, (0, 2, 3, 1)) # this op expects a [N, H, W, C] format
            >>> input_reshape = input_perm.reshape(input_shape) # this op expects [1, 1, NHW, C]
            >>> tt_input = ttnn.from_torch(input_reshape, device=device)
            >>> tt_output = ttnn.adaptive_avg_pool2d(
                            input_tensor=tt_input,
                            batch_size=in_N,
                            input_h=in_H,
                            input_w=in_W,
                            channels=in_C,
                            output_size=[1, 1],  # Global adaptive pooling
                            memory_config=None,
                            applied_shard_scheme=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                            in_place_halo=False,
                        )
        )doc",
        ttnn::pybind_overload_t{
            [](const decltype(ttnn::adaptive_avg_pool2d)& self,
               const ttnn::Tensor& input_tensor,
               uint32_t batch_size,
               uint32_t input_h,
               uint32_t input_w,
               uint32_t channels,
               std::array<uint32_t, 2> output_size,
               const std::optional<const MemoryConfig>& memory_config,
               const std::optional<const ttnn::TensorMemoryLayout> applied_shard_scheme,
               bool in_place_halo,
               bool deallocate_input,
               bool reallocate_output,
               QueueId queue_id) -> ttnn::Tensor {
                return self(
                    queue_id,
                    input_tensor,
                    batch_size,
                    input_h,
                    input_w,
                    channels,
                    output_size,
                    memory_config,
                    applied_shard_scheme,
                    in_place_halo,
                    deallocate_input,
                    reallocate_output);
            },
            py::arg("input_tensor"),
            py::arg("batch_size"),
            py::arg("input_h"),
            py::arg("input_w"),
            py::arg("channels"),
            py::arg("output_size"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("applied_shard_scheme") = std::nullopt,
            py::arg("in_place_halo") = false,
            py::arg("deallocate_input") = false,
            py::arg("reallocate_output") = true,
            py::arg("queue_id") = DefaultQueueId});
}

void bind_adaptive_max_pool2d_operation(py::module& module) {
    bind_registered_operation(
        module,
        ttnn::adaptive_max_pool2d,
        R"doc(
        Applies experimental adaptive max pooling to the input tensor. Unlike regular pooling, adaptive pooling
        automatically calculates the kernel size and stride to produce the desired output size.
        The input tensor is expected to be in [NHW, C] format and should be on the device.

        Args:
            input_tensor (ttnn.Tensor): the tensor to be pooled.
            batch_size (int): the number of batches (N in a [N, C, H, W] shaped tensor).
            input_h (int): the height of the input tensor (H in a [N, C, H, W] shaped tensor).
            input_w (int): the width of the input tensor (W in a [N, C, H, W] shaped tensor).
            channels (int): the number of channels (C in a [N, C, H, W] shaped tensor).
            output_size (List of [int]): the target (h, w) size of the output tensor.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): the memory configuration for the output tensor. Defaults to `None`.
            applied_shard_scheme (ttnn.TensorMemoryLayout, optional): the sharding scheme to apply to a non-pre-sharded input tensor. Defaults to `None`.
            in_place_halo (bool, optional): whether to perform the halo operation in place. Defaults to `False`.
            deallocate_input (bool, optional): whether to deallocate the input tensor. Defaults to `False`.
            reallocate_output (bool, optional): whether to reallocate the output tensor. Defaults to `True`.
            queue_id (int, optional): the queue id to use for the operation. Defaults to `0`.

        Returns:
            ttnn.Tensor: the experimental adaptive max pooled output tensor.

        Example:
            >>> import ttnn
            >>> import torch
            >>> device = ttnn.open_device(device_id=0, l1_small_size=8192)
            >>> nchw_shape = (1, 256, 64, 64)
            >>> in_N, in_C, in_H, in_W = nchw_shape
            >>> input_shape = (1, 1, in_N * in_H * in_W, in_C)
            >>> input = torch.randn(nchw_shape, dtype=torch.bfloat16)
            >>> input_perm = torch.permute(input, (0, 2, 3, 1)) # this op expects a [N, H, W, C] format
            >>> input_reshape = input_perm.reshape(input_shape) # this op expects [1, 1, NHW, C]
            >>> tt_input = ttnn.from_torch(input_reshape, device=device)
            >>> tt_output = ttnn.adaptive_max_pool2d(
                            input_tensor=tt_input,
                            batch_size=in_N,
                            input_h=in_H,
                            input_w=in_W,
                            channels=in_C,
                            output_size=[7, 7],  # Classifier head size
                            memory_config=None,
                            applied_shard_scheme=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                            in_place_halo=False,
                        )
        )doc",
        ttnn::pybind_overload_t{
            [](const decltype(ttnn::adaptive_max_pool2d)& self,
               const ttnn::Tensor& input_tensor,
               uint32_t batch_size,
               uint32_t input_h,
               uint32_t input_w,
               uint32_t channels,
               std::array<uint32_t, 2> output_size,
               const std::optional<const MemoryConfig>& memory_config,
               const std::optional<const ttnn::TensorMemoryLayout> applied_shard_scheme,
               bool in_place_halo,
               bool deallocate_input,
               bool reallocate_output,
               QueueId queue_id) -> ttnn::Tensor {
                return self(
                    queue_id,
                    input_tensor,
                    batch_size,
                    input_h,
                    input_w,
                    channels,
                    output_size,
                    memory_config,
                    applied_shard_scheme,
                    in_place_halo,
                    deallocate_input,
                    reallocate_output);
            },
            py::arg("input_tensor"),
            py::arg("batch_size"),
            py::arg("input_h"),
            py::arg("input_w"),
            py::arg("channels"),
            py::arg("output_size"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("applied_shard_scheme") = std::nullopt,
            py::arg("in_place_halo") = false,
            py::arg("deallocate_input") = false,
            py::arg("reallocate_output") = true,
            py::arg("queue_id") = DefaultQueueId});
}

}  // namespace ttnn::operations::experimental::adaptive_pool
