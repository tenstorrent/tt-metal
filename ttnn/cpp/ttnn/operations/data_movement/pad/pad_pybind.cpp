// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "pad_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"

#include "pad.hpp"

namespace ttnn::operations::data_movement::detail {
namespace py = pybind11;

void bind_pad(py::module& module) {
    auto doc =
        R"doc(

            Returns a padded tensor, with a specified value at the specified location. If the input tensor is on host, the pad will be performed on host, and if its on device it will be performed on device.
            Any rank of tensor is supported, however tensors with rank > 4 can only apply padding to the lower 3 dimensions.

            Args:
                * :attr:`input_tensor`: (ttnn.Tensor): the input tensor.
                * :attr:`padding`: (list[Tuple[int,int]]): padding to apply. Each element of padding should be a tuple of 2 integers, with the first integer specifying the number of values to add before the tensor and the second integer specifying the number of values to add after the tensor. Mutually exclusive to output_tensor_shape and input_tensor_start.
                * :attr:`value`: (Union[float,int]): value to pad with.

            Keyword Args:
                * :attr:`use_multicore`: (Optional[bool]) switch to use multicore implementation
                * :attr:`memory_config`: (Optional[ttnn.MemoryConfig]): Memory configuration for the operation. Defaults to `None`.
                * :attr:`queue_id`: (Optional[int]): command queue id. Defaults to `0`.

            Returns:
               List of ttnn.Tensor: the output tensor.

            Example:
                .. code-block:: python

                    input_tensor = ttnn.pad(pad_input, [(0, 0), (0, 0), (0, 12), (0, 12)], 0)
                    assert (ttnn.to_torch(input_tensor[:, :, 20:32, 20:32]) == 0).all()
                    assert input_tensor.shape == Shape([1, 8, 32, 32])

        )doc";

    using OperationType = decltype(ttnn::pad);
    ttnn::bind_registered_operation(
        module,
        ttnn::pad,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               ttnn::SmallVector<std::pair<uint32_t, uint32_t>> padding,
               const float value,
               const bool use_multicore,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               QueueId queue_id) { return self(queue_id, input_tensor, padding, value, use_multicore, memory_config); },
            py::arg("input_tensor"),
            py::arg("padding"),
            py::arg("value"),
            py::kw_only(),
            py::arg("use_multicore") = true,
            py::arg("memory_config") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId,
        },
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const tt::tt_metal::Array4D& output_padded_shape,
               const tt::tt_metal::Array4D& input_tensor_start,
               const float value,
               const bool use_multicore,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               QueueId queue_id) {
                return self(
                    queue_id,
                    input_tensor,
                    output_padded_shape,
                    input_tensor_start,
                    value,
                    use_multicore,
                    memory_config);
            },
            py::arg("input_tensor"),
            py::arg("output_padded_shape"),
            py::arg("input_tensor_start"),
            py::arg("value"),
            py::kw_only(),
            py::arg("use_multicore") = false,
            py::arg("memory_config") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId});
}
}  // namespace ttnn::operations::data_movement::detail
