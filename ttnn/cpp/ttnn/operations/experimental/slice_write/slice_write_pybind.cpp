// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "slice_write_pybind.hpp"

#include <array>
#include <cstdint>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
#include "slice_write.hpp"

namespace ttnn::operations::experimental::slice_write {
namespace py = pybind11;

void bind_slice_write(py::module& module) {
    auto doc =
        R"doc(
            Writes the input tensor to a slice of the output tensor.

            Constraints:
                Input & Output must have rank == 4.
                DType must be bfloat16.
                Supports only Row Major Tensors.
                Output Tensor must be interleaved.
                Input Tensor can be interleaved, height sharded or block sharded.
                Steps must be all ones.
                Slicing along the last dimension is not supported.

            Args:
                input_tensor: Input Tensor.
                output_tensor: Output Tensor.
                slice_start: Start indices of input tensor. Values along each dim must be < input_tensor_shape[i].
                slice_end: End indices of input tensor. Values along each dim must be < input_tensor_shape[i].
                slice_step: (Optional[List[int[tensor rank]]) Step size for each dim. Default is None, which works out be 1 for each dimension.

            Keyword Args:
                memory_config Memory Config of the output tensor
                queue_id (uint8, optional) command queue id

            Returns:
                ttnn.Tensor: the output tensor after writing the input tensor to it.

            Example:
                >>> ttnn.slice_write(ttnn_input_tensor, ttnn_output_tensor, output_start_indices, output_end_indices, strides)
                )doc";

    // TODO: implementing the array version and overloading the pybind with all the possible array sizes is better than
    // a vector with a fixed size default value
    using OperationType = decltype(ttnn::experimental::slice_write);
    ttnn::bind_registered_operation(
        module,
        ttnn::experimental::slice_write,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::Tensor& output_tensor,
               const std::array<uint32_t, 4>& start,
               const std::array<uint32_t, 4>& end,
               const std::array<uint32_t, 4>& step,
               QueueId queue_id) { return self(queue_id, input_tensor, output_tensor, start, end, step); },
            py::arg("input_tensor"),
            py::arg("output_tensor"),
            py::arg("start"),
            py::arg("end"),
            py::arg("step"),
            py::kw_only(),
            py::arg("queue_id") = DefaultQueueId,
        });
}
}  // namespace ttnn::operations::experimental::slice_write
