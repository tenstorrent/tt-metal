// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "slice_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"

#include "slice.hpp"

namespace ttnn::operations::data_movement::detail {
namespace py = pybind11;

void bind_slice(py::module& module) {
    auto doc =
        R"doc(
            Returns a sliced tensor. If the input tensor is on host, the slice will be performed on host, and if its on device it will be performed on device.

            Args:
                input_tensor: Input Tensor.
                slice_start: Start indices of input tensor. Values along each dim must be < input_tensor_shape[i].
                slice_end: End indices of input tensor. Values along each dim must be < input_tensor_shape[i].
                slice_step: (Optional[List[int[tensor rank]]) Step size for each dim. Default is None, which works out be 1 for each dimension.

            Keyword Args:
                memory_config Memory Config of the output tensor
                queue_id (uint8, optional) command queue id
                pad_value: Optional value to fill padding for tiled tensors. Padding values are unmodified (and undefined) by default

            Returns:
                ttnn.Tensor: the output tensor.

            Example:
                >>> tensor = ttnn.slice(ttnn.from_torch(torch.zeros((1, 1, 64, 32), dtype=torch.bfloat16), device=device), [0, 0, 0, 0], [1, 1, 64, 16], [1, 1, 2, 1])
                >>> print(tensor.shape)
                [1, 1, 32, 16]
                >>> input = ttnn.from_torch(torch.zeros((1, 1, 64, 32), dtype=torch.bfloat16), device=device)
                >>> output = ttnn.slice(input, [0, 0, 0, 0], [1, 1, 32, 32])
                >>> print(output.shape)
                [1, 1, 32, 32]
                )doc";

    // TODO: implementing the array version and overloading the pybind with all the possible array sizes is better than
    // a vector with a fixed size default value
    using OperationType = decltype(ttnn::slice);
    ttnn::bind_registered_operation(
        module,
        ttnn::slice,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::Tensor& slice_start,
               const ttnn::Tensor& slice_end,
               const std::optional<ttnn::SmallVector<uint32_t>>& step,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<Tensor>& optional_output_tensor,
               const std::optional<float>& pad_value,
               QueueId queue_id) {
                return self(
                    queue_id,
                    input_tensor,
                    slice_start,
                    slice_end,
                    step,
                    memory_config,
                    optional_output_tensor,
                    pad_value);
            },
            py::arg("input_tensor"),
            py::arg("starts"),
            py::arg("ends"),
            py::arg("slice_step") = std::nullopt,  // should consider a better default value
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("output_tensor") = std::nullopt,
            py::arg("pad_value") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId,
        },
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const std::array<uint32_t, 4>& begins,
               const std::array<uint32_t, 4>& ends,
               const std::array<uint32_t, 4>& step,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<Tensor>& optional_output_tensor,
               const std::optional<float>& pad_value,
               QueueId queue_id) {
                return self(
                    queue_id, input_tensor, begins, ends, step, memory_config, optional_output_tensor, pad_value);
            },
            py::arg("input_tensor"),
            py::arg("starts"),
            py::arg("ends"),
            py::arg("steps"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("output_tensor") = std::nullopt,
            py::arg("pad_value") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId,
        },
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::SmallVector<int>& slice_start,
               const ttnn::SmallVector<int>& slice_end,
               const std::optional<ttnn::SmallVector<int>>& step,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<Tensor>& optional_output_tensor,
               QueueId queue_id) {
                const auto step_value = step.value_or(ttnn::SmallVector<int>(slice_end.size(), 1));
                return self(
                    queue_id, input_tensor, slice_start, slice_end, step_value, memory_config, optional_output_tensor);
            },
            py::arg("input_tensor"),
            py::arg("slice_start"),
            py::arg("slice_end"),
            py::arg("slice_step") = std::nullopt,  // should consider a better default value
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("output_tensor") = std::nullopt,
            py::arg("queue_id") = 0,
        }

    );
}
}  // namespace ttnn::operations::data_movement::detail
