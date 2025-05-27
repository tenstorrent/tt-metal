// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "padded_slice_pybind.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "cpp/ttnn-pybind/decorators.hpp"

namespace ttnn::operations::experimental::padded_slice {
namespace py = pybind11;

void bind_padded_slice(py::module& module) {
    auto doc =
        R"doc(
            Returns a padded_sliced tensor. If the input tensor is on host, the padded_slice will be performed on host, and if its on device it will be performed on device.

            Args:
                input_tensor: Input Tensor.
                padded_slice_start: Start indices of input tensor. Values along each dim must be < input_tensor_shape[i].
                padded_slice_end: End indices of input tensor. Values along each dim must be < input_tensor_shape[i].
                padded_slice_step: (Optional[List[int[tensor rank]]) Step size for each dim. Default is None, which works out be 1 for each dimension.
                memory_config: Memory Config of the output tensor. This must be either height or block sharded.

            Keyword Args:
                queue_id (uint8, optional):command queue id

            Returns:
                ttnn.Tensor: the output tensor.

            Example:
                >>> tensor = ttnn.padded_slice(ttnn.from_torch(torch.zeros((1, 1, 64, 32), dtype=torch.bfloat16), device=device), [0, 0, 0, 0], [1, 1, 64, 16], [1, 1, 2, 1])
                >>> print(tensor.shape)
                [1, 1, 32, 16]
                >>> input = ttnn.from_torch(torch.zeros((1, 1, 64, 32), dtype=torch.bfloat16), device=device)
                >>> output = ttnn.padded_slice(input, [0, 0, 0, 0], [1, 1, 32, 32])
                >>> print(output.shape)
                [1, 1, 32, 32]
                )doc";

    // TODO: implementing the array version and overloading the pybind with all the possible array sizes is better than
    // a vector with a fixed size default value
    using OperationType = decltype(ttnn::experimental::padded_slice);
    ttnn::bind_registered_operation(
        module,
        ttnn::experimental::padded_slice,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::SmallVector<int>& padded_slice_start,
               const ttnn::SmallVector<int>& padded_slice_end,
               const std::optional<ttnn::SmallVector<int>>& step,
               const MemoryConfig& memory_config,
               const std::optional<Tensor>& optional_output_tensor,
               const std::optional<float>& pad_value,
               QueueId queue_id) {
                const auto step_value = step.value_or(ttnn::SmallVector<int>(padded_slice_end.size(), 1));
                return self(
                    queue_id,
                    input_tensor,
                    padded_slice_start,
                    padded_slice_end,
                    step_value,
                    memory_config,
                    optional_output_tensor);
            },
            py::arg("input_tensor"),
            py::arg("padded_slice_start"),
            py::arg("padded_slice_end"),
            py::arg("padded_slice_step") = std::nullopt,  // should consider a better default value
            py::kw_only(),
            py::arg("memory_config"),
            py::arg("output_tensor") = std::nullopt,
            py::arg("pad_value") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId,
        });
}
}  // namespace ttnn::operations::experimental::padded_slice
