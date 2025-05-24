// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "permute_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"

#include "permute.hpp"

namespace ttnn::operations::data_movement::detail {
namespace py = pybind11;

void bind_permute(py::module& module) {
    auto doc =
        R"doc(permute(input_tensor: ttnn.Tensor, dims: List[int], memory_config: Optional[MemoryConfig] = std::nullopt, queue_id: int = 0) -> ttnn.Tensor

            Permutes the dimensions of the input tensor according to the specified permutation.

            Args:
                input_tensor (ttnn.Tensor): the input tensor.
                dim (number): tthe permutation of the dimensions of the input tensor.

            Keyword Args:
                memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
                queue_id (int, optional): command queue id. Defaults to `0`.
                pad_value (float, optional): padding value for when tiles are broken in a transpose. Defaults to `0.0`. If set to None, it will be random garbage values.

           Returns:
               List of ttnn.Tensor: the output tensor.

            Example:

                >>> tensor = ttnn.to_device(ttnn.from_torch(torch.zeros((1, 1, 64, 32), dtype=torch.bfloat16)), device)
                >>> output = ttnn.permute(tensor, (0, 1, 3, 2))
                >>> print(output.shape)
                [1, 1, 32, 64])doc";

    using OperationType = decltype(ttnn::permute);
    ttnn::bind_registered_operation(
        module,
        ttnn::permute,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::SmallVector<int64_t>& dims,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               QueueId queue_id,
               const std::optional<float>& pad_value) {
                return self(queue_id, input_tensor, dims, memory_config, pad_value);
            },
            py::arg("input_tensor").noconvert(),
            py::arg("dims"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId,
            py::arg("pad_value") = 0.0f,
        });
}

}  // namespace ttnn::operations::data_movement::detail
