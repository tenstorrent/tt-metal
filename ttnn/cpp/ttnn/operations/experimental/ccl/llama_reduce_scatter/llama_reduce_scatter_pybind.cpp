// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "llama_reduce_scatter_pybind.hpp"
#include "llama_reduce_scatter.hpp"

namespace ttnn::operations::ccl::detail {
namespace py = pybind11;

void bind_llama_reduce_scatter(py::module& module) {
    auto doc =
        R"doc(llama_reduce_scatter(input_tensor: ttnn.Tensor, dims: List[int], memory_config: Optional[MemoryConfig] = std::nullopt, queue_id: int = 0) -> ttnn.Tensor

            Reduces and scatters.

            Args:
                input_tensor (ttnn.Tensor): the input tensor.
                dim (number): the reduce dimension

            Keyword Args:
                memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
                queue_id (int, optional): command queue id. Defaults to `0`.

           Returns:
               List of ttnn.Tensor: the output tensor.

            Example:

                >>> tensor = ttnn.to_device(ttnn.from_torch(torch.zeros((1, 1, 64, 32), dtype=torch.bfloat16)), device)
                >>> output = ttnn.llama_reduce_scatter(tensor, (0, 1, 3, 2)))doc";

    using OperationType = decltype(ttnn::experimental::llama_reduce_scatter);
    ttnn::bind_registered_operation(
        module,
        ttnn::experimental::llama_reduce_scatter,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               uint32_t dim,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               QueueId queue_id) { return self(queue_id, input_tensor, dim, memory_config); },
            py::arg("input_tensor").noconvert(),
            py::arg("dim"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId,
        });
}

}  // namespace ttnn::operations::ccl::detail
