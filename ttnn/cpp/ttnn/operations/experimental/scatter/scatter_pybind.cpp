// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "scatter_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cpp/pybind11/decorators.hpp"
#include "ttnn-pybind/decorators.hpp"

#include "ttnn/operations/experimental/scatter/scatter.hpp"
#include "ttnn/operations/experimental/scatter/scatter_enums.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimnetal::scatter::detail {

void bind_scatter_operation(py::module& module) {
    auto doc =
        R"doc(
            Scatters the source tensor's values along a given dimension according
            to the index tensor.

            Parameters:
                * `dim` (int): The dimension to scatter along.
                * `input_tensor` (Tensor): The input tensor to be sorted.
                * `index_tensor` (Tensor): The input tensor to be sorted.

            Keyword Arguments:
                * `opt_reduction` (ScatterReductionType, optional): TODO(jbbieniekTT): finish
                * `opt_memory_config` (MemoryConfig, optional): Specifies the memory configuration for the output tensor. Defaults to `None`.
                * `out` (tuple of Tensors, optional): Preallocated output tensors for the sorted values and indices. Defaults to `None`.

            Constraints:
                *

            Additional info:
                * For now the `stable` argument is not supported.

            Example:

            .. code-block:: python

                import ttnn

                # Create a tensor
                input_tensor = ttnn.Tensor([3, 1, 2])

                # Sort the tensor in ascending order
                sorted_tensor, indices = ttnn.experimental.sort(input_tensor)

                # Sort the tensor in descending order
                sorted_tensor_desc, indices_desc = ttnn.experimental.sort(input_tensor, descending=True)

                # Sort along a specific dimension
                input_tensor_2d = ttnn.Tensor([[3, 1, 2], [6, 5, 4]])
                sorted_tensor_dim, indices_dim = ttnn.experimental.sort(input_tensor_2d, dim=1)
        )doc";

    using OperationType = decltype(ttnn::experimental::scatter_);
    bind_registered_operation(
        module,
        ttnn::experimental::scatter_,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const int32_t& dim,
               const ttnn::Tensor& index_tensor,
               const ttnn::Tensor& src_tensor,
               const std::optional<experimental::scatter::ScatterReductionType>& opt_reduction,
               const std::optional<tt::tt_metal::MemoryConfig>& opt_out_memory_config,
               std::optional<ttnn::Tensor>& opt_output,
               QueueId queue_id) -> Tensor {
                return self(
                    queue_id,
                    input_tensor,
                    dim,
                    index_tensor,
                    src_tensor,
                    opt_reduction,
                    opt_out_memory_config,
                    opt_output);
            },
            py::arg("input").noconvert(),
            py::arg("dim"),
            py::arg("index"),
            py::arg("src"),
            py::kw_only(),
            py::arg("reduce") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("out") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId});
}

}  // namespace ttnn::operations::experimnetal::scatter::detail
