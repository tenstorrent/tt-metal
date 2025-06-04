// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "scatter_pybind.hpp"

#include "ttnn-pybind/decorators.hpp"

#include "scatter.hpp"
#include "scatter_enums.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::scatter::detail {

void bind_scatter_operation(py::module& module) {
    auto doc =
        R"doc(
            Scatters the source tensor's values along a given dimension according
            to the index tensor.

            Parameters:
                * `input` (Tensor): The input tensor to scatter values onto.
                * `dim` (int): The dimension to scatter along.
                * `index` (Tensor): The tensor specifying indices where values from the source tensor must go to.
                * `src` (Tensor): The tensor containing the source values to be scattered onto input.

            Keyword Arguments:
                * `reduce` (ScatterReductionType, optional): currently not supported - this is the option to reduce numbers going to the same destination in output with a function like `amax`, `amin`, `sum`, etc.
                * `memory_config` (MemoryConfig, optional): Specifies the memory configuration for the output tensor. Defaults to `None`.
                * `out` (Tensor, optional): Preallocated output tensor where scatter result should go to (should be the same shape as the input tensor). Defaults to `None`.

            Additional info:
                * Up until this time, no reductions have been implemented.

            Example:

            .. code-block:: python

                import ttnn
                import torch

                input_torch = torch.randn([10,20,30,20,10], dtype=torch.float32)
                index_torch = torch.randint(0, 10, [10,20,30,20,5], dtype=torch.int64)
                source_torch = torch.randn([10,20,30,20,10], dtype=input_torch.dtype)

                device = ttnn.open_device(device_id=0)
                # input tensors must be interleaved, tiled and on device
                input_ttnn = ttnn.from_torch(input_torch, dtype=ttnn.float32, device=device, layout=ttnn.Layout.TILE)
                index_ttnn = ttnn.from_torch(index_torch, dtype=ttnn.int32, device=device, layout=ttnn.Layout.TILE)
                source_ttnn = ttnn.from_torch(source_torch, dtype=ttnn.float32, device=device, layout=ttnn.Layout.TILE)
                dim = -1

                output = ttnn.experimental.scatter_(input_ttnn, dim, index_ttnn, source_ttnn)

                output_preallocated = ttnn.zeros_like(input_ttnn)
                another_output = ttnn.experimental.scatter_(input_ttnn, dim, index_ttnn, source_ttnn, out=output_preallocated)
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
               const ttnn::Tensor& source_tensor,
               const std::optional<tt::tt_metal::MemoryConfig>& opt_out_memory_config,
               const std::optional<experimental::scatter::ScatterReductionType>& opt_reduction,
               std::optional<ttnn::Tensor>& opt_output,
               const QueueId& queue_id = DefaultQueueId) -> Tensor {
                return self(
                    queue_id,
                    input_tensor,
                    dim,
                    index_tensor,
                    source_tensor,
                    opt_out_memory_config,
                    opt_reduction,
                    opt_output);
            },
            py::arg("input").noconvert(),
            py::arg("dim"),
            py::arg("index").noconvert(),
            py::arg("src").noconvert(),
            py::kw_only(),
            py::arg("reduce") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("out") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId});
}

}  // namespace ttnn::operations::experimental::scatter::detail
