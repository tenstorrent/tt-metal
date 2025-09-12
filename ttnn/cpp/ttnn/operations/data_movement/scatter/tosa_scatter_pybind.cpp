// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tosa_scatter_pybind.hpp"

#include "ttnn-pybind/decorators.hpp"

#include "tosa_scatter.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement::detail {

void bind_tosa_scatter(py::module& module) {
    auto doc =
        R"doc(
            Scatters the source tensor's values along a given dimension according
            to the index tensor.

            Parameters:
                * `input` (Tensor): The input tensor to scatter values onto. Must be of rank 3.
                * `index` (Tensor): The tensor specifying indices where values from the source tensor must go to. Must be of rank 2.
                * `src` (Tensor): The tensor containing the source values to be scattered onto input. Must be of rank 3.

            Keyword Arguments:
                * `memory_config` (MemoryConfig, optional): Specifies the memory configuration for the output tensor. Defaults to `None`.

            Example:

            .. code-block:: python

                import ttnn
                import torch

                input_torch = torch.randn([10,20,30,20,10], dtype=torch.float32)
                index_torch = torch.randint(0, 10, [10,20,30,20,5], dtype=torch.int64)
                source_torch = torch.randn([10,20,30,20,10], dtype=input_torch.dtype)

                device = ttnn.open_device(device_id=0)
                # input tensors must be interleaved and on device
                input_ttnn = ttnn.from_torch(input_torch, dtype=ttnn.float32, device=device, layout=ttnn.Layout.TILE)
                index_ttnn = ttnn.from_torch(index_torch, dtype=ttnn.int32, device=device, layout=ttnn.Layout.TILE)
                source_ttnn = ttnn.from_torch(source_torch, dtype=ttnn.float32, device=device, layout=ttnn.Layout.TILE)

                output = ttnn.tosa_scatter(input_ttnn, index_ttnn, source_ttnn)
        )doc";

    using OperationType = decltype(ttnn::tosa_scatter);
    bind_registered_operation(
        module,
        ttnn::tosa_scatter,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::Tensor& index_tensor,
               const ttnn::Tensor& source_tensor,
               const std::optional<tt::tt_metal::MemoryConfig>& opt_out_memory_config) -> Tensor {
                return self(input_tensor, index_tensor, source_tensor, opt_out_memory_config);
            },
            py::arg("input").noconvert(),
            py::arg("index").noconvert(),
            py::arg("src").noconvert(),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt});
}

}  // namespace ttnn::operations::data_movement::detail
