// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "scatter_nanobind.hpp"

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"

#include "scatter.hpp"
#include "scatter_enums.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement::detail {

void bind_scatter(nb::module_& mod) {
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

                output = ttnn.scatter(input_ttnn, dim, index_ttnn, source_ttnn)
        )doc";

    using OperationType = decltype(ttnn::scatter);
    bind_registered_operation(
        mod,
        ttnn::scatter,
        doc,
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const int32_t& dim,
               const ttnn::Tensor& index_tensor,
               const ttnn::Tensor& source_tensor,
               const std::optional<tt::tt_metal::MemoryConfig>& opt_out_memory_config,
               const std::optional<scatter::ScatterReductionType>& opt_reduction) -> Tensor {
                return self(input_tensor, dim, index_tensor, source_tensor, opt_out_memory_config, opt_reduction);
            },
            nb::arg("input").noconvert(),
            nb::arg("dim"),
            nb::arg("index").noconvert(),
            nb::arg("src").noconvert(),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("reduce") = nb::none()});
}

}  // namespace ttnn::operations::data_movement::detail
