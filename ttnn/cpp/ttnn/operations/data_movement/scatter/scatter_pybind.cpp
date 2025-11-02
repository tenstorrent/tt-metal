// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "scatter_pybind.hpp"

#include "ttnn-pybind/decorators.hpp"

#include "scatter.hpp"
#include "scatter_enums.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement::detail {

void bind_scatter(py::module& module) {
    auto doc =
        R"doc(
        Scatters the source tensor's values along a given dimension according to the index tensor.

        Args:
            input (ttnn.Tensor): the input tensor to scatter values onto.
            dim (int): the dimension to scatter along.
            index (ttnn.Tensor): the tensor specifying indices where values from the source tensor must go to.
            src (ttnn.Tensor): the tensor containing the source values to be scattered onto input.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the output tensor. Defaults to `None`.
            reduce (ttnn.ScatterReductionType, optional): reduction operation to apply when multiple values are scattered to the same location (e.g., amax, amin, sum). Currently not supported. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor with scattered values.

        Note:
            * Input tensors must be interleaved, tiled and on device.
            * No reduction operations have been implemented yet.

        Example:
            >>> input_torch = torch.randn([10, 20, 30, 20, 10], dtype=torch.float32)
            >>> index_torch = torch.randint(0, 10, [10, 20, 30, 20, 5], dtype=torch.int64)
            >>> source_torch = torch.randn([10, 20, 30, 20, 10], dtype=input_torch.dtype)
            >>> device = ttnn.open_device(device_id=0)
            >>> input_ttnn = ttnn.from_torch(input_torch, dtype=ttnn.float32, device=device, layout=ttnn.TILE_LAYOUT)
            >>> index_ttnn = ttnn.from_torch(index_torch, dtype=ttnn.int32, device=device, layout=ttnn.TILE_LAYOUT)
            >>> source_ttnn = ttnn.from_torch(source_torch, dtype=ttnn.float32, device=device, layout=ttnn.TILE_LAYOUT)
            >>> dim = -1
            >>> output = ttnn.scatter(input_ttnn, dim, index_ttnn, source_ttnn)
        )doc";

    using OperationType = decltype(ttnn::scatter);
    bind_registered_operation(
        module,
        ttnn::scatter,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const int32_t& dim,
               const ttnn::Tensor& index_tensor,
               const ttnn::Tensor& source_tensor,
               const std::optional<tt::tt_metal::MemoryConfig>& opt_out_memory_config,
               const std::optional<scatter::ScatterReductionType>& opt_reduction) -> Tensor {
                return self(input_tensor, dim, index_tensor, source_tensor, opt_out_memory_config, opt_reduction);
            },
            py::arg("input").noconvert(),
            py::arg("dim"),
            py::arg("index").noconvert(),
            py::arg("src").noconvert(),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("reduce") = std::nullopt});
}

}  // namespace ttnn::operations::data_movement::detail
