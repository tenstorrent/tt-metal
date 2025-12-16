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
    const auto* doc =
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
            sub_core_grids (ttnn.CoreRangeSet, optional): specifies which cores scatter should run on. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor with scattered values.

        Note:
            * Input tensors must be interleaved and on device.
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
               const std::optional<std::string>& opt_reduction,
               const std::optional<CoreRangeSet>& sub_core_grid) -> Tensor {
                return self(
                    input_tensor,
                    dim,
                    index_tensor,
                    source_tensor,
                    opt_out_memory_config,
                    opt_reduction,
                    sub_core_grid);
            },
            py::arg("input").noconvert(),
            py::arg("dim"),
            py::arg("index").noconvert(),
            py::arg("src").noconvert(),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("reduce") = std::nullopt,
            py::arg("sub_core_grids") = std::nullopt});
}

void bind_scatter_add(py::module& module) {
    const auto* doc =
        R"doc(
        Scatters the source tensor's values along a given dimension according to the index tensor, adding source values associated with according repeated indices.

        Args:
            input (ttnn.Tensor): the input tensor to scatter values onto.
            dim (int): the dimension to scatter along.
            index (ttnn.Tensor): the tensor specifying indices where values from the source tensor must go to.
            src (ttnn.Tensor): the tensor containing the source values to be scattered onto input.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the output tensor. Defaults to `None`.
            sub_core_grids (ttnn.CoreRangeSet, optional): specifies which cores scatter should run on. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor with scattered values.

        Note:
            * Input tensors must be interleaved and on device.
        )doc";

    using OperationType = decltype(ttnn::scatter_add);
    bind_registered_operation(
        module,
        ttnn::scatter_add,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const int32_t& dim,
               const ttnn::Tensor& index_tensor,
               const ttnn::Tensor& source_tensor,
               const std::optional<tt::tt_metal::MemoryConfig>& opt_out_memory_config,
               const std::optional<CoreRangeSet>& sub_core_grid) -> Tensor {
                return self(input_tensor, dim, index_tensor, source_tensor, opt_out_memory_config, sub_core_grid);
            },
            py::arg("input").noconvert(),
            py::arg("dim"),
            py::arg("index").noconvert(),
            py::arg("src").noconvert(),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("sub_core_grids") = std::nullopt});
}

}  // namespace ttnn::operations::data_movement::detail
