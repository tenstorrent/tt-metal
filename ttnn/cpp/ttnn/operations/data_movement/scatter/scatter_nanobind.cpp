// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "scatter_nanobind.hpp"

#include <cstdint>
#include <optional>
#include <string>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include "ttnn-nanobind/decorators.hpp"

#include "scatter.hpp"
#include "scatter_enums.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement::detail {

void bind_scatter(nb::module_& mod) {
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
            nb::arg("input").noconvert(),
            nb::arg("dim"),
            nb::arg("index").noconvert(),
            nb::arg("src").noconvert(),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("reduce") = nb::none(),
            nb::arg("sub_core_grids") = nb::none()});
}

void bind_scatter_add(nb::module_& mod) {
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
        mod,
        ttnn::scatter_add,
        doc,
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const int32_t& dim,
               const ttnn::Tensor& index_tensor,
               const ttnn::Tensor& source_tensor,
               const std::optional<tt::tt_metal::MemoryConfig>& opt_out_memory_config,
               const std::optional<CoreRangeSet>& sub_core_grid) -> Tensor {
                return self(input_tensor, dim, index_tensor, source_tensor, opt_out_memory_config, sub_core_grid);
            },
            nb::arg("input").noconvert(),
            nb::arg("dim"),
            nb::arg("index").noconvert(),
            nb::arg("src").noconvert(),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("sub_core_grids") = nb::none()});
}

}  // namespace ttnn::operations::data_movement::detail
