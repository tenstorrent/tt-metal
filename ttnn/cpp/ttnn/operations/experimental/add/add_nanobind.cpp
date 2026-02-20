// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "add_nanobind.hpp"

#include <optional>
#include <utility>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/experimental/add/add.hpp"

namespace ttnn::operations::experimental::binary::detail {

void bind_add(nb::module_& mod) {
    auto operation = ttnn::operations::experimental::binary::add;
    using OperationType = decltype(operation);

    auto doc = fmt::format(
        R"doc(
        Selects elements from `true_values` or `false_values` based on a boolean `condition` and returns the tensor with the same layout as `condition`

        Args:
            a (ttnn.Tensor): A tensor to be added.
            b (ttnn.Tensor): B tensor to be added.



        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the operation. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): preallocated output tensor. Defaults to `None`.


        Note:
            Supported dtypes, layouts, and ranks:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - BFLOAT16, BFLOAT8_B, FLOAT32
                 - TILE

            bfloat8_b/bfloat4_b supports only on TILE_LAYOUT

        Example:
            >>> a = ttnn.from_torch(torch.tensor([[1, 0], [1, 0]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> b = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> output = {1}(a, b)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name());

    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const Tensor& a,
               const Tensor& b,
               const std::optional<const DataType>& output_dtype,
               const std::optional<MemoryConfig>& memory_config,
               std::optional<Tensor> output_tensor) {
                return self(a, b, output_dtype, memory_config, std::move(output_tensor));
            },
            nb::arg("a"),
            nb::arg("b"),
            nb::kw_only(),
            nb::arg("dtype").noconvert() = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none()});
}
}  // namespace ttnn::operations::experimental::binary::detail
