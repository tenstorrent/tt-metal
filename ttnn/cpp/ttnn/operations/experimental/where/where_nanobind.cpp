// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/where/where_nanobind.hpp"

#include <optional>
#include <utility>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/experimental/where/where.hpp"

namespace ttnn::operations::experimental::ternary::detail {

void bind_where(nb::module_& mod) {
    auto operation = ttnn::operations::experimental::ternary::where;
    using OperationType = decltype(ttnn::operations::experimental::ternary::where);

    auto doc = fmt::format(
        R"doc(
        Selects elements from `true_values` or `false_values` based on a boolean `condition` and returns the tensor with the same layout as `condition`

        Args:
            condition (ttnn.Tensor): A boolean array where each element determines which value to choose.
            true_values (ttnn.Tensor or Number): Values to select where `condition` is True.
            false_values (ttnn.Tensor or Number): Values to select where `condition` is False.


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
            >>> condition = ttnn.from_torch(torch.tensor([[1, 0], [1, 0]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> true_values = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> false_values = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> output = {1}(condition, true_values, false_values)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name());

    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const Tensor& condition,
               const Tensor& true_value,
               const Tensor& false_value,
               std::optional<const DataType> output_dtype,
               const std::optional<MemoryConfig>& memory_config,
               std::optional<Tensor> output_tensor) {
                return self(condition, true_value, false_value, output_dtype, memory_config, std::move(output_tensor));
            },
            nb::arg("condition"),
            nb::arg("true_value"),
            nb::arg("false_value"),
            nb::kw_only(),
            nb::arg("dtype").noconvert() = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none()},
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const Tensor& condition,
               const float true_value,
               const Tensor& false_value,
               std::optional<const DataType> output_dtype,
               const std::optional<MemoryConfig>& memory_config,
               std::optional<Tensor> output_tensor) {
                return self(condition, true_value, false_value, output_dtype, memory_config, std::move(output_tensor));
            },
            nb::arg("condition"),
            nb::arg("true_value"),
            nb::arg("false_value"),
            nb::kw_only(),
            nb::arg("dtype").noconvert() = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none()},
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const Tensor& condition,
               const Tensor& true_value,
               const float false_value,
               std::optional<const DataType> output_dtype,
               const std::optional<MemoryConfig>& memory_config,
               std::optional<Tensor> output_tensor) {
                return self(condition, true_value, false_value, output_dtype, memory_config, std::move(output_tensor));
            },
            nb::arg("condition"),
            nb::arg("true_value"),
            nb::arg("false_value"),
            nb::kw_only(),
            nb::arg("dtype").noconvert() = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none()},
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const Tensor& condition,
               const float true_value,
               const float false_value,
               std::optional<const DataType> output_dtype,
               const std::optional<MemoryConfig>& memory_config,
               std::optional<Tensor> output_tensor) {
                return self(condition, true_value, false_value, output_dtype, memory_config, std::move(output_tensor));
            },
            nb::arg("condition"),
            nb::arg("true_value"),
            nb::arg("false_value"),
            nb::kw_only(),
            nb::arg("dtype").noconvert() = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none()});
}

}  // namespace ttnn::operations::experimental::ternary::detail
