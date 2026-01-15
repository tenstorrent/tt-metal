// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ternary_nanobind.hpp"

#include <string>
#include <optional>
#include <variant>

#include <fmt/format.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/eltwise/ternary/ternary_composite.hpp"
#include "ttnn/operations/eltwise/ternary/ternary.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::ternary {

namespace {

template <typename ternary_operation_t>
void bind_ternary_composite_float(
    nb::module_& mod,
    const ternary_operation_t& operation,
    const std::string& description,
    const std::string& supported_dtype = "BFLOAT16") {
    auto doc = fmt::format(
        R"doc(
        {2}

        Args:
            input_tensor_a (ttnn.Tensor): the input tensor.
            input_tensor_b (ttnn.Tensor): the input tensor.
            input_tensor_c (ttnn.Tensor or Number): the input tensor.


        Keyword Args:
            value (float, optional): scalar value to be multiplied.
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the operation. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

        Note:
            Supported dtypes, layouts, and ranks:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
                 - Ranks
               * - {3}
                 - TILE
                 - 2, 3, 4

            bfloat8_b/bfloat4_b supports only on TILE_LAYOUT
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        description,
        supported_dtype);

    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const ternary_operation_t& self,
               const Tensor& input_tensor_a,
               const Tensor& input_tensor_b,
               const Tensor& input_tensor_c,
               float value,
               const std::optional<MemoryConfig>& memory_config) {
                return self(input_tensor_a, input_tensor_b, input_tensor_c, value, memory_config);
            },
            nb::arg("input_tensor_a"),
            nb::arg("input_tensor_b"),
            nb::arg("input_tensor_c"),
            nb::kw_only(),
            nb::arg("value"),
            nb::arg("memory_config") = nb::none()});
}

template <typename ternary_operation_t>
void bind_ternary_where(nb::module_& mod, const ternary_operation_t& operation, const std::string& description) {
    auto doc = fmt::format(
        R"doc(
            {2}

        Args:
            condition (ttnn.Tensor): the condition tensor must contain only 0's or 1's.
            true_value (ttnn.Tensor or Number): The value selected if the corresponding element in condition is 1.
            false_value (ttnn.Tensor or Number): The value selected if the corresponding element in condition is 0.


        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the operation. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): preallocated output tensor. Defaults to `None`.
            sub_core_grids (ttnn.CoreRangeSet, optional): sub core grids for the operation. Defaults to `None`.


        Note:
            Supported dtypes, layouts, and ranks:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
                 - Ranks
               * - BFLOAT16, BFLOAT8_B, FLOAT32, INT32
                 - TILE
                 - 1, 2, 3, 4, 5

            bfloat8_b/bfloat4_b supports only on TILE_LAYOUT
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        description);

    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const ternary_operation_t& self,
               const Tensor& predicate,
               const TensorScalarVariant& true_value,
               const TensorScalarVariant& false_value,
               const std::optional<MemoryConfig>& memory_config,
               std::optional<Tensor> output_tensor,
               const std::optional<CoreRangeSet>& sub_core_grids) {
                return self(predicate, true_value, false_value, memory_config, output_tensor, sub_core_grids);
            },
            nb::arg("predicate"),
            nb::arg("true_value"),
            nb::arg("false_value"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none(),
            nb::arg("sub_core_grids") = nb::none()},

        ttnn::nanobind_overload_t{
            [](const ternary_operation_t& self,
               const Tensor& predicate,
               const int32_t& true_value,
               const int32_t& false_value,
               const std::optional<MemoryConfig>& memory_config,
               std::optional<Tensor> output_tensor,
               const std::optional<CoreRangeSet>& sub_core_grids) {
                return self(predicate, true_value, false_value, memory_config, output_tensor, sub_core_grids);
            },
            nb::arg("predicate"),
            nb::arg("true_value"),
            nb::arg("false_value"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none(),
            nb::arg("sub_core_grids") = nb::none()},

        ttnn::nanobind_overload_t{
            [](const ternary_operation_t& self,
               const Tensor& predicate,
               const uint32_t& true_value,
               const uint32_t& false_value,
               const std::optional<MemoryConfig>& memory_config,
               std::optional<Tensor> output_tensor,
               const std::optional<CoreRangeSet>& sub_core_grids) {
                return self(predicate, true_value, false_value, memory_config, output_tensor, sub_core_grids);
            },
            nb::arg("predicate"),
            nb::arg("true_value"),
            nb::arg("false_value"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none(),
            nb::arg("sub_core_grids") = nb::none()});
}

template <typename ternary_operation_t>
void bind_ternary_lerp(nb::module_& mod, const ternary_operation_t& operation, const std::string& description) {
    auto doc = fmt::format(
        R"doc(
        {2}

        .. math::
            \mathrm{{output\_tensor}} = \verb|{0}|(\mathrm{{input, end, weight}})

        Args:
            input  (ttnn.Tensor): the input tensor with the starting points.
            end    (ttnn.Tensor): the tensor with the ending points.
            weight (ttnn.Tensor or float): the weight for the interpolation formula.


        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the operation. Defaults to `None`.


        Note:
            Supported dtypes, layouts, and ranks:

            .. list-table::
                :header-rows: 1

                * - Dtypes
                  - Layouts
                  - Ranks
                * - BFLOAT16, BFLOAT8_B
                  - TILE
                  - 2, 3, 4

            bfloat8_b/bfloat4_b supports only on TILE_LAYOUT

            end, weight tensors should have same dtype as input
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        description);

    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const ternary_operation_t& self,
               const Tensor& input,
               const Tensor& end,
               const Tensor& weight,
               const std::optional<MemoryConfig>& memory_config) { return self(input, end, weight, memory_config); },
            nb::arg("input"),
            nb::arg("end"),
            nb::arg("weight"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()},

        ttnn::nanobind_overload_t{
            [](const ternary_operation_t& self,
               const Tensor& input,
               const Tensor& end,
               float weight,
               const std::optional<MemoryConfig>& memory_config) { return self(input, end, weight, memory_config); },
            nb::arg("input"),
            nb::arg("end"),
            nb::arg("weight"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()});
}

template <typename ternary_operation_t>
void bind_ternary_addcmul(nb::module_& mod, const ternary_operation_t& operation, const std::string& description) {
    auto doc = fmt::format(
        R"doc(
            {2}

        Args:
            input_a (ttnn.Tensor): the first input tensor.
            input_b (ttnn.Tensor): the second input tensor.
            input_c (ttnn.Tensor): the third input tensor.

        Keyword Args:
            value (float, optional): scalar value to multiply with input_b * input_c. Defaults to 1.0.
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the operation. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): preallocated output tensor. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

        Note:
            Supported dtypes, layouts, and ranks:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
                 - Ranks
               * - FLOAT32, BFLOAT16, BFLOAT8_B, INT32
                 - TILE
                 - 2, 3, 4

            Only TTT (tensor-tensor-tensor) variant is supported.
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        description);

    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const ternary_operation_t& self,
               const Tensor& input_a,
               const Tensor& input_b,
               const Tensor& input_c,
               float value,
               const std::optional<MemoryConfig>& memory_config,
               std::optional<Tensor> output_tensor) {
                return self(input_a, input_b, input_c, value, memory_config, output_tensor);
            },
            nb::arg("input_a"),
            nb::arg("input_b"),
            nb::arg("input_c"),
            nb::kw_only(),
            nb::arg("value") = 1.0f,
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none()});
}

template <typename ternary_operation_t>
void bind_ternary_mac(nb::module_& mod, const ternary_operation_t& operation, const std::string& description) {
    auto doc = fmt::format(
        R"doc(
            {2}

        Args:
            input_tensor_a (ttnn.Tensor): the input tensor.
            input_tensor_b (ttnn.Tensor or Number): the input tensor.
            input_tensor_c (ttnn.Tensor or Number): the input tensor.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the operation. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

        Note:
            Supported dtypes, layouts, and ranks:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
                 - Ranks
               * - BFLOAT16, BFLOAT8_B
                 - TILE
                 - 2, 3, 4

            bfloat8_b/bfloat4_b supports only on TILE_LAYOUT
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        description);

    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const ternary_operation_t& self,
               const Tensor& input_tensor_a,
               const Tensor& input_tensor_b,
               const Tensor& input_tensor_c,
               const std::optional<MemoryConfig>& memory_config) {
                return self(input_tensor_a, input_tensor_b, input_tensor_c, memory_config);
            },
            nb::arg("input_tensor_a"),
            nb::arg("input_tensor_b"),
            nb::arg("input_tensor_c"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()},

        ttnn::nanobind_overload_t{
            [](const ternary_operation_t& self,
               const Tensor& input_tensor_a,
               float value1,
               float value2,
               const std::optional<MemoryConfig>& memory_config) {
                return self(input_tensor_a, value1, value2, memory_config);
            },
            nb::arg("input_tensor_a"),
            nb::arg("value1"),
            nb::arg("value2"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()});
}

}  // namespace

void py_module(nb::module_& mod) {
    // new imported
    bind_ternary_addcmul(mod, ttnn::addcmul, R"doc(Computes addcmul: output = input_a + value * input_b * input_c)doc");

    bind_ternary_composite_float(
        mod,
        ttnn::addcdiv,
        R"doc(Computes Addcdiv on :attr:`input_tensor_a`, :attr:`input_tensor_b` and :attr:`input_tensor_c` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc");

    bind_ternary_where(
        mod,
        ttnn::where,
        R"doc(Selects elements from :attr:`true_value` or :attr:`false_value` depending on the corresponding value in :attr:`condition`. For each element, if the corresponding entry in :attr:`condition` is 1, the output element is taken from :attr:`true_value`; otherwise, it is taken from :attr:`false_value`.)doc");

    bind_ternary_lerp(
        mod,
        ttnn::lerp,
        R"doc(Computes Lerp on :attr:`input`, :attr:`end` and :attr:`weight` and returns the tensor with the same layout as :attr:`input`)doc");

    bind_ternary_mac(
        mod,
        ttnn::mac,
        R"doc(Computes Mac on :attr:`input_tensor_a`, :attr:`input_tensor_b` and :attr:`input_tensor_c` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc");
}

}  // namespace ttnn::operations::ternary
