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
#include "ttnn/operations/eltwise/ternary/where/where.hpp"
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

        Example:
            >>> value = 1.0
            >>> tensor1 = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> tensor2 = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> tensor3 = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> output = {1}(tensor1, tensor2, tensor3, value)
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
            input_tensor_a (ttnn.Tensor): the input tensor.
            input_tensor_b (ttnn.Tensor or Number): the input tensor.
            input_tensor_c (ttnn.Tensor or Number): the input tensor.


        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the operation. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): preallocated output tensor. Defaults to `None`.


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

        Example:
            >>> tensor1 = ttnn.from_torch(torch.tensor([[1, 0], [1, 0]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> tensor2 = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> tensor3 = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> output = {1}(tensor1, tensor2, tensor3)
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
               const std::variant<float, Tensor>& true_value,
               const std::variant<float, Tensor>& false_value,
               const std::optional<MemoryConfig>& memory_config,
               std::optional<Tensor> output_tensor) {
                return self(predicate, true_value, false_value, memory_config, output_tensor);
            },
            nb::arg("predicate"),
            nb::arg("true_value"),
            nb::arg("false_value"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none()});
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

        Example:
            >>> tensor1 = ttnn.from_torch(torch.tensor([[1, 0], [1, 0]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> tensor2 = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> tensor3 = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> output = {1}(tensor1, tensor2, tensor3/scalar)
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

        Example:
            >>> tensor1 = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> tensor2 = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> tensor3 = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> output = {1}(tensor1, tensor2/scalar, tensor3/scalar)
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
    bind_ternary_composite_float(
        mod,
        ttnn::addcmul,
        R"doc(Computes Addcmul on :attr:`input_tensor_a`, :attr:`input_tensor_b` and :attr:`input_tensor_c` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    bind_ternary_composite_float(
        mod,
        ttnn::addcdiv,
        R"doc(Computes Addcdiv on :attr:`input_tensor_a`, :attr:`input_tensor_b` and :attr:`input_tensor_c` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc");

    bind_ternary_where(
        mod,
        ttnn::where,
        R"doc(Computes Where on :attr:`input_tensor_a`, :attr:`input_tensor_b` and :attr:`input_tensor_c` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc");

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
