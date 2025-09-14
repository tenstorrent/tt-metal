// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ternary_pybind.hpp"

#include <string>

#include <fmt/format.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
#include "ttnn/operations/eltwise/ternary/ternary_composite.hpp"
#include "ttnn/operations/eltwise/ternary/where/where.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::ternary {

namespace {

template <typename ternary_operation_t>
void bind_ternary_composite_float(
    py::module& module,
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
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const ternary_operation_t& self,
               const Tensor& input_tensor_a,
               const Tensor& input_tensor_b,
               const Tensor& input_tensor_c,
               float value,
               const std::optional<MemoryConfig>& memory_config) {
                return self(input_tensor_a, input_tensor_b, input_tensor_c, value, memory_config);
            },
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::arg("input_tensor_c"),
            py::kw_only(),
            py::arg("value"),
            py::arg("memory_config") = std::nullopt});
}

template <typename ternary_operation_t>
void bind_ternary_where(py::module& module, const ternary_operation_t& operation, const std::string& description) {
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
            queue_id (int, optional): command queue id. Defaults to `0`.


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
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const ternary_operation_t& self,
               const Tensor& predicate,
               const std::variant<float, Tensor>& true_value,
               const std::variant<float, Tensor>& false_value,
               const std::optional<MemoryConfig>& memory_config,
               std::optional<Tensor> output_tensor,
               QueueId queue_id) {
                return self(queue_id, predicate, true_value, false_value, memory_config, output_tensor);
            },
            py::arg("predicate"),
            py::arg("true_value"),
            py::arg("false_value"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("output_tensor") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId});
}

template <typename ternary_operation_t>
void bind_ternary_lerp(py::module& module, const ternary_operation_t& operation, const std::string& description) {
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
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const ternary_operation_t& self,
               const Tensor& input,
               const Tensor& end,
               const Tensor& weight,
               const std::optional<MemoryConfig>& memory_config) { return self(input, end, weight, memory_config); },
            py::arg("input"),
            py::arg("end"),
            py::arg("weight"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt},

        ttnn::pybind_overload_t{
            [](const ternary_operation_t& self,
               const Tensor& input,
               const Tensor& end,
               float weight,
               const std::optional<MemoryConfig>& memory_config) { return self(input, end, weight, memory_config); },
            py::arg("input"),
            py::arg("end"),
            py::arg("weight"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt});
}

template <typename ternary_operation_t>
void bind_ternary_mac(py::module& module, const ternary_operation_t& operation, const std::string& description) {
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
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const ternary_operation_t& self,
               const Tensor& input_tensor_a,
               const Tensor& input_tensor_b,
               const Tensor& input_tensor_c,
               const std::optional<MemoryConfig>& memory_config) {
                return self(input_tensor_a, input_tensor_b, input_tensor_c, memory_config);
            },
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::arg("input_tensor_c"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt},

        ttnn::pybind_overload_t{
            [](const ternary_operation_t& self,
               const Tensor& input_tensor_a,
               float value1,
               float value2,
               const std::optional<MemoryConfig>& memory_config) {
                return self(input_tensor_a, value1, value2, memory_config);
            },
            py::arg("input_tensor_a"),
            py::arg("value1"),
            py::arg("value2"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt});
}

}  // namespace

void py_module(py::module& module) {
    // new imported
    bind_ternary_composite_float(
        module,
        ttnn::addcmul,
        R"doc(Computes Addcmul on :attr:`input_tensor_a`, :attr:`input_tensor_b` and :attr:`input_tensor_c` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    bind_ternary_composite_float(
        module,
        ttnn::addcdiv,
        R"doc(Computes Addcdiv on :attr:`input_tensor_a`, :attr:`input_tensor_b` and :attr:`input_tensor_c` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc");

    bind_ternary_where(
        module,
        ttnn::where,
        R"doc(Computes Where on :attr:`input_tensor_a`, :attr:`input_tensor_b` and :attr:`input_tensor_c` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc");

    bind_ternary_lerp(
        module,
        ttnn::lerp,
        R"doc(Computes Lerp on :attr:`input`, :attr:`end` and :attr:`weight` and returns the tensor with the same layout as :attr:`input`)doc");

    bind_ternary_mac(
        module,
        ttnn::mac,
        R"doc(Computes Mac on :attr:`input_tensor_a`, :attr:`input_tensor_b` and :attr:`input_tensor_c` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc");
}

}  // namespace ttnn::operations::ternary
