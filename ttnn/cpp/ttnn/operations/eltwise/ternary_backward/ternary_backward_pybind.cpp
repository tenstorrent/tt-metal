// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ternary_backward_pybind.hpp"

#include <optional>
#include <string>
#include <string_view>

#include <fmt/format.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
#include "ttnn/operations/eltwise/ternary_backward/ternary_backward.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::ternary_backward {

namespace {

template <typename ternary_backward_operation_t>
void bind_ternary_backward(
    py::module& module,
    const ternary_backward_operation_t& operation,
    const std::string_view description,
    const std::string& supported_dtype = "BFLOAT16",
    const std::string_view note = "") {
    auto doc = fmt::format(
        R"doc(

        {2}

        Args:
            grad_tensor (ttnn.Tensor): the input gradient tensor.
            input_tensor_a (ttnn.Tensor): the input tensor.
            input_tensor_b (ttnn.Tensor): the input tensor.
            input_tensor_c (ttnn.Tensor): the input tensor.
            alpha (float): the alpha value.

        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the operation. Defaults to `None`.

        Returns:
            List of ttnn.Tensor: the output tensor.

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

            {4}

        Example:
            >>> value = 1.0
            >>> grad_tensor = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> tensor1 = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device)
            >>> tensor2 = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device)
            >>> tensor3 = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device)
            >>> output = {1}(grad_tensor, tensor1, tensor2, tensor3, value)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        description,
        supported_dtype,
        note);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const ternary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor_a,
               const ttnn::Tensor& input_tensor_b,
               const ttnn::Tensor& input_tensor_c,
               float alpha,
               const std::optional<ttnn::MemoryConfig>& memory_config) {
                auto output_memory_config = memory_config.value_or(input_tensor_a.memory_config());
                return self(grad_tensor, input_tensor_a, input_tensor_b, input_tensor_c, alpha, output_memory_config);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::arg("input_tensor_c"),
            py::arg("alpha"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt});
}

template <typename ternary_backward_operation_t>
void bind_ternary_backward_op(
    py::module& module,
    const ternary_backward_operation_t& operation,
    const std::string_view description,
    const std::string& supported_dtype = "BFLOAT16") {
    auto doc = fmt::format(
        R"doc(
        {2}


        Args:
            grad_tensor (ttnn.Tensor): the input gradient tensor.
            input_tensor_a (ttnn.Tensor): the input tensor.
            input_tensor_b (ttnn.Tensor): the input tensor.
            input_tensor_c (ttnn.Tensor or Number): the input tensor.


        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the operation. Defaults to `None`.


        Returns:
            List of ttnn.Tensor: the output tensor.


        Note:
            Supported dtypes, layouts, and ranks:

            For Inputs : :attr:`input_tensor_a`, :attr:`input_tensor_b` and :attr:`input_tensor_c`

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
                 - Ranks
               * - BFLOAT16
                 - TILE
                 - 2, 3, 4

            For Inputs : :attr:`input_tensor_a`, :attr:`input_tensor_b` and :attr:`scalar`

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
                 - Ranks
               * - {3}
                 - TILE
                 - 2, 3, 4

            bfloat8_b/bfloat4_b is only supported on TILE_LAYOUT


        Example:
            >>> grad_tensor = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> tensor1 = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device)
            >>> tensor2 = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device)
            >>> tensor3 = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device)
            >>> output = {1}(grad_tensor, tensor1, tensor2, tensor3/scalar)

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
            [](const ternary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor_a,
               const ttnn::Tensor& input_tensor_b,
               const ttnn::Tensor& input_tensor_c,
               const std::optional<ttnn::MemoryConfig>& memory_config) {
                return self(grad_tensor, input_tensor_a, input_tensor_b, input_tensor_c, memory_config);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::arg("input_tensor_c"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt},

        ttnn::pybind_overload_t{
            [](const ternary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor_a,
               const ttnn::Tensor& input_tensor_b,
               const float scalar,
               const std::optional<ttnn::MemoryConfig>& memory_config) {
                return self(grad_tensor, input_tensor_a, input_tensor_b, scalar, memory_config);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::arg("scalar"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt});
}

template <typename ternary_backward_operation_t>
void bind_ternary_backward_optional_output(
    py::module& module,
    const ternary_backward_operation_t& operation,
    const std::string_view description,
    const std::string& supported_dtype = "BFLOAT16") {
    auto doc = fmt::format(
        R"doc(

        {2}

        Args:
            grad_tensor (ttnn.Tensor): the input gradient tensor.
            input_tensor_a (ttnn.Tensor): the input tensor.
            input_tensor_b (ttnn.Tensor): the input tensor.
            input_tensor_c (ttnn.Tensor): the input tensor.

        Keyword args:
            are_required_outputs (List[bool], optional): list of required outputs. Defaults to `[True, True]`.
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the operation. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): preallocated output tensor. Defaults to `None`.
            queue_id (int, optional): command queue id. Defaults to `0`.

        Returns:
            List of ttnn.Tensor: the output tensor.


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


        Example:
            >>> grad_tensor = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> tensor1 = ttnn.from_torch(torch.tensor([[1, 0], [1, 0]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device)
            >>> tensor2 = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device)
            >>> tensor3 = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device)
            >>> output = {1}(grad_tensor, tensor1, tensor2, tensor3)
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
            [](const ternary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor_a,
               const ttnn::Tensor& input_tensor_b,
               const ttnn::Tensor& input_tensor_c,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::vector<bool>& are_required_outputs,
               const std::optional<ttnn::Tensor>& input_a_grad,
               const std::optional<ttnn::Tensor>& input_b_grad,
               QueueId queue_id) -> std::vector<std::optional<ttnn::Tensor>> {
                return self(
                    queue_id,
                    grad_tensor,
                    input_tensor_a,
                    input_tensor_b,
                    input_tensor_c,
                    memory_config,
                    are_required_outputs,
                    input_a_grad,
                    input_b_grad);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::arg("input_tensor_c"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("are_required_outputs") = std::vector<bool>{true, true},
            py::arg("input_a_grad") = std::nullopt,
            py::arg("input_b_grad") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId});
}
}  // namespace

void py_module(py::module& module) {
    bind_ternary_backward(
        module,
        ttnn::addcmul_bw,
        R"doc(Performs backward operations for addcmul of :attr:`input_tensor_a`, :attr:`input_tensor_b` and :attr:`input_tensor_c` with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    bind_ternary_backward(
        module,
        ttnn::addcdiv_bw,
        R"doc(Performs backward operations for addcdiv of :attr:`input_tensor_a`, :attr:`input_tensor_b` and :attr:`input_tensor_c` with given :attr:`grad_tensor`.)doc",
        "",
        R"doc(For more details about BFLOAT8_B, refer to the `BFLOAT8_B limitations <../tensor.html#limitation-of-bfloat8-b>`_.)doc");

    bind_ternary_backward_optional_output(
        module,
        ttnn::where_bw,
        R"doc(Performs backward operations for where of :attr:`input_tensor_a`, :attr:`input_tensor_b` and :attr:`input_tensor_c` with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    bind_ternary_backward_op(
        module,
        ttnn::lerp_bw,
        R"doc(Performs backward operations for lerp of :attr:`input_tensor_a`, :attr:`input_tensor_b` and :attr:`input_tensor_c` or :attr:`scalar` with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");
}

}  // namespace ttnn::operations::ternary_backward
