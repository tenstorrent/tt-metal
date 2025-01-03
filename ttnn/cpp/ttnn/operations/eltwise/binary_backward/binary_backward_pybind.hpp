// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
#include "ttnn/operations/eltwise/binary_backward/binary_backward.hpp"
#include "ttnn/operations/eltwise/ternary_backward/ternary_backward.hpp"
#include "ttnn/types.hpp"
#include "ttnn/cpp/ttnn/common/constants.hpp"

namespace py = pybind11;

namespace ttnn {
namespace operations {
namespace binary_backward {

namespace detail {

template <typename binary_backward_operation_t>
void bind_binary_backward_ops(
    py::module& module,
    const binary_backward_operation_t& operation,
    const std::string_view description,
    const std::string_view supported_dtype = "BFLOAT16",
    const std::string_view note = "") {
    auto doc = fmt::format(
        R"doc(
        {2}

        Args:
            grad_tensor (ttnn.Tensor): the input gradient tensor.
            input_tensor_a (ttnn.Tensor): the input tensor.
            input_tensor_b (ttnn.Tensor): the input tensor.

        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.

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

            bfloat8_b/bfloat4_b is only supported on TILE_LAYOUT

            {4}

        Example:
            >>> grad_tensor = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> tensor1 = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device)
            >>> tensor2 = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device)
            >>> output = {1}(grad_tensor, tensor1, tensor2)


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
            [](const binary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor_a,
               const ttnn::Tensor& input_tensor_b,
               const std::optional<ttnn::MemoryConfig>& memory_config) -> std::vector<ttnn::Tensor> {
                auto output_memory_config = memory_config.value_or(input_tensor_a.memory_config());
                return self(grad_tensor, input_tensor_a, input_tensor_b, output_memory_config);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt});
}

template <typename binary_backward_operation_t>
void bind_binary_backward_concat(
    py::module& module,
    const binary_backward_operation_t& operation,
    const std::string& parameter_name,
    const std::string& parameter_doc,
    int parameter_value,
    const std::string_view description,
    const std::string_view supported_dtype = "BFLOAT16") {
    auto doc = fmt::format(
        R"doc(
        {5}

        Args:
            grad_tensor (ttnn.Tensor): the input gradient tensor.
            input_tensor_a (ttnn.Tensor): the input tensor.
            input_tensor_b (ttnn.Tensor): the input tensor.
            {2} (int): {3}. Defaults to `{4}`.


        Keyword args:
            are_required_outputs (List[bool], optional): List of required outputs. Defaults to `[True, True]`.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            input_grad (ttnn.Tensor, optional): Preallocated output tensor for gradient of `input_tensor_a`. Defaults to `None`.
            other_grad (ttnn.Tensor, optional): Preallocated output tensor for gradient of `input_tensor_b`. Defaults to `None`.
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
               * - {6}
                 - TILE
                 - 4

            bfloat8_b/bfloat4_b is only supported on TILE_LAYOUT


        Example:
            >>> grad_tensor = ttnn.from_torch(torch.rand([14, 1, 30, 32], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> tensor1 = ttnn.from_torch(torch.rand([12, 1, 30, 32], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device)
            >>> tensor2 = ttnn.from_torch(torch.rand([2, 1, 30, 32], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device)
            >>> {2} = {4}
            >>> output = ttnn.concat_bw(grad_tensor, tensor1, tensor2, {2})


        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        parameter_name,
        parameter_doc,
        parameter_value,
        description,
        supported_dtype);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const binary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor_a,
               const ttnn::Tensor& input_tensor_b,
               int parameter,
               const std::vector<bool>& are_required_outputs,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& input_grad,
               const std::optional<ttnn::Tensor>& other_grad,
               const uint8_t& queue_id) -> std::vector<std::optional<ttnn::Tensor>> {
                return self(
                    grad_tensor,
                    input_tensor_a,
                    input_tensor_b,
                    parameter,
                    are_required_outputs,
                    memory_config,
                    input_grad,
                    other_grad);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::arg(parameter_name.c_str()) = parameter_value,
            py::kw_only(),
            py::arg("are_required_outputs") = std::vector<bool>{true, true},
            py::arg("memory_config") = std::nullopt,
            py::arg("input_a_grad") = std::nullopt,
            py::arg("input_b_grad") = std::nullopt,
            py::arg("queue_id") = ttnn::DefaultQueueId});
}

template <typename binary_backward_operation_t>
void bind_binary_backward_addalpha(
    py::module& module,
    const binary_backward_operation_t& operation,
    const std::string& parameter_name,
    const std::string& parameter_doc,
    float parameter_value,
    const std::string_view description,
    const std::string_view supported_dtype = "BFLOAT16") {
    auto doc = fmt::format(
        R"doc(
        {5}


        Args:
            grad_tensor (ttnn.Tensor): the input gradient tensor.
            input_tensor_a (ttnn.Tensor): the input tensor.
            input_tensor_b (ttnn.Tensor): the input tensor.
            {2} (float): {3}. Defaults to `{4}`.


        Keyword args:
            are_required_outputs (List[bool], optional): List of required outputs. Defaults to `[True, True]`.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            input_grad (ttnn.Tensor, optional): Preallocated output tensor for gradient of `input_tensor_a`. Defaults to `None`.
            other_grad (ttnn.Tensor, optional): Preallocated output tensor for gradient of `input_tensor_b`. Defaults to `None`.
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
               * - {6}
                 - TILE
                 - 2, 3, 4

            bfloat8_b/bfloat4_b is only supported on TILE_LAYOUT


        Example:
            >>> grad_tensor = ttnn.from_torch(torch.tensor([[1, 2], [3,4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> tensor1 = ttnn.from_torch(torch.tensor([[1, 2], [3,4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device)
            >>> tensor2 = ttnn.from_torch(torch.tensor([[1, 2], [3,4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device)
            >>> {2} = {4}
            >>> output = ttnn.addalpha_bw(grad_tensor, tensor1, tensor2, {2})


        )doc",

        operation.base_name(),
        operation.python_fully_qualified_name(),
        parameter_name,
        parameter_doc,
        parameter_value,
        description,
        supported_dtype);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const binary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor_a,
               const ttnn::Tensor& input_tensor_b,
               float parameter,
               const std::vector<bool>& are_required_outputs,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& input_a_grad,
               const std::optional<ttnn::Tensor>& input_b_grad,
               const uint8_t& queue_id) -> std::vector<std::optional<ttnn::Tensor>> {
                return self(
                    queue_id,
                    grad_tensor,
                    input_tensor_a,
                    input_tensor_b,
                    parameter,
                    are_required_outputs,
                    memory_config,
                    input_a_grad,
                    input_b_grad);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::arg(parameter_name.c_str()) = parameter_value,
            py::kw_only(),
            py::arg("are_required_outputs") = std::vector<bool>{true, true},
            py::arg("memory_config") = std::nullopt,
            py::arg("input_a_grad") = std::nullopt,
            py::arg("input_b_grad") = std::nullopt,
            py::arg("queue_id") = ttnn::DefaultQueueId});
}

template <typename binary_backward_operation_t>
void bind_binary_backward_bias_gelu(
    py::module& module,
    const binary_backward_operation_t& operation,
    const std::string& parameter_name_a,
    const std::string& parameter_a_doc,
    const std::string& parameter_name_b,
    const std::string& parameter_b_doc,
    string parameter_b_value,
    const std::string& description,
    const std::string& supported_dtype = "BFLOAT16") {
    auto doc = fmt::format(
        R"doc(

        {7}

        Args:
            grad_tensor (ttnn.Tensor): the input gradient tensor.
            input_tensor_a (ttnn.Tensor): the input tensor.
            input_tensor_b (ttnn.Tensor or Number): the input tensor.

        Keyword args:
            {4} (string): {5}. Defaults to `{6}`.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.

        Returns:
            List of ttnn.Tensor: the output tensor.

        Note:
            Supported dtypes, layouts, and ranks:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
                 - Ranks
               * - {8}
                 - TILE
                 - 2, 3, 4

            bfloat8_b/bfloat4_b is only supported on TILE_LAYOUT

        Example:

            >>> grad_tensor = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> tensor1 = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device)
            >>> tensor2 = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device)
            >>> approximate = "none"
            >>> output = ttnn.bias_gelu_bw(grad_tensor, tensor1, tensor2, approximate)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        parameter_name_a,
        parameter_a_doc,
        parameter_name_b,
        parameter_b_doc,
        parameter_b_value,
        description,
        supported_dtype);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const binary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor_a,
               const ttnn::Tensor& input_tensor_b,
               string parameter_b,
               const std::optional<MemoryConfig>& memory_config) {
                return self(grad_tensor, input_tensor_a, input_tensor_b, parameter_b, memory_config);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::kw_only(),
            py::arg(parameter_name_b.c_str()) = parameter_b_value,
            py::arg("memory_config") = std::nullopt},

        ttnn::pybind_overload_t{
            [](const binary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor,
               float parameter_a,
               string parameter_b,
               const std::optional<MemoryConfig>& memory_config) {
                return self(grad_tensor, input_tensor, parameter_a, parameter_b, memory_config);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor"),
            py::arg(parameter_name_a.c_str()),
            py::kw_only(),
            py::arg(parameter_name_b.c_str()) = parameter_b_value,
            py::arg("memory_config") = std::nullopt});
}

template <typename binary_backward_operation_t>
void bind_binary_backward_sub_alpha(
    py::module& module,
    const binary_backward_operation_t& operation,
    const std::string& parameter_name,
    const std::string& parameter_doc,
    float parameter_value,
    const std::string_view description,
    const std::string_view supported_dtype = "BFLOAT16") {
    auto doc = fmt::format(
        R"doc(

        {5}

        Args:
            grad_tensor (ttnn.Tensor): the input gradient tensor.
            input_tensor_a (ttnn.Tensor): the input tensor.
            input_tensor_b (ttnn.Tensor): the input tensor.
            {2} (float): {3}. Defaults to `{4}`.

        Keyword args:
            are_required_outputs (List[bool], optional): List of required outputs. Defaults to `[True, True]`.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            input_grad (ttnn.Tensor, optional): Preallocated output tensor for gradient of `input_tensor_a`. Defaults to `None`.
            other_grad (ttnn.Tensor, optional): Preallocated output tensor for gradient of `input_tensor_b`. Defaults to `None`.
            queue_id (int, optional): command queue id. Defaults to `0`.

        Note:
            Supported dtypes, layouts, and ranks:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
                 - Ranks
               * - {6}
                 - TILE
                 - 2, 3, 4

            bfloat8_b/bfloat4_b is only supported on TILE_LAYOUT

        Example:

            >>> grad_tensor = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> tensor1 = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device)
            >>> tensor2 = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device)
            >>> alpha = 1
            >>> output = ttnn.subalpha_bw(grad_tensor, tensor1, tensor2, alpha)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        parameter_name,
        parameter_doc,
        parameter_value,
        description,
        supported_dtype);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const binary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor,
               const ttnn::Tensor& other_tensor,
               float alpha,
               const std::vector<bool>& are_required_outputs,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& input_grad,
               const std::optional<ttnn::Tensor>& other_grad,
               const uint8_t& queue_id) -> std::vector<std::optional<ttnn::Tensor>> {
                return self(
                    queue_id,
                    grad_tensor,
                    input_tensor,
                    other_tensor,
                    alpha,
                    are_required_outputs,
                    memory_config,
                    input_grad,
                    other_grad);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::arg(parameter_name.c_str()) = parameter_value,
            py::kw_only(),
            py::arg("are_required_outputs") = std::vector<bool>{true, true},
            py::arg("memory_config") = std::nullopt,
            py::arg("input_grad") = std::nullopt,
            py::arg("other_grad") = std::nullopt,
            py::arg("queue_id") = ttnn::DefaultQueueId});
}

template <typename binary_backward_operation_t>
void bind_binary_backward_rsub(
    py::module& module,
    const binary_backward_operation_t& operation,
    const std::string_view description,
    const std::string_view supported_dtype = "BFLOAT16") {
    auto doc = fmt::format(
        R"doc(

        {2}

        Args:
            grad_tensor (ttnn.Tensor): the input gradient tensor.
            input_tensor_a (ttnn.Tensor): the input tensor.
            input_tensor_b (ttnn.Tensor): the input tensor.

        Keyword args:
            are_required_outputs (List[bool], optional): List of required outputs. Defaults to `[True, True]`.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            input_grad (ttnn.Tensor, optional): Preallocated output tensor for gradient of `input_tensor_a`. Defaults to `None`.
            other_grad (ttnn.Tensor, optional): Preallocated output tensor for gradient of `input_tensor_b`. Defaults to `None`.
            queue_id (int, optional): command queue id. Defaults to `0`.

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

            bfloat8_b/bfloat4_b is only supported on TILE_LAYOUT

        Example:

            >>> grad_tensor = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> tensor1 = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device)
            >>> tensor2 = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device)
            >>> output = ttnn.rsub_bw(grad_tensor, tensor1, tensor2)
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
            [](const binary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor,
               const ttnn::Tensor& other_tensor,
               const std::vector<bool>& are_required_outputs,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& input_grad,
               const std::optional<ttnn::Tensor>& other_grad,
               const uint8_t& queue_id) -> std::vector<std::optional<ttnn::Tensor>> {
                return self(
                    queue_id,
                    grad_tensor,
                    input_tensor,
                    other_tensor,
                    are_required_outputs,
                    memory_config,
                    input_grad,
                    other_grad);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::kw_only(),
            py::arg("are_required_outputs") = std::vector<bool>{true, true},
            py::arg("memory_config") = std::nullopt,
            py::arg("input_grad") = std::nullopt,
            py::arg("other_grad") = std::nullopt,
            py::arg("queue_id") = ttnn::DefaultQueueId});
}

template <typename binary_backward_operation_t>
void bind_binary_bw_mul(
    py::module& module,
    const binary_backward_operation_t& operation,
    const std::string_view description,
    const std::string_view supported_dtype = "BFLOAT16") {
    auto doc = fmt::format(
        R"doc(
        {2}

        Args:
            grad_tensor (ComplexTensor or ttnn.Tensor): the input gradient tensor.
            input_tensor_a (ComplexTensor or ttnn.Tensor): the input tensor.
            input_tensor_b (ComplexTensor or ttnn.Tensor or Number): the input tensor.

        Keyword args:
            are_required_outputs (List[bool], optional): List of required outputs. Defaults to `[True, True]`.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            input_grad (ttnn.Tensor, optional): Preallocated output tensor for gradient of `input_tensor_a`. Defaults to `None`.
            other_grad (ttnn.Tensor, optional): Preallocated output tensor for gradient of `input_tensor_b`. Defaults to `None`.
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

            bfloat8_b/bfloat4_b is only supported on TILE_LAYOUT

        Example:
            >>> grad_tensor = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> tensor1 = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device)
            >>> tensor2 = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device)
            >>> output = ttnn.mul_bw(grad_tensor, tensor1, tensor2)

            >>> grad_tensor = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> tensor1 = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device)
            >>> scalar = 4
            >>> output = ttnn.mul_bw(grad_tensor, tensor1, scalar)

        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        description,
        supported_dtype);

    bind_registered_operation(
        module,
        operation,
        doc,
        // tensor and scalar
        ttnn::pybind_overload_t{
            [](const binary_backward_operation_t& self,
               const Tensor& grad_tensor,
               const Tensor& input_tensor_a,
               const float scalar,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& input_grad,
               const uint8_t& queue_id) -> std::vector<std::optional<ttnn::Tensor>> {
                return self(queue_id, grad_tensor, input_tensor_a, scalar, memory_config, input_grad);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor_a"),
            py::arg("scalar"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("input_grad") = std::nullopt,
            py::arg("queue_id") = ttnn::DefaultQueueId},

        // tensor and tensor
        ttnn::pybind_overload_t{
            [](const binary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor,
               const ttnn::Tensor& other_tensor,
               const std::vector<bool>& are_required_outputs,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& input_grad,
               const std::optional<ttnn::Tensor>& other_grad,
               const uint8_t& queue_id) -> std::vector<std::optional<ttnn::Tensor>> {
                return self(
                    queue_id,
                    grad_tensor,
                    input_tensor,
                    other_tensor,
                    are_required_outputs,
                    memory_config,
                    input_grad,
                    other_grad);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor"),
            py::arg("other_tensor"),
            py::kw_only(),
            py::arg("are_required_outputs") = std::vector<bool>{true, true},
            py::arg("memory_config") = std::nullopt,
            py::arg("input_grad") = std::nullopt,
            py::arg("other_grad") = std::nullopt,
            py::arg("queue_id") = ttnn::DefaultQueueId},

        // complex tensor
        ttnn::pybind_overload_t{
            [](const binary_backward_operation_t& self,
               const ComplexTensor& grad_tensor,
               const ComplexTensor& input_tensor_a,
               const ComplexTensor& input_tensor_b,
               const MemoryConfig& memory_config) {
                return self(grad_tensor, input_tensor_a, input_tensor_b, memory_config);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt});
}

template <typename binary_backward_operation_t>
void bind_binary_bw(
    py::module& module,
    const binary_backward_operation_t& operation,
    const std::string_view description,
    const std::string_view supported_dtype = "BFLOAT16") {
    auto doc = fmt::format(
        R"doc(

        {2}
        Supports broadcasting.

        Args:
            grad_tensor (ComplexTensor or ttnn.Tensor): the input gradient tensor.
            input_tensor_a (ComplexTensor or ttnn.Tensor): the input tensor.
            input_tensor_b (ComplexTensor or ttnn.Tensor or Number): the input tensor.

        Keyword args:
            are_required_outputs (List[bool], optional): List of required outputs. Defaults to `[True, True]`.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            input_grad (ttnn.Tensor, optional): Preallocated output tensor for gradient of `input_tensor_a`. Defaults to `None`.
            other_grad (ttnn.Tensor, optional): Preallocated output tensor for gradient of `input_tensor_b`. Defaults to `None`.
            queue_id (int, optional): command queue id. Defaults to `0`.

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

            bfloat8_b/bfloat4_b is only supported on TILE_LAYOUT

        Example:
            >>> grad_tensor = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> tensor1 = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device)
            >>> tensor2 = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device)
            >>> output = {1}(grad_tensor, tensor1, tensor2)

            >>> grad_tensor = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> tensor1 = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device)
            >>> scalar = 2
            >>> output = {1}(grad_tensor, tensor1, scalar)

        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        description,
        supported_dtype);

    bind_registered_operation(
        module,
        operation,
        doc,
        // tensor and scalar
        ttnn::pybind_overload_t{
            [](const binary_backward_operation_t& self,
               const Tensor& grad_tensor,
               const Tensor& input_tensor_a,
               const float scalar,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& input_grad,
               const uint8_t& queue_id) -> std::vector<std::optional<ttnn::Tensor>> {
                return self(queue_id, grad_tensor, input_tensor_a, scalar, memory_config, input_grad);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor_a"),
            py::arg("scalar"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("input_grad") = std::nullopt,
            py::arg("queue_id") = ttnn::DefaultQueueId},

        // tensor and tensor
        ttnn::pybind_overload_t{
            [](const binary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor,
               const ttnn::Tensor& other_tensor,
               const std::vector<bool>& are_required_outputs,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& input_grad,
               const std::optional<ttnn::Tensor>& other_grad,
               const uint8_t& queue_id) -> std::vector<std::optional<ttnn::Tensor>> {
                return self(
                    queue_id,
                    grad_tensor,
                    input_tensor,
                    other_tensor,
                    are_required_outputs,
                    memory_config,
                    input_grad,
                    other_grad);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor"),
            py::arg("other_tensor"),
            py::kw_only(),
            py::arg("are_required_outputs") = std::vector<bool>{true, true},
            py::arg("memory_config") = std::nullopt,
            py::arg("input_grad") = std::nullopt,
            py::arg("other_grad") = std::nullopt,
            py::arg("queue_id") = ttnn::DefaultQueueId},

        // complex tensor
        ttnn::pybind_overload_t{
            [](const binary_backward_operation_t& self,
               const ComplexTensor& grad_tensor,
               const ComplexTensor& input_tensor_a,
               const ComplexTensor& input_tensor_b,
               float alpha,
               const std::optional<MemoryConfig>& memory_config) {
                return self(grad_tensor, input_tensor_a, input_tensor_b, alpha, memory_config);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::arg("alpha"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt});
}

template <typename binary_backward_operation_t>
void bind_binary_bw_div(
    py::module& module,
    const binary_backward_operation_t& operation,
    const std::string_view description,
    const std::string_view supported_dtype = "BFLOAT16") {
    auto doc = fmt::format(
        R"doc(

        {2}

        Args:
            grad_tensor (ComplexTensor or ttnn.Tensor): the input gradient tensor.
            input_tensor_a (ComplexTensor or ttnn.Tensor): the input tensor.
            input_tensor_b (ComplexTensor or ttnn.Tensor or Number): the input tensor.

        Keyword args:
            round_mode (str, optional): Round mode for the operation (when input tensors are not ComplexTensor type). Can be  None, "trunc", "floor". Defaults to `None`.
            are_required_outputs (List[bool], optional): List of required outputs. Defaults to `[True, True]`.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            input_grad (ttnn.Tensor, optional): Preallocated output tensor for gradient of `input_tensor`. Defaults to `None`.
            other_grad (ttnn.Tensor, optional): Preallocated output tensor for gradient of `other_tensor`. Defaults to `None`.
            queue_id (int, optional): command queue id. Defaults to `0`.

        Returns:
            List of ttnn.Tensor: the output tensor.

        Supports broadcasting.

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

            bfloat8_b/bfloat4_b is only supported on TILE_LAYOUT

        Example:
            >>> grad_tensor = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> tensor1 = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device)
            >>> tensor2 = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device)
            >>> output = ttnn.div_bw(grad_tensor, tensor1, tensor2, round_mode = None)

            >>> grad_tensor = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> tensor = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device)
            >>> scalar = 2
            >>> output = ttnn.div_bw(grad_tensor, tensor, scalar, round_mode = None)

        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        description,
        supported_dtype);

    bind_registered_operation(
        module,
        operation,
        doc,
        // tensor and scalar
        ttnn::pybind_overload_t{
            [](const binary_backward_operation_t& self,
               const Tensor& grad_tensor,
               const Tensor& input_tensor_a,
               const float scalar,
               const std::optional<std::string> round_mode,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& input_grad,
               const uint8_t& queue_id) -> std::vector<std::optional<ttnn::Tensor>> {
                return self(queue_id, grad_tensor, input_tensor_a, scalar, round_mode, memory_config, input_grad);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor_a"),
            py::arg("scalar"),
            py::kw_only(),
            py::arg("round_mode") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("input_grad") = std::nullopt,
            py::arg("queue_id") = ttnn::DefaultQueueId},

        // tensor and tensor
        ttnn::pybind_overload_t{
            [](const binary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor,
               const ttnn::Tensor& other_tensor,
               const std::optional<std::string> round_mode,
               const std::vector<bool>& are_required_outputs,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& input_grad,
               const std::optional<ttnn::Tensor>& other_grad,
               const uint8_t& queue_id) -> std::vector<std::optional<ttnn::Tensor>> {
                return self(
                    queue_id,
                    grad_tensor,
                    input_tensor,
                    other_tensor,
                    round_mode,
                    are_required_outputs,
                    memory_config,
                    input_grad,
                    other_grad);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor"),
            py::arg("other_tensor"),
            py::kw_only(),
            py::arg("round_mode") = std::nullopt,
            py::arg("are_required_outputs") = std::vector<bool>{true, true},
            py::arg("memory_config") = std::nullopt,
            py::arg("input_grad") = std::nullopt,
            py::arg("other_grad") = std::nullopt,
            py::arg("queue_id") = ttnn::DefaultQueueId},

        // complex tensor
        ttnn::pybind_overload_t{
            [](const binary_backward_operation_t& self,
               const ComplexTensor& grad_tensor,
               const ComplexTensor& input_tensor_a,
               const ComplexTensor& input_tensor_b,
               const MemoryConfig& memory_config) {
                return self(grad_tensor, input_tensor_a, input_tensor_b, memory_config);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt});
}

template <typename binary_backward_operation_t>
void bind_binary_backward_overload(
    py::module& module,
    const binary_backward_operation_t& operation,
    const std::string& description,
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& note = "") {
    auto doc = fmt::format(
        R"doc(

        {2}

        Args:
            grad_tensor (ttnn.Tensor): the input gradient tensor.
            input_tensor_a (ttnn.Tensor): the input tensor.
            input_tensor_b (ttnn.Tensor or Number): the input tensor.

        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.

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

            >>> grad_tensor = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> tensor1 = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device)
            >>> tensor2 = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device)
            >>> output = {1}(grad_tensor, tensor1, tensor2)

            >>> grad_tensor = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> tensor1 = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device)
            >>> scalar = 2
            >>> output = {1}(grad_tensor, tensor1, scalar)
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
        // tensor and scalar
        ttnn::pybind_overload_t{
            [](const binary_backward_operation_t& self,
               const Tensor& grad_tensor,
               const Tensor& input_tensor_a,
               const float scalar,
               const std::optional<ttnn::MemoryConfig>& memory_config) -> std::vector<ttnn::Tensor> {
                return self(grad_tensor, input_tensor_a, scalar, memory_config);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor_a"),
            py::arg("scalar"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt},

        // tensor and tensor
        ttnn::pybind_overload_t{
            [](const binary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor_a,
               const ttnn::Tensor& input_tensor_b,
               const std::optional<ttnn::MemoryConfig>& memory_config) -> std::vector<ttnn::Tensor> {
                return self(grad_tensor, input_tensor_a, input_tensor_b, memory_config);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt});
}

template <typename binary_backward_operation_t>
void bind_binary_backward_assign(
    py::module& module,
    const binary_backward_operation_t& operation,
    const std::string_view description,
    const std::string_view supported_dtype = "BFLOAT16") {
    auto doc = fmt::format(
        R"doc(

        {2}

        Args:
            grad_tensor (ttnn.Tensor): the input gradient tensor.
            input_tensor_a (ttnn.Tensor): the input tensor.
            input_tensor_b (ttnn.Tensor): the input tensor.

        Keyword args:
            are_required_outputs (List[bool], optional): List of required outputs. Defaults to `[True, True]`.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            input_grad (ttnn.Tensor, optional): Preallocated output tensor for gradient of `input_tensor`. Defaults to `None`.
            other_grad (ttnn.Tensor, optional): Preallocated output tensor for gradient of `other_tensor`. Defaults to `None`.
            queue_id (int, optional): command queue id. Defaults to `0`.
            round_mode (str, optional): Round mode for the operation. Defaults to `None`.

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
            >>> tensor1 = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device)
            >>> tensor2 = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device)
            >>> output = ttnn.assign_bw(grad_tensor, tensor1, tensor2)

            >>> grad_tensor = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> tensor1 = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device)
            >>> output = ttnn.assign_bw(grad_tensor, tensor1)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        description,
        supported_dtype);

    bind_registered_operation(
        module,
        operation,
        doc,
        // tensor
        ttnn::pybind_overload_t{
            [](const binary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& input_grad,
               const uint8_t& queue_id) -> std::vector<std::optional<ttnn::Tensor>> {
                return self(queue_id, grad_tensor, input_tensor, memory_config, input_grad);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("input_a_grad") = std::nullopt,
            py::arg("queue_id") = ttnn::DefaultQueueId},

        // tensor and tensor
        ttnn::pybind_overload_t{
            [](const binary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor_a,
               const ttnn::Tensor& input_tensor_b,
               const std::vector<bool>& are_required_outputs,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& input_a_grad,
               const std::optional<ttnn::Tensor>& input_b_grad,
               const uint8_t& queue_id) -> std::vector<std::optional<ttnn::Tensor>> {
                return self(
                    queue_id,
                    grad_tensor,
                    input_tensor_a,
                    input_tensor_b,
                    are_required_outputs,
                    memory_config,
                    input_a_grad,
                    input_b_grad);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::kw_only(),
            py::arg("are_required_outputs") = std::vector<bool>{true, true},
            py::arg("memory_config") = std::nullopt,
            py::arg("input_a_grad") = std::nullopt,
            py::arg("input_b_grad") = std::nullopt,
            py::arg("queue_id") = ttnn::DefaultQueueId});
}

}  // namespace detail

void py_module(py::module& module) {
    detail::bind_binary_bw_mul(
        module,
        ttnn::mul_bw,
        R"doc(Performs backward operations for multiply on :attr:`input_tensor_a`, :attr:`input_tensor_b`, with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_binary_bw(
        module,
        ttnn::add_bw,
        R"doc(Performs backward operations for add of :attr:`input_tensor_a` and :attr:`input_tensor_b` or :attr:`scalar` with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_binary_bw(
        module,
        ttnn::sub_bw,
        R"doc(Performs backward operations for subtract of :attr:`input_tensor_a` and :attr:`input_tensor_b` or :attr:`scalar` with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_binary_bw_div(
        module,
        ttnn::div_bw,
        R"doc(Performs backward operations for divide on :attr:`input_tensor`, :attr:`alpha` or :attr:`input_tensor_a`, :attr:`input_tensor_b`, :attr:`round_mode`,  with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16)doc");

    detail::bind_binary_backward_overload(
        module,
        ttnn::remainder_bw,
        R"doc(Performs backward operations for remainder of :attr:`input_tensor_a`, :attr:`scalar` or :attr:`input_tensor_b` with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16)doc",
        R"doc(Supported only in WHB0.)doc");

    detail::bind_binary_backward_overload(
        module,
        ttnn::fmod_bw,
        R"doc(Performs backward operations for fmod of :attr:`input_tensor_a`, :attr:`scalar` or :attr:`input_tensor_b` with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16)doc");

    detail::bind_binary_backward_assign(
        module,
        ttnn::assign_bw,
        R"doc(Performs backward operations for assign of :attr:`input_tensor_a`, :attr:`input_tensor_b` with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_binary_backward_ops(
        module,
        ttnn::atan2_bw,
        R"doc(Performs backward operations for atan2 of :attr:`input_tensor_a` and :attr:`input_tensor_b` with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16)doc");

    detail::bind_binary_backward_sub_alpha(
        module,
        ttnn::subalpha_bw,
        "alpha",
        "Alpha value",
        1.0f,
        R"doc(Performs backward operations for subalpha of :attr:`input_tensor_a` and :attr:`input_tensor_b` with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_binary_backward_addalpha(
        module,
        ttnn::addalpha_bw,
        "alpha",
        "Alpha value",
        1.0f,
        R"doc(Performs backward operations for addalpha on :attr:`input_tensor_b` , :attr:`input_tensor_a` and :attr:`alpha` with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_binary_backward_ops(
        module,
        ttnn::xlogy_bw,
        R"doc(Performs backward operations for xlogy of :attr:`input_tensor_a` and :attr:`input_tensor_b` with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16)doc");

    detail::bind_binary_backward_ops(
        module,
        ttnn::hypot_bw,
        R"doc(Performs backward operations for hypot of :attr:`input_tensor_a` and :attr:`input_tensor_b` with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16)doc");

    detail::bind_binary_backward_ops(
        module,
        ttnn::ldexp_bw,
        R"doc(Performs backward operations for ldexp of :attr:`input_tensor_a` and :attr:`input_tensor_b` with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16)doc",
        R"doc(Recommended input range : [-80, 80]. Performance of the PCC may degrade if the input falls outside this range.)doc");

    detail::bind_binary_backward_ops(
        module,
        ttnn::logaddexp_bw,
        R"doc(Performs backward operations for logaddexp of :attr:`input_tensor_a` and :attr:`input_tensor_b` with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16)doc");

    detail::bind_binary_backward_ops(
        module,
        ttnn::logaddexp2_bw,
        R"doc(Performs backward operations for logaddexp2 of :attr:`input_tensor_a` and :attr:`input_tensor_b` with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16)doc");

    detail::bind_binary_backward_ops(
        module,
        ttnn::squared_difference_bw,
        R"doc(Performs backward operations for squared_difference of :attr:`input_tensor_a` and :attr:`input_tensor_b` with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_binary_backward_concat(
        module,
        ttnn::concat_bw,
        "dim",
        "Dimension to concatenate",
        0,
        R"doc(Performs backward operations for concat on :attr:`input_tensor_a` and :attr:`input_tensor_b` with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16)doc");

    detail::bind_binary_backward_rsub(
        module,
        ttnn::rsub_bw,
        R"doc(Performs backward operations for subraction of :attr:`input_tensor_a` from :attr:`input_tensor_b` with given :attr:`grad_tensor` (reversed order of subtraction operator).)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_binary_backward_ops(
        module,
        ttnn::min_bw,
        R"doc(Performs backward operations for minimum of :attr:`input_tensor_a` and :attr:`input_tensor_b` with given :attr:`grad_tensor`.)doc");

    detail::bind_binary_backward_ops(
        module,
        ttnn::max_bw,
        R"doc(Performs backward operations for maximum of :attr:`input_tensor_a` and :attr:`input_tensor_b` with given :attr:`grad_tensor`.)doc");

    detail::bind_binary_backward_bias_gelu(
        module,
        ttnn::bias_gelu_bw,
        "bias",
        "Bias value",
        "approximate",
        "Approximation type",
        "none",
        R"doc(Performs backward operations for bias_gelu on :attr:`input_tensor_a` and :attr:`input_tensor_b` or :attr:`input_tensor` and :attr:`bias`, with given :attr:`grad_tensor` using given :attr:`approximate` mode.
        :attr:`approximate` mode can be 'none', 'tanh'.)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");
}

}  // namespace binary_backward
}  // namespace operations
}  // namespace ttnn
