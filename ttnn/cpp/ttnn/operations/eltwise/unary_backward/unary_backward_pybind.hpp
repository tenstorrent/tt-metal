// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
#include "ttnn/operations/eltwise/binary_backward/binary_backward.hpp"
#include "ttnn/operations/eltwise/unary_backward/unary_backward.hpp"
#include "ttnn/types.hpp"

namespace py = pybind11;

namespace ttnn {
namespace operations {
namespace unary_backward {

namespace detail {

template <typename unary_backward_operation_t>
void bind_unary_backward_two_float(
    py::module& module,
    const unary_backward_operation_t& operation,
    const std::string_view description,
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& note = "") {
    auto doc = fmt::format(
        R"doc(

        {2}

        Args:
            grad_tensor (ttnn.Tensor): the input gradient tensor.
            input_tensor (ttnn.Tensor): the input tensor.
            threshold (float): the input threshold value.
            value (float): the input value.

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

            >>> grad_tensor = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> input = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device)
            >>> threshold = 1.0
            >>> value = 1.0
            >>> output = {1}(grad_tensor, input, threshold, value)
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
            [](const unary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor,
               float min,
               float max,
               const std::optional<MemoryConfig>& memory_config) {
                return self(grad_tensor, input_tensor, min, max, memory_config);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor"),
            py::arg("min"),
            py::arg("max"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt});
}

template <typename unary_backward_operation_t>
void bind_unary_backward_op(
    py::module& module,
    const unary_backward_operation_t& operation,
    const std::string& description,
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& layout = "TILE",
    const std::string& note = "") {
    auto doc = fmt::format(
        R"doc(
        {2}

        Args:
            grad_tensor (ttnn.Tensor): the input gradient tensor.
            input_tensor_a (ttnn.Tensor): the input tensor.

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
                 - {4}
                 - 2, 3, 4

            {5}

        Example:

            >>> grad_tensor = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> input = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device)
            >>> output = {1}(grad_tensor, input)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        description,
        supported_dtype,
        layout,
        note);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const unary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor,
               const std::optional<MemoryConfig>& memory_config) {
                return self(grad_tensor, input_tensor, memory_config);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt});
}

template <typename unary_backward_operation_t>
void bind_unary_backward_rsqrt(
    py::module& module,
    const unary_backward_operation_t& operation,
    const std::string_view description,
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& note = "") {
    auto doc = fmt::format(
        R"doc(

        {2}

        Args:
            grad_tensor (ttnn.Tensor): the input gradient tensor.
            input_tensor (ttnn.Tensor): the input tensor.

        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the operation. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): preallocated output tensor. Defaults to `None`.
            queue_id (uint8, optional): command queue id. Defaults to `0`.

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
            >>> input = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device)
            >>> output = {1}(grad_tensor, input)
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
            [](const unary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& input_grad,
               const uint8_t& queue_id) -> std::vector<std::optional<ttnn::Tensor>> {
                return self(queue_id, grad_tensor, input_tensor, memory_config, input_grad);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("input_grad") = std::nullopt,
            py::arg("queue_id") = ttnn::DefaultQueueId});
}

template <typename unary_backward_operation_t>
void bind_unary_backward_op_reciprocal(
    py::module& module,
    const unary_backward_operation_t& operation,
    const std::string& description,
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& note = "") {
    auto doc = fmt::format(
        R"doc(
        {2}

        Args:
            grad_tensor (ComplexTensor or ttnn.Tensor): the input gradient tensor.
            input_tensor_a (ComplexTensor or ttnn.Tensor): the input tensor.

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

            >>> grad_tensor = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> input = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device)
            >>> output = {1}(grad_tensor, input)
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
            [](const unary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor,
               const std::optional<MemoryConfig>& memory_config) {
                return self(grad_tensor, input_tensor, memory_config);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt},

        ttnn::pybind_overload_t{
            [](const unary_backward_operation_t& self,
               const ComplexTensor& grad_tensor,
               const ComplexTensor& input_tensor,
               const MemoryConfig& memory_config) { return self(grad_tensor, input_tensor, memory_config); },
            py::arg("grad_tensor"),
            py::arg("input_tensor"),
            py::kw_only(),
            py::arg("memory_config")});
}

template <typename unary_backward_operation_t>
void bind_unary_backward_op_overload_abs(
    py::module& module,
    const unary_backward_operation_t& operation,
    const std::string_view description,
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& note = "") {
    auto doc = fmt::format(
        R"doc(
        {2}

        Args:
            grad_tensor (ttnn.Tensor): the input gradient tensor.
            input_tensor (ComplexTensor or ttnn.Tensor): the input tensor.

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

            >>> grad_tensor = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> input = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device)
            >>> output = {1}(grad_tensor, input)
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
            [](const unary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor,
               const std::optional<MemoryConfig>& memory_config) {
                return self(grad_tensor, input_tensor, memory_config);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt},

        ttnn::pybind_overload_t{
            [](const unary_backward_operation_t& self,
               const Tensor& grad_tensor,
               const ComplexTensor& input_tensor,
               const MemoryConfig& memory_config) { return self(grad_tensor, input_tensor, memory_config); },
            py::arg("grad_tensor"),
            py::arg("input_tensor"),
            py::kw_only(),
            py::arg("memory_config")});
}

template <typename unary_backward_operation_t>
void bind_unary_backward_float(
    py::module& module,
    const unary_backward_operation_t& operation,
    const std::string& description,
    const std::string& parameter_name_a,
    const std::string& parameter_a_doc,
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& note = "") {
    auto doc = fmt::format(
        R"doc(
        {2}

        Args:
            grad_tensor (ttnn.Tensor): the input gradient tensor.
            input_tensor (ttnn.Tensor): the input tensor.
            {3} (float): {4}.

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
               * - {5}
                 - TILE
                 - 2, 3, 4

            {6}

        Example:

            >>> grad_tensor = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device=device)
            >>> input = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device=device)
            >>> output = {1}(grad_tensor, input, float)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        description,
        parameter_name_a,
        parameter_a_doc,
        supported_dtype,
        note);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const unary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor,
               float parameter_a,
               const std::optional<MemoryConfig>& memory_config) {
                return self(grad_tensor, input_tensor, parameter_a, memory_config);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor"),
            py::arg(parameter_name_a.c_str()),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt}

    );
}

template <typename unary_backward_operation_t>
void bind_unary_backward_two_float_with_default(
    py::module& module,
    const unary_backward_operation_t& operation,
    const std::string& parameter_name_a,
    const std::string& parameter_a_doc,
    float parameter_a_value,
    const std::string& parameter_name_b,
    const std::string& parameter_b_doc,
    float parameter_b_value,
    const std::string_view description,
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& note = "") {
    auto doc = fmt::format(
        R"doc(
        {8}

        Args:
            grad_tensor (ttnn.Tensor): the input gradient tensor.
            input_tensor (ttnn.Tensor): the input tensor.

        Keyword args:
            {2} (float, optional): {3}. Defaults to `{4}`.
            {5} (float, optional): {6}. Defaults to `{7}`.
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
               * - {9}
                 - TILE
                 - 2, 3, 4

            {10}

        Example:

            >>> grad_tensor = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> input = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device)
            >>> output = {1}(grad_tensor, input, {2} = {4}, {5} = {7}
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        parameter_name_a,
        parameter_a_doc,
        parameter_a_value,
        parameter_name_b,
        parameter_b_doc,
        parameter_b_value,
        description,
        supported_dtype,
        note);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const unary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor,
               float parameter_a,
               float parameter_b,
               const std::optional<MemoryConfig>& memory_config) {
                return self(grad_tensor, input_tensor, parameter_a, parameter_b, memory_config);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor"),
            py::kw_only(),
            py::arg(parameter_name_a.c_str()) = parameter_a_value,
            py::arg(parameter_name_b.c_str()) = parameter_b_value,
            py::arg("memory_config") = std::nullopt});
}

template <typename unary_backward_operation_t>
void bind_unary_backward_float_with_default(
    py::module& module,
    const unary_backward_operation_t& operation,
    const std::string& parameter_name_a,
    const std::string& parameter_a_doc,
    float parameter_a_value,
    const std::string& description,
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& note = "") {
    auto doc = fmt::format(
        R"doc(
        {5}

        Args:
            grad_tensor (ttnn.Tensor): the input gradient tensor.
            input_tensor_a (ttnn.Tensor): the input tensor.


        Keyword args:
            {2} (float, optional): {3}. Defaults to `{4}`.
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
               * - {6}
                 - TILE
                 - 2, 3, 4

            {7}

        Example:

            >>> grad_tensor = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> input = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device)
            >>> output = {1}(grad_tensor, input, {2} = {4})
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        parameter_name_a,
        parameter_a_doc,
        parameter_a_value,
        description,
        supported_dtype,
        note);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const unary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor,
               float parameter_a,
               const std::optional<MemoryConfig>& memory_config) {
                return self(grad_tensor, input_tensor, parameter_a, memory_config);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor"),
            py::kw_only(),
            py::arg(parameter_name_a.c_str()) = parameter_a_value,
            py::arg("memory_config") = std::nullopt});
}

template <typename unary_backward_operation_t>
void bind_unary_backward_optional_float_params_with_default(
    py::module& module,
    const unary_backward_operation_t& operation,
    const std::string& parameter_name_a,
    const std::string& parameter_a_doc,
    std::optional<float> parameter_a_value,
    const std::string& parameter_name_b,
    const std::string& parameter_b_doc,
    std::optional<float> parameter_b_value,
    const std::string& description,
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& note = "") {
    auto doc = fmt::format(
        R"doc(
        {8}

        Args:
            grad_tensor (ttnn.Tensor): the input gradient tensor.
            input_tensor (ttnn.Tensor): the input tensor.

        Keyword args:
            {2} (float, optional): {3}. Defaults to `None`.
            {5} (float, optional): {6}. Defaults to `None`.
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
               * - {9}
                 - TILE
                 - 2, 3, 4

            {10}


        Example:

            >>> grad_tensor = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> input = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device)
            >>> {2} = 0.5
            >>> {5} = 2.0
            >>> output = {1}(grad_tensor, input, {2}, {5})
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        parameter_name_a,
        parameter_a_doc,
        parameter_a_value,
        parameter_name_b,
        parameter_b_doc,
        parameter_b_value,
        description,
        supported_dtype,
        note);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const unary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor,
               std::optional<float> parameter_a,
               std::optional<float> parameter_b,
               const std::optional<MemoryConfig>& memory_config) {
                return self(grad_tensor, input_tensor, parameter_a, parameter_b, memory_config);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor"),
            py::arg(parameter_name_a.c_str()) = parameter_a_value,
            py::arg(parameter_name_b.c_str()) = parameter_b_value,
            py::kw_only(),
            py::arg("memory_config") = std::nullopt},

        ttnn::pybind_overload_t{
            [](const unary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor,
               std::optional<Tensor> parameter_a,
               std::optional<Tensor> parameter_b,
               const std::optional<MemoryConfig>& memory_config) {
                return self(grad_tensor, input_tensor, parameter_a, parameter_b, memory_config);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor"),
            py::arg(parameter_name_a.c_str()) = parameter_a_value,
            py::arg(parameter_name_b.c_str()) = parameter_b_value,
            py::kw_only(),
            py::arg("memory_config") = std::nullopt});
}

template <typename unary_backward_operation_t>
void bind_unary_backward_rdiv(
    py::module& module,
    const unary_backward_operation_t& operation,
    const std::string& parameter_name_a,
    const std::string& parameter_a_doc,
    const std::string& parameter_name_b,
    const std::string& parameter_b_doc,
    string parameter_b_value,
    const std::string_view description,
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& note = "") {
    auto doc = fmt::format(
        R"doc(
        {7}

        Args:
            grad_tensor (ttnn.Tensor): the input gradient tensor.
            input_tensor_a (ttnn.Tensor): the input tensor.
            {2} (float): {3}.

        Keyword args:
            {4} (string, optional): {5}. Defaults to None.
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
               * - {8}
                 - TILE
                 - 2, 3, 4

            {9}

        Example:

            >>> grad_tensor = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> input = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device)
            >>> {2} = 0.5
            >>> output = {1}(grad_tensor, input, {2}, {4} = None)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        parameter_name_a,
        parameter_a_doc,
        parameter_name_b,
        parameter_b_doc,
        parameter_b_value,
        description,
        supported_dtype,
        note);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const unary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor,
               float parameter_a,
               const std::optional<string> parameter_b,
               const std::optional<MemoryConfig>& memory_config) {
                return self(grad_tensor, input_tensor, parameter_a, parameter_b, memory_config);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor"),
            py::arg(parameter_name_a.c_str()),
            py::kw_only(),
            py::arg(parameter_name_b.c_str()) = std::nullopt,
            py::arg("memory_config") = std::nullopt});
}

template <typename unary_backward_operation_t>
void bind_unary_backward_unary_optional_float(
    py::module& module,
    const unary_backward_operation_t& operation,
    const std::string& parameter_name,
    const std::string& parameter_doc,
    const std::string_view description,
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& note = "") {
    auto doc = fmt::format(
        R"doc(
        {4}

        Args:
            grad_tensor (ttnn.Tensor): the input gradient tensor.
            input_tensor_a (ttnn.Tensor): the input tensor.
            {2} (Number): {3}.

        Keyword args:
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
               * - {5}
                 - TILE
                 - 2, 3, 4

            {6}

        Example:

            >>> grad_tensor = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> input = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device)
            >>> output = {1}(grad_tensor, tensor, {2})
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        parameter_name,
        parameter_doc,
        description,
        supported_dtype,
        note);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const unary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor,
               float parameter,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& input_grad,
               const uint8_t& queue_id) -> std::vector<std::optional<ttnn::Tensor>> {
                return self(queue_id, grad_tensor, input_tensor, parameter, memory_config, input_grad);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor"),
            py::arg(parameter_name.c_str()),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("input_grad") = std::nullopt,
            py::arg("queue_id") = 0});
}

template <typename unary_backward_operation_t>
void bind_unary_backward_shape(
    py::module& module,
    const unary_backward_operation_t& operation,
    const std::string& parameter_name_a,
    const std::string& parameter_a_doc,
    const std::string_view description,
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& note = "") {
    auto doc = fmt::format(
        R"doc(
        {4}

        Args:
            grad_tensor (ttnn.Tensor): the input gradient tensor.
            input_tensor (ttnn.Tensor): the input tensor.
            {2} (List[int]): {3}.

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
               * - {5}
                 - TILE
                 - 4

            {6}

        Example:

            >>> grad_tensor = ttnn.from_torch(torch.rand([1, 1, 32, 32], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> input = ttnn.from_torch(torch.rand([2, 1, 32, 32], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device)
            >>> output = {1}(grad_tensor, input, {2})
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        parameter_name_a,
        parameter_a_doc,
        description,
        supported_dtype,
        note);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const unary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor,
               const tt::tt_metal::LegacyShape& parameter_a,
               const std::optional<MemoryConfig>& memory_config) {
                return self(grad_tensor, input_tensor, parameter_a, memory_config);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor"),
            py::arg(parameter_name_a.c_str()),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt});
}

template <typename unary_backward_operation_t>
void bind_unary_backward_optional(
    py::module& module,
    const unary_backward_operation_t& operation,
    const std::string_view description,
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& layout = "TILE",
    const std::string& note = "") {
    auto doc = fmt::format(
        R"doc(
        {2}

        Args:
            grad_tensor (ttnn.Tensor): the input gradient tensor.
            input_tensor (ttnn.Tensor): the input tensor.

        Keyword args:
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
                 - {4}
                 - 2, 3, 4

            {5}

        Example:

            >>> grad_tensor = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> input = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device)
            >>> output = {1}(grad_tensor, tensor)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        description,
        supported_dtype,
        layout,
        note);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const unary_backward_operation_t& self,
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
            py::arg("input_grad") = std::nullopt,
            py::arg("queue_id") = ttnn::DefaultQueueId});
}

template <typename unary_backward_operation_t>
void bind_unary_backward_neg(
    py::module& module,
    const unary_backward_operation_t& operation,
    const std::string_view description,
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& note = "") {
    auto doc = fmt::format(
        R"doc(
        {2}

        Args:
            grad_tensor (ttnn.Tensor): the input gradient tensor.
            input_tensor (ttnn.Tensor): the input tensor.

        Keyword args:
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

            {4}

        Example:

            >>> grad_tensor = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> input = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device)
            >>> output = {1}(grad_tensor, tensor)
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
            [](const unary_backward_operation_t& self,
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
            py::arg("input_grad") = std::nullopt,
            py::arg("queue_id") = ttnn::DefaultQueueId});
}

template <typename unary_backward_operation_t>
void bind_unary_backward_prod_bw(py::module& module, const unary_backward_operation_t& operation) {
    auto doc = fmt::format(
        R"doc(
        Performs backward operations for prod on :attr:`input_tensor` with given :attr:`grad_tensor` along `all_dimensions` or a particular `dim`.

        Args:
            grad_tensor (ttnn.Tensor): the input gradient tensor.
            input_tensor (ttnn.Tensor): the input tensor.

        Keyword args:
            all_dimensions (bool, optional): perform prod backward along all dimensions, ignores dim param. Defaults to `True`.
            dim (int, optional): dimension to perform prod backward. Defaults to `0`.
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
               * - BFLOAT16
                 - TILE
                 - 4

        Example:

            >>> grad_tensor = ttnn.from_torch(torch.rand([1, 1, 32, 32], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> input = ttnn.from_torch(torch.rand([1, 1, 32, 32], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device)
            >>> all_dimensions = True
            >>> dim =0
            >>> output = {1}(grad_tensor, input, all_dimensions, dim)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name());

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const unary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor,
               bool all_dimensions,
               int64_t dim,
               const std::optional<MemoryConfig>& memory_config) {
                return self(grad_tensor, input_tensor, all_dimensions, dim, memory_config);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor"),
            py::kw_only(),
            py::arg("all_dimensions") = true,
            py::arg("dim") = 0,
            py::arg("memory_config") = std::nullopt});
}

template <typename unary_backward_operation_t>
void bind_unary_backward_gelu(
    py::module& module,
    const unary_backward_operation_t& operation,
    const std::string& parameter_name_a,
    const std::string& parameter_a_doc,
    string parameter_a_value,
    const std::string_view description,
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& note = "") {
    auto doc = fmt::format(
        R"doc(
        {5}

        Args:
            grad_tensor (ttnn.Tensor): the input gradient tensor.
            input_tensor (ttnn.Tensor): the input tensor.

        Keyword args:
            {2} (string): {3}. Defaults to `{4}`.
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the operation. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): preallocated output tensor. Defaults to `None`.
            queue_id (uint8, optional): command queue id. Defaults to `0`.

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

            {7}

        Example:

            >>> grad_tensor = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> input = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True), layout=ttnn.TILE_LAYOUT, device=device)
            >>> output = {1}(grad_tensor, input, {2} = {4})
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        parameter_name_a,
        parameter_a_doc,
        parameter_a_value,
        description,
        supported_dtype,
        note);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const unary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor,
               string parameter_a,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& input_grad,
               const uint8_t& queue_id) -> std::vector<std::optional<ttnn::Tensor>> {
                return self(queue_id, grad_tensor, input_tensor, parameter_a, memory_config, input_grad);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor"),
            py::kw_only(),
            py::arg(parameter_name_a.c_str()) = parameter_a_value,
            py::arg("memory_config") = std::nullopt,
            py::arg("input_grad") = std::nullopt,
            py::arg("queue_id") = ttnn::DefaultQueueId});
}

}  // namespace detail

void py_module(py::module& module) {
    detail::bind_unary_backward_optional_float_params_with_default(
        module,
        ttnn::clamp_bw,
        "min",
        "Minimum value",
        std::nullopt,
        "max",
        "Maximum value",
        std::nullopt,
        R"doc(Performs backward operations for clamp on :attr:`input_tensor`, :attr:`min`, :attr:`max` with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16)doc",
        R"doc(Only one of `min` or `max` value can be `None`.)doc");

    detail::bind_unary_backward_optional_float_params_with_default(
        module,
        ttnn::clip_bw,
        "min",
        "Minimum value",
        std::nullopt,
        "max",
        "Maximum value",
        std::nullopt,
        R"doc(Performs backward operations for clip on :attr:`input_tensor`, :attr:`min`, :attr:`max` with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16)doc",
        R"doc(Only one of `min` or `max` value can be `None`.)doc");

    detail::bind_unary_backward_two_float_with_default(
        module,
        ttnn::hardtanh_bw,
        "min",
        "Minimum value",
        -1.0,
        "max",
        "Maximum value",
        1.0,
        R"doc(Performs backward operations for hardtanh activation function on :attr:`input_tensor`, :attr:`min`, :attr:`max` with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_unary_backward_float_with_default(
        module,
        ttnn::hardshrink_bw,
        "lambd",
        "Lambda value for the hardshrink formula ",
        0.5,
        R"doc(Performs backward operations for hardshrink on :attr:`input_tensor`, :attr:`lambd`, with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_unary_backward_float_with_default(
        module,
        ttnn::softshrink_bw,
        "lambd",
        "Lambda value for the softshrink formula ",
        0.5,
        R"doc(Performs backward operations for softshrink on :attr:`input_tensor`, :attr:`lambd`, with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_unary_backward_float_with_default(
        module,
        ttnn::leaky_relu_bw,
        "negative_slope",
        "negative_slope value for the hardshrink formula ",
        0.01,
        R"doc(Performs backward operations for leaky_relu on :attr:`input_tensor`, :attr:`negative_slope`, with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_unary_backward_float_with_default(
        module,
        ttnn::elu_bw,
        "alpha",
        "alpha value for the elu formula ",
        1.0,
        R"doc(Performs backward operations for elu on :attr:`input_tensor`, :attr:`alpha`, with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward_float_with_default(
        module,
        ttnn::celu_bw,
        "alpha",
        "alpha value for the celu formula ",
        1.0,
        R"doc(Performs backward operations for celu on :attr:`input_tensor`, :attr:`alpha`, with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward_float_with_default(
        module,
        ttnn::logiteps_bw,
        "eps",
        "eps value for the logiteps formula ",
        0.0,
        R"doc(Performs backward operations for logiteps on :attr:`input_tensor`, :attr:`eps`, with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward_two_float(
        module,
        ttnn::threshold_bw,
        R"doc(Performs backward operations for threshold on :attr:`input_tensor`, :attr:`threshold`, :attr:`value` with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward_two_float_with_default(
        module,
        ttnn::softplus_bw,
        "beta",
        "Beta value for the Softplus formula ",
        1.0,
        "threshold",
        "Threshold value",
        20.0,
        R"doc(Performs backward operations for softplus on :attr:`input_tensor`, :attr:`beta`, :attr:`threshold` with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward_rdiv(
        module,
        ttnn::rdiv_bw,
        "scalar",
        "divisor",
        "round_mode",
        "Mode of Rounding",
        "None",
        R"doc(Performs backward operations for Unary rdiv on :attr:`input_tensor`, :attr:`scalar` with given :attr:`grad_tensor` using given :attr:`round_mode`.
        :attr:`round_mode` can be 'None', 'trunc', or 'floor'.)doc");

    detail::bind_unary_backward_shape(
        module,
        ttnn::repeat_bw,
        "shape",
        "Shape of tensor",
        R"doc(Performs backward operations for repeat on :attr:`input_tensor`, with given :attr:`grad_tensor` using given :attr:`shape`.)doc");

    detail::bind_unary_backward_gelu(
        module,
        ttnn::gelu_bw,
        "approximate",
        "Approximation type",
        "none",
        R"doc(Performs backward operations for gelu on :attr:`input_tensor`, with given :attr:`grad_tensor` using given :attr:`approximate` mode.
        :attr:`approximate` mode can be 'none', 'tanh'.)doc");

    detail::bind_unary_backward_unary_optional_float(
        module,
        ttnn::pow_bw,
        "exponent",
        "Exponent value [must be non-negative]",
        R"doc(Performs backward operations for power on :attr:`input_tensor`, :attr:`exponent` with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_unary_backward_optional(
        module,
        ttnn::exp_bw,
        R"doc(Performs backward operations for exponential function on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_unary_backward_optional(
        module,
        ttnn::tanh_bw,
        R"doc(Performs backward operations for hyperbolic tangent (tanh) function on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward_optional(
        module,
        ttnn::sqrt_bw,
        R"doc(Performs backward operations for square-root on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::multigammaln_bw,
        R"doc(Performs backward operations for multivariate logarithmic gamma function (also referred to as mvlgamma) on :attr:`input_tensor` with given :attr:`grad_tensor`.
        The dimensionality is set to 4.)doc",
        R"doc(BFLOAT16)doc",
        R"doc(TILE)doc",
        R"doc(Input value must be greater than 2.5f)doc");

    detail::bind_unary_backward_prod_bw(module, ttnn::prod_bw);

    detail::bind_unary_backward_op(
        module,
        ttnn::lgamma_bw,
        R"doc(Performs backward operations for lgamma on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward_optional(
        module,
        ttnn::fill_bw,
        R"doc(Performs backward operations for fill on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16)doc",
        R"doc(TILE, ROW MAJOR)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::hardsigmoid_bw,
        R"doc(Performs backward operations for hardsigmoid on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::cos_bw,
        R"doc(Performs backward operations for cosine on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::acosh_bw,
        R"doc(Performs backward operations for inverse hyperbolic cosine (acosh) on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::acos_bw,
        R"doc(Performs backward operations for inverse cosine (acos) on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::atan_bw,
        R"doc(Performs backward operations for inverse tangenr (atan) on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::rad2deg_bw,
        R"doc(Performs backward operations for radian to degree conversion (rad2deg) on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::frac_bw,
        R"doc(Performs backward operations for frac on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16)doc",
        R"doc(TILE, ROW MAJOR)doc",
        R"doc(BFLOAT8_B is supported for TILE layout.)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::trunc_bw,
        R"doc(Performs backward operations for truncation on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16)doc",
        R"doc(TILE, ROW MAJOR)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::log_sigmoid_bw,
        R"doc(Performs backward operations for log sigmoid on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::fill_zero_bw,
        R"doc(Performs backward operations for fill zero on :attr:`input_tensor` with given :attr:`grad_tensor`. Returns an tensor of zeros like :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16)doc",
        R"doc(TILE, ROW MAJOR)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::i0_bw,
        R"doc(Performs backward operations for i0 on :attr:`input_tensor` with given :attr:`grad_tensor`. Supported input range is (-10, 10))doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::tan_bw,
        R"doc(Performs backward operations for tan on :attr:`input_tensor` with given :attr:`grad_tensor`. Supported input range is (-1.45, 1.45))doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::sigmoid_bw,
        R"doc(Performs backward operations for sigmoid on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_unary_backward_rsqrt(
        module,
        ttnn::rsqrt_bw,
        R"doc(Performs backward operations for reciprocal of square-root on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_unary_backward_neg(
        module,
        ttnn::neg_bw,
        R"doc(Performs backward operations for neg on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::relu_bw,
        R"doc(Performs backward operations for relu on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::logit_bw,
        R"doc(Performs backward operations for logit on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::floor_bw,
        R"doc(Performs backward operations for floor on :attr:`input_tensor` with given :attr:`grad_tensor`)doc",
        R"doc(BFLOAT16)doc",
        R"doc(TILE, ROW MAJOR)doc");

    detail::bind_unary_backward_float(
        module,
        ttnn::rpow_bw,
        R"doc(Performs backward operations for rpow on :attr:`input_tensor`, :attr:`exponent` with given :attr:`grad_tensor`.)doc",
        "exponent",
        "Exponent value",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::round_bw,
        R"doc(Performs backward operations for round on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16)doc",
        R"doc(TILE, ROW MAJOR)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::log_bw,
        R"doc(Performs backward operations for logarithm on :attr:`input_tensor` with given :attr:`grad_tensor`)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::relu6_bw,
        R"doc(Performs backward operations for relu6 on :attr:`input_tensor` with given :attr:`grad_tensor`)doc");

    detail::bind_unary_backward_op_overload_abs(
        module,
        ttnn::abs_bw,
        R"doc(Performs backward operations for abs on :attr:`input_tensor` with given :attr:`grad_tensor`)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_unary_backward_neg(
        module,
        ttnn::silu_bw,
        R"doc(Performs backward operations for silu on :attr:`input_tensor` with given :attr:`grad_tensor`)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::selu_bw,
        R"doc(Performs backward operations for selu on :attr:`input_tensor` with given :attr:`grad_tensor`)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::square_bw,
        R"doc(Performs backward operations for square on :attr:`input_tensor` with given :attr:`grad_tensor`)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::hardswish_bw,
        R"doc(Performs backward operations for  hardswish on :attr:`input_tensor` with given :attr:`grad_tensor`)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::tanhshrink_bw,
        R"doc(Performs backward operations for  tanhshrink on :attr:`input_tensor` with given :attr:`grad_tensor`)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::atanh_bw,
        R"doc(Performs backward operations for inverse hyperbolic tangent (atanh) on :attr:`input_tensor` with given :attr:`grad_tensor`)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::asin_bw,
        R"doc(Performs backward operations for inverse sine (asin) on :attr:`input_tensor` with given :attr:`grad_tensor`)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::asinh_bw,
        R"doc(Performs backward operations for inverse hyperbolic sine (asinh) on :attr:`input_tensor` with given :attr:`grad_tensor`)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::sin_bw,
        R"doc(Performs backward operations for sin on :attr:`input_tensor` with given :attr:`grad_tensor`)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::sinh_bw,
        R"doc(Performs backward operations for hyperbolic sine (sinh) on :attr:`input_tensor` with given :attr:`grad_tensor`)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::log10_bw,
        R"doc(Performs backward operations for log10 on :attr:`input_tensor` with given :attr:`grad_tensor`)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::log1p_bw,
        R"doc(Performs backward operations for log1p on :attr:`input_tensor` with given :attr:`grad_tensor`)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::erfc_bw,
        R"doc(Performs backward operations for erfc on :attr:`input_tensor` with given :attr:`grad_tensor`)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::ceil_bw,
        R"doc(Performs backward operations for ceil on :attr:`input_tensor` with given :attr:`grad_tensor`)doc",
        R"doc(BFLOAT16)doc",
        R"doc(TILE, ROW MAJOR)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::softsign_bw,
        R"doc(Performs backward operations for softsign on :attr:`input_tensor` with given :attr:`grad_tensor`)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::cosh_bw,
        R"doc(Performs backward operations for hyperbolic cosine (cosh) on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::log2_bw,
        R"doc(Performs backward operations for log2 on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::sign_bw,
        R"doc(Performs backward operations for sign on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16)doc",
        R"doc(TILE, ROW MAJOR)doc");

    detail::bind_unary_backward_float(
        module,
        ttnn::div_no_nan_bw,
        R"doc(Performs backward operations for div_no_nan on :attr:`input_tensor`, :attr:`scalar` with given :attr:`grad_tensor`.)doc",
        "scalar",
        "Denominator value",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::exp2_bw,
        R"doc(Performs backward operations for exp2 on :attr:`input_tensor` with given :attr:`grad_tensor`)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::expm1_bw,
        R"doc(Performs backward operations for expm1 on :attr:`input_tensor` with given :attr:`grad_tensor`)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_unary_backward_op_reciprocal(
        module,
        ttnn::reciprocal_bw,
        R"doc(Performs backward operations for reciprocal on :attr:`input_tensor` with given :attr:`grad_tensor`)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::digamma_bw,
        R"doc(Performs backward operations for digamma on :attr:`input_tensor` with given :attr:`grad_tensor`)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::erfinv_bw,
        R"doc(Performs backward operations for erfinv on :attr:`input_tensor` with given :attr:`grad_tensor`)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::erf_bw,
        R"doc(Performs backward operations for erf on :attr:`input_tensor` with given :attr:`grad_tensor`)doc");

    detail::bind_unary_backward_op(
        module,
        ttnn::deg2rad_bw,
        R"doc(Performs backward operations for degree to radian conversion (deg2rad) on :attr:`input_tensor` with given :attr:`grad_tensor`)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_unary_backward_float(
        module,
        ttnn::polygamma_bw,
        R"doc(Performs backward operations for polygamma on :attr:`input_tensor`, :attr:`scalar` with given :attr:`grad_tensor`.)doc",
        "n",
        "Order of polygamma function");
}

}  // namespace unary_backward
}  // namespace operations
}  // namespace ttnn
