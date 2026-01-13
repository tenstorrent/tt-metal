// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "binary_backward_nanobind.hpp"

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <fmt/format.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/common/constants.hpp"
#include "ttnn/operations/eltwise/binary_backward/binary_backward.hpp"
#include "ttnn/operations/eltwise/ternary_backward/ternary_backward.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::binary_backward {

namespace {

template <typename binary_backward_operation_t>
void bind_binary_backward_ops(
    nb::module_& mod,
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
        )doc",

        operation.base_name(),
        operation.python_fully_qualified_name(),
        description,
        supported_dtype,
        note);

    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const binary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor_a,
               const ttnn::Tensor& input_tensor_b,
               const std::optional<ttnn::MemoryConfig>& memory_config) -> std::vector<ttnn::Tensor> {
                auto output_memory_config = memory_config.value_or(input_tensor_a.memory_config());
                return self(grad_tensor, input_tensor_a, input_tensor_b, output_memory_config);
            },
            nb::arg("grad_tensor"),
            nb::arg("input_tensor_a"),
            nb::arg("input_tensor_b"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()});
}

template <typename binary_backward_operation_t>
void bind_binary_backward_concat(
    nb::module_& mod,
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
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        parameter_name,
        parameter_doc,
        parameter_value,
        description,
        supported_dtype);

    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const binary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor_a,
               const ttnn::Tensor& input_tensor_b,
               int parameter,
               const std::vector<bool>& are_required_outputs,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& input_grad,
               const std::optional<ttnn::Tensor>& other_grad) -> std::vector<std::optional<ttnn::Tensor>> {
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
            nb::arg("grad_tensor"),
            nb::arg("input_tensor_a"),
            nb::arg("input_tensor_b"),
            nb::arg(parameter_name.c_str()) = parameter_value,
            nb::kw_only(),
            nb::arg("are_required_outputs") = std::vector<bool>{true, true},
            nb::arg("memory_config") = nb::none(),
            nb::arg("input_a_grad") = nb::none(),
            nb::arg("input_b_grad") = nb::none()});
}

template <typename binary_backward_operation_t>
void bind_binary_backward_addalpha(
    nb::module_& mod,
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
        )doc",

        operation.base_name(),
        operation.python_fully_qualified_name(),
        parameter_name,
        parameter_doc,
        parameter_value,
        description,
        supported_dtype);

    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const binary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor_a,
               const ttnn::Tensor& input_tensor_b,
               float parameter,
               const std::vector<bool>& are_required_outputs,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& input_a_grad,
               const std::optional<ttnn::Tensor>& input_b_grad) -> std::vector<std::optional<ttnn::Tensor>> {
                return self(
                    grad_tensor,
                    input_tensor_a,
                    input_tensor_b,
                    parameter,
                    are_required_outputs,
                    memory_config,
                    input_a_grad,
                    input_b_grad);
            },
            nb::arg("grad_tensor"),
            nb::arg("input_tensor_a"),
            nb::arg("input_tensor_b"),
            nb::arg(parameter_name.c_str()) = parameter_value,
            nb::kw_only(),
            nb::arg("are_required_outputs") = std::vector<bool>{true, true},
            nb::arg("memory_config") = nb::none(),
            nb::arg("input_a_grad") = nb::none(),
            nb::arg("input_b_grad") = nb::none()});
}

template <typename binary_backward_operation_t>
void bind_binary_backward_bias_gelu(
    nb::module_& mod,
    const binary_backward_operation_t& operation,
    const std::string& parameter_name_a,
    const std::string& parameter_a_doc,
    const std::string& parameter_name_b,
    const std::string& parameter_b_doc,
    std::string parameter_b_value,
    const std::string& description,
    const std::string& supported_dtype = "BFLOAT16",
    const std::string_view note = "") {
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

            {9}

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
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const binary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor_a,
               const ttnn::Tensor& input_tensor_b,
               std::string parameter_b,
               const std::optional<MemoryConfig>& memory_config) {
                return self(grad_tensor, input_tensor_a, input_tensor_b, parameter_b, memory_config);
            },
            nb::arg("grad_tensor"),
            nb::arg("input_tensor_a"),
            nb::arg("input_tensor_b"),
            nb::kw_only(),
            nb::arg(parameter_name_b.c_str()) = parameter_b_value,
            nb::arg("memory_config") = nb::none()},

        ttnn::nanobind_overload_t{
            [](const binary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor,
               float parameter_a,
               std::string parameter_b,
               const std::optional<MemoryConfig>& memory_config) {
                return self(grad_tensor, input_tensor, parameter_a, parameter_b, memory_config);
            },
            nb::arg("grad_tensor"),
            nb::arg("input_tensor"),
            nb::arg(parameter_name_a.c_str()),
            nb::kw_only(),
            nb::arg(parameter_name_b.c_str()) = parameter_b_value,
            nb::arg("memory_config") = nb::none()});
}

template <typename binary_backward_operation_t>
void bind_binary_backward_sub_alpha(
    nb::module_& mod,
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
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        parameter_name,
        parameter_doc,
        parameter_value,
        description,
        supported_dtype);

    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const binary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor,
               const ttnn::Tensor& other_tensor,
               float alpha,
               const std::vector<bool>& are_required_outputs,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& input_grad,
               const std::optional<ttnn::Tensor>& other_grad) -> std::vector<std::optional<ttnn::Tensor>> {
                return self(
                    grad_tensor,
                    input_tensor,
                    other_tensor,
                    alpha,
                    are_required_outputs,
                    memory_config,
                    input_grad,
                    other_grad);
            },
            nb::arg("grad_tensor"),
            nb::arg("input_tensor_a"),
            nb::arg("input_tensor_b"),
            nb::arg(parameter_name.c_str()) = parameter_value,
            nb::kw_only(),
            nb::arg("are_required_outputs") = std::vector<bool>{true, true},
            nb::arg("memory_config") = nb::none(),
            nb::arg("input_grad") = nb::none(),
            nb::arg("other_grad") = nb::none()});
}

template <typename binary_backward_operation_t>
void bind_binary_backward_rsub(
    nb::module_& mod,
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
            [](const binary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor,
               const ttnn::Tensor& other_tensor,
               const std::vector<bool>& are_required_outputs,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& input_grad,
               const std::optional<ttnn::Tensor>& other_grad) -> std::vector<std::optional<ttnn::Tensor>> {
                return self(
                    grad_tensor,
                    input_tensor,
                    other_tensor,
                    are_required_outputs,
                    memory_config,
                    input_grad,
                    other_grad);
            },
            nb::arg("grad_tensor"),
            nb::arg("input_tensor_a"),
            nb::arg("input_tensor_b"),
            nb::kw_only(),
            nb::arg("are_required_outputs") = std::vector<bool>{true, true},
            nb::arg("memory_config") = nb::none(),
            nb::arg("input_grad") = nb::none(),
            nb::arg("other_grad") = nb::none()});
}

template <typename binary_backward_operation_t>
void bind_binary_bw_mul(
    nb::module_& mod,
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
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        description,
        supported_dtype);

    bind_registered_operation(
        mod,
        operation,
        doc,
        // tensor and scalar
        ttnn::nanobind_overload_t{
            [](const binary_backward_operation_t& self,
               const Tensor& grad_tensor,
               const Tensor& input_tensor_a,
               const float scalar,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& input_grad) -> std::vector<std::optional<ttnn::Tensor>> {
                return self(grad_tensor, input_tensor_a, scalar, memory_config, input_grad);
            },
            nb::arg("grad_tensor"),
            nb::arg("input_tensor_a"),
            nb::arg("scalar"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("input_grad") = nb::none()},

        // tensor and tensor
        ttnn::nanobind_overload_t{
            [](const binary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor,
               const ttnn::Tensor& other_tensor,
               const std::vector<bool>& are_required_outputs,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& input_grad,
               const std::optional<ttnn::Tensor>& other_grad) -> std::vector<std::optional<ttnn::Tensor>> {
                return self(
                    grad_tensor,
                    input_tensor,
                    other_tensor,
                    are_required_outputs,
                    memory_config,
                    input_grad,
                    other_grad);
            },
            nb::arg("grad_tensor"),
            nb::arg("input_tensor"),
            nb::arg("other_tensor"),
            nb::kw_only(),
            nb::arg("are_required_outputs") = std::vector<bool>{true, true},
            nb::arg("memory_config") = nb::none(),
            nb::arg("input_grad") = nb::none(),
            nb::arg("other_grad") = nb::none()},

        // complex tensor
        ttnn::nanobind_overload_t{
            [](const binary_backward_operation_t& self,
               const ComplexTensor& grad_tensor,
               const ComplexTensor& input_tensor_a,
               const ComplexTensor& input_tensor_b,
               const MemoryConfig& memory_config) {
                return self(grad_tensor, input_tensor_a, input_tensor_b, memory_config);
            },
            nb::arg("grad_tensor"),
            nb::arg("input_tensor_a"),
            nb::arg("input_tensor_b"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()});
}

template <typename binary_backward_operation_t>
void bind_binary_bw(
    nb::module_& mod,
    const binary_backward_operation_t& operation,
    const std::string_view description,
    const std::string_view supported_dtype = "BFLOAT16",
    const std::string_view note = "") {
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
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        description,
        supported_dtype,
        note);

    bind_registered_operation(
        mod,
        operation,
        doc,
        // tensor and scalar
        ttnn::nanobind_overload_t{
            [](const binary_backward_operation_t& self,
               const Tensor& grad_tensor,
               const Tensor& input_tensor_a,
               const float scalar,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& input_grad) -> std::vector<std::optional<ttnn::Tensor>> {
                return self(grad_tensor, input_tensor_a, scalar, memory_config, input_grad);
            },
            nb::arg("grad_tensor"),
            nb::arg("input_tensor_a"),
            nb::arg("scalar"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("input_grad") = nb::none()},

        // tensor and tensor
        ttnn::nanobind_overload_t{
            [](const binary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor,
               const ttnn::Tensor& other_tensor,
               const std::vector<bool>& are_required_outputs,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& input_grad,
               const std::optional<ttnn::Tensor>& other_grad) -> std::vector<std::optional<ttnn::Tensor>> {
                return self(
                    grad_tensor,
                    input_tensor,
                    other_tensor,
                    are_required_outputs,
                    memory_config,
                    input_grad,
                    other_grad);
            },
            nb::arg("grad_tensor"),
            nb::arg("input_tensor"),
            nb::arg("other_tensor"),
            nb::kw_only(),
            nb::arg("are_required_outputs") = std::vector<bool>{true, true},
            nb::arg("memory_config") = nb::none(),
            nb::arg("input_grad") = nb::none(),
            nb::arg("other_grad") = nb::none()},

        // complex tensor
        ttnn::nanobind_overload_t{
            [](const binary_backward_operation_t& self,
               const ComplexTensor& grad_tensor,
               const ComplexTensor& input_tensor_a,
               const ComplexTensor& input_tensor_b,
               float alpha,
               const std::optional<MemoryConfig>& memory_config) {
                return self(grad_tensor, input_tensor_a, input_tensor_b, alpha, memory_config);
            },
            nb::arg("grad_tensor"),
            nb::arg("input_tensor_a"),
            nb::arg("input_tensor_b"),
            nb::arg("alpha"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()});
}

template <typename binary_backward_operation_t>
void bind_binary_bw_div(
    nb::module_& mod,
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
            rounding_mode (str, optional): Round mode for the operation (when input tensors are not ComplexTensor type). Can be  None, "trunc", "floor". Defaults to `None`.
            are_required_outputs (List[bool], optional): List of required outputs. Defaults to `[True, True]`.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            input_grad (ttnn.Tensor, optional): Preallocated output tensor for gradient of `input_tensor`. Defaults to `None`.
            other_grad (ttnn.Tensor, optional): Preallocated output tensor for gradient of `other_tensor`. Defaults to `None`.


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

            Performance of the PCC may degrade when using BFLOAT8_B. For more details, refer to the `BFLOAT8_B limitations <../tensor.html#limitation-of-bfloat8-b>`_.

        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        description,
        supported_dtype);

    bind_registered_operation(
        mod,
        operation,
        doc,
        // tensor and scalar
        ttnn::nanobind_overload_t{
            [](const binary_backward_operation_t& self,
               const Tensor& grad_tensor,
               const Tensor& input_tensor_a,
               const float scalar,
               const std::optional<std::string>& rounding_mode,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& input_grad) -> std::vector<std::optional<ttnn::Tensor>> {
                return self(grad_tensor, input_tensor_a, scalar, rounding_mode, memory_config, input_grad);
            },
            nb::arg("grad_tensor"),
            nb::arg("input_tensor_a"),
            nb::arg("scalar"),
            nb::kw_only(),
            nb::arg("rounding_mode") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("input_grad") = nb::none()},

        // tensor and tensor
        ttnn::nanobind_overload_t{
            [](const binary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor,
               const ttnn::Tensor& other_tensor,
               const std::optional<std::string>& rounding_mode,
               const std::vector<bool>& are_required_outputs,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& input_grad,
               const std::optional<ttnn::Tensor>& other_grad) -> std::vector<std::optional<ttnn::Tensor>> {
                return self(
                    grad_tensor,
                    input_tensor,
                    other_tensor,
                    rounding_mode,
                    are_required_outputs,
                    memory_config,
                    input_grad,
                    other_grad);
            },
            nb::arg("grad_tensor"),
            nb::arg("input_tensor"),
            nb::arg("other_tensor"),
            nb::kw_only(),
            nb::arg("rounding_mode") = nb::none(),
            nb::arg("are_required_outputs") = std::vector<bool>{true, true},
            nb::arg("memory_config") = nb::none(),
            nb::arg("input_grad") = nb::none(),
            nb::arg("other_grad") = nb::none()},

        // complex tensor
        ttnn::nanobind_overload_t{
            [](const binary_backward_operation_t& self,
               const ComplexTensor& grad_tensor,
               const ComplexTensor& input_tensor_a,
               const ComplexTensor& input_tensor_b,
               const MemoryConfig& memory_config) {
                return self(grad_tensor, input_tensor_a, input_tensor_b, memory_config);
            },
            nb::arg("grad_tensor"),
            nb::arg("input_tensor_a"),
            nb::arg("input_tensor_b"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()});
}

template <typename binary_backward_operation_t>
void bind_binary_backward_overload(
    nb::module_& mod,
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
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        description,
        supported_dtype,
        note);

    bind_registered_operation(
        mod,
        operation,
        doc,
        // tensor and scalar
        ttnn::nanobind_overload_t{
            [](const binary_backward_operation_t& self,
               const Tensor& grad_tensor,
               const Tensor& input_tensor_a,
               const float scalar,
               const std::optional<ttnn::MemoryConfig>& memory_config) -> std::vector<ttnn::Tensor> {
                return self(grad_tensor, input_tensor_a, scalar, memory_config);
            },
            nb::arg("grad_tensor"),
            nb::arg("input_tensor_a"),
            nb::arg("scalar"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()},

        // tensor and tensor
        ttnn::nanobind_overload_t{
            [](const binary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor_a,
               const ttnn::Tensor& input_tensor_b,
               const std::optional<ttnn::MemoryConfig>& memory_config) -> std::vector<ttnn::Tensor> {
                return self(grad_tensor, input_tensor_a, input_tensor_b, memory_config);
            },
            nb::arg("grad_tensor"),
            nb::arg("input_tensor_a"),
            nb::arg("input_tensor_b"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()});
}

template <typename binary_backward_operation_t>
void bind_binary_backward_assign(
    nb::module_& mod,
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

            rounding_mode (str, optional): Round mode for the operation. Defaults to `None`.

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

        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        description,
        supported_dtype);

    bind_registered_operation(
        mod,
        operation,
        doc,
        // tensor
        ttnn::nanobind_overload_t{
            [](const binary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& input_grad) -> std::vector<std::optional<ttnn::Tensor>> {
                return self(grad_tensor, input_tensor, memory_config, input_grad);
            },
            nb::arg("grad_tensor"),
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("input_a_grad") = nb::none()},

        // tensor and tensor
        ttnn::nanobind_overload_t{
            [](const binary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor_a,
               const ttnn::Tensor& input_tensor_b,
               const std::vector<bool>& are_required_outputs,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& input_a_grad,
               const std::optional<ttnn::Tensor>& input_b_grad) -> std::vector<std::optional<ttnn::Tensor>> {
                return self(
                    grad_tensor,
                    input_tensor_a,
                    input_tensor_b,
                    are_required_outputs,
                    memory_config,
                    input_a_grad,
                    input_b_grad);
            },
            nb::arg("grad_tensor"),
            nb::arg("input_tensor_a"),
            nb::arg("input_tensor_b"),
            nb::kw_only(),
            nb::arg("are_required_outputs") = std::vector<bool>{true, true},
            nb::arg("memory_config") = nb::none(),
            nb::arg("input_a_grad") = nb::none(),
            nb::arg("input_b_grad") = nb::none()});
}

}  // namespace

void py_module(nb::module_& mod) {
    bind_binary_bw_mul(
        mod,
        ttnn::mul_bw,
        R"doc(Performs backward operations for multiply on :attr:`input_tensor_a`, :attr:`input_tensor_b`, with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    bind_binary_bw(
        mod,
        ttnn::add_bw,
        R"doc(Performs backward operations for add of :attr:`input_tensor_a` and :attr:`input_tensor_b` or :attr:`scalar` with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc",
        R"doc(Sharding is not supported if both inputs are tensors.)doc");

    bind_binary_bw(
        mod,
        ttnn::sub_bw,
        R"doc(Performs backward operations for subtract of :attr:`input_tensor_a` and :attr:`input_tensor_b` or :attr:`scalar` with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    bind_binary_bw_div(
        mod,
        ttnn::div_bw,
        R"doc(Performs backward operations for divide on :attr:`input_tensor`, :attr:`alpha` or :attr:`input_tensor_a`, :attr:`input_tensor_b`, :attr:`rounding_mode`,  with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    bind_binary_backward_overload(
        mod,
        ttnn::remainder_bw,
        R"doc(Performs backward operations for remainder of :attr:`input_tensor_a`, :attr:`scalar` or :attr:`input_tensor_b` with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16)doc",
        R"doc(Supported only in WHB0. For more details about BFLOAT8_B, refer to the `BFLOAT8_B limitations <../tensor.html#limitation-of-bfloat8-b>`_.)doc");

    bind_binary_backward_overload(
        mod,
        ttnn::fmod_bw,
        R"doc(Performs backward operations for fmod of :attr:`input_tensor_a`, :attr:`scalar` or :attr:`input_tensor_b` with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16)doc",
        R"doc(For more details about BFLOAT8_B, refer to the `BFLOAT8_B limitations <../tensor.html#limitation-of-bfloat8-b>`_.)doc");

    bind_binary_backward_assign(
        mod,
        ttnn::assign_bw,
        R"doc(Performs backward operations for assign of :attr:`input_tensor_a`, :attr:`input_tensor_b` with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    bind_binary_backward_ops(
        mod,
        ttnn::atan2_bw,
        R"doc(Performs backward operations for atan2 of :attr:`input_tensor_a` and :attr:`input_tensor_b` with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16)doc");

    bind_binary_backward_sub_alpha(
        mod,
        ttnn::subalpha_bw,
        "alpha",
        "Alpha value",
        1.0f,
        R"doc(Performs backward operations for subalpha of :attr:`input_tensor_a` and :attr:`input_tensor_b` with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    bind_binary_backward_addalpha(
        mod,
        ttnn::addalpha_bw,
        "alpha",
        "Alpha value",
        1.0f,
        R"doc(Performs backward operations for addalpha on :attr:`input_tensor_b` , :attr:`input_tensor_a` and :attr:`alpha` with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    bind_binary_backward_ops(
        mod,
        ttnn::xlogy_bw,
        R"doc(Performs backward operations for xlogy of :attr:`input_tensor_a` and :attr:`input_tensor_b` with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc",
        R"doc(For more details about BFLOAT8_B, refer to the `BFLOAT8_B limitations <../tensor.html#limitation-of-bfloat8-b>`_.)doc");

    bind_binary_backward_ops(
        mod,
        ttnn::hypot_bw,
        R"doc(Performs backward operations for hypot of :attr:`input_tensor_a` and :attr:`input_tensor_b` with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16)doc",
        R"doc(Performance of the PCC may degrade when using BFLOAT8_B. For more details, refer to the `BFLOAT8_B limitations <../tensor.html#limitation-of-bfloat8-b>`_.)doc");

    bind_binary_backward_ops(
        mod,
        ttnn::ldexp_bw,
        R"doc(Performs backward operations for ldexp of :attr:`input_tensor_a` and :attr:`input_tensor_b` with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16)doc",
        R"doc(Recommended input range : [-80, 80]. Performance of the PCC may degrade if the input falls outside this range.)doc");

    bind_binary_backward_ops(
        mod,
        ttnn::logaddexp_bw,
        R"doc(Performs backward operations for logaddexp of :attr:`input_tensor_a` and :attr:`input_tensor_b` with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16)doc",
        R"doc(For more details about BFLOAT8_B, refer to the `BFLOAT8_B limitations <../tensor.html#limitation-of-bfloat8-b>`_.)doc");

    bind_binary_backward_ops(
        mod,
        ttnn::logaddexp2_bw,
        R"doc(Performs backward operations for logaddexp2 of :attr:`input_tensor_a` and :attr:`input_tensor_b` with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16)doc",
        R"doc(For more details about BFLOAT8_B, refer to the `BFLOAT8_B limitations <../tensor.html#limitation-of-bfloat8-b>`_.)doc");

    bind_binary_backward_ops(
        mod,
        ttnn::squared_difference_bw,
        R"doc(Performs backward operations for squared_difference of :attr:`input_tensor_a` and :attr:`input_tensor_b` with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    bind_binary_backward_concat(
        mod,
        ttnn::concat_bw,
        "dim",
        "Dimension to concatenate",
        0,
        R"doc(Performs backward operations for concat on :attr:`input_tensor_a` and :attr:`input_tensor_b` with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16)doc");

    bind_binary_backward_rsub(
        mod,
        ttnn::rsub_bw,
        R"doc(Performs backward operations for subraction of :attr:`input_tensor_a` from :attr:`input_tensor_b` with given :attr:`grad_tensor` (reversed order of subtraction operator).)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    bind_binary_backward_ops(
        mod,
        ttnn::min_bw,
        R"doc(Performs backward operations for minimum of :attr:`input_tensor_a` and :attr:`input_tensor_b` with given :attr:`grad_tensor`.)doc");

    bind_binary_backward_ops(
        mod,
        ttnn::max_bw,
        R"doc(Performs backward operations for maximum of :attr:`input_tensor_a` and :attr:`input_tensor_b` with given :attr:`grad_tensor`.)doc");

    bind_binary_backward_bias_gelu(
        mod,
        ttnn::bias_gelu_bw,
        "bias",
        "Bias value",
        "approximate",
        "Approximation type",
        "none",
        R"doc(Performs backward operations for bias_gelu on :attr:`input_tensor_a` and :attr:`input_tensor_b` or :attr:`input_tensor` and :attr:`bias`, with given :attr:`grad_tensor` using given :attr:`approximate` mode.
        :attr:`approximate` mode can be 'none', 'tanh'.)doc",
        R"doc(BFLOAT16)doc",
        R"doc(For more details about BFLOAT8_B, refer to the `BFLOAT8_B limitations <../tensor.html#limitation-of-bfloat8-b>`_.)doc");
}

}  // namespace ttnn::operations::binary_backward
