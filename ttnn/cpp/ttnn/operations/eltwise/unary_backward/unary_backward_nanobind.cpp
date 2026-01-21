// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "unary_backward_nanobind.hpp"

#include <optional>
#include <string>
#include <string_view>

#include <fmt/format.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn-nanobind/pytensor.hpp"
#include "ttnn/operations/eltwise/binary_backward/binary_backward.hpp"
#include "ttnn/operations/eltwise/unary_backward/unary_backward.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::unary_backward {

namespace {

template <typename unary_backward_operation_t>
void bind_unary_backward_two_float(
    nb::module_& mod,
    const unary_backward_operation_t& operation,
    const std::string_view description,
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& note = "") {
    auto doc = fmt::format(
        R"doc(
        {2}

        .. math::
            \mathrm{{{{output\_tensor}}}}_i = \verb|{0}|(\mathrm{{{{grad\_tensor}}}}_i, \mathrm{{{{input\_tensor}}}}_i, \verb|threshold|, \verb|value|)

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
            [](const unary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor,
               float min,
               float max,
               const std::optional<MemoryConfig>& memory_config) {
                return self(grad_tensor, input_tensor, min, max, memory_config);
            },
            nb::arg("grad_tensor"),
            nb::arg("input_tensor"),
            nb::arg("min"),
            nb::arg("max"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()});
}

template <typename unary_backward_operation_t>
void bind_unary_backward_op(
    nb::module_& mod,
    const unary_backward_operation_t& operation,
    const std::string& description,
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& layout = "TILE",
    const std::string& note = "") {
    auto doc = fmt::format(
        R"doc(
        {2}

        .. math::
            \mathrm{{{{output\_tensor}}}}_i = \verb|{0}|(\mathrm{{{{grad\_tensor}}}}_i, \mathrm{{{{input\_tensor}}}}_i)

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
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        description,
        supported_dtype,
        layout,
        note);

    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const unary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor,
               const std::optional<MemoryConfig>& memory_config) {
                return self(grad_tensor, input_tensor, memory_config);
            },
            nb::arg("grad_tensor"),
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()});
}

template <typename unary_backward_operation_t>
void bind_unary_backward_rsqrt(
    nb::module_& mod,
    const unary_backward_operation_t& operation,
    const std::string_view description,
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& note = "") {
    auto doc = fmt::format(
        R"doc(
        {2}

        .. math::
            \mathrm{{{{output\_tensor}}}}_i = \verb|{0}|(\mathrm{{{{grad\_tensor}}}}_i, \mathrm{{{{input\_tensor}}}}_i)

        Args:
            grad_tensor (ttnn.Tensor): the input gradient tensor.
            input_tensor (ttnn.Tensor): the input tensor.

        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the operation. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): preallocated output tensor. Defaults to `None`.

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
        ttnn::nanobind_overload_t{
            [](const unary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& input_grad) -> std::vector<std::optional<ttnn::Tensor>> {
                return self(grad_tensor, input_tensor, memory_config, input_grad);
            },
            nb::arg("grad_tensor"),
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("input_grad") = nb::none()});
}

template <typename unary_backward_operation_t>
void bind_unary_backward_op_reciprocal(
    nb::module_& mod,
    const unary_backward_operation_t& operation,
    const std::string& description,
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& note = "") {
    auto doc = fmt::format(
        R"doc(
        {2}

        .. math::
            \mathrm{{{{output\_tensor}}}}_i = \verb|{0}|(\mathrm{{{{grad\_tensor}}}}_i, \mathrm{{{{input\_tensor}}}}_i)

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
            [](const unary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor,
               const std::optional<MemoryConfig>& memory_config) {
                return self(grad_tensor, input_tensor, memory_config);
            },
            nb::arg("grad_tensor"),
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()},

        ttnn::nanobind_overload_t{
            [](const unary_backward_operation_t& self,
               const ComplexTensor& grad_tensor,
               const ComplexTensor& input_tensor,
               const MemoryConfig& memory_config) { return self(grad_tensor, input_tensor, memory_config); },
            nb::arg("grad_tensor"),
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg("memory_config")});
}

template <typename unary_backward_operation_t>
void bind_unary_backward_op_overload_abs(
    nb::module_& mod,
    const unary_backward_operation_t& operation,
    const std::string_view description,
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& note = "") {
    auto doc = fmt::format(
        R"doc(
        {2}

        .. math::
            \mathrm{{{{output\_tensor}}}}_i = \verb|{0}|(\mathrm{{{{grad\_tensor}}}}_i, \mathrm{{{{input\_tensor}}}}_i)

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
            [](const unary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor,
               const std::optional<MemoryConfig>& memory_config) {
                return self(grad_tensor, input_tensor, memory_config);
            },
            nb::arg("grad_tensor"),
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()},

        ttnn::nanobind_overload_t{
            [](const unary_backward_operation_t& self,
               const Tensor& grad_tensor,
               const ComplexTensor& input_tensor,
               const MemoryConfig& memory_config) { return self(grad_tensor, input_tensor, memory_config); },
            nb::arg("grad_tensor"),
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg("memory_config")});
}

template <typename unary_backward_operation_t>
void bind_unary_backward_float(
    nb::module_& mod,
    const unary_backward_operation_t& operation,
    const std::string& description,
    const std::string& parameter_name_a,
    const std::string& parameter_a_doc,
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& note = "") {
    auto doc = fmt::format(
        R"doc(
        {2}

        .. math::
            \mathrm{{{{output\_tensor}}}}_i = \verb|{0}|(\mathrm{{{{grad\_tensor}}}}_i, \mathrm{{{{input\_tensor}}}}_i, \verb|{3}|)

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
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        description,
        parameter_name_a,
        parameter_a_doc,
        supported_dtype,
        note);

    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const unary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor,
               float parameter_a,
               const std::optional<MemoryConfig>& memory_config) {
                return self(grad_tensor, input_tensor, parameter_a, memory_config);
            },
            nb::arg("grad_tensor"),
            nb::arg("input_tensor"),
            nb::arg(parameter_name_a.c_str()),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()}

    );
}

template <typename unary_backward_operation_t>
void bind_unary_backward_two_float_with_default(
    nb::module_& mod,
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
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const unary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor,
               float parameter_a,
               float parameter_b,
               const std::optional<MemoryConfig>& memory_config) {
                return self(grad_tensor, input_tensor, parameter_a, parameter_b, memory_config);
            },
            nb::arg("grad_tensor"),
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg(parameter_name_a.c_str()) = parameter_a_value,
            nb::arg(parameter_name_b.c_str()) = parameter_b_value,
            nb::arg("memory_config") = nb::none()});
}

template <typename unary_backward_operation_t>
void bind_unary_backward_float_with_default(
    nb::module_& mod,
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
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const unary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor,
               float parameter_a,
               const std::optional<MemoryConfig>& memory_config) {
                return self(grad_tensor, input_tensor, parameter_a, memory_config);
            },
            nb::arg("grad_tensor"),
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg(parameter_name_a.c_str()) = parameter_a_value,
            nb::arg("memory_config") = nb::none()});
}

template <typename unary_backward_operation_t>
void bind_unary_backward_optional_float_params_with_default(
    nb::module_& mod,
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
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const unary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor,
               std::optional<float> parameter_a,
               std::optional<float> parameter_b,
               const std::optional<MemoryConfig>& memory_config) {
                return self(grad_tensor, input_tensor, parameter_a, parameter_b, memory_config);
            },
            nb::arg("grad_tensor"),
            nb::arg("input_tensor"),
            nb::arg(parameter_name_a.c_str()) = parameter_a_value,
            nb::arg(parameter_name_b.c_str()) = parameter_b_value,
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()},

        ttnn::nanobind_overload_t{
            [](const unary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor,
               std::optional<Tensor> parameter_a,
               std::optional<Tensor> parameter_b,
               const std::optional<MemoryConfig>& memory_config) {
                return self(grad_tensor, input_tensor, parameter_a, parameter_b, memory_config);
            },
            nb::arg("grad_tensor"),
            nb::arg("input_tensor"),
            nb::arg(parameter_name_a.c_str()) = parameter_a_value,
            nb::arg(parameter_name_b.c_str()) = parameter_b_value,
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()});
}

template <typename unary_backward_operation_t>
void bind_unary_backward_rdiv(
    nb::module_& mod,
    const unary_backward_operation_t& operation,
    const std::string& parameter_name_a,
    const std::string& parameter_a_doc,
    const std::string& parameter_name_b,
    const std::string& parameter_b_doc,
    std::string parameter_b_value,
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

            Performance of the PCC may degrade when using BFLOAT8_B. For more details, refer to the `BFLOAT8_B limitations <../tensor.html#limitation-of-bfloat8-b>`_.
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
            [](const unary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor,
               float parameter_a,
               const std::optional<std::string>& parameter_b,
               const std::optional<MemoryConfig>& memory_config) {
                return self(grad_tensor, input_tensor, parameter_a, parameter_b, memory_config);
            },
            nb::arg("grad_tensor"),
            nb::arg("input_tensor"),
            nb::arg(parameter_name_a.c_str()),
            nb::kw_only(),
            nb::arg(parameter_name_b.c_str()) = nb::none(),
            nb::arg("memory_config") = nb::none()});
}

template <typename unary_backward_operation_t>
void bind_unary_backward_unary_optional_float(
    nb::module_& mod,
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
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        parameter_name,
        parameter_doc,
        description,
        supported_dtype,
        note);

    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const unary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor,
               float parameter,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& input_grad) -> std::vector<std::optional<ttnn::Tensor>> {
                return self(grad_tensor, input_tensor, parameter, memory_config, input_grad);
            },
            nb::arg("grad_tensor"),
            nb::arg("input_tensor"),
            nb::arg(parameter_name.c_str()),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("input_grad") = nb::none()});
}

template <typename unary_backward_operation_t>
void bind_unary_backward_shape(
    nb::module_& mod,
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
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        parameter_name_a,
        parameter_a_doc,
        description,
        supported_dtype,
        note);

    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const unary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor,
               const ttnn::Shape& parameter_a,
               const std::optional<MemoryConfig>& memory_config) {
                return self(grad_tensor, input_tensor, parameter_a, memory_config);
            },
            nb::arg("grad_tensor"),
            nb::arg("input_tensor"),
            nb::arg(parameter_name_a.c_str()),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()});
}

template <typename unary_backward_operation_t>
void bind_unary_backward_optional(
    nb::module_& mod,
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
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        description,
        supported_dtype,
        layout,
        note);

    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const unary_backward_operation_t& self,
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
            nb::arg("input_grad") = nb::none()});
}

template <typename unary_backward_operation_t>
void bind_unary_backward_neg(
    nb::module_& mod,
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
        ttnn::nanobind_overload_t{
            [](const unary_backward_operation_t& self,
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
            nb::arg("input_grad") = nb::none()});
}

template <typename unary_backward_operation_t>
void bind_unary_backward_prod_bw(nb::module_& mod, const unary_backward_operation_t& operation) {
    auto doc = fmt::format(
        R"doc(
        Performs backward operations for prod on :attr:`input_tensor` with given :attr:`grad_tensor` along a particular `dim`.
        If no `dim` is provided, the prod is taken over all dimensions.

        Args:
            grad_tensor (ttnn.Tensor): the input gradient tensor.
            input_tensor (ttnn.Tensor): the input tensor.

        Keyword args:
            dim (int, optional): dimension to perform prod backward. Defaults to `None`.
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

            For more details about BFLOAT8_B, refer to the `BFLOAT8_B limitations <../tensor.html#limitation-of-bfloat8-b>`_.
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name());

    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const unary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor,
               const std::optional<int64_t> dim,
               const std::optional<MemoryConfig>& memory_config) {
                return self(grad_tensor, input_tensor, dim, memory_config);
            },
            nb::arg("grad_tensor"),
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg("dim") = nb::none(),
            nb::arg("memory_config") = nb::none()});
}

template <typename unary_backward_operation_t>
void bind_unary_backward_gelu(
    nb::module_& mod,
    const unary_backward_operation_t& operation,
    const std::string& parameter_name_a,
    const std::string& parameter_a_doc,
    std::string parameter_a_value,
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
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const unary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor,
               std::string parameter_a,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& input_grad) -> std::vector<std::optional<ttnn::Tensor>> {
                return self(grad_tensor, input_tensor, parameter_a, memory_config, input_grad);
            },
            nb::arg("grad_tensor"),
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg(parameter_name_a.c_str()) = parameter_a_value,
            nb::arg("memory_config") = nb::none(),
            nb::arg("input_grad") = nb::none()});
}

}  // namespace

void py_module(nb::module_& mod) {
    bind_unary_backward_optional_float_params_with_default(
        mod,
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

    bind_unary_backward_optional_float_params_with_default(
        mod,
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

    bind_unary_backward_two_float_with_default(
        mod,
        ttnn::hardtanh_bw,
        "min",
        "Minimum value",
        -1.0,
        "max",
        "Maximum value",
        1.0,
        R"doc(Performs backward operations for hardtanh activation function on :attr:`input_tensor`, :attr:`min`, :attr:`max` with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    bind_unary_backward_float_with_default(
        mod,
        ttnn::hardshrink_bw,
        "lambd",
        "Lambda value for the hardshrink formula ",
        0.5,
        R"doc(Performs backward operations for hardshrink on :attr:`input_tensor`, :attr:`lambd`, with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    bind_unary_backward_float_with_default(
        mod,
        ttnn::softshrink_bw,
        "lambd",
        "Lambda value for the softshrink formula ",
        0.5,
        R"doc(Performs backward operations for softshrink on :attr:`input_tensor`, :attr:`lambd`, with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    bind_unary_backward_float_with_default(
        mod,
        ttnn::leaky_relu_bw,
        "negative_slope",
        "negative_slope value for the hardshrink formula ",
        0.01,
        R"doc(Performs backward operations for leaky_relu on :attr:`input_tensor`, :attr:`negative_slope`, with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    bind_unary_backward_float_with_default(
        mod,
        ttnn::elu_bw,
        "alpha",
        "alpha value for the elu formula ",
        1.0,
        R"doc(Performs backward operations for elu on :attr:`input_tensor`, :attr:`alpha`, with given :attr:`grad_tensor`.)doc");

    bind_unary_backward_float_with_default(
        mod,
        ttnn::celu_bw,
        "alpha",
        "alpha value for the celu formula ",
        1.0,
        R"doc(Performs backward operations for celu on :attr:`input_tensor`, :attr:`alpha`, with given :attr:`grad_tensor`.)doc");

    bind_unary_backward_float_with_default(
        mod,
        ttnn::logiteps_bw,
        "eps",
        "eps value for the logiteps formula ",
        0.0,
        R"doc(Performs backward operations for logiteps on :attr:`input_tensor`, :attr:`eps`, with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16)doc",
        R"doc(For more details about BFLOAT8_B, refer to the `BFLOAT8_B limitations <../tensor.html#limitation-of-bfloat8-b>`_.)doc");

    bind_unary_backward_two_float(
        mod,
        ttnn::threshold_bw,
        R"doc(Performs backward operations for threshold on :attr:`input_tensor`, :attr:`threshold`, :attr:`value` with given :attr:`grad_tensor`.)doc");

    bind_unary_backward_two_float_with_default(
        mod,
        ttnn::softplus_bw,
        "beta",
        "Beta value for the Softplus formula ",
        1.0,
        "threshold",
        "Threshold value",
        20.0,
        R"doc(Performs backward operations for softplus on :attr:`input_tensor`, :attr:`beta`, :attr:`threshold` with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16)doc",
        R"doc(For more details about BFLOAT8_B, refer to the `BFLOAT8_B limitations <../tensor.html#limitation-of-bfloat8-b>`_.)doc");

    bind_unary_backward_rdiv(
        mod,
        ttnn::rdiv_bw,
        "scalar",
        "divisor",
        "rounding_mode",
        "Mode of Rounding",
        "None",
        R"doc(Performs backward operations for Unary rdiv on :attr:`input_tensor`, :attr:`scalar` with given :attr:`grad_tensor` using given :attr:`rounding_mode`.
        :attr:`rounding_mode` can be 'None', 'trunc', or 'floor'.)doc");

    bind_unary_backward_shape(
        mod,
        ttnn::repeat_bw,
        "shape",
        "Shape of tensor",
        R"doc(Performs backward operations for repeat on :attr:`input_tensor`, with given :attr:`grad_tensor` using given :attr:`shape`.)doc");

    bind_unary_backward_gelu(
        mod,
        ttnn::gelu_bw,
        "approximate",
        "Approximation type",
        "none",
        R"doc(Performs backward operations for gelu on :attr:`input_tensor`, with given :attr:`grad_tensor` using given :attr:`approximate` mode.
        :attr:`approximate` mode can be 'none', 'tanh'.)doc",
        R"doc(BFLOAT16)doc",
        R"doc(For more details about BFLOAT8_B, refer to the `BFLOAT8_B limitations <../tensor.html#limitation-of-bfloat8-b>`_.)doc");

    bind_unary_backward_unary_optional_float(
        mod,
        ttnn::pow_bw,
        "exponent",
        "Exponent value [must be non-negative]",
        R"doc(Performs backward operations for power on :attr:`input_tensor`, :attr:`exponent` with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc",
        R"doc(For more details about BFLOAT8_B, refer to the `BFLOAT8_B limitations <../tensor.html#limitation-of-bfloat8-b>`_.)doc");

    bind_unary_backward_optional(
        mod,
        ttnn::exp_bw,
        R"doc(Performs backward operations for exponential function on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    bind_unary_backward_optional(
        mod,
        ttnn::tanh_bw,
        R"doc(Performs backward operations for hyperbolic tangent (tanh) function on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc");

    bind_unary_backward_optional(
        mod,
        ttnn::sqrt_bw,
        R"doc(Performs backward operations for square-root on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc",
        R"doc(TILE)doc",
        R"doc(For more details about BFLOAT8_B, refer to the `BFLOAT8_B limitations <../tensor.html#limitation-of-bfloat8-b>`_.)doc");

    bind_unary_backward_op(
        mod,
        ttnn::multigammaln_bw,
        R"doc(Performs backward operations for multivariate logarithmic gamma function (also referred to as mvlgamma) on :attr:`input_tensor` with given :attr:`grad_tensor`.
        The dimensionality is set to 4.)doc",
        R"doc(BFLOAT16)doc",
        R"doc(TILE)doc",
        R"doc(Input value must be greater than 2.5f)doc");

    bind_unary_backward_prod_bw(mod, ttnn::prod_bw);

    bind_unary_backward_op(
        mod,
        ttnn::lgamma_bw,
        R"doc(Performs backward operations for lgamma on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc");

    bind_unary_backward_optional(
        mod,
        ttnn::fill_bw,
        R"doc(Performs backward operations for fill on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16)doc",
        R"doc(TILE, ROW MAJOR)doc");

    bind_unary_backward_op(
        mod,
        ttnn::hardsigmoid_bw,
        R"doc(Performs backward operations for hardsigmoid on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc");

    bind_unary_backward_op(
        mod,
        ttnn::cos_bw,
        R"doc(Performs backward operations for cosine on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    bind_unary_backward_op(
        mod,
        ttnn::acosh_bw,
        R"doc(Performs backward operations for inverse hyperbolic cosine (acosh) on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc");

    bind_unary_backward_op(
        mod,
        ttnn::acos_bw,
        R"doc(Performs backward operations for inverse cosine (acos) on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc");

    bind_unary_backward_op(
        mod,
        ttnn::atan_bw,
        R"doc(Performs backward operations for inverse tangent (atan) on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    bind_unary_backward_op(
        mod,
        ttnn::rad2deg_bw,
        R"doc(Performs backward operations for radian to degree conversion (rad2deg) on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    bind_unary_backward_op(
        mod,
        ttnn::frac_bw,
        R"doc(Performs backward operations for frac on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16)doc",
        R"doc(TILE, ROW MAJOR)doc",
        R"doc(BFLOAT8_B is supported for TILE layout.)doc");

    bind_unary_backward_op(
        mod,
        ttnn::trunc_bw,
        R"doc(Performs backward operations for truncation on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16)doc",
        R"doc(TILE, ROW MAJOR)doc");

    bind_unary_backward_op(
        mod,
        ttnn::log_sigmoid_bw,
        R"doc(Performs backward operations for log sigmoid on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc",
        R"doc(TILE)doc",
        R"doc(For more details about BFLOAT8_B, refer to the `BFLOAT8_B limitations <../tensor.html#limitation-of-bfloat8-b>`_.)doc");

    bind_unary_backward_op(
        mod,
        ttnn::fill_zero_bw,
        R"doc(Performs backward operations for fill zero on :attr:`input_tensor` with given :attr:`grad_tensor`. Returns an tensor of zeros like :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16)doc",
        R"doc(TILE, ROW MAJOR)doc");

    bind_unary_backward_op(
        mod,
        ttnn::i0_bw,
        R"doc(Performs backward operations for i0 on :attr:`input_tensor` with given :attr:`grad_tensor`. Supported input range is (-10, 10))doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    bind_unary_backward_op(
        mod,
        ttnn::tan_bw,
        R"doc(Performs backward operations for tan on :attr:`input_tensor` with given :attr:`grad_tensor`. Supported input range is (-1.45, 1.45))doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    bind_unary_backward_op(
        mod,
        ttnn::sigmoid_bw,
        R"doc(Performs backward operations for sigmoid on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    bind_unary_backward_rsqrt(
        mod,
        ttnn::rsqrt_bw,
        R"doc(Performs backward operations for reciprocal of square-root on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    bind_unary_backward_neg(
        mod,
        ttnn::neg_bw,
        R"doc(Performs backward operations for neg on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    bind_unary_backward_op(
        mod,
        ttnn::relu_bw,
        R"doc(Performs backward operations for relu on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    bind_unary_backward_op(
        mod,
        ttnn::logit_bw,
        R"doc(Performs backward operations for logit on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16)doc",
        R"doc(TILE)doc",
        R"doc(For more details about BFLOAT8_B, refer to the `BFLOAT8_B limitations <../tensor.html#limitation-of-bfloat8-b>`_.)doc");

    bind_unary_backward_op(
        mod,
        ttnn::floor_bw,
        R"doc(Performs backward operations for floor on :attr:`input_tensor` with given :attr:`grad_tensor`)doc",
        R"doc(BFLOAT16)doc",
        R"doc(TILE, ROW MAJOR)doc");

    bind_unary_backward_float(
        mod,
        ttnn::rpow_bw,
        R"doc(Performs backward operations for rpow on :attr:`input_tensor`, :attr:`exponent` with given :attr:`grad_tensor`.)doc",
        "exponent",
        "Exponent value",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    bind_unary_backward_op(
        mod,
        ttnn::round_bw,
        R"doc(Performs backward operations for round on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16)doc",
        R"doc(TILE, ROW MAJOR)doc");

    bind_unary_backward_op(
        mod,
        ttnn::log_bw,
        R"doc(Performs backward operations for logarithm on :attr:`input_tensor` with given :attr:`grad_tensor`)doc",
        R"doc(BFLOAT16)doc",
        R"doc(TILE)doc",
        R"doc(For more details about BFLOAT8_B, refer to the `BFLOAT8_B limitations <../tensor.html#limitation-of-bfloat8-b>`_.)doc");

    bind_unary_backward_op(
        mod,
        ttnn::relu6_bw,
        R"doc(Performs backward operations for relu6 on :attr:`input_tensor` with given :attr:`grad_tensor`)doc");

    bind_unary_backward_op_overload_abs(
        mod,
        ttnn::abs_bw,
        R"doc(Performs backward operations for abs on :attr:`input_tensor` with given :attr:`grad_tensor`)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc",
        R"doc(For more details about BFLOAT8_B, refer to the `BFLOAT8_B limitations <../tensor.html#limitation-of-bfloat8-b>`_.)doc");

    bind_unary_backward_neg(
        mod,
        ttnn::silu_bw,
        R"doc(Performs backward operations for silu on :attr:`input_tensor` with given :attr:`grad_tensor`)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    bind_unary_backward_op(
        mod,
        ttnn::selu_bw,
        R"doc(Performs backward operations for selu on :attr:`input_tensor` with given :attr:`grad_tensor`)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    bind_unary_backward_op(
        mod,
        ttnn::square_bw,
        R"doc(Performs backward operations for square on :attr:`input_tensor` with given :attr:`grad_tensor`)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    bind_unary_backward_op(
        mod,
        ttnn::hardswish_bw,
        R"doc(Performs backward operations for  hardswish on :attr:`input_tensor` with given :attr:`grad_tensor`)doc");

    bind_unary_backward_op(
        mod,
        ttnn::tanhshrink_bw,
        R"doc(Performs backward operations for  tanhshrink on :attr:`input_tensor` with given :attr:`grad_tensor`)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    bind_unary_backward_op(
        mod,
        ttnn::atanh_bw,
        R"doc(Performs backward operations for inverse hyperbolic tangent (atanh) on :attr:`input_tensor` with given :attr:`grad_tensor`)doc");

    bind_unary_backward_op(
        mod,
        ttnn::asin_bw,
        R"doc(Performs backward operations for inverse sine (asin) on :attr:`input_tensor` with given :attr:`grad_tensor`)doc");

    bind_unary_backward_op(
        mod,
        ttnn::asinh_bw,
        R"doc(Performs backward operations for inverse hyperbolic sine (asinh) on :attr:`input_tensor` with given :attr:`grad_tensor`)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    bind_unary_backward_op(
        mod,
        ttnn::sin_bw,
        R"doc(Performs backward operations for sin on :attr:`input_tensor` with given :attr:`grad_tensor`)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    bind_unary_backward_op(
        mod,
        ttnn::sinh_bw,
        R"doc(Performs backward operations for hyperbolic sine (sinh) on :attr:`input_tensor` with given :attr:`grad_tensor`)doc");

    bind_unary_backward_op(
        mod,
        ttnn::log10_bw,
        R"doc(Performs backward operations for log10 on :attr:`input_tensor` with given :attr:`grad_tensor`)doc",
        R"doc(BFLOAT16)doc",
        R"doc(TILE)doc",
        R"doc(For more details about BFLOAT8_B, refer to the `BFLOAT8_B limitations <../tensor.html#limitation-of-bfloat8-b>`_.)doc");

    bind_unary_backward_op(
        mod,
        ttnn::log1p_bw,
        R"doc(Performs backward operations for log1p on :attr:`input_tensor` with given :attr:`grad_tensor`)doc",
        R"doc(BFLOAT16)doc",
        R"doc(TILE)doc",
        R"doc(For more details about BFLOAT8_B, refer to the `BFLOAT8_B limitations <../tensor.html#limitation-of-bfloat8-b>`_.)doc");

    bind_unary_backward_op(
        mod,
        ttnn::erfc_bw,
        R"doc(Performs backward operations for erfc on :attr:`input_tensor` with given :attr:`grad_tensor`)doc");

    bind_unary_backward_op(
        mod,
        ttnn::ceil_bw,
        R"doc(Performs backward operations for ceil on :attr:`input_tensor` with given :attr:`grad_tensor`)doc",
        R"doc(BFLOAT16)doc",
        R"doc(TILE, ROW MAJOR)doc");

    bind_unary_backward_op(
        mod,
        ttnn::softsign_bw,
        R"doc(Performs backward operations for softsign on :attr:`input_tensor` with given :attr:`grad_tensor`)doc");

    bind_unary_backward_op(
        mod,
        ttnn::cosh_bw,
        R"doc(Performs backward operations for hyperbolic cosine (cosh) on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc");

    bind_unary_backward_op(
        mod,
        ttnn::log2_bw,
        R"doc(Performs backward operations for log2 on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16)doc",
        R"doc(TILE)doc",
        R"doc(For more details about BFLOAT8_B, refer to the `BFLOAT8_B limitations <../tensor.html#limitation-of-bfloat8-b>`_.)doc");

    bind_unary_backward_op(
        mod,
        ttnn::sign_bw,
        R"doc(Performs backward operations for sign on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc",
        R"doc(BFLOAT16)doc",
        R"doc(TILE, ROW MAJOR)doc");

    bind_unary_backward_float(
        mod,
        ttnn::div_no_nan_bw,
        R"doc(Performs backward operations for div_no_nan on :attr:`input_tensor`, :attr:`scalar` with given :attr:`grad_tensor`.)doc",
        "scalar",
        "Denominator value",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    bind_unary_backward_op(
        mod,
        ttnn::exp2_bw,
        R"doc(Performs backward operations for exp2 on :attr:`input_tensor` with given :attr:`grad_tensor`)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    bind_unary_backward_op(
        mod,
        ttnn::expm1_bw,
        R"doc(Performs backward operations for expm1 on :attr:`input_tensor` with given :attr:`grad_tensor`)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    bind_unary_backward_op_reciprocal(
        mod,
        ttnn::reciprocal_bw,
        R"doc(Performs backward operations for reciprocal on :attr:`input_tensor` with given :attr:`grad_tensor`)doc",
        R"doc(BFLOAT16)doc",
        R"doc(For more details about BFLOAT8_B, refer to the `BFLOAT8_B limitations <../tensor.html#limitation-of-bfloat8-b>`_.)doc");

    bind_unary_backward_op(
        mod,
        ttnn::digamma_bw,
        R"doc(Performs backward operations for digamma on :attr:`input_tensor` with given :attr:`grad_tensor`)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    bind_unary_backward_op(
        mod,
        ttnn::erfinv_bw,
        R"doc(Performs backward operations for erfinv on :attr:`input_tensor` with given :attr:`grad_tensor`)doc");

    bind_unary_backward_op(
        mod,
        ttnn::erf_bw,
        R"doc(Performs backward operations for erf on :attr:`input_tensor` with given :attr:`grad_tensor`)doc");

    bind_unary_backward_op(
        mod,
        ttnn::deg2rad_bw,
        R"doc(Performs backward operations for degree to radian conversion (deg2rad) on :attr:`input_tensor` with given :attr:`grad_tensor`)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    bind_unary_backward_float(
        mod,
        ttnn::polygamma_bw,
        R"doc(Performs backward operations for polygamma on :attr:`input_tensor`, :attr:`scalar` with given :attr:`grad_tensor`.)doc",
        "n",
        "Order of polygamma function");
}

}  // namespace ttnn::operations::unary_backward
