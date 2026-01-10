// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "unary_nanobind.hpp"

#include <cstdint>
#include <string>
#include <optional>

#include <fmt/format.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/eltwise/unary/unary_composite.hpp"
#include "ttnn/operations/eltwise/complex_unary/complex_unary.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operations/eltwise/complex/complex.hpp"
#include "unary.hpp"

namespace ttnn::operations::unary {

namespace {
template <typename unary_operation_t>
void bind_unary_clamp(nb::module_& mod, const unary_operation_t& operation) {
    auto doc = fmt::format(
        R"doc(
        Applies {0} to :attr:`input_tensor` element-wise.

        Args:
            input_tensor (ttnn.Tensor): the input tensor.

        Keyword args:
            min (ttnn.Tensor or number): Minimum value. Defaults to `None`.
            max (ttnn.Tensor or number): Maximum value. Defaults to `None`.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
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
               * - BFLOAT16, BFLOAT8_B, INT32, FLOAT32
                 - TILE
                 - 2, 3, 4

            INT32 is supported only for Tensor-scalar-scalar version.
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name());

    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const unary_operation_t& self,
               const ttnn::Tensor& input_tensor,
               std::optional<Tensor> parameter_a,
               std::optional<Tensor> parameter_b,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor) {
                return self(input_tensor, parameter_a, parameter_b, memory_config, output_tensor);
            },
            nb::arg("input_tensor"),
            nb::arg("min") = nb::none(),
            nb::arg("max") = nb::none(),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none()},

        ttnn::nanobind_overload_t{
            [](const unary_operation_t& self,
               const ttnn::Tensor& input_tensor,
               std::optional<std::variant<float, int32_t>> parameter_a,
               std::optional<std::variant<float, int32_t>> parameter_b,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor) {
                return self(input_tensor, parameter_a, parameter_b, memory_config, output_tensor);
            },
            nb::arg("input_tensor"),
            nb::arg("min") = nb::none(),
            nb::arg("max") = nb::none(),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none()});
}

template <typename unary_operation_t>
void bind_unary_composite_optional_floats_with_default(
    nb::module_& mod,
    const unary_operation_t& operation,
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
            input_tensor (ttnn.Tensor): the input tensor.

        Keyword args:
            {2} (float or ttnn.Tensor): {3}. Defaults to `None`.
            {5} (float or ttnn.Tensor): {6}. Defaults to `None`.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

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
            [](const unary_operation_t& self,
               const ttnn::Tensor& input_tensor,
               std::optional<Tensor> parameter_a,
               std::optional<Tensor> parameter_b,
               const std::optional<MemoryConfig>& memory_config) {
                return self(input_tensor, parameter_a, parameter_b, memory_config);
            },
            nb::arg("input_tensor"),
            nb::arg(parameter_name_a.c_str()) = parameter_a_value,
            nb::arg(parameter_name_b.c_str()) = parameter_b_value,
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()},

        ttnn::nanobind_overload_t{
            [](const unary_operation_t& self,
               const ttnn::Tensor& input_tensor,
               std::optional<float> parameter_a,
               std::optional<float> parameter_b,
               const std::optional<MemoryConfig>& memory_config) {
                return self(input_tensor, parameter_a, parameter_b, memory_config);
            },
            nb::arg("input_tensor"),
            nb::arg(parameter_name_a.c_str()) = parameter_a_value,
            nb::arg(parameter_name_b.c_str()) = parameter_b_value,
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()});
}

template <typename unary_operation_t>
void bind_unary_operation(
    nb::module_& mod,
    const unary_operation_t& operation,
    const std::string& math,
    const std::string& range = "",
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& note = "",
    const std::string& example_tensor = "torch.rand([2, 2], dtype=torch.bfloat16)") {
    auto doc = fmt::format(
        R"doc(
        Applies {0} to :attr:`input_tensor` element-wise.

        .. math::
            {2}

        Args:
            input_tensor (ttnn.Tensor): the input tensor. {3}

        Keyword Args:
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
               * - {4}
                 - TILE
                 - 2, 3, 4

            {5}

        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        math,
        range,
        supported_dtype,
        note,
        example_tensor);

    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const unary_operation_t& self,
               const Tensor& input_tensor,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor) -> ttnn::Tensor {
                return self(input_tensor, memory_config, output_tensor);
            },
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none()});
}

template <typename unary_operation_t>
void bind_unary_operation_subcoregrids(
    nb::module_& mod,
    const unary_operation_t& operation,
    const std::string& math,
    const std::string& range = "",
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& note = "",
    const std::string& example_tensor = "torch.rand([2, 2], dtype=torch.bfloat16)") {
    auto doc = fmt::format(
        R"doc(
        Applies {0} to :attr:`input_tensor` element-wise.

        .. math::
            {2}

        Args:
            input_tensor (ttnn.Tensor): the input tensor. {3}

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the operation. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): preallocated output tensor. Defaults to `None`.
            sub_core_grids (ttnn.CoreRangeSet, optional): sub core grids for the operation. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

        Note:
            Supported dtypes, layouts, and ranks:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
                 - Ranks
               * - {4}
                 - TILE
                 - 2, 3, 4

            {5}

        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        math,
        range,
        supported_dtype,
        note,
        example_tensor);

    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const unary_operation_t& self,
               const Tensor& input_tensor,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor,
               const std::optional<CoreRangeSet>& sub_core_grids) -> ttnn::Tensor {
                return self(input_tensor, memory_config, output_tensor, sub_core_grids);
            },
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none(),
            nb::arg("sub_core_grids") = nb::none()});
}

template <typename unary_operation_t>
void bind_unary_sqrt_operation(
    nb::module_& mod,
    const unary_operation_t& operation,
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& info_doc = "") {
    auto doc = fmt::format(
        R"doc(
        Applies {0} to :attr:`input_tensor` element-wise.

        .. math::
            \mathrm{{output\_tensor}}_i = \verb|{0}|(\mathrm{{input\_tensor}}_i)

        Args:
            input_tensor (ttnn.Tensor): the input tensor.

        Keyword Args:
            fast_approx_mode (bool, optional): use the fast and approximate mode. Defaults to `False`.
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
               * - {2}
                 - TILE
                 - 2, 3, 4

            {3}
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        supported_dtype,
        info_doc);

    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const unary_operation_t& self,
               const Tensor& input_tensor,
               const bool fast_approx_mode,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor) -> ttnn::Tensor {
                return self(input_tensor, fast_approx_mode, memory_config, output_tensor);
            },
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg("fast_approx_mode") = false,
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none()});
}

template <typename unary_operation_t>
void bind_unary_operation_overload_complex(
    nb::module_& mod,
    const unary_operation_t& operation,
    const std::string& math,
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& note = "") {
    auto doc = fmt::format(
        R"doc(
        Applies {0} to :attr:`input_tensor` element-wise.

        .. math::
            {2}

        Args:
            input_tensor (ttnn.Tensor or ComplexTensor): the input tensor.

        Keyword Args:
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
               * - {3}
                 - TILE
                 - 2, 3, 4

            {4}
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        math,
        supported_dtype,
        note);

    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const unary_operation_t& self,
               const Tensor& input_tensor,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor) -> ttnn::Tensor {
                return self(input_tensor, memory_config, output_tensor);
            },
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none()},

        ttnn::nanobind_overload_t{
            [](const unary_operation_t& self,
               const ComplexTensor& input_tensor,
               const ttnn::MemoryConfig& memory_config) -> Tensor { return self(input_tensor, memory_config); },
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg("memory_config")});
}

template <typename unary_operation_t>
void bind_unary_operation_overload_complex_return_complex(
    nb::module_& mod,
    const unary_operation_t& operation,
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& info_doc = "") {
    auto doc = fmt::format(
        R"doc(
        Applies {0} to :attr:`input_tensor` element-wise.

        .. math::
            \mathrm{{output\_tensor}}_i = \verb|{0}|(\mathrm{{input\_tensor}}_i)

        Args:
            input_tensor (ttnn.Tensor or ComplexTensor): the input tensor.

        Keyword Args:
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
               * - {2}
                 - TILE
                 - 2, 3, 4

            {3}
            More information about the `BFLOAT8_B  <../tensor.html#limitation-of-bfloat8-b>`_.
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        supported_dtype,
        info_doc);

    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const unary_operation_t& self,
               const Tensor& input_tensor,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor) -> ttnn::Tensor {
                return self(input_tensor, memory_config, output_tensor);
            },
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none()},

        ttnn::nanobind_overload_t{
            [](const unary_operation_t& self,
               const ComplexTensor& input_tensor,
               const ttnn::MemoryConfig& memory_config) -> ComplexTensor { return self(input_tensor, memory_config); },
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg("memory_config")});
}

template <typename unary_operation_t>
void bind_unary_operation_with_fast_and_approximate_mode(
    nb::module_& mod,
    const unary_operation_t& operation,
    const std::string& range = "",
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& info_doc = "") {
    auto doc = fmt::format(
        R"doc(
        Applies {0} to :attr:`input_tensor` element-wise.

        .. math::
            \mathrm{{output\_tensor}}_i = {0}(\mathrm{{input\_tensor}}_i)

        Args:
            input_tensor (ttnn.Tensor): the input tensor. {2}

        Keyword Args:
            fast_and_approximate_mode (bool, optional): Use the fast and approximate mode. Defaults to `False`.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): preallocated output tensor. Defaults to `None`.
            sub_core_grids (ttnn.CoreRangeSet, optional): sub core grids for the operation. Defaults to `None`.

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

            {4}
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        range,
        supported_dtype,
        info_doc);

    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const unary_operation_t& self,
               const Tensor& input_tensor,
               const bool parameter,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor,
               const std::optional<CoreRangeSet>& sub_core_grids) -> ttnn::Tensor {
                return self(input_tensor, parameter, memory_config, output_tensor, sub_core_grids);
            },
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg("fast_and_approximate_mode") = false,
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none(),
            nb::arg("sub_core_grids") = nb::none()});
}

template <typename unary_operation_t>
void bind_unary_operation_with_float_parameter(
    nb::module_& mod,
    const unary_operation_t& operation,
    const std::string& parameter_name,
    const std::string& parameter_doc,
    const std::string& info_doc,
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& note = "") {
    auto doc = fmt::format(
        R"doc(
        Applies {0} to :attr:`input_tensor` element-wise with {2}.

        {4}

        .. math::
            \mathrm{{output\_tensor}}_i = \verb|{0}|(\mathrm{{input\_tensor}}_i, \verb|{2}|)

        Args:
            input_tensor (ttnn.Tensor): the input tensor.
            {2} (float): {3}.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
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
               * - {5}
                 - TILE
                 - 2, 3, 4

            {6}
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        parameter_name,
        parameter_doc,
        info_doc,
        supported_dtype,
        note);

    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const unary_operation_t& self,
               const Tensor& input_tensor,
               const float parameter,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor) {
                return self(input_tensor, parameter, memory_config, output_tensor);
            },
            nb::arg("input_tensor"),
            nb::arg(parameter_name.c_str()),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none()});
}

template <typename unary_operation_t>
void bind_unary_operation_with_scalar_parameter(
    nb::module_& mod,
    const unary_operation_t& operation,
    const std::string& parameter_name,
    const std::string& parameter_doc,
    const std::string& info_doc,
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& note = "") {
    auto doc = fmt::format(
        R"doc(
        Applies {0} to :attr:`input_tensor` element-wise with {2}.

        {4}

        .. math::
            \mathrm{{output\_tensor}}_i = \verb|{0}|(\mathrm{{input\_tensor}}_i, \verb|{2}|)

        Args:
            input_tensor (ttnn.Tensor): the input tensor.
            {2} (float/int): {3}.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
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
               * - {5}
                 - TILE
                 - 1, 2, 3, 4, 5, 6

            {6}
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        parameter_name,
        parameter_doc,
        info_doc,
        supported_dtype,
        note);

    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const unary_operation_t& self,
               const Tensor& input_tensor,
               ScalarVariant parameter,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor) {
                return self(input_tensor, parameter, memory_config, output_tensor);
            },
            nb::arg("input_tensor"),
            nb::arg(parameter_name.c_str()),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none()});
}

template <typename unary_operation_t>
void bind_unary_operation_with_float_parameter_default(
    nb::module_& mod,
    const unary_operation_t& operation,
    const std::string_view& parameter_name,
    const std::string_view& parameter_doc,
    const float parameter_default,
    const std::string_view& info_doc,
    const std::string_view& supported_dtype = "BFLOAT16",
    const std::string_view& note = "") {
    auto doc = fmt::format(
        R"doc(
        Applies {0} to :attr:`input_tensor` element-wise with {2}.

        {4}

        .. math::
            \mathrm{{output\_tensor}}_i = \verb|{0}|(\mathrm{{input\_tensor}}_i, \verb|{2}|)

        Args:
            input_tensor (ttnn.Tensor): the input tensor.

        Keyword Args:
            {2} (float): Defaults to `{3}`.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
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
               * - {6}
                 - TILE
                 - 2, 3, 4

            {7}
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        parameter_name,
        parameter_default,
        parameter_doc,
        info_doc,
        supported_dtype,
        note);

    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const unary_operation_t& self,
               const Tensor& input_tensor,
               const float parameter,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor) {
                return self(input_tensor, parameter, memory_config, output_tensor);
            },
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg(parameter_name.data()) = parameter_default,
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none()});
}

template <typename unary_operation_t>
void bind_unary_composite_with_default_float(
    nb::module_& mod,
    const unary_operation_t& operation,
    const std::string& parameter_name_a,
    const std::string& parameter_a_doc,
    float parameter_a_value,
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& info_doc = "") {
    auto doc = fmt::format(
        R"doc(
        Performs {0} function on :attr:`input_tensor`, :attr:`{2}`.

        Args:
            input_tensor (ttnn.Tensor): the input tensor.
            {2} (float, optional): {3}. Defaults to `{4}`.

        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

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
        parameter_name_a,
        parameter_a_doc,
        parameter_a_value,
        supported_dtype,
        info_doc);

    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const unary_operation_t& self,
               const ttnn::Tensor& input_tensor,
               float parameter_a,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor) {
                return self(input_tensor, parameter_a, memory_config, output_tensor);
            },
            nb::arg("input_tensor"),
            nb::arg(parameter_name_a.c_str()) = parameter_a_value,
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none()});
}

template <typename unary_operation_t>
void bind_unary_operation_with_int_parameter(
    nb::module_& mod,
    const unary_operation_t& operation,
    const std::string& parameter_name,
    const std::string& parameter_doc,
    const std::string& info_doc,
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& note = "") {
    auto doc = fmt::format(
        R"doc(
        Applies {0} to :attr:`input_tensor` element-wise with {2}.

        {4}

        .. math::
            \mathrm{{output\_tensor}}_i = \verb|{0}|(\mathrm{{input\_tensor}}_i, \verb|{2}|)

        Args:
            input_tensor (ttnn.Tensor): the input tensor.
            {2} (int): {3}.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
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
               * - {5}
                 - TILE
                 - 2, 3, 4

            {6}
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        parameter_name,
        parameter_doc,
        info_doc,
        supported_dtype,
        note);

    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const unary_operation_t& self,
               const Tensor& input_tensor,
               const std::optional<int>& parameter,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor) {
                return self(input_tensor, parameter, memory_config, output_tensor);
            },
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg(parameter_name.c_str()) = 0,
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none()});
}

template <typename unary_operation_t>
void bind_unary_operation_with_dim_parameter(
    nb::module_& mod,
    const unary_operation_t& operation,
    const std::string& parameter_name,
    const std::string& parameter_doc,
    const std::string& info_doc,
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& note = "") {
    auto doc = fmt::format(
        R"doc(
        Applies {0} to :attr:`input_tensor` element-wise.

        {4}

        .. math::
            \mathrm{{output\_tensor}}_i = \verb|{0}|(\mathrm{{input\_tensor}}_i)

        Args:
            input_tensor (ttnn.Tensor): the input tensor.
            {2} (int): {3}. Defaults to `-1`.

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
               * - {5}
                 - TILE
                 - 4

            {6}
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        parameter_name,
        parameter_doc,
        info_doc,
        supported_dtype,
        note);

    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const unary_operation_t& self,
               const Tensor& input_tensor,
               int dim,
               const std::optional<MemoryConfig>& memory_config) { return self(input_tensor, dim, memory_config); },
            nb::arg("input_tensor"),
            nb::arg("dim") = -1,
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()});
}

template <typename unary_operation_t>
void bind_unary_rdiv(
    nb::module_& mod,
    const unary_operation_t& operation,
    const std::string& parameter_name_a,
    const std::string& parameter_a_doc,
    const std::string& parameter_name_b,
    const std::string& parameter_b_doc,
    const std::string& parameter_b_value,
    const std::string& description,
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& note = "") {
    auto doc = fmt::format(
        R"doc(
        {7}

        Args:
            input_tensor (ttnn.Tensor): the input tensor.
            {2} (float): {3}.

        Keyword Args:
            {4} (string): {5}. Can be  None, "trunc", "floor". Defaults to `None`.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
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
               * - {8}
                 - TILE
                 - 2, 3, 4

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
            [](const unary_operation_t& self,
               const ttnn::Tensor& input_tensor,
               float parameter_a,
               const std::optional<std::string>& parameter_b,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor) {
                return self(input_tensor, parameter_a, parameter_b, memory_config, output_tensor);
            },
            nb::arg("input_tensor"),
            nb::arg(parameter_name_a.c_str()),
            nb::kw_only(),
            nb::arg(parameter_name_b.c_str()) = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none()});
}

template <typename unary_operation_t>
void bind_softplus(nb::module_& mod) {
    auto doc = fmt::format(
        R"doc(
        Applies {0} to :attr:`input_tensor` element-wise.

        .. math::
            \mathrm{{output\_tensor}}_i = {0}(\mathrm{{input\_tensor}}_i)

        Args:
            input_tensor (ttnn.Tensor): the input tensor.

        Keyword Args:
            beta (float, optional): Scales the input before applying the Softplus function. By modifying :attr:`beta`, you can adjust the steepness of the function. A higher :attr:`beta` value makes the function steeper, approaching a hard threshold like the ReLU function for large values of :attr:`beta`. Defaults to `1`.
            threshold (float, optional): Used to switch to a linear function for large values to improve numerical stability. This avoids issues with floating-point representation for very large values. Defaults to `20`.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
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
               * - BFLOAT16, BFLOAT8_B
                 - TILE
                 - 2, 3, 4
        )doc",
        ttnn::softplus.base_name(),
        ttnn::softplus.python_fully_qualified_name());

    bind_registered_operation(
        mod,
        ttnn::softplus,
        doc,
        ttnn::nanobind_overload_t{
            [](const unary_operation_t& self,
               const Tensor& input,
               const float beta,
               const float threshold,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<Tensor>& output_tensor) {
                return self(input, beta, threshold, memory_config, output_tensor);
            },
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg("beta") = 1.0f,
            nb::arg("threshold") = 20.0f,
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none()});
}

template <typename unary_operation_t>
void bind_tanh_like(nb::module_& mod, const unary_operation_t& operation) {
    auto doc = fmt::format(
        R"doc(
        Applies {0} to :attr:`input_tensor` element-wise.

        .. math::
            \mathrm{{output\_tensor}}_i = {0}(\mathrm{{input\_tensor}}_i)

        Args:
            input_tensor (ttnn.Tensor): the input tensor.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): preallocated output tensor. Defaults to `None`.
            fast_and_approximate_mode (Boolean, optional): Enables a performance-optimized approximation method. When True, the operation runs faster but may produce results with minor precision differences. Defaults to `False`.

        Returns:
            ttnn.Tensor: the output tensor.

        Note:
            Supported dtypes, layouts, and ranks:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
                 - Ranks
               * - BFLOAT16, BFLOAT8_B, FLOAT32
                 - TILE
                 - 2, 3, 4

            BFLOAT8_B/BFLOAT4_B is supported only for approx=True mode.
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name());

    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const unary_operation_t& self,
               const Tensor& input,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<Tensor>& output_tensor,
               bool approx) { return self(input, memory_config, output_tensor, approx); },
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none(),
            nb::arg("fast_and_approximate_mode") = false});
}

template <typename unary_operation_t>
void bind_sigmoid_accurate(nb::module_& mod) {
    auto doc = fmt::format(
        R"doc(
        Applies {0} to :attr:`input_tensor` element-wise.

        .. math::
            \mathrm{{output\_tensor}}_i = \verb|{0}|(\mathrm{{input\_tensor}}_i)

        Args:
            input_tensor (ttnn.Tensor): the input tensor.
            fast_and_approximate_mode (bool, optional): Enables fast and approximate mode for exponential operation. When False, uses the accurate version of exponential algorithm. Defaults to `False`.

        Keyword Args:
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
               * - BFLOAT16, BFLOAT8_B
                 - TILE
                 - 2, 3, 4
        )doc",
        ttnn::sigmoid_accurate.base_name(),
        ttnn::sigmoid_accurate.python_fully_qualified_name());

    bind_registered_operation(
        mod,
        ttnn::sigmoid_accurate,
        doc,
        ttnn::nanobind_overload_t{
            [](const unary_operation_t& self,
               const Tensor& input_tensor,
               bool fast_and_approximate_mode,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<Tensor>& output_tensor) -> ttnn::Tensor {
                return self(input_tensor, fast_and_approximate_mode, memory_config, output_tensor);
            },
            nb::arg("input_tensor"),
            nb::arg("fast_and_approximate_mode") = false,
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none()});
}

template <typename unary_operation_t>
void bind_sigmoid_mode_appx(nb::module_& mod) {
    auto doc = fmt::format(
        R"doc(
        Applies {0} to :attr:`input_tensor` element-wise.

        .. math::
            \mathrm{{output\_tensor}}_i = \verb|{0}|(\mathrm{{input\_tensor}}_i)

        Args:
            input_tensor (ttnn.Tensor): the input tensor.

        Keyword Args:
            vector_mode (int, optional): Use vector mode to get better performance. Defaults to 4. Use 2 or 4 for different vector modes (2 -> Vector Mode C and 4 -> Vector Mode RC)".
            fast_and_approximate_mode (bool, optional): Use the fast and approximate mode. Defaults to `False`.
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
               * - BFLOAT16
                 - TILE
                 - 2, 3, 4
        )doc",
        ttnn::sigmoid.base_name(),
        ttnn::sigmoid.python_fully_qualified_name());

    bind_registered_operation(
        mod,
        ttnn::sigmoid,
        doc,
        ttnn::nanobind_overload_t{
            [](const unary_operation_t& self,
               const Tensor& input_tensor,
               const int vector_mode,
               const bool parameter,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<Tensor>& output_tensor) -> ttnn::Tensor {
                return self(input_tensor, vector_mode, parameter, memory_config, output_tensor);
            },
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg("vector_mode") = 4,
            nb::arg("fast_and_approximate_mode") = false,
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none()});
}

template <typename unary_operation_t>
void bind_unary_chain(nb::module_& mod) {
    auto doc = fmt::format(
        R"doc(
        Applies {0} to :attr:`input_tensor` element-wise.

        .. math::
            \mathrm{{output\_tensor}}_i = \verb|{0}|(\mathrm{{input\_tensor}}_i)

        Args:
            input_tensor (ttnn.Tensor): the input tensor.
            ops_chain (list[ttnn.UnaryWithParam]): list of unary ops to be chained.

        Keyword Args:
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
               * - BFLOAT16, BFLOAT8_B
                 - TILE
                 - 2, 3, 4
        )doc",
        ttnn::unary_chain.base_name(),
        ttnn::unary_chain.python_fully_qualified_name());

    bind_registered_operation(
        mod,
        ttnn::unary_chain,
        doc,
        ttnn::nanobind_overload_t{
            [](const unary_operation_t& self,
               const Tensor& input_tensor,
               const EltwiseFusedActivations& ops_chain,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<Tensor>& output_tensor) {
                return self(input_tensor, ops_chain, memory_config, output_tensor);
            },
            nb::arg("input_tensor"),
            nb::arg("ops_chain"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none()});
}
template <typename unary_operation_t>
void bind_identity(nb::module_& mod) {
    auto doc = fmt::format(
        R"doc(
        Returns a copy of the :attr:`input_tensor`; useful for profiling the SFPU.
        This shouldn't normally be used. Users should normally use clone operation instead for the same functionality since this results in lower performance.

        .. math::
            \mathrm{{output\_tensor}}_i = \verb|{0}|(\mathrm{{input\_tensor}}_i)

        Args:
            input_tensor (ttnn.Tensor): the input tensor.

        Keyword Args:
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
               * - BFLOAT16, BFLOAT8_B, FLOAT32, UINT32, UINT16, UINT8
                 - TILE
                 - 2, 3, 4
        )doc",
        ttnn::identity.base_name(),
        ttnn::identity.python_fully_qualified_name());

    bind_registered_operation(
        mod,
        ttnn::identity,
        doc,
        ttnn::nanobind_overload_t{
            [](const unary_operation_t& self,
               const Tensor& input_tensor,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<Tensor>& output_tensor) { return self(input_tensor, memory_config, output_tensor); },
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none()});
}

template <typename unary_operation_t>
void bind_unary_composite(
    nb::module_& mod,
    const unary_operation_t& operation,
    const std::string& description,
    const std::string& range = "",
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& supported_layout = "TILE",
    const std::string& supported_rank = "2, 3, 4",
    const std::string& note = "",
    const std::string& example_tensor = "torch.rand([2, 2], dtype=torch.bfloat16)") {
    auto doc = fmt::format(
        R"doc(
        {2}

        .. math::
            \mathrm{{{{output\_tensor}}}}_i = \verb|{0}|(\mathrm{{{{input\_tensor}}}}_i)

        Args:
            input_tensor (ttnn.Tensor): the input tensor. {3}

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
               * - {4}
                 - {5}
                 - {6}

            {7}
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        description,
        range,
        supported_dtype,
        supported_layout,
        supported_rank,
        note,
        example_tensor);

    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const unary_operation_t& self,
               const Tensor& input_tensor,
               const std::optional<MemoryConfig>& memory_config) { return self(input_tensor, memory_config); },
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()});
}

// OpHandler_1int
template <typename unary_operation_t>
void bind_unary_composite_int_with_default(
    nb::module_& mod,
    const unary_operation_t& operation,
    const std::string& parameter_name_a,
    const std::string& parameter_a_doc,
    int32_t parameter_a_value,
    const std::string& description,
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& note = "") {
    auto doc = fmt::format(
        R"doc(
        {5}

        .. math::
            \mathrm{{{{output\_tensor}}}}_i = \verb|{0}|(\mathrm{{{{input\_tensor}}}}_i, \verb|{2}|)

        Args:
            input_tensor (ttnn.Tensor): the input tensor.

        Keyword args:
            {2} (float): {3}. Defaults to `{4}`.
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
            [](const unary_operation_t& self,
               const ttnn::Tensor& input_tensor,
               int32_t parameter_a,
               const std::optional<MemoryConfig>& memory_config) {
                return self(input_tensor, parameter_a, memory_config);
            },
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg(parameter_name_a.c_str()) = parameter_a_value,
            nb::arg("memory_config") = nb::none()});
}

// OpHandler_two_float_with_default
template <typename unary_operation_t>
void bind_unary_composite_floats_with_default(
    nb::module_& mod,
    const unary_operation_t& operation,
    const std::string& parameter_name_a,
    const std::string& parameter_a_doc,
    float parameter_a_value,
    const std::string& parameter_name_b,
    const std::string& parameter_b_doc,
    float parameter_b_value,
    const std::string& supported_dtype = "BFLOAT16, BFLOAT8_B",
    const std::string& info_doc = "") {
    auto doc = fmt::format(
        R"doc(
        Performs {0} function on :attr:`input_tensor`, :attr:`{2}`, :attr:`{5}`.

        Args:
            input_tensor (ttnn.Tensor): the input tensor.

        Keyword args:
            {2} (float, optional): {3}. Defaults to `{4}`.
            {5} (float, optional): {6}. Defaults to `{7}`.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

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
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        parameter_name_a,
        parameter_a_doc,
        parameter_a_value,
        parameter_name_b,
        parameter_b_doc,
        parameter_b_value,
        supported_dtype,
        info_doc);

    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const unary_operation_t& self,
               const ttnn::Tensor& input_tensor,
               float parameter_a,
               float parameter_b,
               const std::optional<MemoryConfig>& memory_config) {
                return self(input_tensor, parameter_a, parameter_b, memory_config);
            },
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg(parameter_name_a.c_str()) = parameter_a_value,
            nb::arg(parameter_name_b.c_str()) = parameter_b_value,
            nb::arg("memory_config") = nb::none()});
}

// OpHandler_one_int
template <typename unary_operation_t>
void bind_unary_composite_int(
    nb::module_& mod,
    const unary_operation_t& operation,
    const std::string& parameter_name_a,
    const std::string& parameter_a_doc,
    const std::string& description) {
    auto doc = fmt::format(
        R"doc(
        {4}

        Args:
            input_tensor (ttnn.Tensor): the input tensor.
            {2} (int): {3}.

        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

        Note:
            Supported dtypes, layouts, and ranks:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
                 - Ranks
               * - BFLOAT16
                 - TILE
                 - 2, 3, 4
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        parameter_name_a,
        parameter_a_doc,
        description);

    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const unary_operation_t& self,
               const ttnn::Tensor& input_tensor,
               int32_t parameter_a,
               const std::optional<MemoryConfig>& memory_config) {
                return self(input_tensor, parameter_a, memory_config);
            },
            nb::arg("input_tensor"),
            nb::arg(parameter_name_a.c_str()),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()});
}

// OpHandler_threshold
template <typename unary_operation_t>
void bind_unary_threshold(
    nb::module_& mod,
    const unary_operation_t& operation,
    const std::string& parameter_name_a,
    const std::string& parameter_a_doc,
    const std::string& parameter_name_b,
    const std::string& parameter_b_doc,
    const std::string& description) {
    auto doc = fmt::format(
        R"doc(
        {6}

        .. math::
            \mathrm{{{{output\_tensor}}}}_i = \verb|{0}|(\mathrm{{{{input\_tensor}}}}_i, \verb|{2}|, \verb|{4}|)

        Args:
            input_tensor (ttnn.Tensor): the input tensor.
            {2} (float): {3}.
            {4} (float): {5}.

        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

        Note:
            Supported dtypes, layouts, and ranks:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
                 - Ranks
               * - BFLOAT16
                 - TILE
                 - 2, 3, 4
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        parameter_name_a,
        parameter_a_doc,
        parameter_name_b,
        parameter_b_doc,
        description);

    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const unary_operation_t& self,
               const ttnn::Tensor& input_tensor,
               float parameter_a,
               float parameter_b,
               const std::optional<MemoryConfig>& memory_config) {
                return self(input_tensor, parameter_a, parameter_b, memory_config);
            },
            nb::arg("input_tensor"),
            nb::arg(parameter_name_a.c_str()),
            nb::arg(parameter_name_b.c_str()),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()});
}

// OpHandler_float_with_default
template <typename unary_operation_t>
void bind_unary_composite_float_with_default(
    nb::module_& mod,
    const unary_operation_t& operation,
    const std::string& parameter_name_a,
    const std::string& parameter_a_doc,
    float parameter_a_value,
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& info_doc = "") {
    auto doc = fmt::format(
        R"doc(
        Performs {0} function on :attr:`input_tensor`, :attr:`{2}`.

        .. math::
            \mathrm{{{{output\_tensor}}}}_i = \verb|{0}|(\mathrm{{{{input\_tensor}}}}_i, \verb|{2}|)

        Args:
            input_tensor (ttnn.Tensor): the input tensor.

        Keyword args:
            {2} (float, optional): {3}. Defaults to `{4}`.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

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
        parameter_name_a,
        parameter_a_doc,
        parameter_a_value,
        supported_dtype,
        info_doc);

    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const unary_operation_t& self,
               const ttnn::Tensor& input_tensor,
               float parameter_a,
               const std::optional<MemoryConfig>& memory_config) {
                return self(input_tensor, parameter_a, memory_config);
            },
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg(parameter_name_a.c_str()) = parameter_a_value,
            nb::arg("memory_config") = nb::none()});
}

template <typename unary_operation_t>
void bind_unary_logit(nb::module_& mod, const unary_operation_t& operation, const std::string& info_doc = "") {
    auto doc = fmt::format(
        R"doc(
        Performs {0} function on :attr:`input_tensor`, :attr:`eps`.

        Args:
            input_tensor (ttnn.Tensor): the input tensor.

        Keyword args:
            eps (float, optional): The epsilon for input clamp bound. Defaults to `None`.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

        Note:
            Supported dtypes, layouts, and ranks:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
                 - Ranks
               * - FLOAT32, BFLOAT16, BFLOAT8_B
                 - TILE
                 - 2, 3, 4


        Example:
            >>> tensor = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            >>> output = {1}(tensor, eps = None)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        info_doc);

    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const unary_operation_t& self,
               const ttnn::Tensor& input_tensor,
               std::optional<float> eps,
               const std::optional<MemoryConfig>& memory_config) { return self(input_tensor, eps, memory_config); },
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg("eps") = nb::none(),
            nb::arg("memory_config") = nb::none()});
}

template <typename unary_operation_t>
void bind_unary_composite_rpow(
    nb::module_& mod,
    const unary_operation_t& operation,
    const std::string& parameter_name_a,
    const std::string& parameter_a_doc,
    const std::string& description,
    const std::string& range,
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& info_doc = "") {
    auto doc = fmt::format(
        R"doc(
        {4}

        Args:
            input_tensor (ttnn.Tensor): the input tensor. {5}
            {2} (float): {3}

        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

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
        description,
        range,
        supported_dtype,
        info_doc);
    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const unary_operation_t& self,
               const ttnn::Tensor& input_tensor,
               float parameter_a,
               const std::optional<MemoryConfig>& memory_config) {
                return self(input_tensor, parameter_a, memory_config);
            },
            nb::arg("input_tensor"),
            nb::arg(parameter_name_a.c_str()),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()});
}
}  // namespace

void py_module(nb::module_& mod) {
    bind_unary_operation_overload_complex(
        mod,
        ttnn::abs,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|abs|(\mathrm{{input\_tensor}}_i))doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");
    bind_unary_operation(
        mod,
        ttnn::acos,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|acos|(\mathrm{{input\_tensor}}_i))doc",
        "",
        R"doc(FLOAT32, BFLOAT16, BFLOAT8_B)doc");
    bind_unary_operation(
        mod,
        ttnn::asin,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|asin|(\mathrm{{input\_tensor}}_i))doc",
        "",
        R"doc(FLOAT32, BFLOAT16, BFLOAT8_B)doc");
    bind_unary_operation(
        mod,
        ttnn::atan,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|atan|(\mathrm{{input\_tensor}}_i))doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B)doc");
    bind_unary_composite(
        mod,
        ttnn::atanh,
        R"doc(Performs atanh function on :attr:`input_tensor`.)doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");
    bind_unary_operation(
        mod,
        ttnn::cos,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|cos|(\mathrm{{input\_tensor}}_i))doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B)doc");
    bind_unary_operation(
        mod,
        ttnn::acosh,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|acosh|(\mathrm{{input\_tensor}}_i))doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");

    bind_unary_operation(
        mod,
        ttnn::erfinv,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|erfinv|(\mathrm{{input\_tensor}}_i))doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    bind_unary_operation(
        mod,
        ttnn::exp2,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|exp2|(\mathrm{{input\_tensor}}_i))doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    bind_unary_operation(
        mod,
        ttnn::expm1,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|expm1|(\mathrm{{input\_tensor}}_i))doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    bind_unary_operation_subcoregrids(
        mod,
        ttnn::floor,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|floor|(\mathrm{{input\_tensor}}_i))doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B)doc");
    bind_unary_operation_subcoregrids(
        mod,
        ttnn::trunc,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|trunc|(\mathrm{{input\_tensor}}_i))doc",
        "",
        R"doc(FLOAT32, BFLOAT16, BFLOAT8_B)doc");
    bind_unary_operation(
        mod,
        ttnn::frac,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|frac|(\mathrm{{input\_tensor}}_i))doc",
        "",
        R"doc(FLOAT32, BFLOAT16, BFLOAT8_B)doc");
    bind_unary_operation(
        mod,
        ttnn::eqz,
        R"doc(\mathrm{{output\_tensor}}_i = (\mathrm{{input\_tensor_i\ == 0}}))doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B, INT32, UINT16, UINT32)doc");
    bind_unary_operation(
        mod,
        ttnn::ceil,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|ceil|(\mathrm{{input\_tensor}}_i))doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B)doc");
    bind_unary_operation(
        mod,
        ttnn::mish,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|mish|(\mathrm{{input\_tensor}}_i))doc",
        "[Supported range -20 to inf]",
        R"doc(BFLOAT16, BFLOAT8_B)doc");
    bind_unary_operation(
        mod,
        ttnn::hardmish,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|hardmish|(\mathrm{{input\_tensor}}_i))doc",
        "[Supported range -20 to inf]",
        R"doc(BFLOAT16, BFLOAT8_B)doc");
    bind_unary_operation(
        mod,
        ttnn::gez,
        R"doc(\mathrm{{output\_tensor}}_i = (\mathrm{{input\_tensor_i\ >= 0}}))doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B, INT32)doc");
    bind_unary_operation(
        mod,
        ttnn::gtz,
        R"doc(\mathrm{{output\_tensor}}_i= (\mathrm{{input\_tensor_i\ > 0}}))doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B, INT32)doc");

    bind_unary_operation(
        mod,
        ttnn::i0,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|i0|(\mathrm{{input\_tensor}}_i))doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B)doc");
    bind_unary_operation(
        mod,
        ttnn::i1,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|i1|(\mathrm{{input\_tensor}}_i))doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B)doc");
    bind_unary_operation(
        mod,
        ttnn::isfinite,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|isfinite|(\mathrm{{input\_tensor}}_i))doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B)doc");
    bind_unary_operation(
        mod,
        ttnn::isinf,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|isinf|(\mathrm{{input\_tensor}}_i))doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B)doc");
    bind_unary_operation(
        mod,
        ttnn::isnan,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|isnan|(\mathrm{{input\_tensor}}_i))doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B)doc");
    bind_unary_operation(
        mod,
        ttnn::isneginf,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|isneginf|(\mathrm{{input\_tensor}}_i))doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B)doc");
    bind_unary_operation(
        mod,
        ttnn::isposinf,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|isposinf|(\mathrm{{input\_tensor}}_i))doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B)doc");
    bind_unary_operation(
        mod,
        ttnn::lez,
        R"doc(\mathrm{{output\_tensor}}_i = (\mathrm{{input\_tensor_i\ <= 0}}))doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B, INT32)doc");
    bind_unary_operation(
        mod,
        ttnn::logical_not,
        R"doc(\mathrm{{output\_tensor}}_i = \mathrm{{!input\_tensor_i}})doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B, INT32, UINT16 (range: 0 - 65535), UINT32 (range: 0 - 4294967295))doc");
    bind_unary_operation_subcoregrids(
        mod,
        ttnn::ltz,
        R"doc(\mathrm{{output\_tensor}}_i = (\mathrm{{input\_tensor_i\ < 0}}))doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B, INT32)doc");
    bind_unary_operation(
        mod,
        ttnn::neg,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|neg|(\mathrm{{input\_tensor}}_i))doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B)doc");
    bind_unary_operation(
        mod,
        ttnn::nez,
        R"doc(\mathrm{{output\_tensor}}_i = (\mathrm{{input\_tensor_i\ != 0}}))doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B, INT32, UINT16, UINT32)doc");

    bind_unary_operation_overload_complex_return_complex(
        mod,
        ttnn::reciprocal,
        R"doc(BFLOAT16, BFLOAT8_B)doc",
        R"doc(BFLOAT8_B is supported only for non-zero inputs. Inputs containing zero may produce inaccurate results due to the characteristics of the block-FP format.)doc");
    bind_unary_operation(
        mod,
        ttnn::relu,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|relu|(\mathrm{{input\_tensor}}_i))doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B)doc");
    bind_unary_operation(
        mod,
        ttnn::relu6,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|relu6|(\mathrm{{input\_tensor}}_i))doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B)doc");
    bind_unary_operation(
        mod,
        ttnn::sign,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|sign|(\mathrm{{input\_tensor}}_i))doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B)doc");
    bind_unary_operation(
        mod,
        ttnn::signbit,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|signbit|(\mathrm{{input\_tensor}}_i))doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B, INT32, FLOAT32)doc");
    bind_unary_operation(
        mod,
        ttnn::silu,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|silu|(\mathrm{{input\_tensor}}_i))doc",
        "",
        R"doc(FLOAT32, BFLOAT16, BFLOAT8_B)doc");
    bind_unary_operation(
        mod,
        ttnn::swish,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|swish|(\mathrm{{input\_tensor}}_i))doc",
        "",
        R"doc(FLOAT32, BFLOAT16, BFLOAT8_B)doc");
    bind_unary_operation(
        mod,
        ttnn::sin,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|sin|(\mathrm{{input\_tensor}}_i))doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    bind_unary_operation(
        mod,
        ttnn::square,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|square|(\mathrm{{input\_tensor}}_i))doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B, INT32, UINT32, UINT16 [0,255])doc");
    bind_unary_operation(
        mod,
        ttnn::tan,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|tan|(\mathrm{{input\_tensor}}_i))doc",
        "Supported input range is (-1.45, 1.45)",
        R"doc(BFLOAT16, BFLOAT8_B)doc");
    bind_unary_operation(
        mod,
        ttnn::log_sigmoid,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|log_sigmoid|(\mathrm{{input\_tensor}}_i))doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B)doc");
    bind_unary_operation(
        mod,
        ttnn::bitwise_not,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|bitwise_not|(\mathrm{{input\_tensor}}_i))doc",
        R"doc(Supported input range is [-2147483647, 2147483647].)doc",
        R"doc(INT32)doc",
        R"doc(torch.tensor([[1, 2], [3, 4]], dtype=torch.int32))doc");
    bind_unary_operation(
        mod,
        ttnn::alt_complex_rotate90,
        R"doc((\mathrm{{output\_tensor}}_{2i}, \mathrm{{output\_tensor}}_{2i+1}) = (-\mathrm{{input\_tensor}}_{2i+1}, \mathrm{{input\_tensor}}_{2i}))doc",
        R"doc(FLOAT32, BFLOAT16, BFLOAT8_B, BFLOAT4_B)doc",
        "",
        R"doc(The last dimension of the input tensor must be even.)doc");
    bind_unary_operation(
        mod,
        ttnn::deg2rad,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|deg2rad|(\mathrm{{input\_tensor}}_i))doc",
        "",
        R"doc(FLOAT32, BFLOAT16, BFLOAT8_B)doc");
    bind_unary_operation(
        mod,
        ttnn::rad2deg,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|rad2deg|(\mathrm{{input\_tensor}}_i))doc",
        "",
        R"doc(FLOAT32, BFLOAT16, BFLOAT8_B)doc");
    bind_unary_operation(
        mod,
        ttnn::asinh,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|asinh|(\mathrm{{input\_tensor}}_i))doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");
    bind_unary_operation(
        mod,
        ttnn::hardsigmoid,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|hardsigmoid|(\mathrm{{input\_tensor}}_i))doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");
    bind_unary_operation(
        mod,
        ttnn::hardswish,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|hardswish|(\mathrm{{input\_tensor}}_i))doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");
    bind_unary_operation(
        mod,
        ttnn::softsign,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|softsign|(\mathrm{{input\_tensor}}_i))doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");
    bind_unary_operation(
        mod,
        ttnn::cbrt,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|cbrt|(\mathrm{{input\_tensor}}_i))doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");

    //  Unaries with fast_and_approximate_mode
    bind_unary_operation_with_fast_and_approximate_mode(mod, ttnn::sqrt, "", R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");
    bind_unary_operation_with_fast_and_approximate_mode(mod, ttnn::rsqrt, "", R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");
    bind_unary_operation_with_fast_and_approximate_mode(mod, ttnn::exp, "", R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");
    bind_unary_operation_with_fast_and_approximate_mode(mod, ttnn::erf, "", R"doc(BFLOAT16, BFLOAT8_B)doc");
    bind_unary_operation_with_fast_and_approximate_mode(mod, ttnn::erfc, "", R"doc(BFLOAT16, BFLOAT8_B)doc");
    bind_unary_operation_with_fast_and_approximate_mode(mod, ttnn::gelu, "", R"doc(BFLOAT16, BFLOAT8_B)doc");
    bind_unary_operation_with_fast_and_approximate_mode(mod, ttnn::log, "", R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");
    bind_unary_operation_with_fast_and_approximate_mode(mod, ttnn::log10, "", R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");
    bind_unary_operation_with_fast_and_approximate_mode(mod, ttnn::log2, "", R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");
    bind_unary_operation_with_fast_and_approximate_mode(
        mod, ttnn::log1p, R"doc([Supported range: [-1, 1e7]])doc", R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");

    // Unaries with float parameter
    bind_unary_composite_with_default_float(
        mod, ttnn::elu, "alpha", "The alpha parameter for the ELU function", 1.0f, R"doc(BFLOAT16, BFLOAT8_B)doc");
    bind_unary_operation_with_float_parameter(
        mod,
        ttnn::heaviside,
        "value",
        "The value parameter for the Heaviside function",
        "",
        R"doc(BFLOAT16, BFLOAT8_B)doc");
    bind_unary_operation_with_float_parameter(
        mod,
        ttnn::leaky_relu,
        "negative_slope",
        "The slope parameter for the Leaky ReLU function",
        "",
        R"doc(FLOAT32, BFLOAT16, BFLOAT8_B)doc");
    bind_unary_operation_with_float_parameter(
        mod,
        ttnn::relu_max,
        "upper_limit",
        "The max value for ReLU function",
        "This function caps off the input to a max value and a min value of 0",
        R"doc(BFLOAT16, BFLOAT8_B)doc",
        R"doc(System memory is not supported.)doc");
    bind_unary_operation_with_float_parameter(
        mod,
        ttnn::relu_min,
        "lower_limit",
        "The min value for ReLU function",
        "This will carry out ReLU operation at min value instead of the standard 0",
        R"doc(BFLOAT16)doc",
        R"doc(System memory is not supported.)doc");
    bind_unary_operation_with_float_parameter(
        mod, ttnn::rpow, "exponent", "exponent value. Non-positive values are not supported.", "");
    bind_unary_operation_with_float_parameter_default(
        mod,
        ttnn::celu,
        "alpha",
        "The alpha parameter for the CELU function",
        1.0f,
        "",
        R"doc(FLOAT32, BFLOAT16, BFLOAT8_B)doc");

    bind_unary_operation_with_scalar_parameter(
        mod,
        ttnn::fill,
        "fill_value",
        "The value to be filled in the output tensor",
        "This will create a tensor of same shape and dtype as input reference tensor with fill_value.",
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32, INT32, UINT32)doc",
        R"doc(Host memory is not supported.)doc");

    // Unary ops with dim parameter
    bind_unary_operation_with_dim_parameter(
        mod,
        ttnn::glu,
        "dim",
        "Dimension to split input tensor. Supported only for last dimension (dim = -1 or 3)",
        "Split the tensor into two parts, apply the GLU function on the second tensor, and then perform multiplication "
        "with the first tensor.",
        R"doc(BFLOAT16, BFLOAT8_B)doc",
        R"doc(System memory is not supported.

           Last dimension of input tensor should be divisible by 64.

        )doc");

    bind_unary_operation_with_dim_parameter(
        mod,
        ttnn::reglu,
        "dim",
        "Dimension to split input tensor. Supported only for last dimension (dim = -1 or 3)",
        "Split the tensor into two parts, apply the ReLU function on the second tensor, and then perform "
        "multiplication with the first tensor.",
        R"doc(BFLOAT16, BFLOAT8_B)doc",
        R"doc(System memory is not supported.

           Last dimension of input tensor should be divisible by 64.

        )doc");

    bind_unary_operation_with_dim_parameter(
        mod,
        ttnn::geglu,
        "dim",
        "Dimension to split input tensor. Supported only for last dimension (dim = -1 or 3)",
        "Split the tensor into two parts, apply the GELU function on the second tensor, and then perform "
        "multiplication with the first tensor.",
        R"doc(BFLOAT16, BFLOAT8_B)doc",
        R"doc(System memory is not supported.

           Last dimension of input tensor should be divisible by 64.

        )doc");

    bind_unary_operation_with_dim_parameter(
        mod,
        ttnn::swiglu,
        "dim",
        "Dimension to split input tensor. Supported only for last dimension (dim = -1 or 3)",
        "Split the tensor into two parts, apply the SiLU function on the second tensor, and then perform "
        "multiplication with the first tensor.",
        R"doc(BFLOAT16, BFLOAT8_B)doc",
        R"doc(System memory is not supported.

           Last dimension of input tensor should be divisible by 64.

        )doc");

    // Other unaries (unary chain operations)
    bind_softplus<decltype(ttnn::softplus)>(mod);
    bind_tanh_like(mod, ttnn::tanh);
    bind_tanh_like(mod, ttnn::tanhshrink);
    bind_sigmoid_accurate<decltype(ttnn::sigmoid_accurate)>(mod);
    bind_sigmoid_mode_appx<decltype(ttnn::sigmoid)>(mod);

    bind_unary_chain<decltype(ttnn::unary_chain)>(mod);
    bind_identity<decltype(ttnn::identity)>(mod);

    // unary composite imported into ttnn
    bind_unary_composite(
        mod,
        ttnn::cosh,
        R"doc(Performs cosh function on :attr:`input_tensor`.)doc",
        "[supported range -9 to 9]",
        R"doc(BFLOAT16, BFLOAT8_B)doc");
    bind_unary_composite(
        mod,
        ttnn::digamma,
        R"doc(Performs digamma function on :attr:`input_tensor`.)doc",
        "[supported for values greater than 0].",
        R"doc(BFLOAT16, BFLOAT8_B)doc",
        R"doc(TILE)doc",
        R"doc(2, 3, 4)doc",
        "",
        R"doc(torch.tensor([[2, 3], [4, 5]], dtype=torch.bfloat16))doc");
    bind_unary_composite(
        mod,
        ttnn::lgamma,
        R"doc(Performs lgamma function on :attr:`input_tensor`.)doc",
        "[supported for value greater than 0].",
        R"doc(BFLOAT16)doc");
    bind_unary_composite(
        mod,
        ttnn::multigammaln,
        R"doc(Performs multigammaln function on :attr:`input_tensor`.)doc",
        "[supported range 1.6 to inf].",
        R"doc(BFLOAT16)doc",
        R"doc(TILE)doc",
        R"doc(2, 3, 4)doc",
        "",
        R"doc(torch.tensor([[2, 3], [4, 5]], dtype=torch.bfloat16))doc");
    bind_unary_composite(
        mod,
        ttnn::sinh,
        R"doc(Performs sinh function on :attr:`input_tensor`.)doc",
        "[supported range -9 to 9].",
        R"doc(BFLOAT16, BFLOAT8_B)doc");
    bind_unary_composite(mod, ttnn::var_hw, R"doc(Performs var_hw function on :attr:`input_tensor`.)doc");
    bind_unary_composite(mod, ttnn::std_hw, R"doc(Performs std_hw function on :attr:`input_tensor`.)doc");
    bind_unary_composite(
        mod,
        ttnn::normalize_hw,
        R"doc(Performs normalize_hw function on :attr:`input_tensor`.)doc",
        "",
        R"doc(BFLOAT16)doc",
        R"doc(ROW_MAJOR, TILE)doc",
        R"doc(4)doc",
        "",
        R"doc(torch.rand([1, 1, 32, 32], dtype=torch.bfloat16))doc");
    bind_unary_composite(
        mod,
        ttnn::logical_not_,
        R"doc(Performs logical_not inplace function on :attr:`input_tensor`.)doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B, INT32, UINT32)doc");
    bind_unary_composite(
        mod,
        ttnn::normalize_global,
        R"doc(Performs normalize_global function on :attr:`input_tensor`.)doc",
        "",
        R"doc(BFLOAT16)doc",
        R"doc(ROW_MAJOR, TILE)doc",
        R"doc(4)doc",
        "",
        R"doc(torch.rand([1, 1, 32, 32], dtype=torch.bfloat16))doc");

    bind_unary_composite_optional_floats_with_default(
        mod,
        ttnn::clip,
        "min",
        "Minimum value",
        std::nullopt,
        "max",
        "Maximum value",
        std::nullopt,
        R"doc(Performs clip function on :attr:`input_tensor`, :attr:`min`, :attr:`max`. Only one of 'min' or 'max' value can be None.)doc");
    bind_unary_clamp(mod, ttnn::clamp);
    bind_unary_composite_floats_with_default(
        mod, ttnn::selu, "scale", "Scale value", 1.0507, "alpha", "Alpha value", 1.67326);
    bind_unary_composite_floats_with_default(
        mod,
        ttnn::hardtanh,
        "min_val",
        "min value",
        -1.0f,
        "max_val",
        "max value",
        1.0f,
        R"doc(FLOAT32, BFLOAT16, BFLOAT8_B)doc");
    bind_unary_threshold(
        mod,
        ttnn::threshold,
        "threshold",
        "Threshold value",
        "value",
        "Replacing value",
        R"doc(Performs threshold function on :attr:`input_tensor`, :attr:`threshold`, :attr:`value`.)doc");
    bind_unary_composite_int_with_default(
        mod,
        ttnn::tril,
        "diagonal",
        "diagonal value",
        0,
        R"doc(Performs tril function on :attr:`input_tensor`, :attr:`diagonal`.)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");
    bind_unary_composite_int_with_default(
        mod,
        ttnn::triu,
        "diagonal",
        "diagonal value",
        0,
        R"doc(Performs triu function on :attr:`input_tensor`, :attr:`diagonal`.)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");
    bind_unary_operation_with_int_parameter(
        mod,
        ttnn::round,
        "decimals",
        "No. of decimal places to round off to [supported range -6 to 7], Defaults to 0.",
        R"doc(Round the input tensor to `decimals` decimal places.)doc",
        R"doc(FLOAT32, BFLOAT16, BFLOAT8_B)doc");
    bind_unary_composite_int(
        mod,
        ttnn::polygamma,
        "k",
        "k value",
        R"doc(Performs polygamma function on :attr:`input_tensor`, :attr:`decimals`. it is supported for range 1 to 10 only)doc");

    // unary composite with float imported into ttnn
    bind_unary_composite_float_with_default(
        mod, ttnn::hardshrink, "lambd", "lambd value", 0.5f, R"doc(FLOAT32, BFLOAT16, BFLOAT8_B)doc");

    bind_unary_composite_float_with_default(
        mod, ttnn::softshrink, "lambd", "lambd value", 0.5f, R"doc(FLOAT32, BFLOAT16, BFLOAT8_B)doc");

    bind_unary_logit(mod, ttnn::logit);

    bind_unary_rdiv(
        mod,
        ttnn::rdiv,
        "value",
        "denominator that is considered as numerator, which should be a non-zero float value",
        "round_mode",
        "rounding_mode value",
        "None",
        R"doc(Performs the element-wise division of a scalar ``value`` by a tensor ``input`` and rounds the result using round_mode.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.)doc",

        R"doc(BFLOAT16)doc",
        R"doc(System memory is not supported.)doc");

    // Bind bitcast operation
    auto bitcast_doc = fmt::format(
        R"doc(
        Bitcast reinterprets the bit pattern without conversion (unlike typecast which converts values).

        Args:
            input_tensor (ttnn.Tensor): the input tensor.
            dtype (ttnn.DataType): output data type. Must have the same bit size as input dtype. Supported pairs: UINT16 <-> BFLOAT16 (both 16 bits), UINT32 <-> FLOAT32 (both 32 bits), UINT32 <-> INT32 (both 32 bits).

        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
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
               * - BFLOAT16, FLOAT32, INT32, UINT16, UINT32
                 - TILE
                 - 2, 3, 4
        )doc",
        ttnn::bitcast.base_name());

    using BitcastType = decltype(ttnn::bitcast);
    bind_registered_operation(
        mod,
        ttnn::bitcast,
        bitcast_doc,
        ttnn::nanobind_overload_t{
            [](const BitcastType& self,
               const ttnn::Tensor& input_tensor,
               const DataType dtype,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor) -> ttnn::Tensor {
                return self(input_tensor, dtype, memory_config, output_tensor);
            },
            nb::arg("input_tensor"),
            nb::arg("dtype"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none()});
}

}  // namespace ttnn::operations::unary
