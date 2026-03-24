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

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/eltwise/unary/unary_composite.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operations/eltwise/complex/complex.hpp"
#include "unary.hpp"

namespace ttnn::operations::unary {

namespace {
template <Tensor (*Func)(
    const Tensor&,
    const std::optional<MemoryConfig>&,
    const std::optional<Tensor>&,
    const std::optional<CoreRangeSet>&)>
Tensor unary_3param_wrapper(const Tensor& t, const std::optional<MemoryConfig>& m, const std::optional<Tensor>& o) {
    return Func(t, m, o, std::nullopt);
}

template <Tensor (*Func)(
    const Tensor&,
    const std::optional<MemoryConfig>&,
    const std::optional<Tensor>&,
    const std::optional<CoreRangeSet>&)>
Tensor unary_composite_2param_wrapper(const Tensor& t, const std::optional<MemoryConfig>& m) {
    return Func(t, m, std::nullopt, std::nullopt);
}

template <auto Func>
Tensor unary_4param_to_5param_wrapper(
    const Tensor& input_tensor,
    float parameter,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output_tensor) {
    return Func(input_tensor, parameter, memory_config, output_tensor, std::nullopt);
}

template <auto Func>
Tensor unary_3param_to_4param_wrapper(
    const Tensor& input_tensor, float parameter, const std::optional<MemoryConfig>& memory_config) {
    return Func(input_tensor, parameter, memory_config, std::nullopt);
}

template <auto Func>
Tensor unary_composite_3param_to_4param_wrapper(
    const Tensor& input_tensor,
    float parameter_a,
    float parameter_b,
    const std::optional<MemoryConfig>& memory_config) {
    return Func(input_tensor, parameter_a, parameter_b, memory_config, std::nullopt);
}

void bind_unary_clamp(nb::module_& mod) {
    const char* doc = R"doc(
        Applies clamp to :attr:`input_tensor` element-wise.

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
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - BFLOAT16, BFLOAT8_B, INT32, FLOAT32
                 - TILE, ROW_MAJOR

            INT32 is supported only for Tensor-scalar-scalar version.
        )doc";

    ttnn::bind_function<"clamp">(
        mod,
        doc,
        ttnn::overload_t(
            nb::overload_cast<
                const Tensor&,
                std::optional<Tensor>,
                std::optional<Tensor>,
                const std::optional<MemoryConfig>&,
                const std::optional<Tensor>&>(&ttnn::clamp),
            nb::arg("input_tensor"),
            nb::arg("min") = nb::none(),
            nb::arg("max") = nb::none(),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none()),
        ttnn::overload_t(
            nb::overload_cast<
                const Tensor&,
                std::optional<std::variant<float, int32_t>>,
                std::optional<std::variant<float, int32_t>>,
                const std::optional<MemoryConfig>&,
                const std::optional<Tensor>&>(&ttnn::clamp),
            nb::arg("input_tensor"),
            nb::arg("min") = nb::none(),
            nb::arg("max") = nb::none(),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none()));
}

void bind_unary_clip(nb::module_& mod) {
    const char* doc = R"doc(
        Performs clip function on :attr:`input_tensor`, :attr:`min`, :attr:`max`. Only one of 'min' or 'max' value can be None.

        Args:
            input_tensor (ttnn.Tensor): the input tensor.

        Keyword args:
            min (float or ttnn.Tensor): Minimum value. Defaults to `None`.
            max (float or ttnn.Tensor): Maximum value. Defaults to `None`.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

        Note:
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - BFLOAT16
                 - TILE, ROW_MAJOR

        )doc";

    ttnn::bind_function<"clip">(
        mod,
        doc,
        ttnn::overload_t{
            nb::overload_cast<
                const Tensor&,
                std::optional<Tensor>,
                std::optional<Tensor>,
                const std::optional<MemoryConfig>&>(&ttnn::clip),
            nb::arg("input_tensor"),
            nb::arg("min") = nb::none(),
            nb::arg("max") = nb::none(),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()},
        ttnn::overload_t{
            nb::overload_cast<
                const Tensor&,
                std::optional<float>,
                std::optional<float>,
                const std::optional<MemoryConfig>&>(&ttnn::clip),
            nb::arg("input_tensor"),
            nb::arg("min") = nb::none(),
            nb::arg("max") = nb::none(),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()});
}

template <
    ttnn::unique_string OpName,
    Tensor (*Func)(const Tensor&, const std::optional<MemoryConfig>&, const std::optional<Tensor>&)>
void bind_unary_operation_3param(
    nb::module_& mod,
    const std::string& math,
    const std::string& range = "",
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& note = "") {
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
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - {4}
                 - TILE, ROW_MAJOR

            {5}

        )doc",
        std::string(OpName),
        std::string("ttnn.") + std::string(OpName),
        math,
        range,
        supported_dtype,
        note);

    ttnn::bind_function<OpName>(
        mod,
        doc.c_str(),
        Func,
        nb::arg("input_tensor"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("output_tensor") = nb::none());
}

template <
    ttnn::unique_string OpName,
    Tensor (*Func)(
        const Tensor&,
        const std::optional<MemoryConfig>&,
        const std::optional<Tensor>&,
        const std::optional<CoreRangeSet>&)>
void bind_unary_operation(
    nb::module_& mod,
    const std::string& math,
    const std::string& range = "",
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& note = "") {
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
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - {4}
                 - TILE, ROW_MAJOR

            {5}

        )doc",
        std::string(OpName),
        std::string("ttnn.") + std::string(OpName),
        math,
        range,
        supported_dtype,
        note);

    ttnn::bind_function<OpName>(
        mod,
        doc.c_str(),
        &unary_3param_wrapper<Func>,
        nb::arg("input_tensor"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("output_tensor") = nb::none());
}

template <ttnn::unique_string OpName, typename Func>
void bind_unary_operation_subcoregrids(
    nb::module_& mod,
    Func func,
    const std::string& math,
    const std::string& range = "",
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& note = "") {
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
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - {4}
                 - TILE, ROW_MAJOR

            {5}

        )doc",
        std::string(OpName),
        std::string("ttnn.") + std::string(OpName),
        math,
        range,
        supported_dtype,
        note);

    ttnn::bind_function<OpName>(
        mod,
        doc.c_str(),
        func,
        nb::arg("input_tensor"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("output_tensor") = nb::none(),
        nb::arg("sub_core_grids") = nb::none());
}

template <ttnn::unique_string OpName, typename FuncTensor, typename FuncComplex>
void bind_unary_operation_overload_complex(
    nb::module_& mod,
    FuncTensor func_tensor,
    FuncComplex func_complex,
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
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - {3}
                 - TILE, ROW_MAJOR

            {4}
        )doc",
        std::string(OpName),
        std::string("ttnn.") + std::string(OpName),
        math,
        supported_dtype,
        note);

    ttnn::bind_function<OpName>(
        mod,
        doc.c_str(),
        ttnn::overload_t(
            func_tensor,
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none()),
        ttnn::overload_t(func_complex, nb::arg("input_tensor"), nb::kw_only(), nb::arg("memory_config")));
}

template <ttnn::unique_string OpName>
void bind_unary_operation_overload_complex_return_complex(
    nb::module_& mod, const std::string& supported_dtype = "BFLOAT16", const std::string& info_doc = "") {
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
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - {2}
                 - TILE, ROW_MAJOR

            {3}
            More information about the `BFLOAT8_B  <../tensor.html#limitation-of-bfloat8-b>`_.
        )doc",
        std::string(OpName),
        std::string("ttnn.") + std::string(OpName),
        supported_dtype,
        info_doc);

    ttnn::bind_function<OpName>(
        mod,
        doc.c_str(),
        ttnn::overload_t{
            &unary_3param_wrapper<ttnn::reciprocal>,
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none()},
        ttnn::overload_t{
            nb::overload_cast<const ComplexTensor&, const ttnn::MemoryConfig&>(&ttnn::reciprocal),
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg("memory_config")});
}

template <ttnn::unique_string OpName, auto Func>
void bind_unary_operation_with_fast_and_approximate_mode(
    nb::module_& mod,
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
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - {3}
                 - TILE, ROW_MAJOR

            {4}
        )doc",
        std::string(OpName),
        std::string("ttnn.") + std::string(OpName),
        range,
        supported_dtype,
        info_doc);

    ttnn::bind_function<OpName>(
        mod,
        doc.c_str(),
        ttnn::overload_t{
            Func,
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg("fast_and_approximate_mode") = false,
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none(),
            nb::arg("sub_core_grids") = nb::none()});
}

template <ttnn::unique_string OpName, auto Func>
void bind_unary_operation_with_float_parameter(
    nb::module_& mod,
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
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - {5}
                 - TILE, ROW_MAJOR

            {6}
        )doc",
        std::string(OpName),
        std::string("ttnn.") + std::string(OpName),
        parameter_name,
        parameter_doc,
        info_doc,
        supported_dtype,
        note);

    ttnn::bind_function<OpName>(
        mod,
        doc.c_str(),
        ttnn::overload_t{
            &unary_4param_to_5param_wrapper<Func>,
            nb::arg("input_tensor"),
            nb::arg(parameter_name.c_str()),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none()});
}

template <ttnn::unique_string OpName, auto Func>
void bind_unary_operation_with_scalar_parameter(
    nb::module_& mod,
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
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - {5}
                 - TILE, ROW_MAJOR

            {6}
        )doc",
        std::string(OpName),
        std::string("ttnn.") + std::string(OpName),
        parameter_name,
        parameter_doc,
        info_doc,
        supported_dtype,
        note);

    ttnn::bind_function<OpName>(
        mod,
        doc.c_str(),
        ttnn::overload_t{
            Func,
            nb::arg("input_tensor"),
            nb::arg(parameter_name.c_str()),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none()});
}

template <ttnn::unique_string OpName, auto Func>
void bind_unary_operation_with_float_parameter_default(
    nb::module_& mod,
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
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - {6}
                 - TILE, ROW_MAJOR

            {7}
        )doc",
        std::string(OpName),
        std::string("ttnn.") + std::string(OpName),
        parameter_name,
        parameter_default,
        parameter_doc,
        info_doc,
        supported_dtype,
        note);

    ttnn::bind_function<OpName>(
        mod,
        doc.c_str(),
        ttnn::overload_t{
            &unary_4param_to_5param_wrapper<Func>,
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg(parameter_name.data()) = parameter_default,
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none()});
}

template <ttnn::unique_string OpName, auto Func>
void bind_unary_composite_with_default_float(
    nb::module_& mod,
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
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - {5}
                 - TILE, ROW_MAJOR

            {6}
        )doc",
        std::string(OpName),
        std::string("ttnn.") + std::string(OpName),
        parameter_name_a,
        parameter_a_doc,
        parameter_a_value,
        supported_dtype,
        info_doc);

    ttnn::bind_function<OpName>(
        mod,
        doc.c_str(),
        ttnn::overload_t{
            Func,
            nb::arg("input_tensor"),
            nb::arg(parameter_name_a.c_str()) = parameter_a_value,
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none()});
}

template <ttnn::unique_string OpName, auto Func>
void bind_unary_operation_with_int_parameter(
    nb::module_& mod,
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
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - {5}
                 - TILE, ROW_MAJOR

            {6}
        )doc",
        std::string(OpName),
        std::string("ttnn.") + std::string(OpName),
        parameter_name,
        parameter_doc,
        info_doc,
        supported_dtype,
        note);

    ttnn::bind_function<OpName>(
        mod,
        doc.c_str(),
        ttnn::overload_t{
            Func,
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg(parameter_name.c_str()) = 0,
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none()});
}

template <ttnn::unique_string OpName, auto Func>
void bind_unary_operation_with_dim_parameter(
    nb::module_& mod,
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
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - {5}
                 - TILE, ROW_MAJOR

            {6}
        )doc",
        std::string(OpName),
        std::string("ttnn.") + std::string(OpName),
        parameter_name,
        parameter_doc,
        info_doc,
        supported_dtype,
        note);

    ttnn::bind_function<OpName>(
        mod,
        doc.c_str(),
        ttnn::overload_t{
            Func, nb::arg("input_tensor"), nb::arg("dim") = -1, nb::kw_only(), nb::arg("memory_config") = nb::none()});
}

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
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - BFLOAT16, BFLOAT8_B, FLOAT32
                 - TILE, ROW_MAJOR
        )doc",
        "softplus",
        "ttnn.softplus");

    ttnn::bind_function<"softplus">(
        mod,
        doc.c_str(),
        &ttnn::softplus,
        nb::arg("input_tensor"),
        nb::kw_only(),
        nb::arg("beta") = 1.0f,
        nb::arg("threshold") = 20.0f,
        nb::arg("memory_config") = nb::none(),
        nb::arg("output_tensor") = nb::none());
}

void bind_xielu(nb::module_& mod) {
    auto doc = fmt::format(
        R"doc(
        Applies xIELU (Expanded Integral of the Exponential Linear Unit) to :attr:`input_tensor` element-wise.
        This is a custom piecewise trainable activation function derived from "Deriving Activation Functions Using Integration" paper:
        https://arxiv.org/abs/2411.13010

        With beta = 0.5 and eps = -1e-6:
            x > 0 :  alpha_p * x^2 + beta * x
            x <= 0:  alpha_n * (expm1(minimum(x, eps))) - (alpha_n * x) + 0.5 * x

        Args:
            input_tensor (ttnn.Tensor): the input tensor.

        Keyword Args:
            alpha_p (float, optional): Alpha positive constant. Defaults to `0.8`.
            alpha_n (float, optional): Alpha negative constant. Defaults to `0.8`.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): preallocated output tensor. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

        Note:
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - BFLOAT16, FLOAT32
                 - TILE, ROW_MAJOR
        )doc");

    ttnn::bind_function<"xielu">(
        mod,
        doc.c_str(),
        ttnn::overload_t(
            &ttnn::xielu,
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg("alpha_p") = 0.8f,
            nb::arg("alpha_n") = 0.8f,
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none()));
}

template <ttnn::unique_string OpName, auto Func>
void bind_unary_rdiv(
    nb::module_& mod,
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
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - {8}
                 - TILE, ROW_MAJOR

            {9}
        )doc",
        std::string(OpName),
        std::string("ttnn.") + std::string(OpName),
        parameter_name_a,
        parameter_a_doc,
        parameter_name_b,
        parameter_b_doc,
        parameter_b_value,
        description,
        supported_dtype,
        note);

    ttnn::bind_function<OpName>(
        mod,
        doc.c_str(),
        ttnn::overload_t{
            Func,
            nb::arg("input_tensor"),
            nb::arg(parameter_name_a.c_str()),
            nb::kw_only(),
            nb::arg(parameter_name_b.c_str()) = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none()});
}

template <ttnn::unique_string OpName, auto Func>
void bind_tanh_like(nb::module_& mod) {
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
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - BFLOAT16, BFLOAT8_B, FLOAT32
                 - TILE, ROW_MAJOR

            BFLOAT8_B/BFLOAT4_B is supported only for approx=True mode.
        )doc",
        std::string(OpName),
        std::string("ttnn.") + std::string(OpName));

    ttnn::bind_function<OpName>(
        mod,
        doc.c_str(),
        ttnn::overload_t{
            Func,
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none(),
            nb::arg("fast_and_approximate_mode") = false});
}

void bind_sigmoid_accurate(nb::module_& mod) {
    auto doc = fmt::format(
        R"doc(
        Deprecated in favor of ttnn.sigmoid.

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
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - BFLOAT16, BFLOAT8_B, FLOAT32
                 - TILE, ROW_MAJOR
        )doc",
        "sigmoid_accurate",
        "ttnn.sigmoid_accurate");

    ttnn::bind_function<"sigmoid_accurate">(
        mod,
        doc.c_str(),
        &ttnn::sigmoid_accurate,
        nb::arg("input_tensor"),
        nb::arg("fast_and_approximate_mode") = false,
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("output_tensor") = nb::none());
}

void bind_sigmoid(nb::module_& mod) {
    auto doc = fmt::format(
        R"doc(
        Applies {0} to :attr:`input_tensor` element-wise.

        .. math::
            \mathrm{{output\_tensor}}_i = \verb|{0}|(\mathrm{{input\_tensor}}_i)

        Args:
            input_tensor (ttnn.Tensor): the input tensor.

        Keyword Args:
            vector_mode (int, optional): Use vector mode to get better performance. Defaults to 4. Use 2 or 4 for different vector modes (2 -> Vector Mode C and 4 -> Vector Mode RC)".
            mode (ttnn.SigmoidMode, optional): Select sigmoid mode to use. Defaults to `SigmoidMode.Accurate`.
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the operation. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): preallocated output tensor. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

        Note:
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - BFLOAT16
                 - TILE, ROW_MAJOR
        )doc",
        "sigmoid",
        "ttnn.sigmoid");

    mod.attr("SigmoidMode") =
        nb::enum_<SigmoidMode>(mod, "SigmoidMode")
            .value("Accurate", SigmoidMode::ACCURATE, "Most accurate, but least performant.")
            .value(
                "AccurateWithFastExp",
                SigmoidMode::ACCURATE_FAST_EXP,
                "Similar to accurate, but uses fast and approximate exp")
            .value("FastApproximate", SigmoidMode::FAST_APPROXIMATE, "Fastest, but least accurate.");

    ttnn::bind_function<"sigmoid">(
        mod,
        doc.c_str(),
        &ttnn::sigmoid,
        nb::arg("input_tensor"),
        nb::kw_only(),
        nb::arg("vector_mode") = 4,
        nb::arg("mode") = SigmoidMode::ACCURATE,
        nb::arg("memory_config") = nb::none(),
        nb::arg("output_tensor") = nb::none());
}

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
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - BFLOAT16, BFLOAT8_B, FLOAT32
                 - TILE, ROW_MAJOR
        )doc",
        "unary_chain",
        "ttnn.unary_chain");

    ttnn::bind_function<"unary_chain">(
        mod,
        doc.c_str(),
        &ttnn::unary_chain,
        nb::arg("input_tensor"),
        nb::arg("ops_chain"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("output_tensor") = nb::none());
}
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
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - BFLOAT16, BFLOAT8_B, FLOAT32, UINT32, UINT16, UINT8
                 - TILE, ROW_MAJOR
        )doc",
        "identity",
        "ttnn.identity");

    ttnn::bind_function<"identity">(
        mod,
        doc.c_str(),
        &ttnn::identity,
        nb::arg("input_tensor"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("output_tensor") = nb::none());
}

template <ttnn::unique_string OpName, Tensor (*Func)(const Tensor&, const std::optional<MemoryConfig>&)>
void bind_unary_composite_2param(
    nb::module_& mod,
    const std::string& description,
    const std::string& range = "",
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& supported_layout = "TILE",
    const std::string& note = "") {
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
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - {4}
                 - {5}

            {6}
        )doc",
        std::string(OpName),
        std::string("ttnn.") + std::string(OpName),
        description,
        range,
        supported_dtype,
        supported_layout,
        note);

    ttnn::bind_function<OpName>(
        mod, doc.c_str(), Func, nb::arg("input_tensor"), nb::kw_only(), nb::arg("memory_config") = nb::none());
}

template <
    ttnn::unique_string OpName,
    Tensor (*Func)(
        const Tensor&,
        const std::optional<MemoryConfig>&,
        const std::optional<Tensor>&,
        const std::optional<CoreRangeSet>&)>
void bind_unary_composite(
    nb::module_& mod,
    const std::string& description,
    const std::string& range = "",
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& supported_layout = "TILE",
    const std::string& note = "") {
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
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - {4}
                 - {5}

            {6}
        )doc",
        std::string(OpName),
        std::string("ttnn.") + std::string(OpName),
        description,
        range,
        supported_dtype,
        supported_layout,
        note);

    ttnn::bind_function<OpName>(
        mod,
        doc.c_str(),
        &unary_composite_2param_wrapper<Func>,
        nb::arg("input_tensor"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none());
}

// OpHandler_1int
template <ttnn::unique_string OpName, auto Func>
void bind_unary_composite_int_with_default(
    nb::module_& mod,
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
            \mathrm{{{{output\_tensor}}}}_i = \verb|{0}|(\mathrm{{{{input\_tensor}}}}_i, \mathrm{{{2}}})

        Args:
            input_tensor (ttnn.Tensor): the input tensor.

        Keyword args:
            {2} (int, optional): {3}. Defaults to `{4}`.
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the operation. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

        Note:
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - {6}
                 - TILE, ROW_MAJOR

            {7}
        )doc",
        std::string(OpName),
        std::string("ttnn.") + std::string(OpName),
        parameter_name_a,
        parameter_a_doc,
        parameter_a_value,
        description,
        supported_dtype,
        note);

    ttnn::bind_function<OpName>(
        mod,
        doc.c_str(),
        ttnn::overload_t{
            Func,
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg(parameter_name_a.c_str()) = parameter_a_value,
            nb::arg("memory_config") = nb::none()});
}

// OpHandler_two_float_with_default
template <ttnn::unique_string OpName, auto Func>
void bind_unary_composite_floats_with_default(
    nb::module_& mod,
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
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - {8}
                 - TILE, ROW_MAJOR

            {9}
        )doc",
        std::string(OpName),
        std::string("ttnn.") + std::string(OpName),
        parameter_name_a,
        parameter_a_doc,
        parameter_a_value,
        parameter_name_b,
        parameter_b_doc,
        parameter_b_value,
        supported_dtype,
        info_doc);

    ttnn::bind_function<OpName>(
        mod,
        doc.c_str(),
        ttnn::overload_t{
            &unary_composite_3param_to_4param_wrapper<Func>,
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg(parameter_name_a.c_str()) = parameter_a_value,
            nb::arg(parameter_name_b.c_str()) = parameter_b_value,
            nb::arg("memory_config") = nb::none()});
}

// OpHandler_one_int
template <ttnn::unique_string OpName, auto Func>
void bind_unary_composite_int(
    nb::module_& mod,
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
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - BFLOAT16
                 - TILE, ROW_MAJOR
        )doc",
        std::string(OpName),
        std::string("ttnn.") + std::string(OpName),
        parameter_name_a,
        parameter_a_doc,
        description);

    ttnn::bind_function<OpName>(
        mod,
        doc.c_str(),
        ttnn::overload_t{
            Func,
            nb::arg("input_tensor"),
            nb::arg(parameter_name_a.c_str()),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()});
}

// OpHandler_threshold
template <ttnn::unique_string OpName, auto Func>
void bind_unary_threshold(
    nb::module_& mod,
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
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - BFLOAT16
                 - TILE, ROW_MAJOR
        )doc",
        std::string(OpName),
        std::string("ttnn.") + std::string(OpName),
        parameter_name_a,
        parameter_a_doc,
        parameter_name_b,
        parameter_b_doc,
        description);

    ttnn::bind_function<OpName>(
        mod,
        doc.c_str(),
        ttnn::overload_t{
            &unary_composite_3param_to_4param_wrapper<Func>,
            nb::arg("input_tensor"),
            nb::arg(parameter_name_a.c_str()),
            nb::arg(parameter_name_b.c_str()),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()});
}

// OpHandler_float_with_default
template <ttnn::unique_string OpName, auto Func>
void bind_unary_composite_float_with_default(
    nb::module_& mod,
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
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - {5}
                 - TILE, ROW_MAJOR

            {6}
        )doc",
        std::string(OpName),
        std::string("ttnn.") + std::string(OpName),
        parameter_name_a,
        parameter_a_doc,
        parameter_a_value,
        supported_dtype,
        info_doc);

    ttnn::bind_function<OpName>(
        mod,
        doc.c_str(),
        ttnn::overload_t{
            &unary_3param_to_4param_wrapper<Func>,
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg(parameter_name_a.c_str()) = parameter_a_value,
            nb::arg("memory_config") = nb::none()});
}

namespace {
Tensor logit_wrapper(const Tensor& t, std::optional<float> eps, const std::optional<MemoryConfig>& memory_config) {
    return ttnn::logit(t, eps, memory_config, std::nullopt);
}
}  // namespace

void bind_unary_logit(nb::module_& mod, const std::string& info_doc = "") {
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
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - FLOAT32, BFLOAT16, BFLOAT8_B
                 - TILE, ROW_MAJOR


        Example:
            >>> tensor = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            >>> output = {1}(tensor, eps = None)
        )doc",
        "logit",
        "ttnn.logit",
        info_doc);

    ttnn::bind_function<"logit">(
        mod,
        doc.c_str(),
        &logit_wrapper,
        nb::arg("input_tensor"),
        nb::kw_only(),
        nb::arg("eps") = nb::none(),
        nb::arg("memory_config") = nb::none());
}

template <ttnn::unique_string OpName, auto Func>
void bind_unary_composite_rpow(
    nb::module_& mod,
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
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - {6}
                 - TILE, ROW_MAJOR

            {7}
        )doc",
        std::string(OpName),
        std::string("ttnn.") + std::string(OpName),
        parameter_name_a,
        parameter_a_doc,
        description,
        range,
        supported_dtype,
        info_doc);
    ttnn::bind_function<OpName>(
        mod,
        doc.c_str(),
        ttnn::overload_t{
            Func,
            nb::arg("input_tensor"),
            nb::arg(parameter_name_a.c_str()),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()});
}
}  // namespace

void py_module(nb::module_& mod) {
    ttnn::bind_function<"abs">(
        mod,
        R"doc(
        Applies abs to :attr:`input_tensor` element-wise.

        .. math::
            \mathrm{{output\_tensor}}_i = \verb|abs|(\mathrm{{input\_tensor}}_i)

        Args:
            input_tensor (ttnn.Tensor or ComplexTensor): the input tensor.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the operation. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): preallocated output tensor. Defaults to `None`.
            sub_core_grids (ttnn.CoreRangeSet, optional): sub core grids for the operation. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

        Note:
            Supported dtypes: BFLOAT16, BFLOAT8_B, FLOAT32
        )doc",
        ttnn::overload_t(
            nb::overload_cast<
                const Tensor&,
                const std::optional<MemoryConfig>&,
                const std::optional<Tensor>&,
                const std::optional<CoreRangeSet>&>(&ttnn::abs),
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none(),
            nb::arg("sub_core_grids") = nb::none()),
        ttnn::overload_t(
            nb::overload_cast<const ttnn::operations::complex::ComplexTensor&, const MemoryConfig&>(&ttnn::abs),
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg("memory_config")));
    bind_unary_operation_subcoregrids<"acos">(
        mod,
        &ttnn::acos,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|acos|(\mathrm{{input\_tensor}}_i))doc",
        "",
        R"doc(FLOAT32, BFLOAT16, BFLOAT8_B)doc");
    bind_unary_operation_subcoregrids<"asin">(
        mod,
        &ttnn::asin,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|asin|(\mathrm{{input\_tensor}}_i))doc",
        "",
        R"doc(FLOAT32, BFLOAT16, BFLOAT8_B)doc");
    bind_unary_operation_subcoregrids<"atan">(
        mod,
        &ttnn::atan,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|atan|(\mathrm{{input\_tensor}}_i))doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");
    bind_unary_operation_subcoregrids<"atanh">(
        mod,
        &ttnn::atanh,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|atanh|(\mathrm{{input\_tensor}}_i))doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");
    bind_unary_operation_subcoregrids<"cos">(
        mod,
        &ttnn::cos,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|cos|(\mathrm{{input\_tensor}}_i))doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");
    bind_unary_operation_subcoregrids<"acosh">(
        mod,
        &ttnn::acosh,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|acosh|(\mathrm{{input\_tensor}}_i))doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");
    bind_unary_operation_subcoregrids<"erfinv">(
        mod,
        &ttnn::erfinv,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|erfinv|(\mathrm{{input\_tensor}}_i))doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");
    bind_unary_operation_subcoregrids<"exp2">(
        mod,
        &ttnn::exp2,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|exp2|(\mathrm{{input\_tensor}}_i))doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");
    bind_unary_operation_subcoregrids<"expm1">(
        mod,
        &ttnn::expm1,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|expm1|(\mathrm{{input\_tensor}}_i))doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");

    bind_unary_operation_subcoregrids<"floor">(
        mod,
        &ttnn::floor,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|floor|(\mathrm{{input\_tensor}}_i))doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");
    bind_unary_operation_subcoregrids<"trunc">(
        mod,
        &ttnn::trunc,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|trunc|(\mathrm{{input\_tensor}}_i))doc",
        "",
        R"doc(FLOAT32, BFLOAT16, BFLOAT8_B)doc");
    bind_unary_operation_subcoregrids<"frac">(
        mod,
        &ttnn::frac,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|frac|(\mathrm{{input\_tensor}}_i))doc",
        "",
        R"doc(FLOAT32, BFLOAT16, BFLOAT8_B)doc");
    bind_unary_operation_3param<"eqz", &ttnn::eqz>(
        mod,
        R"doc(\mathrm{{output\_tensor}}_i = (\mathrm{{input\_tensor_i\ == 0}}))doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32, INT32, UINT16, UINT32)doc");
    bind_unary_operation_subcoregrids<"ceil">(
        mod,
        &ttnn::ceil,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|ceil|(\mathrm{{input\_tensor}}_i))doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");
    bind_unary_operation_3param<"hardmish", &ttnn::hardmish>(
        mod,
        R"doc(\mathrm{{output\_tensor}}_i = \mathrm{{input\_tensor}}_i \times \frac{{\min(\max(\mathrm{{input\_tensor}}_i + 2.8, 0), 5)}}{{5}})doc",
        "[Supported range -20 to inf]",
        R"doc(BFLOAT16, BFLOAT8_B)doc",
        R"doc(Computes the Hard Mish activation function. Hard Mish is a piecewise-linear approximation of the Mish activation function, offering improved computational efficiency while maintaining similar performance characteristics.)doc");
    bind_unary_operation_subcoregrids<"gez">(
        mod,
        &ttnn::gez,
        R"doc(\mathrm{{output\_tensor}}_i = (\mathrm{{input\_tensor_i\ >= 0}}))doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B, INT32)doc");
    bind_unary_operation_subcoregrids<"gtz">(
        mod,
        &ttnn::gtz,
        R"doc(\mathrm{{output\_tensor}}_i= (\mathrm{{input\_tensor_i\ > 0}}))doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B, INT32)doc");
    bind_unary_operation_subcoregrids<"i0">(
        mod,
        &ttnn::i0,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|i0|(\mathrm{{input\_tensor}}_i))doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");
    bind_unary_operation_subcoregrids<"i1">(
        mod,
        &ttnn::i1,
        R"doc(\mathrm{{output\_tensor}}_i = I_1(\mathrm{{input\_tensor}}_i))doc",
        "[Validated range: -10 to 10]",
        R"doc(BFLOAT16, BFLOAT8_B)doc",
        R"doc(Computes the modified Bessel function of the first kind of order 1.)doc");
    bind_unary_operation_subcoregrids<"isfinite">(
        mod,
        &ttnn::isfinite,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|isfinite|(\mathrm{{input\_tensor}}_i))doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");
    bind_unary_operation_subcoregrids<"isinf">(
        mod,
        &ttnn::isinf,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|isinf|(\mathrm{{input\_tensor}}_i))doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");
    bind_unary_operation_subcoregrids<"isnan">(
        mod,
        &ttnn::isnan,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|isnan|(\mathrm{{input\_tensor}}_i))doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");
    bind_unary_operation_subcoregrids<"isneginf">(
        mod,
        &ttnn::isneginf,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|isneginf|(\mathrm{{input\_tensor}}_i))doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");
    bind_unary_operation_subcoregrids<"isposinf">(
        mod,
        &ttnn::isposinf,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|isposinf|(\mathrm{{input\_tensor}}_i))doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");
    bind_unary_operation_subcoregrids<"lez">(
        mod,
        &ttnn::lez,
        R"doc(\mathrm{{output\_tensor}}_i = (\mathrm{{input\_tensor_i\ <= 0}}))doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B, INT32)doc");
    bind_unary_operation_subcoregrids<"logical_not">(
        mod,
        &ttnn::logical_not,
        R"doc(\mathrm{{output\_tensor}}_i = \mathrm{{!input\_tensor_i}})doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32, INT32, UINT16 (range: 0 - 65535), UINT32 (range: 0 - 4294967295))doc");
    bind_unary_operation_subcoregrids<"ltz">(
        mod,
        &ttnn::ltz,
        R"doc(\mathrm{{output\_tensor}}_i = (\mathrm{{input\_tensor_i\ < 0}}))doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B, INT32)doc");
    bind_unary_operation_subcoregrids<"neg">(
        mod,
        &ttnn::neg,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|neg|(\mathrm{{input\_tensor}}_i))doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");
    bind_unary_operation_subcoregrids<"nez">(
        mod,
        &ttnn::nez,
        R"doc(\mathrm{{output\_tensor}}_i = (\mathrm{{input\_tensor_i\ != 0}}))doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32, INT32, UINT16, UINT32)doc");

    bind_unary_operation_overload_complex_return_complex<"reciprocal">(
        mod,
        R"doc(BFLOAT16, BFLOAT8_B)doc",
        R"doc(BFLOAT8_B is supported only for non-zero inputs. Inputs containing zero may produce inaccurate results due to the characteristics of the block-FP format.)doc");
    bind_unary_operation_subcoregrids<"relu">(
        mod,
        &ttnn::relu,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|relu|(\mathrm{{input\_tensor}}_i))doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");
    bind_unary_operation_subcoregrids<"relu6">(
        mod,
        &ttnn::relu6,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|relu6|(\mathrm{{input\_tensor}}_i))doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");
    bind_unary_operation_subcoregrids<"sign">(
        mod,
        &ttnn::sign,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|sign|(\mathrm{{input\_tensor}}_i))doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");
    bind_unary_operation_subcoregrids<"signbit">(
        mod,
        &ttnn::signbit,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|signbit|(\mathrm{{input\_tensor}}_i))doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B, INT32, FLOAT32)doc");
    bind_unary_operation_subcoregrids<"silu">(
        mod,
        &ttnn::silu,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|silu|(\mathrm{{input\_tensor}}_i))doc",
        "",
        R"doc(FLOAT32, BFLOAT16, BFLOAT8_B)doc");
    bind_unary_operation_3param<"swish", &ttnn::swish>(
        mod,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|swish|(\mathrm{{input\_tensor}}_i))doc",
        "",
        R"doc(FLOAT32, BFLOAT16, BFLOAT8_B)doc");
    bind_unary_operation_subcoregrids<"sin">(
        mod,
        &ttnn::sin,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|sin|(\mathrm{{input\_tensor}}_i))doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");
    bind_unary_operation_subcoregrids<"square">(
        mod,
        &ttnn::square,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|square|(\mathrm{{input\_tensor}}_i))doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B, INT32, UINT32, UINT16 [0,255])doc");
    bind_unary_operation_subcoregrids<"tan">(
        mod,
        &ttnn::tan,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|tan|(\mathrm{{input\_tensor}}_i))doc",
        "Supported input range is (-1.45, 1.45)",
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");
    bind_unary_operation_3param<"log_sigmoid", &ttnn::log_sigmoid>(
        mod,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|log_sigmoid|(\mathrm{{input\_tensor}}_i))doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");
    bind_unary_operation_subcoregrids<"bitwise_not">(
        mod,
        &ttnn::bitwise_not,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|bitwise_not|(\mathrm{{input\_tensor}}_i))doc",
        R"doc(Supported input range is [-2147483647, 2147483647].)doc",
        R"doc(INT32)doc",
        R"doc(torch.tensor([[1, 2], [3, 4]], dtype=torch.int32))doc");
    bind_unary_operation_subcoregrids<"alt_complex_rotate90">(
        mod,
        &ttnn::alt_complex_rotate90,
        R"doc((\mathrm{{output\_tensor}}_{2i}, \mathrm{{output\_tensor}}_{2i+1}) = (-\mathrm{{input\_tensor}}_{2i+1}, \mathrm{{input\_tensor}}_{2i}))doc",
        R"doc(FLOAT32, BFLOAT16, BFLOAT8_B, BFLOAT4_B)doc",
        "",
        R"doc(The last dimension of the input tensor must be even.)doc");
    bind_unary_operation_3param<"deg2rad", &ttnn::deg2rad>(
        mod,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|deg2rad|(\mathrm{{input\_tensor}}_i))doc",
        "",
        R"doc(FLOAT32, BFLOAT16, BFLOAT8_B)doc");
    bind_unary_operation_3param<"rad2deg", &ttnn::rad2deg>(
        mod,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|rad2deg|(\mathrm{{input\_tensor}}_i))doc",
        "",
        R"doc(FLOAT32, BFLOAT16, BFLOAT8_B)doc");
    bind_unary_operation_subcoregrids<"asinh">(
        mod,
        &ttnn::asinh,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|asinh|(\mathrm{{input\_tensor}}_i))doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");
    bind_unary_operation_subcoregrids<"hardsigmoid">(
        mod,
        &ttnn::hardsigmoid,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|hardsigmoid|(\mathrm{{input\_tensor}}_i))doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");
    bind_unary_operation_subcoregrids<"hardswish">(
        mod,
        &ttnn::hardswish,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|hardswish|(\mathrm{{input\_tensor}}_i))doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");
    bind_unary_operation_subcoregrids<"softsign">(
        mod,
        &ttnn::softsign,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|softsign|(\mathrm{{input\_tensor}}_i))doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");
    bind_unary_operation_subcoregrids<"cbrt">(
        mod,
        &ttnn::cbrt,
        R"doc(\mathrm{{output\_tensor}}_i = \verb|cbrt|(\mathrm{{input\_tensor}}_i))doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");

    //  Unaries with fast_and_approximate_mode
    bind_unary_operation_with_fast_and_approximate_mode<"sqrt", &ttnn::sqrt>(
        mod, "", R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");
    bind_unary_operation_with_fast_and_approximate_mode<"rsqrt", &ttnn::rsqrt>(
        mod, "", R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");
    bind_unary_operation_with_fast_and_approximate_mode<"exp", &ttnn::exp>(
        mod, "", R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");
    bind_unary_operation_with_fast_and_approximate_mode<"erf", &ttnn::erf>(mod, "", R"doc(BFLOAT16, BFLOAT8_B)doc");
    bind_unary_operation_with_fast_and_approximate_mode<"erfc", &ttnn::erfc>(mod, "", R"doc(BFLOAT16, BFLOAT8_B)doc");
    bind_unary_operation_with_fast_and_approximate_mode<"gelu", &ttnn::gelu>(mod, "", R"doc(BFLOAT16, BFLOAT8_B)doc");
    bind_unary_operation_with_fast_and_approximate_mode<"log", &ttnn::log>(
        mod, "", R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");
    bind_unary_operation_with_fast_and_approximate_mode<"log10", &ttnn::log10>(
        mod, "", R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");
    bind_unary_operation_with_fast_and_approximate_mode<"log2", &ttnn::log2>(
        mod, "", R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");
    bind_unary_operation_with_fast_and_approximate_mode<"log1p", &ttnn::log1p>(
        mod, R"doc([Supported range: [-1, 1e7]])doc", R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");
    bind_unary_operation_with_fast_and_approximate_mode<"mish", &ttnn::mish>(
        mod, "[Supported range -20 to inf]", R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");
    // Unaries with float parameter
    bind_unary_composite_with_default_float<"elu", &ttnn::elu>(
        mod, "alpha", "The alpha parameter for the ELU function", 1.0f, R"doc(BFLOAT16, BFLOAT8_B)doc");
    bind_unary_operation_with_float_parameter<"heaviside", &ttnn::heaviside>(
        mod, "value", "The value parameter for the Heaviside function", "", R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");
    bind_unary_operation_with_float_parameter<"leaky_relu", &ttnn::leaky_relu>(
        mod,
        "negative_slope",
        "The slope parameter for the Leaky ReLU function",
        "",
        R"doc(FLOAT32, BFLOAT16, BFLOAT8_B)doc");
    bind_unary_operation_with_float_parameter<"relu_max", &ttnn::relu_max>(
        mod,
        "upper_limit",
        "The max value for ReLU function",
        "This function caps off the input to a max value and a min value of 0",
        R"doc(BFLOAT16, BFLOAT8_B)doc",
        R"doc(System memory is not supported.)doc");
    bind_unary_operation_with_float_parameter<"relu_min", &ttnn::relu_min>(
        mod,
        "lower_limit",
        "The min value for ReLU function",
        "This will carry out ReLU operation at min value instead of the standard 0",
        R"doc(BFLOAT16, FLOAT32)doc",
        R"doc(System memory is not supported.)doc");
    bind_unary_operation_with_float_parameter<"rpow", &ttnn::rpow>(
        mod, "exponent", "exponent value. Non-positive values are not supported.", "");
    bind_unary_operation_with_float_parameter_default<"celu", &ttnn::celu>(
        mod, "alpha", "The alpha parameter for the CELU function", 1.0f, "", R"doc(FLOAT32, BFLOAT16, BFLOAT8_B)doc");

    bind_unary_operation_with_scalar_parameter<"fill", &ttnn::fill>(
        mod,
        "fill_value",
        "The value to be filled in the output tensor",
        "This will create a tensor of same shape and dtype as input reference tensor with fill_value.",
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32, INT32, UINT32, UINT16)doc",
        R"doc(Host memory is not supported.)doc");

    // Unary ops with dim parameter
    bind_unary_operation_with_dim_parameter<"glu", &ttnn::glu>(
        mod,
        "dim",
        "Dimension to split input tensor. Supported only for last dimension (dim = -1 or 3)",
        "Split the tensor into two parts, apply the GLU function on the second tensor, and then perform multiplication "
        "with the first tensor.",
        R"doc(BFLOAT16, BFLOAT8_B)doc",
        R"doc(System memory is not supported.

           Last dimension of input tensor should be divisible by 64.

        )doc");

    bind_unary_operation_with_dim_parameter<"reglu", &ttnn::reglu>(
        mod,
        "dim",
        "Dimension to split input tensor. Supported only for last dimension (dim = -1 or 3)",
        "Split the tensor into two parts, apply the ReLU function on the second tensor, and then perform "
        "multiplication with the first tensor.",
        R"doc(BFLOAT16, BFLOAT8_B)doc",
        R"doc(System memory is not supported.

           Last dimension of input tensor should be divisible by 64.

        )doc");

    bind_unary_operation_with_dim_parameter<"geglu", &ttnn::geglu>(
        mod,
        "dim",
        "Dimension to split input tensor. Supported only for last dimension (dim = -1 or 3)",
        "Split the tensor into two parts, apply the GELU function on the second tensor, and then perform "
        "multiplication with the first tensor.",
        R"doc(BFLOAT16, BFLOAT8_B)doc",
        R"doc(System memory is not supported.

           Last dimension of input tensor should be divisible by 64.

        )doc");

    bind_unary_operation_with_dim_parameter<"swiglu", &ttnn::swiglu>(
        mod,
        "dim",
        "Dimension to split input tensor. Supported only for last dimension (dim = -1 or 3)",
        "Split the tensor into two parts, apply the SiLU function on the second tensor, and then perform "
        "multiplication with the first tensor.",
        R"doc(BFLOAT16, BFLOAT8_B)doc",
        R"doc(System memory is not supported.

           Last dimension of input tensor should be divisible by 64.

        )doc");

    // Other unaries (unary chain operations)
    bind_softplus(mod);
    bind_xielu(mod);
    bind_tanh_like<"tanh", &ttnn::tanh>(mod);
    bind_tanh_like<"tanhshrink", &ttnn::tanhshrink>(mod);
    bind_sigmoid_accurate(mod);
    bind_sigmoid(mod);

    bind_unary_chain(mod);
    bind_unary_operation<"lgamma", &ttnn::lgamma>(
        mod,
        R"doc(Computes natural logarithm of the gamma function on :attr:`input_tensor`.)doc",
        "",
        R"doc(BFLOAT16, FLOAT32)doc");
    bind_identity(mod);

    // unary composite imported into ttnn
    bind_unary_composite<"cosh", &ttnn::cosh>(
        mod,
        R"doc(Performs cosh function on :attr:`input_tensor`.)doc",
        "[supported range -9 to 9]",
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");
    bind_unary_composite_2param<"digamma", &ttnn::digamma>(
        mod,
        R"doc(Performs digamma function on :attr:`input_tensor`.)doc",
        "[supported for values greater than 0].",
        R"doc(BFLOAT16, BFLOAT8_B)doc",
        R"doc(TILE)doc");
    bind_unary_composite_2param<"multigammaln", &ttnn::multigammaln>(
        mod,
        R"doc(Performs multigammaln function on :attr:`input_tensor`.)doc",
        "[supported range 1.6 to inf].",
        R"doc(BFLOAT16, FLOAT32)doc",
        R"doc(TILE)doc");
    bind_unary_composite<"sinh", &ttnn::sinh>(
        mod,
        R"doc(Performs sinh function on :attr:`input_tensor`.)doc",
        "[supported range -9 to 9].",
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");
    bind_unary_composite_2param<"var_hw", &ttnn::var_hw>(
        mod,
        R"doc(Computes the variance across the height (H) and width (W) dimensions for each batch and channel. The variance is calculated as :math:`\mathrm{Var}[X] = E[(X - \mu)^2]` where :math:`\mu` is the mean over H and W dimensions. Output shape: [N, C, 1, 1].)doc");
    bind_unary_composite_2param<"std_hw", &ttnn::std_hw>(
        mod,
        R"doc(Computes the standard deviation across the height (H) and width (W) dimensions for each batch and channel. The standard deviation is calculated as :math:`\sigma = \sqrt{\mathrm{Var}[X]}` where the variance is computed over H and W dimensions. Output shape: [N, C, 1, 1].)doc");
    bind_unary_composite_2param<"normalize_hw", &ttnn::normalize_hw>(
        mod,
        R"doc(Performs normalize_hw function on :attr:`input_tensor`.)doc",
        "",
        R"doc(BFLOAT16, FLOAT32)doc",
        R"doc(ROW_MAJOR, TILE)doc");
    bind_unary_composite_2param<"logical_not_", &ttnn::logical_not_>(
        mod,
        R"doc(Performs logical_not inplace function on :attr:`input_tensor`.)doc",
        "",
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32, INT32, UINT16 (range: 0 - 65535), UINT32 (range: 0 - 4294967295))doc");
    bind_unary_composite_2param<"normalize_global", &ttnn::normalize_global>(
        mod,
        R"doc(Performs normalize_global function on :attr:`input_tensor`.)doc",
        "",
        R"doc(BFLOAT16, FLOAT32)doc",
        R"doc(ROW_MAJOR, TILE)doc");

    bind_unary_clip(mod);
    bind_unary_clamp(mod);
    bind_unary_composite_floats_with_default<"selu", &ttnn::selu>(
        mod, "scale", "Scale value", 1.0507, "alpha", "Alpha value", 1.67326);
    bind_unary_composite_floats_with_default<"hardtanh", &ttnn::hardtanh>(
        mod, "min_val", "min value", -1.0f, "max_val", "max value", 1.0f, R"doc(FLOAT32, BFLOAT16, BFLOAT8_B)doc");
    bind_unary_threshold<"threshold", &ttnn::threshold>(
        mod,
        "threshold",
        "Threshold value",
        "value",
        "Replacing value",
        R"doc(Performs threshold function on :attr:`input_tensor`, :attr:`threshold`, :attr:`value`.)doc");
    bind_unary_composite_int_with_default<"tril", &ttnn::tril>(
        mod,
        "diagonal",
        "diagonal value",
        0,
        R"doc(
        Returns the lower triangular part of :attr:`input_tensor` by zeroing out elements above the specified :attr:`diagonal`.
        Elements on and below the given :attr:`diagonal` are preserved, while elements above it are set to zero.

        - ``diagonal = 0`` selects the main diagonal (keeps elements on and below the main diagonal)
        - ``diagonal > 0`` selects a diagonal above the main diagonal (keeps more elements)
        - ``diagonal < 0`` selects a diagonal below the main diagonal (keeps fewer elements)
        )doc",
        R"doc(FLOAT32, BFLOAT16, BFLOAT8_B)doc");
    bind_unary_composite_int_with_default<"triu", &ttnn::triu>(
        mod,
        "diagonal",
        "diagonal value",
        0,
        R"doc(
        Returns the upper triangular part of :attr:`input_tensor` by zeroing out elements below the specified :attr:`diagonal`.
        Elements on and above the given :attr:`diagonal` are preserved, while elements below it are set to zero.

        - ``diagonal = 0`` selects the main diagonal (keeps elements on and above the main diagonal)
        - ``diagonal > 0`` selects a diagonal above the main diagonal (keeps fewer elements)
        - ``diagonal < 0`` selects a diagonal below the main diagonal (keeps more elements)
        )doc",
        R"doc(FLOAT32, BFLOAT16, BFLOAT8_B)doc");
    bind_unary_operation_with_int_parameter<"round", &ttnn::round>(
        mod,
        "decimals",
        "No. of decimal places to round off to [supported range -6 to 7], Defaults to 0.",
        R"doc(Round the input tensor to `decimals` decimal places.)doc",
        R"doc(FLOAT32, BFLOAT16, BFLOAT8_B)doc");
    bind_unary_composite_int<"polygamma", &ttnn::polygamma>(
        mod,
        "k",
        "k value",
        R"doc(Performs polygamma function on :attr:`input_tensor`, :attr:`decimals`. it is supported for range 1 to 10 only)doc");

    // unary composite with float imported into ttnn
    bind_unary_composite_float_with_default<"hardshrink", &ttnn::hardshrink>(
        mod, "lambd", "lambd value", 0.5f, R"doc(FLOAT32, BFLOAT16, BFLOAT8_B)doc");

    bind_unary_composite_float_with_default<"softshrink", &ttnn::softshrink>(
        mod, "lambd", "lambd value", 0.5f, R"doc(FLOAT32, BFLOAT16, BFLOAT8_B)doc");

    bind_unary_logit(mod);

    bind_unary_rdiv<"rdiv", &ttnn::rdiv>(
        mod,
        "value",
        "denominator that is considered as numerator, which should be a non-zero float value",
        "rounding_mode",
        "rounding_mode value",
        "None",
        R"doc(Performs the element-wise division of a scalar ``value`` by a tensor ``input`` and rounds the result using rounding_mode.

        Input tensor must have BFLOAT16 or FLOAT32 data type.

        Output tensor will have BFLOAT16 or FLOAT32 data type.)doc",

        R"doc(BFLOAT16, FLOAT32)doc",
        R"doc(System memory is not supported.)doc");

    // Bind bitcast operation
    const char* bitcast_doc = R"doc(
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
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - BFLOAT16, FLOAT32, INT32, UINT16, UINT32
                 - TILE, ROW_MAJOR
        )doc";

    ttnn::bind_function<"bitcast">(
        mod,
        bitcast_doc,
        &ttnn::bitcast,
        nb::arg("input_tensor"),
        nb::arg("dtype"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("output_tensor") = nb::none());
}

}  // namespace ttnn::operations::unary
