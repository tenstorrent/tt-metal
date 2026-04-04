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

// BEGIN: Binding functions disabled during SFPU nuke.
// These reference operations (reciprocal, softplus, xielu, sigmoid, etc.) that have been
// removed and not yet reimplemented. They will be re-enabled as operations are regenerated.
#if 0
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

#endif  // end first #if 0 block (reciprocal-related templates)

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

#if 0  // resume disabled block (float_parameter, softplus, xielu, sigmoid, etc.)

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

#endif
// END: Binding functions disabled during SFPU nuke.

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
    // NOTE: Most SFPU unary operation bindings have been removed during batch nuke.
    // They will be re-added as operations are regenerated.

    bind_identity(mod);
    bind_unary_logit(mod);
    bind_unary_chain(mod);

    bind_unary_operation_with_fast_and_approximate_mode<"mish", &ttnn::mish>(
        mod, "[Supported range -20 to inf]", R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");

    bind_unary_operation<"cosh", &ttnn::cosh>(
        mod,
        R"doc(\mathrm{{output\_tensor}}_i = \cosh(\mathrm{{input\_tensor}}_i))doc",
        "[supported range -9 to 9]",
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");
}

}  // namespace ttnn::operations::unary
