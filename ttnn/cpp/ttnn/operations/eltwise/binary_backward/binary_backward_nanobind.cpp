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

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/common/constants.hpp"
#include "ttnn/operations/eltwise/binary_backward/binary_backward.hpp"
#include "ttnn/operations/eltwise/ternary_backward/ternary_backward.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::binary_backward {

namespace {

template <ttnn::unique_string Name, typename Func>
void bind_binary_backward_ops(
    nb::module_& mod,
    Func func,
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
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - {3}
                 - TILE, ROW_MAJOR

            bfloat8_b/bfloat4_b is only supported on TILE_LAYOUT

            {4}
        )doc",

        std::string(Name),
        "ttnn." + std::string(Name),
        description,
        supported_dtype,
        note);

    ttnn::bind_function<Name>(
        mod,
        doc.c_str(),
        ttnn::overload_t(
            func,
            nb::arg("grad_tensor"),
            nb::arg("input_tensor_a"),
            nb::arg("input_tensor_b"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()));
}

template <ttnn::unique_string Name>
void bind_binary_backward_concat(
    nb::module_& mod,
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
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - {6}
                 - TILE, ROW_MAJOR

            bfloat8_b/bfloat4_b is only supported on TILE_LAYOUT
        )doc",
        std::string(Name),
        "ttnn." + std::string(Name),
        parameter_name,
        parameter_doc,
        parameter_value,
        description,
        supported_dtype);

    ttnn::bind_function<Name>(
        mod,
        doc.c_str(),
        ttnn::overload_t(
            &ttnn::concat_bw,
            nb::arg("grad_tensor"),
            nb::arg("input_tensor_a"),
            nb::arg("input_tensor_b"),
            nb::arg(parameter_name.c_str()) = parameter_value,
            nb::kw_only(),
            nb::arg("are_required_outputs") = std::vector<bool>{true, true},
            nb::arg("memory_config") = nb::none(),
            nb::arg("input_a_grad") = nb::none(),
            nb::arg("input_b_grad") = nb::none()));
}

template <ttnn::unique_string Name>
void bind_binary_backward_addalpha(
    nb::module_& mod,
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
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - {6}
                 - TILE, ROW_MAJOR

            bfloat8_b/bfloat4_b is only supported on TILE_LAYOUT
        )doc",

        std::string(Name),
        "ttnn." + std::string(Name),
        parameter_name,
        parameter_doc,
        parameter_value,
        description,
        supported_dtype);

    ttnn::bind_function<Name>(
        mod,
        doc.c_str(),
        ttnn::overload_t(
            &ttnn::addalpha_bw,
            nb::arg("grad_tensor"),
            nb::arg("input_tensor_a"),
            nb::arg("input_tensor_b"),
            nb::arg(parameter_name.c_str()) = parameter_value,
            nb::kw_only(),
            nb::arg("are_required_outputs") = std::vector<bool>{true, true},
            nb::arg("memory_config") = nb::none(),
            nb::arg("input_a_grad") = nb::none(),
            nb::arg("input_b_grad") = nb::none()));
}

template <ttnn::unique_string Name>
void bind_binary_backward_bias_gelu(
    nb::module_& mod,
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
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - {8}
                 - TILE, ROW_MAJOR

            bfloat8_b/bfloat4_b is only supported on TILE_LAYOUT

            {9}

        )doc",
        std::string(Name),
        "ttnn." + std::string(Name),
        parameter_name_a,
        parameter_a_doc,
        parameter_name_b,
        parameter_b_doc,
        parameter_b_value,
        description,
        supported_dtype,
        note);

    ttnn::bind_function<Name>(
        mod,
        doc.c_str(),
        ttnn::overload_t(
            nb::overload_cast<
                const ttnn::Tensor&,
                const ttnn::Tensor&,
                const ttnn::Tensor&,
                const std::string&,
                const std::optional<MemoryConfig>&>(&ttnn::bias_gelu_bw),
            nb::arg("grad_tensor"),
            nb::arg("input_tensor_a"),
            nb::arg("input_tensor_b"),
            nb::kw_only(),
            nb::arg(parameter_name_b.c_str()) = parameter_b_value,
            nb::arg("memory_config") = nb::none()),
        ttnn::overload_t(
            nb::overload_cast<
                const ttnn::Tensor&,
                const ttnn::Tensor&,
                float,
                const std::string&,
                const std::optional<MemoryConfig>&>(&ttnn::bias_gelu_bw),
            nb::arg("grad_tensor"),
            nb::arg("input_tensor"),
            nb::arg(parameter_name_a.c_str()),
            nb::kw_only(),
            nb::arg(parameter_name_b.c_str()) = parameter_b_value,
            nb::arg("memory_config") = nb::none()));
}

template <ttnn::unique_string Name>
void bind_binary_backward_sub_alpha(
    nb::module_& mod,
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
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - {6}
                 - TILE, ROW_MAJOR

            bfloat8_b/bfloat4_b is only supported on TILE_LAYOUT
        )doc",
        std::string(Name),
        "ttnn." + std::string(Name),
        parameter_name,
        parameter_doc,
        parameter_value,
        description,
        supported_dtype);

    ttnn::bind_function<Name>(
        mod,
        doc.c_str(),
        ttnn::overload_t(
            &ttnn::subalpha_bw,
            nb::arg("grad_tensor"),
            nb::arg("input_tensor_a"),
            nb::arg("input_tensor_b"),
            nb::arg(parameter_name.c_str()) = parameter_value,
            nb::kw_only(),
            nb::arg("are_required_outputs") = std::vector<bool>{true, true},
            nb::arg("memory_config") = nb::none(),
            nb::arg("input_grad") = nb::none(),
            nb::arg("other_grad") = nb::none()));
}

template <ttnn::unique_string Name>
void bind_binary_backward_rsub(
    nb::module_& mod, const std::string_view description, const std::string_view supported_dtype = "BFLOAT16") {
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
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - {3}
                 - TILE, ROW_MAJOR

            bfloat8_b/bfloat4_b is only supported on TILE_LAYOUT
        )doc",
        std::string(Name),
        "ttnn." + std::string(Name),
        description,
        supported_dtype);

    ttnn::bind_function<Name>(
        mod,
        doc.c_str(),
        ttnn::overload_t(
            &ttnn::rsub_bw,
            nb::arg("grad_tensor"),
            nb::arg("input_tensor_a"),
            nb::arg("input_tensor_b"),
            nb::kw_only(),
            nb::arg("are_required_outputs") = std::vector<bool>{true, true},
            nb::arg("memory_config") = nb::none(),
            nb::arg("input_grad") = nb::none(),
            nb::arg("other_grad") = nb::none()));
}

template <ttnn::unique_string Name>
void bind_binary_bw_mul(
    nb::module_& mod, const std::string_view description, const std::string_view supported_dtype = "BFLOAT16") {
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
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - {3}
                 - TILE, ROW_MAJOR

            bfloat8_b/bfloat4_b is only supported on TILE_LAYOUT
        )doc",
        std::string(Name),
        "ttnn." + std::string(Name),
        description,
        supported_dtype);

    ttnn::bind_function<Name>(
        mod,
        doc.c_str(),
        ttnn::overload_t(
            nb::overload_cast<
                const Tensor&,
                const Tensor&,
                float,
                const std::optional<MemoryConfig>&,
                std::optional<Tensor>>(&ttnn::mul_bw),
            nb::arg("grad_tensor"),
            nb::arg("input_tensor_a"),
            nb::arg("scalar"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("input_grad") = nb::none()),
        ttnn::overload_t(
            nb::overload_cast<
                const Tensor&,
                const Tensor&,
                const Tensor&,
                const std::vector<bool>&,
                const std::optional<MemoryConfig>&,
                std::optional<Tensor>,
                std::optional<Tensor>>(&ttnn::mul_bw),
            nb::arg("grad_tensor"),
            nb::arg("input_tensor"),
            nb::arg("other_tensor"),
            nb::kw_only(),
            nb::arg("are_required_outputs") = std::vector<bool>{true, true},
            nb::arg("memory_config") = nb::none(),
            nb::arg("input_grad") = nb::none(),
            nb::arg("other_grad") = nb::none()),
        ttnn::overload_t(
            nb::overload_cast<const ComplexTensor&, const ComplexTensor&, const ComplexTensor&, const MemoryConfig&>(
                &ttnn::mul_bw),
            nb::arg("grad_tensor"),
            nb::arg("input_tensor_a"),
            nb::arg("input_tensor_b"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()));
}

template <ttnn::unique_string Name>
void bind_binary_bw_add(
    nb::module_& mod,
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
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - {3}
                 - TILE, ROW_MAJOR

            bfloat8_b/bfloat4_b is only supported on TILE_LAYOUT

            {4}
        )doc",
        std::string(Name),
        "ttnn." + std::string(Name),
        description,
        supported_dtype,
        note);

    ttnn::bind_function<Name>(
        mod,
        doc.c_str(),
        ttnn::overload_t(
            nb::overload_cast<
                const Tensor&,
                const Tensor&,
                float,
                const std::optional<MemoryConfig>&,
                std::optional<Tensor>>(&ttnn::add_bw),
            nb::arg("grad_tensor"),
            nb::arg("input_tensor_a"),
            nb::arg("scalar"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("input_grad") = nb::none()),
        ttnn::overload_t(
            nb::overload_cast<
                const Tensor&,
                const Tensor&,
                const Tensor&,
                const std::vector<bool>&,
                const std::optional<MemoryConfig>&,
                std::optional<Tensor>,
                std::optional<Tensor>>(&ttnn::add_bw),
            nb::arg("grad_tensor"),
            nb::arg("input_tensor"),
            nb::arg("other_tensor"),
            nb::kw_only(),
            nb::arg("are_required_outputs") = std::vector<bool>{true, true},
            nb::arg("memory_config") = nb::none(),
            nb::arg("input_grad") = nb::none(),
            nb::arg("other_grad") = nb::none()),
        ttnn::overload_t(
            nb::overload_cast<
                const ComplexTensor&,
                const ComplexTensor&,
                const ComplexTensor&,
                float,
                const std::optional<MemoryConfig>&>(&ttnn::add_bw),
            nb::arg("grad_tensor"),
            nb::arg("input_tensor_a"),
            nb::arg("input_tensor_b"),
            nb::arg("alpha"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()));
}

template <ttnn::unique_string Name>
void bind_binary_bw_sub(
    nb::module_& mod,
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
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - {3}
                 - TILE, ROW_MAJOR

            bfloat8_b/bfloat4_b is only supported on TILE_LAYOUT

            {4}
        )doc",
        std::string(Name),
        "ttnn." + std::string(Name),
        description,
        supported_dtype,
        note);

    ttnn::bind_function<Name>(
        mod,
        doc.c_str(),
        ttnn::overload_t(
            nb::overload_cast<
                const Tensor&,
                const Tensor&,
                float,
                const std::optional<MemoryConfig>&,
                std::optional<Tensor>>(&ttnn::sub_bw),
            nb::arg("grad_tensor"),
            nb::arg("input_tensor_a"),
            nb::arg("scalar"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("input_grad") = nb::none()),
        ttnn::overload_t(
            nb::overload_cast<
                const Tensor&,
                const Tensor&,
                const Tensor&,
                const std::vector<bool>&,
                const std::optional<MemoryConfig>&,
                std::optional<Tensor>,
                std::optional<Tensor>>(&ttnn::sub_bw),
            nb::arg("grad_tensor"),
            nb::arg("input_tensor"),
            nb::arg("other_tensor"),
            nb::kw_only(),
            nb::arg("are_required_outputs") = std::vector<bool>{true, true},
            nb::arg("memory_config") = nb::none(),
            nb::arg("input_grad") = nb::none(),
            nb::arg("other_grad") = nb::none()),
        ttnn::overload_t(
            nb::overload_cast<
                const ComplexTensor&,
                const ComplexTensor&,
                const ComplexTensor&,
                float,
                const std::optional<MemoryConfig>&>(&ttnn::sub_bw),
            nb::arg("grad_tensor"),
            nb::arg("input_tensor_a"),
            nb::arg("input_tensor_b"),
            nb::arg("alpha"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()));
}

template <ttnn::unique_string Name>
void bind_binary_bw_div(
    nb::module_& mod,
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
            rounding_mode (string, optional): Rounding mode. Defaults to `None`.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            input_grad (ttnn.Tensor, optional): Preallocated output tensor for gradient of `input_tensor_a`. Defaults to `None`.
            other_grad (ttnn.Tensor, optional): Preallocated output tensor for gradient of `input_tensor_b`. Defaults to `None`.


        Note:
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - {3}
                 - TILE, ROW_MAJOR

            bfloat8_b/bfloat4_b is only supported on TILE_LAYOUT

            {4}
        )doc",
        std::string(Name),
        "ttnn." + std::string(Name),
        description,
        supported_dtype,
        note);

    ttnn::bind_function<Name>(
        mod,
        doc.c_str(),
        ttnn::overload_t(
            nb::overload_cast<
                const Tensor&,
                const Tensor&,
                float,
                const std::optional<std::string>&,
                const std::optional<MemoryConfig>&,
                std::optional<Tensor>>(&ttnn::div_bw),
            nb::arg("grad_tensor"),
            nb::arg("input_tensor_a"),
            nb::arg("scalar"),
            nb::arg("rounding_mode") = nb::none(),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("input_grad") = nb::none()),
        ttnn::overload_t(
            nb::overload_cast<
                const Tensor&,
                const Tensor&,
                const Tensor&,
                const std::optional<std::string>&,
                const std::vector<bool>&,
                const std::optional<MemoryConfig>&,
                std::optional<Tensor>,
                std::optional<Tensor>>(&ttnn::div_bw),
            nb::arg("grad_tensor"),
            nb::arg("input_tensor"),
            nb::arg("other_tensor"),
            nb::arg("rounding_mode") = nb::none(),
            nb::kw_only(),
            nb::arg("are_required_outputs") = std::vector<bool>{true, true},
            nb::arg("memory_config") = nb::none(),
            nb::arg("input_grad") = nb::none(),
            nb::arg("other_grad") = nb::none()),
        ttnn::overload_t(
            nb::overload_cast<const ComplexTensor&, const ComplexTensor&, const ComplexTensor&, const MemoryConfig&>(
                &ttnn::div_bw),
            nb::arg("grad_tensor"),
            nb::arg("input_tensor_a"),
            nb::arg("input_tensor_b"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()));
}

void bind_binary_backward_assign(nb::module_& mod) {
    ttnn::bind_function<"assign_bw">(
        mod,
        R"doc(
        assign_bw(grad_tensor: ttnn.Tensor, input_tensor: ttnn.Tensor, *, memory_config: Optional[ttnn.MemoryConfig] = None, input_grad: Optional[ttnn.Tensor] = None) -> List[Optional[ttnn.Tensor]]

        Returns the gradient of assign operation.
        )doc",
        ttnn::overload_t(
            nb::overload_cast<const Tensor&, const Tensor&, const std::optional<MemoryConfig>&, std::optional<Tensor>>(
                &ttnn::assign_bw),
            nb::arg("grad_tensor"),
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("input_grad") = nb::none()),
        ttnn::overload_t(
            nb::overload_cast<
                const Tensor&,
                const Tensor&,
                const Tensor&,
                const std::vector<bool>&,
                const std::optional<MemoryConfig>&,
                std::optional<Tensor>,
                std::optional<Tensor>>(&ttnn::assign_bw),
            nb::arg("grad_tensor"),
            nb::arg("input_tensor"),
            nb::arg("other_tensor"),
            nb::kw_only(),
            nb::arg("are_required_outputs") = std::vector<bool>{true, true},
            nb::arg("memory_config") = nb::none(),
            nb::arg("input_grad") = nb::none(),
            nb::arg("other_grad") = nb::none()));
}

}  // namespace

void py_module(nb::module_& module) {
    bind_binary_backward_ops<"atan2_bw">(module, &ttnn::atan2_bw, "Returns the gradient of atan2 operation.");
    bind_binary_backward_ops<"xlogy_bw">(module, &ttnn::xlogy_bw, "Returns the gradient of xlogy operation.");
    bind_binary_backward_ops<"hypot_bw">(module, &ttnn::hypot_bw, "Returns the gradient of hypot operation.");
    bind_binary_backward_ops<"ldexp_bw">(module, &ttnn::ldexp_bw, "Returns the gradient of ldexp operation.");
    bind_binary_backward_ops<"logaddexp_bw">(
        module, &ttnn::logaddexp_bw, "Returns the gradient of logaddexp operation.");
    bind_binary_backward_ops<"logaddexp2_bw">(
        module, &ttnn::logaddexp2_bw, "Returns the gradient of logaddexp2 operation.");
    bind_binary_backward_ops<"squared_difference_bw">(
        module, &ttnn::squared_difference_bw, "Returns the gradient of squared_difference operation.");
    bind_binary_backward_ops<"min_bw">(module, &ttnn::min_bw, "Returns the gradient of min operation.");
    bind_binary_backward_ops<"max_bw">(module, &ttnn::max_bw, "Returns the gradient of max operation.");

    bind_binary_backward_addalpha<"addalpha_bw">(
        module, "alpha", "Alpha value", 1.0f, "Returns the gradient of addalpha operation.");
    bind_binary_backward_sub_alpha<"subalpha_bw">(
        module, "alpha", "Alpha value", 1.0f, "Returns the gradient of subalpha operation.");

    bind_binary_backward_rsub<"rsub_bw">(module, "Returns the gradient of rsub operation.");

    bind_binary_backward_concat<"concat_bw">(
        module, "dim", "Dimension to concat", 0, "Returns the gradient of concat operation.");

    bind_binary_bw_mul<"mul_bw">(module, "Returns the gradient of mul operation.");
    bind_binary_bw_add<"add_bw">(module, "Returns the gradient of add operation.");
    bind_binary_bw_sub<"sub_bw">(module, "Returns the gradient of sub operation.");
    bind_binary_bw_div<"div_bw">(module, "Returns the gradient of div operation.");

    bind_binary_backward_assign(module);

    bind_binary_backward_bias_gelu<"bias_gelu_bw">(
        module,
        "bias",
        "Bias value",
        "approximate",
        "Approximation type",
        "none",
        "Returns the gradient of bias_gelu operation.");

    bind_binary_backward_ops<"remainder_bw">(
        module,
        nb::overload_cast<const Tensor&, const Tensor&, const Tensor&, const std::optional<MemoryConfig>&>(
            &ttnn::remainder_bw),
        "Returns the gradient of remainder operation.");

    bind_binary_backward_ops<"fmod_bw">(
        module,
        nb::overload_cast<const Tensor&, const Tensor&, const Tensor&, const std::optional<MemoryConfig>&>(
            &ttnn::fmod_bw),
        "Returns the gradient of fmod operation.");
}

}  // namespace ttnn::operations::binary_backward
