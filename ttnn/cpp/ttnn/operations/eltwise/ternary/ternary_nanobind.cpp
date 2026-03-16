// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ternary_nanobind.hpp"

#include <string>
#include <optional>
#include <variant>

#include <fmt/format.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/eltwise/ternary/ternary_composite.hpp"
#include "ttnn/operations/eltwise/ternary/ternary.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::ternary {

namespace {

void bind_ternary_where(nb::module_& mod, const std::string& description) {
    auto doc = fmt::format(
        R"doc(
            {2}

        Args:
            condition (ttnn.Tensor): the condition tensor must contain only 0's or 1's.
            true_value (ttnn.Tensor or Number): The value selected if the corresponding element in condition is 1.
            false_value (ttnn.Tensor or Number): The value selected if the corresponding element in condition is 0.


        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the operation. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): preallocated output tensor. Defaults to `None`.
            sub_core_grids (ttnn.CoreRangeSet, optional): sub core grids for the operation. Defaults to `None`.


        Note:
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - BFLOAT16, BFLOAT8_B, FLOAT32, INT32
                 - TILE

            bfloat8_b/bfloat4_b supports only on TILE_LAYOUT
        )doc",
        "where",
        "ttnn.where",
        description);

    ttnn::bind_function<"where">(
        mod,
        doc.c_str(),
        ttnn::overload_t(
            // using static cast to disambiguate the template `where` and
            // `where` with the fixed argument types.
            static_cast<Tensor (*)(
                const Tensor&,
                const TensorScalarVariant&,
                const TensorScalarVariant&,
                const std::optional<MemoryConfig>&,
                const std::optional<Tensor>&,
                const std::optional<CoreRangeSet>&)>(&ttnn::where),
            nb::arg("predicate"),
            nb::arg("true_value"),
            nb::arg("false_value"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none(),
            nb::arg("sub_core_grids") = nb::none()),
        ttnn::overload_t(
            &ttnn::where<int32_t>,
            nb::arg("predicate"),
            nb::arg("true_value"),
            nb::arg("false_value"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none(),
            nb::arg("sub_core_grids") = nb::none()),
        ttnn::overload_t(
            &ttnn::where<uint32_t>,
            nb::arg("predicate"),
            nb::arg("true_value"),
            nb::arg("false_value"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none(),
            nb::arg("sub_core_grids") = nb::none()));
}

void bind_ternary_lerp(nb::module_& mod, const std::string& description) {
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
            output_tensor (ttnn.Tensor, optional): preallocated output tensor. Defaults to `None`.


        Note:
            Supported dtypes and layouts:

            .. list-table::
                :header-rows: 1

                * - Dtypes
                  - Layouts
                * - BFLOAT16, BFLOAT8_B, FLOAT32
                  - TILE

            bfloat8_b/bfloat4_b supports only on TILE_LAYOUT

            end, weight tensors should have same dtype as input

            output_tensor dtype must match input dtype, or be FLOAT32 when inputs are BFLOAT16, or be BFLOAT16 when inputs are FLOAT32
        )doc",
        "lerp",
        "ttnn.lerp",
        description);

    ttnn::bind_function<"lerp">(
        mod,
        doc.c_str(),
        ttnn::overload_t(
            nb::overload_cast<
                const Tensor&,
                const Tensor&,
                const Tensor&,
                const std::optional<MemoryConfig>&,
                const std::optional<Tensor>&>(&ttnn::lerp),
            nb::arg("input"),
            nb::arg("end"),
            nb::arg("weight"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none()),
        ttnn::overload_t(
            nb::overload_cast<
                const Tensor&,
                const Tensor&,
                float,
                const std::optional<MemoryConfig>&,
                const std::optional<Tensor>&>(&ttnn::lerp),
            nb::arg("input"),
            nb::arg("end"),
            nb::arg("weight"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none()));
}

void bind_ternary_addcmul(
    nb::module_& mod,
    const std::string& description,
    const std::string& math,
    const std::string& supported_dtype = "BFLOAT16") {
    auto doc = fmt::format(
        R"doc(
            {2}

        .. math::
            {3}

        Args:
            input_a (ttnn.Tensor): the first input tensor.
            input_b (ttnn.Tensor): the second input tensor.
            input_c (ttnn.Tensor): the third input tensor.

        Keyword Args:
            value (float, optional): scalar value used in the operation. Defaults to 1.0.
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
               * - FLOAT32, BFLOAT16, BFLOAT8_B, INT32
                 - TILE

            Only TTT (tensor-tensor-tensor) variant is supported.
        )doc",
        "addcmul",
        "ttnn.addcmul",
        description,
        math,
        supported_dtype);

    ttnn::bind_function<"addcmul">(
        mod,
        doc.c_str(),
        &ttnn::addcmul,
        nb::arg("input_a"),
        nb::arg("input_b"),
        nb::arg("input_c"),
        nb::kw_only(),
        nb::arg("value") = 1.0f,
        nb::arg("memory_config") = nb::none(),
        nb::arg("output_tensor") = nb::none());
}

void bind_ternary_addcdiv(
    nb::module_& mod,
    const std::string& description,
    const std::string& math,
    const std::string& supported_dtype = "BFLOAT16") {
    auto doc = fmt::format(
        R"doc(
            {2}

        .. math::
            {3}

        Args:
            input_a (ttnn.Tensor): the first input tensor.
            input_b (ttnn.Tensor): the second input tensor.
            input_c (ttnn.Tensor): the third input tensor.

        Keyword Args:
            value (float, optional): scalar value used in the operation. Defaults to 1.0.
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
               * - FLOAT32, BFLOAT16, BFLOAT8_B
                 - TILE

            Only TTT (tensor-tensor-tensor) variant is supported.
        )doc",
        "addcdiv",
        "ttnn.addcdiv",
        description,
        math,
        supported_dtype);

    ttnn::bind_function<"addcdiv">(
        mod,
        doc.c_str(),
        &ttnn::addcdiv,
        nb::arg("input_a"),
        nb::arg("input_b"),
        nb::arg("input_c"),
        nb::kw_only(),
        nb::arg("value") = 1.0f,
        nb::arg("memory_config") = nb::none(),
        nb::arg("output_tensor") = nb::none());
}

void bind_ternary_mac(nb::module_& mod, const std::string& description) {
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
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - BFLOAT16, BFLOAT8_B, FLOAT32
                 - TILE

            bfloat8_b/bfloat4_b supports only on TILE_LAYOUT
        )doc",
        "mac",
        "ttnn.mac",
        description);

    ttnn::bind_function<"mac">(
        mod,
        doc.c_str(),
        ttnn::overload_t(
            nb::overload_cast<const Tensor&, const Tensor&, const Tensor&, const std::optional<MemoryConfig>&>(
                &ttnn::mac),
            nb::arg("input_tensor_a"),
            nb::arg("input_tensor_b"),
            nb::arg("input_tensor_c"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()),
        ttnn::overload_t(
            nb::overload_cast<const Tensor&, float, float, const std::optional<MemoryConfig>&>(&ttnn::mac),
            nb::arg("input_tensor_a"),
            nb::arg("value1"),
            nb::arg("value2"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()));
}

}  // namespace

void py_module(nb::module_& mod) {
    // new imported
    bind_ternary_addcmul(
        mod,
        R"doc(Multiplies :attr:`input_tensor_b` by a scalar, multiplies the result
            element-wise by :attr:`input_tensor_c`, and adds it to
            :attr:`input_tensor_a`.
            Returns a tensor with the same layout as :attr:`input_tensor_a`.)doc",
        R"doc(\mathrm{{output\_tensor}}_i = \mathrm{{input\_tensor\_a}}_i + (value * \mathrm{input\_tensor\_b}_i * \mathrm{input\_tensor\_c}_i))doc",
        "FLOAT32, BFLOAT16, BFLOAT8_B, INT32");

    bind_ternary_addcdiv(
        mod,
        R"doc(Multiplies :attr:`input_tensor_b` by a scalar, divides the result
            element-wise by :attr:`input_tensor_c`, and adds it to
            :attr:`input_tensor_a`.
            Returns a tensor with the same layout as :attr:`input_tensor_a`.)doc",
        R"doc(\mathrm{{output\_tensor}}_i = \mathrm{{input\_tensor\_a}}_i + \frac{(value * \mathrm{input\_tensor\_b}_i)}{\mathrm{input\_tensor\_c}_i})doc",
        "FLOAT32, BFLOAT16, BFLOAT8_B");
    bind_ternary_where(
        mod,
        R"doc(Selects elements from :attr:`true_value` or :attr:`false_value` depending on the corresponding value in :attr:`condition`. For each element, if the corresponding entry in :attr:`condition` is 1, the output element is taken from :attr:`true_value`; otherwise, it is taken from :attr:`false_value`.)doc");

    bind_ternary_lerp(
        mod,
        R"doc(Computes Lerp on :attr:`input`, :attr:`end` and :attr:`weight` and returns the tensor with the same layout as :attr:`input`)doc");

    bind_ternary_mac(
        mod,
        R"doc(Computes Mac on :attr:`input_tensor_a`, :attr:`input_tensor_b` and :attr:`input_tensor_c` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc");
}

}  // namespace ttnn::operations::ternary
