// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "binary_nanobind.hpp"

#include <array>
#include <string>
#include <optional>
#include <utility>

#include <fmt/format.h>
#include <nanobind/nanobind.h>
#include <nanobind/operators.h>  // testing
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/variant.h>

#include <ttnn-nanobind/small_vector_caster.hpp>
#include <ttnn-nanobind/span_caster.hpp>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn-nanobind/export_enum.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/binary/binary_composite.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::binary {

namespace unary = operations::unary;

namespace detail {

Tensor hypot_composite_wrapper(const Tensor& a, const Tensor& b, const std::optional<MemoryConfig>& m) {
    return ttnn::hypot(a, b, m, std::nullopt);
}

// Common broadcasting and performance documentation for binary operations
constexpr auto BINARY_BROADCAST_DOC = R"doc(
        Binary elementwise operations, C=op(A,B), support input tensors A and B in row major and tile layout, in interleaved or sharded format (height, width or block sharded), in DRAM or L1. A and B are completely independent, and can have different tensor specs.

        Broadcast of A and B operands is supported up to dimension 5 (DNCHW). Any dimensions of size 1 in either A or B will be expanded to match the other input, and data will be duplicated along that dimension. For example, if the shape of A is [2,1,1,32] and B is [1,16,8,1], the output shape will be [2,16,8,32]. The size of dimensions higher than 5 must match between A and B.

        The output C also supports row major and tile layout, interleaved or sharded format (height, width or block sharded), in DRAM or L1. The tensor spec of C is independent of A and B, and can be explicitly set using the optional output tensor input; if not provided, the operation will attempt a best decision at an appropriate tensor spec. The dimensions of C, or equivalently the optional output tensor, must match the broadcast-matched size of A and B.

        Performance considerations:
        Elementwise operations operate natively in tile format, tiled tensors are preferred as an input, and row-major tensors are tilized and untilized during the operation.
        L1 sharded layout is preferred, with no broadcast and matching tensor specs for A, B and C.
)doc";

// Free function template that dispatches to the appropriate ttnn binary operation
template <BinaryOpType Op>
Tensor binary_op_binding_tensor_scalar(
    const Tensor& input_tensor_a,
    float scalar,
    const std::optional<const DataType>& dtype,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::Tensor>& output_tensor,
    ttsl::Span<const unary::EltwiseUnaryWithParam> activations,
    ttsl::Span<const unary::EltwiseUnaryWithParam> input_tensor_a_activations,
    ttsl::Span<const unary::EltwiseUnaryWithParam> input_tensor_b_activations,
    const std::optional<bool>& use_legacy,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    return ttnn::binary_op<Op>(
        input_tensor_a,
        scalar,
        dtype,
        memory_config,
        output_tensor,
        activations,
        input_tensor_a_activations,
        input_tensor_b_activations,
        use_legacy,
        sub_core_grids);
}

template <BinaryOpType Op>
Tensor binary_op_binding_tensor_tensor(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<const DataType>& dtype,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::Tensor>& output_tensor,
    ttsl::Span<const unary::EltwiseUnaryWithParam> activations,
    ttsl::Span<const unary::EltwiseUnaryWithParam> input_tensor_a_activations,
    ttsl::Span<const unary::EltwiseUnaryWithParam> input_tensor_b_activations,
    const std::optional<bool>& use_legacy,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    return ttnn::binary_op<Op>(
        input_tensor_a,
        input_tensor_b,
        dtype,
        memory_config,
        output_tensor,
        activations,
        input_tensor_a_activations,
        input_tensor_b_activations,
        use_legacy,
        sub_core_grids);
}

using InplaceScalarFn = Tensor (*)(
    const Tensor&,
    float,
    tt::stl::Span<const unary::EltwiseUnaryWithParam>,
    tt::stl::Span<const unary::EltwiseUnaryWithParam>,
    tt::stl::Span<const unary::EltwiseUnaryWithParam>,
    std::optional<bool>,
    const std::optional<CoreRangeSet>&);

using InplaceTensorFn = Tensor (*)(
    const Tensor&,
    const Tensor&,
    tt::stl::Span<const unary::EltwiseUnaryWithParam>,
    tt::stl::Span<const unary::EltwiseUnaryWithParam>,
    tt::stl::Span<const unary::EltwiseUnaryWithParam>,
    std::optional<bool>,
    const std::optional<CoreRangeSet>&);

template <InplaceScalarFn Fn>
Tensor inplace_binding_tensor_scalar(
    const Tensor& input_tensor_a,
    float scalar,
    ttsl::Span<const unary::EltwiseUnaryWithParam> activations,
    ttsl::Span<const unary::EltwiseUnaryWithParam> input_tensor_a_activations,
    ttsl::Span<const unary::EltwiseUnaryWithParam> input_tensor_b_activations,
    const std::optional<bool>& use_legacy,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    return Fn(
        input_tensor_a,
        scalar,
        activations,
        input_tensor_a_activations,
        input_tensor_b_activations,
        use_legacy,
        sub_core_grids);
}

template <InplaceTensorFn Fn>
Tensor inplace_binding_tensor_tensor(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    ttsl::Span<const unary::EltwiseUnaryWithParam> activations,
    ttsl::Span<const unary::EltwiseUnaryWithParam> input_tensor_a_activations,
    ttsl::Span<const unary::EltwiseUnaryWithParam> input_tensor_b_activations,
    const std::optional<bool>& use_legacy,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    return Fn(
        input_tensor_a,
        input_tensor_b,
        activations,
        input_tensor_a_activations,
        input_tensor_b_activations,
        use_legacy,
        sub_core_grids);
}

template <ttnn::unique_string Name, typename TensorScalarFn, typename TensorTensorFn>
void bind_binary_inplace_operation(
    nb::module_& mod,
    const std::string& description,
    const std::string& math,
    TensorScalarFn tensor_scalar_fn,
    TensorTensorFn tensor_tensor_fn,
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& note = " ") {
    auto doc = fmt::format(
        R"doc(
        {2}

        .. math::
            {3}

        Args:
            input_tensor_a (ttnn.Tensor): the input tensor.
            input_tensor_b (ttnn.Tensor or Number): the input tensor.

        Keyword args:
            activations (List[str], optional): list of activation functions to apply to the output tensor. Defaults to `None`.
            input_tensor_a_activations (List[str], optional): list of activation functions to apply to input_a. Defaults to `None`.
            input_tensor_b_activations (List[str], optional): list of activation functions to apply to input_b. Defaults to `None`.
            use_legacy (bool, optional): use legacy implementation. Defaults to `None`.
            sub_core_grids (CoreRangeSet, optional): sub core grids. Defaults to `None`.

        {6}

        Note:
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - {4}
                 - TILE, ROW_MAJOR

            If the input tensor is ROW_MAJOR layout, it will be internally converted to TILE layout.

            {5}
        )doc",
        std::string(Name),
        "ttnn." + std::string(Name),
        description,
        math,
        supported_dtype,
        note,
        BINARY_BROADCAST_DOC);

    ttnn::bind_function<Name>(
        mod,
        doc.c_str(),
        ttnn::overload_t(
            tensor_scalar_fn,
            nb::arg("input_tensor_a"),
            nb::arg("input_tensor_b"),
            nb::kw_only(),
            nb::arg("activations") = nb::cast(ttsl::Span<const unary::EltwiseUnaryWithParam>{}),
            nb::arg("input_tensor_a_activations") = nb::cast(ttsl::Span<const unary::EltwiseUnaryWithParam>{}),
            nb::arg("input_tensor_b_activations") = nb::cast(ttsl::Span<const unary::EltwiseUnaryWithParam>{}),
            nb::arg("use_legacy") = nb::none(),
            nb::arg("sub_core_grids") = nb::none()),
        ttnn::overload_t(
            tensor_tensor_fn,
            nb::arg("input_tensor_a"),
            nb::arg("input_tensor_b"),
            nb::kw_only(),
            nb::arg("activations") = nb::cast(ttsl::Span<const unary::EltwiseUnaryWithParam>{}),
            nb::arg("input_tensor_a_activations") = nb::cast(ttsl::Span<const unary::EltwiseUnaryWithParam>{}),
            nb::arg("input_tensor_b_activations") = nb::cast(ttsl::Span<const unary::EltwiseUnaryWithParam>{}),
            nb::arg("use_legacy") = nb::none(),
            nb::arg("sub_core_grids") = nb::none()));
}

template <ttnn::unique_string Name, typename TensorScalarFn, typename TensorTensorFn>
void bind_binary_operation(
    nb::module_& mod,
    const std::string& description,
    const std::string& math,
    TensorScalarFn tensor_scalar_fn,
    TensorTensorFn tensor_tensor_fn,
    const std::string& info = ". ",
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& note = " ") {
    auto doc = fmt::format(
        R"doc(
        {2}

        .. math::
            {3}

        Args:
            input_tensor_a (ttnn.Tensor): the input tensor.
            input_tensor_b (ttnn.Tensor or Number): the input tensor.

        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the operation. Defaults to `None`.
            dtype (ttnn.DataType, optional): data type for the output tensor. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): preallocated output tensor. Defaults to `None`.
            activations (List[str], optional): list of activation functions to apply to the output tensor. Defaults to `None`.


        Returns:
            ttnn.Tensor: the output tensor.

        {7}

        Note:
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - {5}
                 - TILE, ROW_MAJOR

            If the input tensor is ROW_MAJOR layout, it will be internally converted to TILE layout.

            {6}
        )doc",
        std::string(Name),
        "ttnn." + std::string(Name),
        description,
        math,
        info,
        supported_dtype,
        note,
        BINARY_BROADCAST_DOC);

    ttnn::bind_function<Name>(
        mod,
        doc.c_str(),
        ttnn::overload_t(
            tensor_scalar_fn,
            nb::arg("input_tensor_a"),
            nb::arg("input_tensor_b"),
            nb::kw_only(),
            nb::arg("dtype") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none(),
            nb::arg("activations") = nb::cast(ttsl::Span<const unary::EltwiseUnaryWithParam>{}),
            nb::arg("input_tensor_a_activations") = nb::cast(ttsl::Span<const unary::EltwiseUnaryWithParam>{}),
            nb::arg("input_tensor_b_activations") = nb::cast(ttsl::Span<const unary::EltwiseUnaryWithParam>{}),
            nb::arg("use_legacy") = nb::none(),
            nb::arg("sub_core_grids") = nb::none()),
        ttnn::overload_t(
            tensor_tensor_fn,
            nb::arg("input_tensor_a"),
            nb::arg("input_tensor_b"),
            nb::kw_only(),
            nb::arg("dtype") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none(),
            nb::arg("activations") = nb::cast(ttsl::Span<const unary::EltwiseUnaryWithParam>{}),
            nb::arg("input_tensor_a_activations") = nb::cast(ttsl::Span<const unary::EltwiseUnaryWithParam>{}),
            nb::arg("input_tensor_b_activations") = nb::cast(ttsl::Span<const unary::EltwiseUnaryWithParam>{}),
            nb::arg("use_legacy") = nb::none(),
            nb::arg("sub_core_grids") = nb::none()));
}

template <ttnn::unique_string Name, typename Fn>
void bind_binary_gcd_lcm_operation(
    nb::module_& mod,
    const std::string& description,
    const std::string& math,
    Fn fn,
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& note = "") {
    auto doc = fmt::format(
        R"doc(
        {2}

        .. math::
            {3}

        Args:
            input_tensor_a (ttnn.Tensor): the input tensor.
            input_tensor_b (ttnn.Tensor): the input tensor.

        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the operation. Defaults to `None`.
            dtype (ttnn.DataType, optional): data type for the output tensor. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): preallocated output tensor. Defaults to `None`.
            activations (List[str], optional): list of activation functions to apply to the output tensor. Defaults to `None`.


        Returns:
            ttnn.Tensor: the output tensor.

        {6}

        Note:
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - {4}
                 - TILE, ROW_MAJOR

            If the input tensor is ROW_MAJOR layout, it will be internally converted to TILE layout.

            {5}
        )doc",

        std::string(Name),
        "ttnn." + std::string(Name),
        description,
        math,
        supported_dtype,
        note,
        BINARY_BROADCAST_DOC);

    ttnn::bind_function<Name>(
        mod,
        doc.c_str(),
        fn,
        nb::arg("input_tensor_a"),
        nb::arg("input_tensor_b"),
        nb::kw_only(),
        nb::arg("dtype") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("output_tensor") = nb::none(),
        nb::arg("activations") = nb::cast(ttsl::Span<const unary::EltwiseUnaryWithParam>{}),
        nb::arg("input_tensor_a_activations") = nb::cast(ttsl::Span<const unary::EltwiseUnaryWithParam>{}),
        nb::arg("input_tensor_b_activations") = nb::cast(ttsl::Span<const unary::EltwiseUnaryWithParam>{}),
        nb::arg("use_legacy") = nb::none());
}

template <ttnn::unique_string Name, typename TensorScalarFn, typename TensorTensorFn>
void bind_binary_unary_max_operation(
    nb::module_& mod,
    const std::string& description,
    TensorScalarFn tensor_scalar_fn,
    TensorTensorFn tensor_tensor_fn,
    const std::string& note = " ",
    const std::string& supported_dtype = "BFLOAT16, FLOAT32, INT32") {
    auto doc = fmt::format(
        R"doc(
        {2}

        Args:
            input_tensor_a (ttnn.Tensor): the input tensor.
            input_tensor_b (ttnn.Tensor or Number): the input tensor.

        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the operation. Defaults to `None`.
            dtype (ttnn.DataType, optional): data type for the output tensor. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): preallocated output tensor. Defaults to `None`.
            activations (List[str], optional): list of activation functions to apply to the output tensor. Defaults to `None`.


        Returns:
            ttnn.Tensor: the output tensor.

        {5}

        Note:
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - {3}
                 - TILE, ROW_MAJOR

            If the input tensor is ROW_MAJOR layout, it will be internally converted to TILE layout.

            {4}
        )doc",
        std::string(Name),
        "ttnn." + std::string(Name),
        description,
        supported_dtype,
        note,
        BINARY_BROADCAST_DOC);

    auto scalar_overload = ttnn::overload_t(
        tensor_scalar_fn,
        nb::arg("input_tensor_a"),
        nb::arg("input_b"),
        nb::kw_only(),
        nb::arg("dtype") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("output_tensor") = nb::none(),
        nb::arg("activations") = nb::cast(ttsl::Span<const unary::EltwiseUnaryWithParam>{}),
        nb::arg("input_tensor_a_activations") = nb::cast(ttsl::Span<const unary::EltwiseUnaryWithParam>{}),
        nb::arg("input_tensor_b_activations") = nb::cast(ttsl::Span<const unary::EltwiseUnaryWithParam>{}),
        nb::arg("use_legacy") = nb::none());
    auto tensor_overload = ttnn::overload_t(
        tensor_tensor_fn,
        nb::arg("input_tensor_a"),
        nb::arg("input_tensor_b"),
        nb::kw_only(),
        nb::arg("dtype") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("output_tensor") = nb::none(),
        nb::arg("activations") = nb::cast(ttsl::Span<const unary::EltwiseUnaryWithParam>{}),
        nb::arg("input_tensor_a_activations") = nb::cast(ttsl::Span<const unary::EltwiseUnaryWithParam>{}),
        nb::arg("input_tensor_b_activations") = nb::cast(ttsl::Span<const unary::EltwiseUnaryWithParam>{}),
        nb::arg("use_legacy") = nb::none());
    ttnn::bind_function<Name>(mod, doc.c_str(), scalar_overload, tensor_overload);
}

template <ttnn::unique_string Name, typename TensorScalarFn, typename TensorTensorFn>
void bind_binary_unary_operation(
    nb::module_& mod,
    const std::string& description,
    const std::string& math,
    TensorScalarFn tensor_scalar_fn,
    TensorTensorFn tensor_tensor_fn,
    const std::string& info = ". ",
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& note = " ") {
    auto doc = fmt::format(
        R"doc(
        {2}

        .. math::
            {3}

        Args:
            input_tensor_a (ttnn.Tensor): the input tensor.
            input_tensor_b (ttnn.Tensor or Number): the input tensor.

        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the operation. Defaults to `None`.
            dtype (ttnn.DataType, optional): data type for the output tensor. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): preallocated output tensor. Defaults to `None`.
            activations (List[str], optional): list of activation functions to apply to the output tensor. Defaults to `None`.


        Returns:
            ttnn.Tensor: the output tensor.

        {7}

        Note:
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - {5}
                 - TILE, ROW_MAJOR

            If the input tensor is ROW_MAJOR layout, it will be internally converted to TILE layout.

            {6}
        )doc",
        std::string(Name),
        "ttnn." + std::string(Name),
        description,
        math,
        info,
        supported_dtype,
        note,
        BINARY_BROADCAST_DOC);

    ttnn::bind_function<Name>(
        mod,
        doc.c_str(),
        ttnn::overload_t(
            tensor_scalar_fn,
            nb::arg("input_tensor_a"),
            nb::arg("input_b"),
            nb::kw_only(),
            nb::arg("dtype") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none(),
            nb::arg("activations") = nb::cast(ttsl::Span<const unary::EltwiseUnaryWithParam>{}),
            nb::arg("input_tensor_a_activations") = nb::cast(ttsl::Span<const unary::EltwiseUnaryWithParam>{}),
            nb::arg("input_tensor_b_activations") = nb::cast(ttsl::Span<const unary::EltwiseUnaryWithParam>{}),
            nb::arg("use_legacy") = nb::none()),
        ttnn::overload_t(
            tensor_tensor_fn,
            nb::arg("input_tensor_a"),
            nb::arg("input_tensor_b"),
            nb::kw_only(),
            nb::arg("dtype") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none(),
            nb::arg("activations") = nb::cast(ttsl::Span<const unary::EltwiseUnaryWithParam>{}),
            nb::arg("input_tensor_a_activations") = nb::cast(ttsl::Span<const unary::EltwiseUnaryWithParam>{}),
            nb::arg("input_tensor_b_activations") = nb::cast(ttsl::Span<const unary::EltwiseUnaryWithParam>{}),
            nb::arg("use_legacy") = nb::none()));
}

template <ttnn::unique_string Name, typename Fn>
void bind_binary_with_float_param(
    nb::module_& mod,
    const std::string& description,
    const std::string& math,
    Fn fn,
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& note = "") {
    auto doc = fmt::format(
        R"doc(
        {2}

        .. math::
            {3}

        Args:
            input_tensor_a (ttnn.Tensor): the input tensor.
            input_tensor_b (ttnn.Tensor): the input tensor.
            alpha (float): the value to be multiplied.

        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the operation. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): preallocated output tensor. Defaults to `None`.


        Returns:
            ttnn.Tensor: the output tensor.

        {6}

        Note:
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - {4}
                 - TILE, ROW_MAJOR

            If the input tensor is ROW_MAJOR layout, it will be internally converted to TILE layout.

            {5}
        )doc",
        std::string(Name),
        "ttnn." + std::string(Name),
        description,
        math,
        supported_dtype,
        note,
        BINARY_BROADCAST_DOC);

    ttnn::bind_function<Name>(
        mod,
        doc.c_str(),
        fn,
        nb::arg("input_tensor_a"),
        nb::arg("input_tensor_b"),
        nb::arg("alpha"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("output_tensor") = nb::none());
}

template <ttnn::unique_string Name, typename TensorScalarFn, typename TensorTensorFn>
void bind_bitwise_binary_ops_operation(
    nb::module_& mod,
    const std::string& description,
    const std::string& math,
    TensorScalarFn tensor_scalar_fn,
    TensorTensorFn tensor_tensor_fn,
    const std::string& info = ". ",
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& note = " ") {
    auto doc = fmt::format(
        R"doc(
        {2}

        .. math::
            {3}

        Args:
            input_tensor_a (ttnn.Tensor): the input tensor.
            input_tensor_b (ttnn.Tensor or Integer): the input tensor.

        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the operation. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): preallocated output tensor. Defaults to `None`.


        Returns:
            ttnn.Tensor: the output tensor.

        {7}

        Note:
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - {5}
                 - TILE, ROW_MAJOR

            If the input tensor is ROW_MAJOR layout, it will be internally converted to TILE layout.

            {6}
        )doc",
        std::string(Name),
        "ttnn." + std::string(Name),
        description,
        math,
        info,
        supported_dtype,
        note,
        BINARY_BROADCAST_DOC);

    ttnn::bind_function<Name>(
        mod,
        doc.c_str(),
        ttnn::overload_t(
            tensor_scalar_fn,
            nb::arg("input_tensor_a"),
            nb::arg("input_b"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none(),
            nb::arg("activations") = nb::cast(ttsl::Span<const unary::EltwiseUnaryWithParam>{}),
            nb::arg("input_tensor_a_activations") = nb::cast(ttsl::Span<const unary::EltwiseUnaryWithParam>{}),
            nb::arg("input_tensor_b_activations") = nb::cast(ttsl::Span<const unary::EltwiseUnaryWithParam>{}),
            nb::arg("use_legacy") = nb::none()),
        ttnn::overload_t(
            tensor_tensor_fn,
            nb::arg("input_tensor_a"),
            nb::arg("input_tensor_b"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none(),
            nb::arg("activations") = nb::cast(ttsl::Span<const unary::EltwiseUnaryWithParam>{}),
            nb::arg("input_tensor_a_activations") = nb::cast(ttsl::Span<const unary::EltwiseUnaryWithParam>{}),
            nb::arg("input_tensor_b_activations") = nb::cast(ttsl::Span<const unary::EltwiseUnaryWithParam>{}),
            nb::arg("use_legacy") = nb::none()));
}

template <ttnn::unique_string Name, typename Fn>
void bind_binary_composite(
    nb::module_& mod,
    const std::string& description,
    const std::string& math,
    Fn fn,
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& note = "") {
    auto doc = fmt::format(
        R"doc(
        {2}

        .. math::
            {3}

        Args:
            input_tensor_a (ttnn.Tensor): the input tensor.
            input_tensor_b (ttnn.Tensor): the input tensor.

        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the operation. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

        {6}

        Note:
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - {4}
                 - TILE, ROW_MAJOR

            If the input tensor is ROW_MAJOR layout, it will be internally converted to TILE layout.

            {5}
        )doc",
        std::string(Name),
        "ttnn." + std::string(Name),
        description,
        math,
        supported_dtype,
        note,
        BINARY_BROADCAST_DOC);

    ttnn::bind_function<Name>(
        mod,
        doc.c_str(),
        fn,
        nb::arg("input_tensor_a"),
        nb::arg("input_tensor_b"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none());
}

template <ttnn::unique_string Name, typename Fn>
void bind_binary_composite_with_rtol_atol(
    nb::module_& mod, const std::string& description, const std::string& math, Fn fn) {
    auto doc = fmt::format(
        R"doc(
        {2}

        .. math::
            {3}

        Args:
            input_tensor_a (ttnn.Tensor): the input tensor.
            input_tensor_b (ttnn.Tensor): the input tensor.

        Keyword args:
            rtol (float): relative tolerance. Defaults to `1e-05f`.
            atol (float): absolute tolerance. Defaults to `1e-08f`.
            equal_nan (bool): if NaN values should be treated as equal during comparison. Defaults to `False`.
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the operation. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

        {4}

        Note:
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - BFLOAT16
                 - TILE, ROW_MAJOR

            If the input tensor is ROW_MAJOR layout, it will be internally converted to TILE layout.
        )doc",

        std::string(Name),
        "ttnn." + std::string(Name),
        description,
        math,
        BINARY_BROADCAST_DOC);

    ttnn::bind_function<Name>(
        mod,
        doc.c_str(),
        fn,
        nb::arg("input_tensor_a"),
        nb::arg("input_tensor_b"),
        nb::kw_only(),
        nb::arg("rtol") = 1e-05f,
        nb::arg("atol") = 1e-08f,
        nb::arg("equal_nan") = false,
        nb::arg("memory_config") = nb::none());
}

// https://nanobind.readthedocs.io/en/latest/api_extra.html

template <ttnn::unique_string Name, typename TensorTensorFn, typename TensorScalarFn>
void bind_binary_composite_overload(
    nb::module_& mod,
    const std::string& description,
    TensorTensorFn tensor_tensor_fn,
    TensorScalarFn tensor_scalar_fn,
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& note = "") {
    auto doc = fmt::format(
        R"doc(
        {2}

        .. math::
            \mathrm{{output\_tensor}} = \verb|{0}|(\mathrm{{input\_tensor\_a,input\_tensor\_b}})

        Args:
            input_tensor_a (ttnn.Tensor): the input tensor.
            input_tensor_b (ttnn.Tensor or Number): the input tensor.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the operation. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

        {5}

        Note:
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - {3}
                 - TILE, ROW_MAJOR

            If the input tensor is ROW_MAJOR layout, it will be internally converted to TILE layout.

            {4}
        )doc",
        std::string(Name),
        "ttnn." + std::string(Name),
        description,
        supported_dtype,
        note,
        BINARY_BROADCAST_DOC);

    ttnn::bind_function<Name>(
        mod,
        doc.c_str(),
        ttnn::overload_t(
            tensor_tensor_fn,
            nb::arg("input_tensor_a"),
            nb::arg("input_tensor_b"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()),
        ttnn::overload_t(
            tensor_scalar_fn,
            nb::arg("input_tensor_a"),
            nb::arg("value"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()));
}

template <ttnn::unique_string Name, typename TensorTensorFn, typename TensorScalarFn, typename TensorArrayFn>
void bind_prelu(
    nb::module_& mod,
    const std::string& description,
    TensorTensorFn tensor_tensor_fn,
    TensorScalarFn tensor_scalar_fn,
    TensorArrayFn tensor_array_fn,
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& note = "") {
    auto doc = fmt::format(
        R"doc(
        {2}

        .. math::
            \mathrm{{output\_tensor}} = \verb|{0}|(\mathrm{{input\_tensor\_a,input\_tensor\_b}})

        Args:
            input_tensor_a (ttnn.Tensor): the input tensor.
            input_tensor_b (ttnn.Tensor or List[float] of length 1 or Number): weight.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the operation. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

        {5}

        Note:
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - {3}
                 - TILE, ROW_MAJOR

            If the input tensor is ROW_MAJOR layout, it will be internally converted to TILE layout.

            {4}
        )doc",
        std::string(Name),
        "ttnn." + std::string(Name),
        description,
        supported_dtype,
        note,
        BINARY_BROADCAST_DOC);

    ttnn::bind_function<Name>(
        mod,
        doc.c_str(),
        ttnn::overload_t(
            tensor_tensor_fn,
            nb::arg("input_tensor_a"),
            nb::arg("weight"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()),
        ttnn::overload_t(
            tensor_scalar_fn,
            nb::arg("input_tensor_a"),
            nb::arg("weight"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()),
        ttnn::overload_t(
            tensor_array_fn,
            nb::arg("input_tensor_a"),
            nb::arg("weight"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()));
}

template <ttnn::unique_string Name, typename TensorTensorFn, typename TensorScalarFn>
void bind_div(
    nb::module_& mod,
    const std::string& description,
    const std::string& math,
    TensorTensorFn tensor_tensor_fn,
    TensorScalarFn tensor_scalar_fn,
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& note = " ") {
    auto doc = fmt::format(
        R"doc(
        {2}

        .. math::
            {3}

        Args:
            input_tensor_a (ttnn.Tensor): the input tensor.
            input_tensor_b (ttnn.Tensor or Number): the input tensor.

        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the operation. Defaults to `None`.
            fast_and_approximate_mode (bool, optional): `true` if input_tensor_b is non-zero for fast approximation, else `false` for accurate division (Only if the input tensor is not ComplexTensor). Defaults to `false`.
            rounding_mode (string, optional): can be `None`, `floor` and `trunc` (only if the input tensor is not ComplexTensor). Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): preallocated output tensor. Defaults to `None`.


        Returns:
            ttnn.Tensor: the output tensor.

        {6}

        Note:
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - BFLOAT16
                 - TILE, ROW_MAJOR

            If the input tensor is ROW_MAJOR layout, it will be internally converted to TILE layout.

        )doc",

        std::string(Name),
        "ttnn." + std::string(Name),
        description,
        math,
        supported_dtype,
        note,
        BINARY_BROADCAST_DOC);

    auto tensor_tensor_overload = ttnn::overload_t(
        tensor_tensor_fn,
        nb::arg("input_tensor_a"),
        nb::arg("input_tensor_b"),
        nb::kw_only(),
        nb::arg("fast_and_approximate_mode") = false,
        nb::arg("rounding_mode") = nb::none(),
        nb::arg("dtype") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("output_tensor") = nb::none(),
        nb::arg("activations") = nb::cast(ttsl::Span<const unary::EltwiseUnaryWithParam>{}),
        nb::arg("input_tensor_a_activations") = nb::cast(ttsl::Span<const unary::EltwiseUnaryWithParam>{}),
        nb::arg("input_tensor_b_activations") = nb::cast(ttsl::Span<const unary::EltwiseUnaryWithParam>{}),
        nb::arg("use_legacy") = nb::none(),
        nb::arg("sub_core_grids") = nb::none());
    auto tensor_scalar_overload = ttnn::overload_t(
        tensor_scalar_fn,
        nb::arg("input_tensor_a"),
        nb::arg("value"),
        nb::kw_only(),
        nb::arg("fast_and_approximate_mode") = false,
        nb::arg("rounding_mode") = nb::none(),
        nb::arg("dtype") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("output_tensor") = nb::none(),
        nb::arg("activations") = nb::cast(ttsl::Span<const unary::EltwiseUnaryWithParam>{}),
        nb::arg("input_tensor_a_activations") = nb::cast(ttsl::Span<const unary::EltwiseUnaryWithParam>{}),
        nb::arg("input_tensor_b_activations") = nb::cast(ttsl::Span<const unary::EltwiseUnaryWithParam>{}),
        nb::arg("use_legacy") = nb::none(),
        nb::arg("sub_core_grids") = nb::none());
    ttnn::bind_function<Name>(mod, doc.c_str(), tensor_tensor_overload, tensor_scalar_overload);
}

// Free functions for multiply and divide with fast_and_approximate_mode
Tensor multiply_fast_approx_tensor_scalar(
    const Tensor& input_tensor_a,
    float value,
    bool fast_and_approximate_mode,
    const std::optional<const DataType>& dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<ttnn::Tensor>& output_tensor,
    ttsl::Span<const unary::EltwiseUnaryWithParam> activations,
    ttsl::Span<const unary::EltwiseUnaryWithParam> input_tensor_a_activations,
    ttsl::Span<const unary::EltwiseUnaryWithParam> input_tensor_b_activations,
    const std::optional<bool>& use_legacy,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    return ttnn::multiply(
        input_tensor_a,
        value,
        dtype,
        memory_config,
        output_tensor,
        tt::stl::Span<const unary::EltwiseUnaryWithParam>(activations.data(), activations.size()),
        tt::stl::Span<const unary::EltwiseUnaryWithParam>(
            input_tensor_a_activations.data(), input_tensor_a_activations.size()),
        tt::stl::Span<const unary::EltwiseUnaryWithParam>(
            input_tensor_b_activations.data(), input_tensor_b_activations.size()),
        use_legacy,
        fast_and_approximate_mode,
        sub_core_grids);
}

Tensor multiply_fast_approx_tensor_tensor(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    bool fast_and_approximate_mode,
    const std::optional<const DataType>& dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<ttnn::Tensor>& output_tensor,
    ttsl::Span<const unary::EltwiseUnaryWithParam> activations,
    ttsl::Span<const unary::EltwiseUnaryWithParam> input_tensor_a_activations,
    ttsl::Span<const unary::EltwiseUnaryWithParam> input_tensor_b_activations,
    const std::optional<bool>& use_legacy,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    return ttnn::multiply(
        input_tensor_a,
        input_tensor_b,
        dtype,
        memory_config,
        output_tensor,
        tt::stl::Span<const unary::EltwiseUnaryWithParam>(activations.data(), activations.size()),
        tt::stl::Span<const unary::EltwiseUnaryWithParam>(
            input_tensor_a_activations.data(), input_tensor_a_activations.size()),
        tt::stl::Span<const unary::EltwiseUnaryWithParam>(
            input_tensor_b_activations.data(), input_tensor_b_activations.size()),
        use_legacy,
        fast_and_approximate_mode,
        sub_core_grids);
}

Tensor divide_fast_approx_tensor_scalar(
    const Tensor& input_tensor_a,
    float value,
    bool fast_and_approximate_mode,
    const std::optional<const DataType>& dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<ttnn::Tensor>& output_tensor,
    ttsl::Span<const unary::EltwiseUnaryWithParam> activations,
    ttsl::Span<const unary::EltwiseUnaryWithParam> input_tensor_a_activations,
    ttsl::Span<const unary::EltwiseUnaryWithParam> input_tensor_b_activations,
    const std::optional<bool>& use_legacy,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    return ttnn::divide(
        input_tensor_a,
        value,
        dtype,
        memory_config,
        output_tensor,
        tt::stl::Span<const unary::EltwiseUnaryWithParam>(activations.data(), activations.size()),
        tt::stl::Span<const unary::EltwiseUnaryWithParam>(
            input_tensor_a_activations.data(), input_tensor_a_activations.size()),
        tt::stl::Span<const unary::EltwiseUnaryWithParam>(
            input_tensor_b_activations.data(), input_tensor_b_activations.size()),
        use_legacy,
        fast_and_approximate_mode,
        sub_core_grids);
}

Tensor divide_fast_approx_tensor_tensor(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    bool fast_and_approximate_mode,
    const std::optional<const DataType>& dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<ttnn::Tensor>& output_tensor,
    ttsl::Span<const unary::EltwiseUnaryWithParam> activations,
    ttsl::Span<const unary::EltwiseUnaryWithParam> input_tensor_a_activations,
    ttsl::Span<const unary::EltwiseUnaryWithParam> input_tensor_b_activations,
    const std::optional<bool>& use_legacy,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    return ttnn::divide(
        input_tensor_a,
        input_tensor_b,
        dtype,
        memory_config,
        output_tensor,
        tt::stl::Span<const unary::EltwiseUnaryWithParam>(activations.data(), activations.size()),
        tt::stl::Span<const unary::EltwiseUnaryWithParam>(
            input_tensor_a_activations.data(), input_tensor_a_activations.size()),
        tt::stl::Span<const unary::EltwiseUnaryWithParam>(
            input_tensor_b_activations.data(), input_tensor_b_activations.size()),
        use_legacy,
        fast_and_approximate_mode,
        sub_core_grids);
}

template <ttnn::unique_string Name, typename TensorScalarFn, typename TensorTensorFn>
void bind_binary_operation_with_fast_approx(
    nb::module_& mod,
    const std::string& description,
    const std::string& math,
    TensorScalarFn tensor_scalar_fn,
    TensorTensorFn tensor_tensor_fn,
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& note = " ") {
    auto doc = fmt::format(
        R"doc(
        {2}

        .. math::
            {3}

        Args:
            input_tensor_a (ttnn.Tensor): the input tensor.
            input_tensor_b (ttnn.Tensor or Number): the input tensor.

        Keyword args:
            fast_and_approximate_mode (bool, optional): Use the fast and approximate mode. Defaults to `False`.
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the operation. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): preallocated output tensor. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

        {6}

        Note:
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - {4}
                 - TILE, ROW_MAJOR

            If the input tensor is ROW_MAJOR layout, it will be internally converted to TILE layout.

            {5}
        )doc",
        std::string(Name),
        "ttnn." + std::string(Name),
        description,
        math,
        supported_dtype,
        note,
        BINARY_BROADCAST_DOC);

    ttnn::bind_function<Name>(
        mod,
        doc.c_str(),
        ttnn::overload_t(
            tensor_scalar_fn,
            nb::arg("input_tensor_a"),
            nb::arg("input_tensor_b"),
            nb::kw_only(),
            nb::arg("fast_and_approximate_mode") = false,
            nb::arg("dtype") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none(),
            nb::arg("activations") = nb::cast(ttsl::Span<const unary::EltwiseUnaryWithParam>{}),
            nb::arg("input_tensor_a_activations") = nb::cast(ttsl::Span<const unary::EltwiseUnaryWithParam>{}),
            nb::arg("input_tensor_b_activations") = nb::cast(ttsl::Span<const unary::EltwiseUnaryWithParam>{}),
            nb::arg("use_legacy") = nb::none(),
            nb::arg("sub_core_grids") = nb::none()),
        ttnn::overload_t(
            tensor_tensor_fn,
            nb::arg("input_tensor_a"),
            nb::arg("input_tensor_b"),
            nb::kw_only(),
            nb::arg("fast_and_approximate_mode") = false,
            nb::arg("dtype") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none(),
            nb::arg("activations") = nb::cast(ttsl::Span<const unary::EltwiseUnaryWithParam>{}),
            nb::arg("input_tensor_a_activations") = nb::cast(ttsl::Span<const unary::EltwiseUnaryWithParam>{}),
            nb::arg("input_tensor_b_activations") = nb::cast(ttsl::Span<const unary::EltwiseUnaryWithParam>{}),
            nb::arg("use_legacy") = nb::none(),
            nb::arg("sub_core_grids") = nb::none()));
}

template <ttnn::unique_string Name, typename Fn>
void bind_polyval(
    nb::module_& mod,
    const std::string& description,
    const std::string& math,
    Fn fn,
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& note = " ") {
    auto doc = fmt::format(
        R"doc(
        {2}

        .. math::
            {3}

        Args:
            input_tensor (ttnn.Tensor): the input tensor.
            Coeffs (Vector of floats): coefficients of the polynomial.

        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the operation. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

        {6}

        Note:
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - {4}
                 - TILE, ROW_MAJOR

            If the input tensor is ROW_MAJOR layout, it will be internally converted to TILE layout.

            {5}
        )doc",
        std::string(Name),
        "ttnn." + std::string(Name),
        description,
        math,
        supported_dtype,
        note,
        BINARY_BROADCAST_DOC);

    ttnn::bind_function<Name>(
        mod,
        doc.c_str(),
        fn,
        nb::arg("input_tensor_a"),
        nb::arg("coeffs"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none());
}

template <ttnn::unique_string Name, typename TensorScalarFn, typename TensorTensorFn>
void bind_binary_overload_operation(
    nb::module_& mod,
    const std::string& description,
    const std::string& math,
    TensorScalarFn tensor_scalar_fn,
    TensorTensorFn tensor_tensor_fn,
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& note = " ") {
    auto doc = fmt::format(
        R"doc(
        {2}

        .. math::
            {3}

        Args:
            input_tensor_a (ttnn.Tensor): the input tensor.
            input_tensor_b (ttnn.Tensor or Number): the input tensor.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the operation. Defaults to `None`.
            sub_core_grids (ttnn.CoreRangeSet, optional): sub core grids for the operation. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

        {6}

        Note:
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - {4}
                 - TILE, ROW_MAJOR

            If the input tensor is ROW_MAJOR layout, it will be internally converted to TILE layout.

            {5}
        )doc",
        std::string(Name),
        "ttnn." + std::string(Name),
        description,
        math,
        supported_dtype,
        note,
        BINARY_BROADCAST_DOC);

    ttnn::bind_function<Name>(
        mod,
        doc.c_str(),
        ttnn::overload_t(
            tensor_scalar_fn,
            nb::arg("input_tensor_a"),
            nb::arg("scalar"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("sub_core_grids") = nb::none()),
        ttnn::overload_t(
            tensor_tensor_fn,
            nb::arg("input_tensor_a"),
            nb::arg("input_tensor_b"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("sub_core_grids") = nb::none()));
}

template <ttnn::unique_string Name, typename TensorScalarFn, typename TensorTensorFn>
void bind_inplace_operation(
    nb::module_& mod,
    const std::string& description,
    const std::string& math,
    TensorScalarFn tensor_scalar_fn,
    TensorTensorFn tensor_tensor_fn,
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& note = "") {
    auto doc = fmt::format(
        R"doc(
        {2}

        .. math::
            {3}

        Args:
            input_tensor_a (ttnn.Tensor): the input tensor.
            input_tensor_b (ttnn.Tensor or Number): the input tensor.

        Keyword Args:
            sub_core_grids (ttnn.CoreRangeSet, optional): sub core grids for the operation. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

        {6}

        Note:
            Supported dtypes and layouts:

            .. list-table::
                :header-rows: 1

                * - Dtypes
                  - Layouts
                * - {4}
                  - TILE, ROW_MAJOR

            If the input tensor is ROW_MAJOR layout, it will be internally converted to TILE layout.

            {5}
        )doc",
        std::string(Name),
        "ttnn." + std::string(Name),
        description,
        math,
        supported_dtype,
        note,
        BINARY_BROADCAST_DOC);

    ttnn::bind_function<Name>(
        mod,
        doc.c_str(),
        ttnn::overload_t(
            tensor_scalar_fn,
            nb::arg("input_a"),
            nb::arg("input_b"),
            nb::kw_only(),
            nb::arg("activations") = nb::cast(ttsl::Span<const unary::EltwiseUnaryWithParam>{}),
            nb::arg("input_tensor_a_activations") = nb::cast(ttsl::Span<const unary::EltwiseUnaryWithParam>{}),
            nb::arg("input_tensor_b_activations") = nb::cast(ttsl::Span<const unary::EltwiseUnaryWithParam>{}),
            nb::arg("use_legacy") = nb::none(),
            nb::arg("sub_core_grids") = nb::none()),
        ttnn::overload_t(
            tensor_tensor_fn,
            nb::arg("input_a"),
            nb::arg("input_b"),
            nb::kw_only(),
            nb::arg("activations") = nb::cast(ttsl::Span<const unary::EltwiseUnaryWithParam>{}),
            nb::arg("input_tensor_a_activations") = nb::cast(ttsl::Span<const unary::EltwiseUnaryWithParam>{}),
            nb::arg("input_tensor_b_activations") = nb::cast(ttsl::Span<const unary::EltwiseUnaryWithParam>{}),
            nb::arg("use_legacy") = nb::none(),
            nb::arg("sub_core_grids") = nb::none()));
}

// Free functions for inplace multiply and divide with fast_and_approximate_mode
Tensor multiply_inplace_fast_approx_tensor_scalar(
    const Tensor& input_tensor_a,
    float scalar,
    ttsl::Span<const unary::EltwiseUnaryWithParam> activations,
    ttsl::Span<const unary::EltwiseUnaryWithParam> input_tensor_a_activations,
    ttsl::Span<const unary::EltwiseUnaryWithParam> input_tensor_b_activations,
    const std::optional<bool>& use_legacy,
    bool fast_and_approximate_mode,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    return ttnn::multiply_(
        input_tensor_a,
        scalar,
        tt::stl::Span<const unary::EltwiseUnaryWithParam>(activations.data(), activations.size()),
        tt::stl::Span<const unary::EltwiseUnaryWithParam>(
            input_tensor_a_activations.data(), input_tensor_a_activations.size()),
        tt::stl::Span<const unary::EltwiseUnaryWithParam>(
            input_tensor_b_activations.data(), input_tensor_b_activations.size()),
        use_legacy,
        std::optional<bool>(fast_and_approximate_mode),
        sub_core_grids);
}

Tensor multiply_inplace_fast_approx_tensor_tensor(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    ttsl::Span<const unary::EltwiseUnaryWithParam> activations,
    ttsl::Span<const unary::EltwiseUnaryWithParam> input_tensor_a_activations,
    ttsl::Span<const unary::EltwiseUnaryWithParam> input_tensor_b_activations,
    const std::optional<bool>& use_legacy,
    bool fast_and_approximate_mode,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    return ttnn::multiply_(
        input_tensor_a,
        input_tensor_b,
        tt::stl::Span<const unary::EltwiseUnaryWithParam>(activations.data(), activations.size()),
        tt::stl::Span<const unary::EltwiseUnaryWithParam>(
            input_tensor_a_activations.data(), input_tensor_a_activations.size()),
        tt::stl::Span<const unary::EltwiseUnaryWithParam>(
            input_tensor_b_activations.data(), input_tensor_b_activations.size()),
        use_legacy,
        std::optional<bool>(fast_and_approximate_mode),
        sub_core_grids);
}

Tensor divide_inplace_fast_approx_tensor_scalar(
    const Tensor& input_tensor_a,
    float scalar,
    ttsl::Span<const unary::EltwiseUnaryWithParam> activations,
    ttsl::Span<const unary::EltwiseUnaryWithParam> input_tensor_a_activations,
    ttsl::Span<const unary::EltwiseUnaryWithParam> input_tensor_b_activations,
    const std::optional<bool>& use_legacy,
    bool fast_and_approximate_mode,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    return ttnn::divide_(
        input_tensor_a,
        scalar,
        tt::stl::Span<const unary::EltwiseUnaryWithParam>(activations.data(), activations.size()),
        tt::stl::Span<const unary::EltwiseUnaryWithParam>(
            input_tensor_a_activations.data(), input_tensor_a_activations.size()),
        tt::stl::Span<const unary::EltwiseUnaryWithParam>(
            input_tensor_b_activations.data(), input_tensor_b_activations.size()),
        use_legacy,
        std::optional<bool>(fast_and_approximate_mode),
        sub_core_grids);
}

Tensor divide_inplace_fast_approx_tensor_tensor(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    ttsl::Span<const unary::EltwiseUnaryWithParam> activations,
    ttsl::Span<const unary::EltwiseUnaryWithParam> input_tensor_a_activations,
    ttsl::Span<const unary::EltwiseUnaryWithParam> input_tensor_b_activations,
    const std::optional<bool>& use_legacy,
    bool fast_and_approximate_mode,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    return ttnn::divide_(
        input_tensor_a,
        input_tensor_b,
        tt::stl::Span<const unary::EltwiseUnaryWithParam>(activations.data(), activations.size()),
        tt::stl::Span<const unary::EltwiseUnaryWithParam>(
            input_tensor_a_activations.data(), input_tensor_a_activations.size()),
        tt::stl::Span<const unary::EltwiseUnaryWithParam>(
            input_tensor_b_activations.data(), input_tensor_b_activations.size()),
        use_legacy,
        std::optional<bool>(fast_and_approximate_mode),
        sub_core_grids);
}

template <typename InplaceFastApproxOp>
Tensor inplace_fast_approx_binding_tensor_scalar(
    const Tensor& input_tensor_a,
    float scalar,
    ttsl::Span<const unary::EltwiseUnaryWithParam> activations,
    ttsl::Span<const unary::EltwiseUnaryWithParam> input_tensor_a_activations,
    ttsl::Span<const unary::EltwiseUnaryWithParam> input_tensor_b_activations,
    const std::optional<bool>& use_legacy,
    bool fast_and_approximate_mode,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    return InplaceFastApproxOp::tensor_scalar(
        input_tensor_a,
        scalar,
        tt::stl::Span<const unary::EltwiseUnaryWithParam>(activations.data(), activations.size()),
        tt::stl::Span<const unary::EltwiseUnaryWithParam>(
            input_tensor_a_activations.data(), input_tensor_a_activations.size()),
        tt::stl::Span<const unary::EltwiseUnaryWithParam>(
            input_tensor_b_activations.data(), input_tensor_b_activations.size()),
        use_legacy,
        fast_and_approximate_mode,
        sub_core_grids);
}
template <typename InplaceFastApproxOp>
Tensor inplace_fast_approx_binding_tensor_tensor(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    ttsl::Span<const unary::EltwiseUnaryWithParam> activations,
    ttsl::Span<const unary::EltwiseUnaryWithParam> input_tensor_a_activations,
    ttsl::Span<const unary::EltwiseUnaryWithParam> input_tensor_b_activations,
    const std::optional<bool>& use_legacy,
    bool fast_and_approximate_mode,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    return InplaceFastApproxOp::tensor_tensor(
        input_tensor_a,
        input_tensor_b,
        tt::stl::Span<const unary::EltwiseUnaryWithParam>(activations.data(), activations.size()),
        tt::stl::Span<const unary::EltwiseUnaryWithParam>(
            input_tensor_a_activations.data(), input_tensor_a_activations.size()),
        tt::stl::Span<const unary::EltwiseUnaryWithParam>(
            input_tensor_b_activations.data(), input_tensor_b_activations.size()),
        use_legacy,
        fast_and_approximate_mode,
        sub_core_grids);
}

template <ttnn::unique_string Name, typename TensorScalarFn, typename TensorTensorFn>
void bind_inplace_operation_with_fast_approx(
    nb::module_& mod,
    const std::string& description,
    const std::string& math,
    TensorScalarFn tensor_scalar_fn,
    TensorTensorFn tensor_tensor_fn,
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& note = " ") {
    auto doc = fmt::format(
        R"doc(
        {2}

        .. math::
            {3}

        Args:
            input_tensor_a (ttnn.Tensor): the input tensor.
            input_tensor_b (ttnn.Tensor): the input tensor.

        Keyword args:
            fast_and_approximate_mode (bool, optional): Use the fast and approximate mode. Defaults to `False`.
            sub_core_grids (ttnn.CoreRangeSet, optional): sub core grids for the operation. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

        {6}

        Note:
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - {4}
                 - TILE, ROW_MAJOR

            If the input tensor is ROW_MAJOR layout, it will be internally converted to TILE layout.

            {5}
        )doc",
        std::string(Name),
        "ttnn." + std::string(Name),
        description,
        math,
        supported_dtype,
        note,
        BINARY_BROADCAST_DOC);

    ttnn::bind_function<Name>(
        mod,
        doc.c_str(),
        ttnn::overload_t(
            tensor_scalar_fn,
            nb::arg("input_a"),
            nb::arg("input_b"),
            nb::kw_only(),
            nb::arg("activations") = nb::cast(ttsl::Span<const unary::EltwiseUnaryWithParam>{}),
            nb::arg("input_tensor_a_activations") = nb::cast(ttsl::Span<const unary::EltwiseUnaryWithParam>{}),
            nb::arg("input_tensor_b_activations") = nb::cast(ttsl::Span<const unary::EltwiseUnaryWithParam>{}),
            nb::arg("use_legacy") = nb::none(),
            nb::arg("fast_and_approximate_mode") = false,
            nb::arg("sub_core_grids") = nb::none()),
        ttnn::overload_t(
            tensor_tensor_fn,
            nb::arg("input_a"),
            nb::arg("input_b"),
            nb::kw_only(),
            nb::arg("activations") = nb::cast(ttsl::Span<const unary::EltwiseUnaryWithParam>{}),
            nb::arg("input_tensor_a_activations") = nb::cast(ttsl::Span<const unary::EltwiseUnaryWithParam>{}),
            nb::arg("input_tensor_b_activations") = nb::cast(ttsl::Span<const unary::EltwiseUnaryWithParam>{}),
            nb::arg("use_legacy") = nb::none(),
            nb::arg("fast_and_approximate_mode") = false,
            nb::arg("sub_core_grids") = nb::none()));
}

void bind_power(nb::module_& mod, const std::string& note = "") {
    auto doc = fmt::format(
        R"doc(
        Perform element-wise {0} operation on :attr:`input_tensor` with :attr:`exponent`.

        .. math::
            \mathrm{{output\_tensor}}_i = (\mathrm{{input\_tensor}}_i ** \mathrm{{exponent}}_i)

        Args:
            input_tensor (ttnn.Tensor, float): the input tensor.
            exponent (float, int, ttnn.Tensor): the exponent value.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the operation. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): preallocated output tensor. Defaults to `None`.


        Returns:
            ttnn.Tensor: the output tensor.

        {3}

        Note:
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - BFLOAT16, BFLOAT8_B, FLOAT32
                 - TILE, ROW_MAJOR

            If the input tensor is ROW_MAJOR layout, it will be internally converted to TILE layout.

            {2}
        )doc",
        "pow",
        "ttnn.pow",
        note,
        BINARY_BROADCAST_DOC);

    ttnn::bind_function<"pow">(
        mod,
        doc.c_str(),
        ttnn::overload_t(
            static_cast<Tensor (*)(
                const Tensor&, int32_t, const std::optional<MemoryConfig>&, const std::optional<Tensor>&)>(&ttnn::pow),
            nb::arg("input_tensor"),
            nb::arg("exponent"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none()),
        ttnn::overload_t(
            static_cast<Tensor (*)(
                const Tensor&, float, const std::optional<MemoryConfig>&, const std::optional<Tensor>&)>(&ttnn::pow),
            nb::arg("input_tensor"),
            nb::arg("exponent"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none()),
        ttnn::overload_t(
            static_cast<Tensor (*)(
                const Tensor&,
                const Tensor&,
                const std::optional<const DataType>&,
                const std::optional<MemoryConfig>&,
                const std::optional<Tensor>&,
                ttsl::Span<const unary::EltwiseUnaryWithParam>,
                ttsl::Span<const unary::EltwiseUnaryWithParam>,
                ttsl::Span<const unary::EltwiseUnaryWithParam>,
                std::optional<bool>)>(&ttnn::pow),
            nb::arg("input_tensor"),
            nb::arg("exponent"),
            nb::kw_only(),
            nb::arg("dtype") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none(),
            nb::arg("activations") = nb::cast(ttsl::Span<const unary::EltwiseUnaryWithParam>{}),
            nb::arg("input_tensor_a_activations") = nb::cast(ttsl::Span<const unary::EltwiseUnaryWithParam>{}),
            nb::arg("input_tensor_b_activations") = nb::cast(ttsl::Span<const unary::EltwiseUnaryWithParam>{}),
            nb::arg("use_legacy") = nb::none()),
        ttnn::overload_t(
            static_cast<Tensor (*)(
                float,
                const Tensor&,
                const std::optional<const DataType>&,
                const std::optional<MemoryConfig>&,
                const std::optional<Tensor>&,
                ttsl::Span<const unary::EltwiseUnaryWithParam>,
                ttsl::Span<const unary::EltwiseUnaryWithParam>,
                ttsl::Span<const unary::EltwiseUnaryWithParam>,
                std::optional<bool>)>(&ttnn::pow),
            nb::arg("input_tensor"),
            nb::arg("exponent"),
            nb::kw_only(),
            nb::arg("dtype") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none(),
            nb::arg("activations") = nb::cast(ttsl::Span<const unary::EltwiseUnaryWithParam>{}),
            nb::arg("input_tensor_a_activations") = nb::cast(ttsl::Span<const unary::EltwiseUnaryWithParam>{}),
            nb::arg("input_tensor_b_activations") = nb::cast(ttsl::Span<const unary::EltwiseUnaryWithParam>{}),
            nb::arg("use_legacy") = nb::none()));
}
}  // namespace detail

void py_module(nb::module_& mod) {
    export_enum<BinaryOpType>(mod, "BinaryOpType");

    detail::bind_binary_operation<"add">(
        mod,
        R"doc(Adds :attr:`input_tensor_a` to :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{output\_tensor}}_i = \mathrm{{input\_tensor\_a}}_i + \mathrm{{input\_tensor\_b}}_i)doc",
        &detail::binary_op_binding_tensor_scalar<BinaryOpType::ADD>,
        &detail::binary_op_binding_tensor_tensor<BinaryOpType::ADD>,
        R"doc(: :code:`'None'` | :code:`'relu'`. )doc",
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32, INT32, UINT32 (range: [0, 4294967295]), UINT16 (range: [0, 65535]))doc");

    detail::bind_binary_inplace_operation<"add_">(
        mod,
        R"doc(Adds :attr:`input_tensor_a` to :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a` in-place)doc",
        R"doc(\mathrm{{input\_tensor\_a}}_i + \mathrm{{input\_tensor\_b}}_i)doc",
        &detail::inplace_binding_tensor_scalar<static_cast<detail::InplaceScalarFn>(&ttnn::add_)>,
        &detail::inplace_binding_tensor_tensor<static_cast<detail::InplaceTensorFn>(&ttnn::add_)>,
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32, INT32, UINT32 (range: [0, 4294967295]), UINT16 (range: [0, 65535]))doc");

    detail::bind_binary_operation<"subtract">(
        mod,
        R"doc(Subtracts :attr:`input_tensor_b` from :attr:`input_tensor_a` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{output\_tensor}}_i = \mathrm{{input\_tensor\_a}}_i - \mathrm{{input\_tensor\_b}}_i)doc",
        &detail::binary_op_binding_tensor_scalar<BinaryOpType::SUB>,
        &detail::binary_op_binding_tensor_tensor<BinaryOpType::SUB>,
        R"doc(: :code:`'None'` | :code:`'relu'`. )doc",
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32, INT32, UINT16 (range: 0 - 65535), UINT32 (range: 0 - 4294967295))doc");

    detail::bind_binary_inplace_operation<"subtract_">(
        mod,
        R"doc(Subtracts :attr:`input_tensor_b` from :attr:`input_tensor_a` and returns the tensor with the same layout as :attr:`input_tensor_a` in-place)doc",
        R"doc(\mathrm{{input\_tensor\_a}}_i - \mathrm{{input\_tensor\_b}}_i)doc",
        &detail::inplace_binding_tensor_scalar<static_cast<detail::InplaceScalarFn>(&ttnn::subtract_)>,
        &detail::inplace_binding_tensor_tensor<static_cast<detail::InplaceTensorFn>(&ttnn::subtract_)>,
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32, INT32, UINT16 (range: 0 - 65535), UINT32 (range: 0 - 4294967295))doc");

    detail::bind_binary_operation<"eq">(
        mod,
        R"doc(Compares if :attr:`input_tensor_a` is equal to :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{output\_tensor}}_i = (\mathrm{{input\_tensor\_a}}_i == \mathrm{{input\_tensor\_b}}_i))doc",
        &detail::binary_op_binding_tensor_scalar<BinaryOpType::EQ>,
        &detail::binary_op_binding_tensor_tensor<BinaryOpType::EQ>,
        ". ",
        R"doc(Float32, BFLOAT16, BFLOAT8_B, INT32, UINT32, UINT16)doc");

    detail::bind_binary_operation<"ne">(
        mod,
        R"doc(Compares if :attr:`input_tensor_a` is not equal to :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{output\_tensor}}_i = (\mathrm{{input\_tensor\_a}}_i != \mathrm{{input\_tensor\_b}}_i))doc",
        &detail::binary_op_binding_tensor_scalar<BinaryOpType::NE>,
        &detail::binary_op_binding_tensor_tensor<BinaryOpType::NE>,
        ". ",
        R"doc(Float32, BFLOAT16, BFLOAT8_B, INT32, UINT32, UINT16)doc");

    detail::bind_binary_operation<"lt">(
        mod,
        R"doc(Compares if :attr:`input_tensor_a` is less than :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{output\_tensor}}_i = (\mathrm{{input\_tensor\_a}}_i < \mathrm{{input\_tensor\_b}}_i))doc",
        &detail::binary_op_binding_tensor_scalar<BinaryOpType::LT>,
        &detail::binary_op_binding_tensor_tensor<BinaryOpType::LT>,
        ". ",
        R"doc(Float32, BFLOAT16, BFLOAT8_B, INT32)doc",
        "INT32 supported only for tensor-tensor.");

    detail::bind_binary_operation<"le">(
        mod,
        R"doc(Compares if :attr:`input_tensor_a` is less than or equal to :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{output\_tensor}}_i = (\mathrm{{input\_tensor\_a}}_i <= \mathrm{{input\_tensor\_b}}_i))doc",
        &detail::binary_op_binding_tensor_scalar<BinaryOpType::LE>,
        &detail::binary_op_binding_tensor_tensor<BinaryOpType::LE>,
        ". ",
        R"doc(Float32, BFLOAT16, BFLOAT8_B, INT32)doc",
        "INT32 supported only for tensor-tensor.");

    detail::bind_binary_operation<"gt">(
        mod,
        R"doc(Compares if :attr:`input_tensor_a` is greater than :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{output\_tensor}}_i = (\mathrm{{input\_tensor\_a}}_i > \mathrm{{input\_tensor\_b}}_i))doc",
        &detail::binary_op_binding_tensor_scalar<BinaryOpType::GT>,
        &detail::binary_op_binding_tensor_tensor<BinaryOpType::GT>,
        ". ",
        R"doc(Float32, BFLOAT16, BFLOAT8_B, INT32)doc",
        "INT32 supported only for tensor-tensor.");

    detail::bind_binary_operation<"ge">(
        mod,
        R"doc(Compares if :attr:`input_tensor_a` is greater than or equal to :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{output\_tensor}}_i = (\mathrm{{input\_tensor\_a}}_i >= \mathrm{{input\_tensor\_b}}_i))doc",
        &detail::binary_op_binding_tensor_scalar<BinaryOpType::GE>,
        &detail::binary_op_binding_tensor_tensor<BinaryOpType::GE>,
        ". ",
        R"doc(Float32, BFLOAT16, BFLOAT8_B, INT32)doc",
        "INT32 supported only for tensor-tensor.");

    detail::bind_binary_operation<"logical_and">(
        mod,
        R"doc(Computes logical AND of :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{output\_tensor}}_i = \mathrm{{input\_tensor\_a}}_i \, \& \, \mathrm{{input\_tensor\_b}}_i)doc",
        &detail::binary_op_binding_tensor_scalar<BinaryOpType::LOGICAL_AND>,
        &detail::binary_op_binding_tensor_tensor<BinaryOpType::LOGICAL_AND>,
        ". ",
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32, INT32, UINT16)doc",
        "INT32 for tensor-scalar is supported only when use_legacy= False.");

    detail::bind_binary_operation<"logical_or">(
        mod,
        R"doc(Computes logical OR of :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{output\_tensor}}_i = \mathrm{{input\_tensor\_a}}_i \, | \, \mathrm{{input\_tensor\_b}}_i)doc",
        &detail::binary_op_binding_tensor_scalar<BinaryOpType::LOGICAL_OR>,
        &detail::binary_op_binding_tensor_tensor<BinaryOpType::LOGICAL_OR>,
        ". ",
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32, INT32, UINT32, UINT16)doc");

    detail::bind_binary_operation<"ldexp">(
        mod,
        R"doc(Computes ldexp of :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{output\_tensor}} = \verb|ldexp|(\mathrm{{input\_tensor\_a,input\_tensor\_b}}))doc",
        &detail::binary_op_binding_tensor_scalar<BinaryOpType::LDEXP>,
        &detail::binary_op_binding_tensor_tensor<BinaryOpType::LDEXP>,
        ". ",
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");

    detail::bind_binary_operation<"logaddexp">(
        mod,
        R"doc(Computes logaddexp of :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{output\_tensor}} = \verb|logaddexp|(\mathrm{{input\_tensor\_a,input\_tensor\_b}}))doc",
        &detail::binary_op_binding_tensor_scalar<BinaryOpType::LOGADDEXP>,
        &detail::binary_op_binding_tensor_tensor<BinaryOpType::LOGADDEXP>,
        ". ",
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");

    detail::bind_binary_operation<"logaddexp2">(
        mod,
        R"doc(Computes logaddexp2 of :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{output\_tensor}} = \verb|logaddexp2|(\mathrm{{input\_tensor\_a,input\_tensor\_b}}))doc",
        &detail::binary_op_binding_tensor_scalar<BinaryOpType::LOGADDEXP2>,
        &detail::binary_op_binding_tensor_tensor<BinaryOpType::LOGADDEXP2>,
        ". ",
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");

    detail::bind_binary_operation<"squared_difference">(
        mod,
        R"doc(Computes squared difference of :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{output\_tensor}} = \verb|squared_difference|(\mathrm{{input\_tensor\_a,input\_tensor\_b}}))doc",
        &detail::binary_op_binding_tensor_scalar<BinaryOpType::SQUARED_DIFFERENCE>,
        &detail::binary_op_binding_tensor_tensor<BinaryOpType::SQUARED_DIFFERENCE>,
        ". ",
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32, INT32, UINT32, UINT16)doc");

    detail::bind_binary_operation<"bias_gelu">(
        mod,
        R"doc(Computes bias_gelu of :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{output\_tensor}} = \verb|bias_gelu|(\mathrm{{input\_tensor\_a,input\_tensor\_b}}))doc",
        &detail::binary_op_binding_tensor_scalar<BinaryOpType::BIAS_GELU>,
        &detail::binary_op_binding_tensor_tensor<BinaryOpType::BIAS_GELU>,
        ". ",
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");

    detail::bind_binary_operation_with_fast_approx<"multiply">(
        mod,
        R"doc(Multiplies :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{output\_tensor}}_i = \mathrm{{input\_tensor\_a}}_i * \mathrm{{input\_tensor\_b}}_i)doc",
        &detail::multiply_fast_approx_tensor_scalar,
        &detail::multiply_fast_approx_tensor_tensor,
        R"doc(BFLOAT16, FLOAT32, INT32, UINT16, UINT32)doc",
        R"doc(
        When :attr:`fast_and_approximate_mode` is `True` for bfloat16 datatype, the operation uses FPU implementation for better performance.
        When :attr:`fast_and_approximate_mode` is `False` for bfloat16 datatype, the operation uses SFPU with the result rounded to nearest even (RNE).
        )doc");
    detail::bind_binary_operation_with_fast_approx<"divide">(
        mod,
        R"doc(Divides :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{output\_tensor}}_i = (\mathrm{{input\_tensor\_a}}_i / \mathrm{{input\_tensor\_b}}_i))doc",
        &detail::divide_fast_approx_tensor_scalar,
        &detail::divide_fast_approx_tensor_tensor,
        R"doc(BFLOAT16, FLOAT32, INT32, UINT16)doc",
        R"doc(
        When :attr:`fast_and_approximate_mode` is `True`, operation assumes that :attr:`input_tensor_b` is not zero.
        When :attr:`fast_and_approximate_mode` is `False` (default), operation properly handle division by zero.
        When the inputs are INT32, the outputs are FLOAT32 and output datatype conversion is not supported.
        )doc");

    detail::bind_binary_operation<"xlogy">(
        mod,
        R"doc(Computes xlogy :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{output\_tensor}_i = \mathrm{input\_tensor\_a}_i \cdot \log(\mathrm{input\_tensor\_b}_i)
        )doc",
        &detail::binary_op_binding_tensor_scalar<BinaryOpType::XLOGY>,
        &detail::binary_op_binding_tensor_tensor<BinaryOpType::XLOGY>);

    detail::bind_binary_unary_operation<"rsub">(
        mod,
        R"doc(Subtracts :attr:`input_tensor_a` from :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{output\_tensor}}_i = \mathrm{{input\_tensor\_b}}_i - \mathrm{{input\_tensor\_a}}_i)doc",
        static_cast<Tensor (*)(
            const Tensor&,
            float,
            const std::optional<const DataType>&,
            const std::optional<MemoryConfig>&,
            const std::optional<Tensor>&,
            ttsl::Span<const unary::EltwiseUnaryWithParam>,
            ttsl::Span<const unary::EltwiseUnaryWithParam>,
            ttsl::Span<const unary::EltwiseUnaryWithParam>,
            std::optional<bool>)>(&ttnn::rsub),
        static_cast<Tensor (*)(
            const Tensor&,
            const Tensor&,
            const std::optional<const DataType>&,
            const std::optional<MemoryConfig>&,
            const std::optional<Tensor>&,
            ttsl::Span<const unary::EltwiseUnaryWithParam>,
            ttsl::Span<const unary::EltwiseUnaryWithParam>,
            ttsl::Span<const unary::EltwiseUnaryWithParam>,
            std::optional<bool>)>(&ttnn::rsub),
        ". ",
        R"doc(FLOAT32,BFLOAT16, BFLOAT8_B, INT32, UINT32, UINT16)doc");

    detail::bind_bitwise_binary_ops_operation<"bitwise_and">(
        mod,
        R"doc(Perform bitwise_and operation on :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{output\_tensor}}_i = \verb|bitwise_and|(\mathrm{{input\_tensor\_a, input\_tensor\_b}}))doc",
        static_cast<Tensor (*)(
            const Tensor&,
            int32_t,
            const std::optional<MemoryConfig>&,
            const std::optional<Tensor>&,
            ttsl::Span<const unary::EltwiseUnaryWithParam>,
            ttsl::Span<const unary::EltwiseUnaryWithParam>,
            ttsl::Span<const unary::EltwiseUnaryWithParam>,
            std::optional<bool>)>(&ttnn::bitwise_and),
        static_cast<Tensor (*)(
            const Tensor&,
            const Tensor&,
            const std::optional<MemoryConfig>&,
            const std::optional<Tensor>&,
            ttsl::Span<const unary::EltwiseUnaryWithParam>,
            ttsl::Span<const unary::EltwiseUnaryWithParam>,
            ttsl::Span<const unary::EltwiseUnaryWithParam>,
            std::optional<bool>)>(&ttnn::bitwise_and),
        ". ",
        R"doc(INT32, UINT16 (range: 0 - 65535), UINT32)doc");

    detail::bind_bitwise_binary_ops_operation<"bitwise_or">(
        mod,
        R"doc(Perform bitwise_or operation on :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{output\_tensor}}_i = \verb|bitwise_or|(\mathrm{{input\_tensor\_a, input\_tensor\_b}}))doc",
        static_cast<Tensor (*)(
            const Tensor&,
            int32_t,
            const std::optional<MemoryConfig>&,
            const std::optional<Tensor>&,
            ttsl::Span<const unary::EltwiseUnaryWithParam>,
            ttsl::Span<const unary::EltwiseUnaryWithParam>,
            ttsl::Span<const unary::EltwiseUnaryWithParam>,
            std::optional<bool>)>(&ttnn::bitwise_or),
        static_cast<Tensor (*)(
            const Tensor&,
            const Tensor&,
            const std::optional<MemoryConfig>&,
            const std::optional<Tensor>&,
            ttsl::Span<const unary::EltwiseUnaryWithParam>,
            ttsl::Span<const unary::EltwiseUnaryWithParam>,
            ttsl::Span<const unary::EltwiseUnaryWithParam>,
            std::optional<bool>)>(&ttnn::bitwise_or),
        ". ",
        R"doc(INT32, UINT16 (range: 0 - 65535), UINT32)doc");

    detail::bind_bitwise_binary_ops_operation<"bitwise_xor">(
        mod,
        R"doc(Perform bitwise_xor operation on :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{output\_tensor}}_i = \verb|bitwise_xor|(\mathrm{{input\_tensor\_a, input\_tensor\_b}}))doc",
        static_cast<Tensor (*)(
            const Tensor&,
            int32_t,
            const std::optional<MemoryConfig>&,
            const std::optional<Tensor>&,
            ttsl::Span<const unary::EltwiseUnaryWithParam>,
            ttsl::Span<const unary::EltwiseUnaryWithParam>,
            ttsl::Span<const unary::EltwiseUnaryWithParam>,
            std::optional<bool>)>(&ttnn::bitwise_xor),
        static_cast<Tensor (*)(
            const Tensor&,
            const Tensor&,
            const std::optional<MemoryConfig>&,
            const std::optional<Tensor>&,
            ttsl::Span<const unary::EltwiseUnaryWithParam>,
            ttsl::Span<const unary::EltwiseUnaryWithParam>,
            ttsl::Span<const unary::EltwiseUnaryWithParam>,
            std::optional<bool>)>(&ttnn::bitwise_xor),
        ". ",
        R"doc(INT32, UINT16 (range: 0 - 65535), UINT32)doc");

    detail::bind_bitwise_binary_ops_operation<"bitwise_left_shift">(
        mod,
        R"doc(Perform bitwise_left_shift operation on :attr:`input_tensor_a` by :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`. :attr:`input_tensor_b` has shift_bits which are integers within range (0, 31))doc",
        R"doc(\mathrm{{output\_tensor}}_i = \verb|bitwise_and|(\mathrm{{input\_tensor\_a, input\_tensor\_b}}))doc",
        static_cast<Tensor (*)(
            const Tensor&,
            int32_t,
            const std::optional<MemoryConfig>&,
            const std::optional<Tensor>&,
            ttsl::Span<const unary::EltwiseUnaryWithParam>,
            ttsl::Span<const unary::EltwiseUnaryWithParam>,
            ttsl::Span<const unary::EltwiseUnaryWithParam>,
            std::optional<bool>)>(&ttnn::bitwise_left_shift),
        static_cast<Tensor (*)(
            const Tensor&,
            const Tensor&,
            const std::optional<MemoryConfig>&,
            const std::optional<Tensor>&,
            ttsl::Span<const unary::EltwiseUnaryWithParam>,
            ttsl::Span<const unary::EltwiseUnaryWithParam>,
            ttsl::Span<const unary::EltwiseUnaryWithParam>,
            std::optional<bool>)>(&ttnn::bitwise_left_shift),
        ". ",
        R"doc(INT32, UINT32, UINT16)doc");

    detail::bind_bitwise_binary_ops_operation<"bitwise_right_shift">(
        mod,
        R"doc(Perform bitwise_right_shift operation on :attr:`input_tensor_a` by :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`. :attr:`input_tensor_b` has shift_bits which are integers within range (0, 31))doc",
        R"doc(\mathrm{{output\_tensor}}_i = \verb|bitwise_and|(\mathrm{{input\_tensor\_a, input\_tensor\_b}}))doc",
        static_cast<Tensor (*)(
            const Tensor&,
            int32_t,
            const std::optional<MemoryConfig>&,
            const std::optional<Tensor>&,
            ttsl::Span<const unary::EltwiseUnaryWithParam>,
            ttsl::Span<const unary::EltwiseUnaryWithParam>,
            ttsl::Span<const unary::EltwiseUnaryWithParam>,
            std::optional<bool>)>(&ttnn::bitwise_right_shift),
        static_cast<Tensor (*)(
            const Tensor&,
            const Tensor&,
            const std::optional<MemoryConfig>&,
            const std::optional<Tensor>&,
            ttsl::Span<const unary::EltwiseUnaryWithParam>,
            ttsl::Span<const unary::EltwiseUnaryWithParam>,
            ttsl::Span<const unary::EltwiseUnaryWithParam>,
            std::optional<bool>)>(&ttnn::bitwise_right_shift),
        ". ",
        R"doc(INT32, UINT32, UINT16)doc");

    detail::bind_bitwise_binary_ops_operation<"logical_left_shift">(
        mod,
        R"doc(Perform logical_left_shift operation on :attr:`input_tensor_a` by :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`. :attr:`input_tensor_b` has shift_bits which are integers within range (0, 31). Equivalent to multiplying by 2^shift_amt.)doc",
        R"doc(\mathrm{{output\_tensor}}_i = \verb|logical_left_shift|(\mathrm{{input\_tensor\_a, input\_tensor\_b}}))doc",
        static_cast<Tensor (*)(
            const Tensor&,
            int32_t,
            const std::optional<MemoryConfig>&,
            const std::optional<Tensor>&,
            ttsl::Span<const unary::EltwiseUnaryWithParam>,
            ttsl::Span<const unary::EltwiseUnaryWithParam>,
            ttsl::Span<const unary::EltwiseUnaryWithParam>,
            std::optional<bool>)>(&ttnn::logical_left_shift),
        static_cast<Tensor (*)(
            const Tensor&,
            const Tensor&,
            const std::optional<MemoryConfig>&,
            const std::optional<Tensor>&,
            ttsl::Span<const unary::EltwiseUnaryWithParam>,
            ttsl::Span<const unary::EltwiseUnaryWithParam>,
            ttsl::Span<const unary::EltwiseUnaryWithParam>,
            std::optional<bool>)>(&ttnn::logical_left_shift),
        ". ",
        R"doc(INT32, UINT32)doc");

    detail::bind_binary_operation<"logical_right_shift">(
        mod,
        R"doc(Perform logical_right_shift operation on :attr:`input_tensor_a` by :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`. :attr:`input_tensor_b` has shift_bits which are integers within range (0, 31). Logical right shift fills vacated bits with zeros. Equivalent to integer division by 2^shift_amt.)doc",
        R"doc(\mathrm{{output\_tensor}}_i = \verb|logical_right_shift|(\mathrm{{input\_tensor\_a, input\_tensor\_b}}))doc",
        &detail::binary_op_binding_tensor_scalar<BinaryOpType::LOGICAL_RIGHT_SHIFT>,
        &detail::binary_op_binding_tensor_tensor<BinaryOpType::LOGICAL_RIGHT_SHIFT>,
        ". ",
        R"doc(INT32, UINT32)doc");

    detail::bind_binary_composite<"hypot">(
        mod,
        R"doc(Computes hypot :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{output\_tensor}_i = \sqrt{(\mathrm{input\_tensor\_a}_i^2 + \mathrm{input\_tensor\_b}_i^2)})doc",
        &detail::hypot_composite_wrapper,
        R"doc(FLOAT32, BFLOAT16, BFLOAT8_B)doc");

    detail::bind_binary_composite<"nextafter">(
        mod,
        R"doc(Computes nextafter :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{output\_tensor}_i = \begin{cases} \mathrm{next\_float}(\mathrm{input\_tensor\_a}_i, \mathrm{input\_tensor\_b}_i), & \text{if } \mathrm{input\_tensor\_a}_i \neq \mathrm{input\_tensor\_b}_i \\ \mathrm{input\_tensor\_a}_i, & \text{if } \mathrm{input\_tensor\_a}_i = \mathrm{input\_tensor\_b}_i \end{cases}
        )doc",
        &ttnn::nextafter,
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");

    detail::bind_binary_unary_max_operation<"minimum">(
        mod,
        R"doc(Computes minimum for :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        static_cast<Tensor (*)(
            const Tensor&,
            unary::ScalarVariant,
            const std::optional<const DataType>&,
            const std::optional<MemoryConfig>&,
            const std::optional<Tensor>&,
            ttsl::Span<const unary::EltwiseUnaryWithParam>,
            ttsl::Span<const unary::EltwiseUnaryWithParam>,
            ttsl::Span<const unary::EltwiseUnaryWithParam>,
            std::optional<bool>)>(&ttnn::minimum),
        static_cast<Tensor (*)(
            const Tensor&,
            const Tensor&,
            const std::optional<const DataType>&,
            const std::optional<MemoryConfig>&,
            const std::optional<Tensor>&,
            ttsl::Span<const unary::EltwiseUnaryWithParam>,
            ttsl::Span<const unary::EltwiseUnaryWithParam>,
            ttsl::Span<const unary::EltwiseUnaryWithParam>,
            std::optional<bool>)>(&ttnn::minimum));

    detail::bind_binary_composite<"atan2">(
        mod,
        R"doc(Computes atan2 :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{output\_tensor}_i = \arctan\left(\frac{\mathrm{input\_tensor\_a}_i}{\mathrm{input\_tensor\_b}_i}\right)
        )doc",
        &ttnn::atan2,
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc",
        R"doc(Input arguments for the atan2 function are in the format (y, x))doc");

    detail::bind_binary_operation<"logical_xor">(
        mod,
        R"doc(Compute logical_xor :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{output\_tensor}_i = (\mathrm{input\_tensor\_a}_i \land \lnot \mathrm{input\_tensor\_b}_i) \lor (\lnot \mathrm{input\_tensor\_a}_i \land \mathrm{input\_tensor\_b}_i))doc",
        &detail::binary_op_binding_tensor_scalar<BinaryOpType::LOGICAL_XOR>,
        &detail::binary_op_binding_tensor_tensor<BinaryOpType::LOGICAL_XOR>,
        ".",
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32, INT32, UINT32, UINT16)doc");

    detail::bind_inplace_operation<"logical_or_">(
        mod,
        R"doc(Computes inplace logical OR of :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{input\_tensor\_a}}_i | \mathrm{{input\_tensor\_b}}_i)doc",
        &detail::inplace_binding_tensor_scalar<static_cast<detail::InplaceScalarFn>(&ttnn::logical_or_)>,
        &detail::inplace_binding_tensor_tensor<static_cast<detail::InplaceTensorFn>(&ttnn::logical_or_)>,
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32, INT32, UINT32, UINT16)doc");

    detail::bind_inplace_operation<"logical_xor_">(
        mod,
        R"doc(Computes inplace logical XOR of :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{input\_tensor\_a}_i \land \lnot \mathrm{input\_tensor\_b}_i) \lor (\lnot \mathrm{input\_tensor\_a}_i \land \mathrm{input\_tensor\_b}_i)doc",
        &detail::inplace_binding_tensor_scalar<static_cast<detail::InplaceScalarFn>(&ttnn::logical_xor_)>,
        &detail::inplace_binding_tensor_tensor<static_cast<detail::InplaceTensorFn>(&ttnn::logical_xor_)>,
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32, INT32, UINT32, UINT16)doc");

    detail::bind_inplace_operation<"logical_and_">(
        mod,
        R"doc(Computes inplace logical AND of :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{input\_tensor\_a}}_i \& \mathrm{{input\_tensor\_b}}_i)doc",
        &detail::inplace_binding_tensor_scalar<static_cast<detail::InplaceScalarFn>(&ttnn::logical_and_)>,
        &detail::inplace_binding_tensor_tensor<static_cast<detail::InplaceTensorFn>(&ttnn::logical_and_)>,
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32, INT32, UINT32, UINT16)doc");

    detail::bind_binary_gcd_lcm_operation<"gcd">(
        mod,
        R"doc(Computes Greatest common divisor of :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`.
        [supported range [-2147483648, 2147483647]].)doc",
        R"doc(\mathrm{output\_tensor}_i = \verb|gcd|\left(\mathrm{input\_tensor\_a}_i , \mathrm{input\_tensor\_b}_i\right)
        )doc",
        &ttnn::gcd,
        R"doc(INT32)doc");

    detail::bind_binary_gcd_lcm_operation<"lcm">(
        mod,
        R"doc(Computes Least common multiple of :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`.
        [supported range [-32768, 32767]].)doc",
        R"doc(\mathrm{output\_tensor}_i = \verb|lcm|\left(\mathrm{input\_tensor\_a}_i , \mathrm{input\_tensor\_b}_i\right)
        )doc",
        &ttnn::lcm,
        R"doc(INT32)doc");

    detail::bind_binary_with_float_param<"addalpha">(
        mod,
        R"doc(Computes addalpha for :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{output\_tensor}} = \mathrm{{input\_tensor\_a\ + input\_tensor\_b\ * \alpha}})doc",
        &ttnn::addalpha,
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");

    detail::bind_binary_with_float_param<"subalpha">(
        mod,
        R"doc(Computes subalpha for :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{output\_tensor}} = \mathrm{{input\_tensor\_a\ - input\_tensor\_b\ * \alpha}})doc",
        &ttnn::subalpha,
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");

    detail::bind_binary_composite_with_rtol_atol<"isclose">(
        mod,
        R"doc(Computes isclose for :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{output\_tensor} = \begin{cases} 1, & \text{if } |\mathrm{input\_tensor\_a} - \mathrm{input\_tensor\_b}| \leq (\mathrm{atol} + \mathrm{rtol} \times |\mathrm{input\_tensor\_b}|) \\ 0, & \text{otherwise} \end{cases}
        )doc",
        &ttnn::isclose);

    detail::bind_div<"div">(
        mod,
        R"doc(Divides :attr:`input_tensor_a` by :attr:`input_tensor_b` and returns a tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{output}_i = \begin{cases} \mathrm{\left(\frac{\mathrm{input\_tensor\_a}_i}{\mathrm{input\_tensor\_b}_i}\right)}, & \text{if } \mathrm{round\_mode} = \mathrm{None} \\ \mathrm{\text{floor}\left(\frac{\mathrm{input\_tensor\_a}_i}{\mathrm{input\_tensor\_b}_i}\right)}, & \text{if } \mathrm{round\_mode} = \mathrm{floor} \\ \mathrm{\text{trunc}\left(\frac{\mathrm{input\_tensor\_a}_i}{\mathrm{input\_tensor\_b}_i}\right)}, & \text{if } \mathrm{round\_mode} = \mathrm{trunc} \end{cases}
        )doc",
        static_cast<Tensor (*)(
            const Tensor&,
            const Tensor&,
            bool,
            const std::optional<std::string>&,
            const std::optional<const DataType>&,
            const std::optional<MemoryConfig>&,
            const std::optional<Tensor>&,
            ttsl::Span<const unary::EltwiseUnaryWithParam>,
            ttsl::Span<const unary::EltwiseUnaryWithParam>,
            ttsl::Span<const unary::EltwiseUnaryWithParam>,
            const std::optional<bool>&,
            const std::optional<CoreRangeSet>&)>(&ttnn::div),
        static_cast<Tensor (*)(
            const Tensor&,
            float,
            bool,
            const std::optional<std::string>&,
            const std::optional<const DataType>&,
            const std::optional<MemoryConfig>&,
            std::optional<Tensor>,
            ttsl::Span<const unary::EltwiseUnaryWithParam>,
            ttsl::Span<const unary::EltwiseUnaryWithParam>,
            ttsl::Span<const unary::EltwiseUnaryWithParam>,
            const std::optional<bool>&,
            const std::optional<CoreRangeSet>&)>(&ttnn::div),
        R"doc(BFLOAT16, FLOAT32, INT32, UINT16)doc",
        R"doc(
        With INT32 inputs, rounding_mode `None` produces a FLOAT32 output, while `floor` and `trunc` produce an INT32 output.
        When :attr:`fast_and_approximate_mode` is `True`, operation assumes that :attr:`input_tensor_b` is not zero for fast approximation.
        When :attr:`fast_and_approximate_mode` is `False` (default), operation properly handles division by zero (accurate mode).
        )doc");

    detail::bind_binary_composite_overload<"div_no_nan">(
        mod,
        R"doc(Computes div_no_nan for :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        static_cast<Tensor (*)(const Tensor&, const Tensor&, const std::optional<MemoryConfig>&)>(&ttnn::div_no_nan),
        static_cast<Tensor (*)(const Tensor&, float, const std::optional<MemoryConfig>&)>(&ttnn::div_no_nan));

    detail::bind_binary_composite_overload<"floor_div">(
        mod,
        R"doc(Computes floor division for :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        static_cast<Tensor (*)(const Tensor&, const Tensor&, const std::optional<MemoryConfig>&)>(&ttnn::floor_div),
        static_cast<Tensor (*)(const Tensor&, float, const std::optional<MemoryConfig>&)>(&ttnn::floor_div));

    detail::bind_binary_unary_max_operation<"maximum">(
        mod,
        R"doc(Computes maximum for :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        static_cast<Tensor (*)(
            const Tensor&,
            unary::ScalarVariant,
            const std::optional<const DataType>&,
            const std::optional<MemoryConfig>&,
            const std::optional<Tensor>&,
            ttsl::Span<const unary::EltwiseUnaryWithParam>,
            ttsl::Span<const unary::EltwiseUnaryWithParam>,
            ttsl::Span<const unary::EltwiseUnaryWithParam>,
            std::optional<bool>)>(&ttnn::maximum),
        static_cast<Tensor (*)(
            const Tensor&,
            const Tensor&,
            const std::optional<const DataType>&,
            const std::optional<MemoryConfig>&,
            const std::optional<Tensor>&,
            ttsl::Span<const unary::EltwiseUnaryWithParam>,
            ttsl::Span<const unary::EltwiseUnaryWithParam>,
            ttsl::Span<const unary::EltwiseUnaryWithParam>,
            std::optional<bool>)>(&ttnn::maximum),
        R"doc(Supported range for :attr:`input_tensor_b` when its of scalar type is [-16777216, 16777216])doc");

    detail::bind_prelu<"prelu">(
        mod,
        R"doc(Perform an eltwise-prelu operation.)doc",
        static_cast<Tensor (*)(const Tensor&, const Tensor&, const std::optional<MemoryConfig>&)>(&ttnn::prelu),
        static_cast<Tensor (*)(const Tensor&, float, const std::optional<MemoryConfig>&)>(&ttnn::prelu),
        static_cast<Tensor (*)(const Tensor&, const std::array<float, 1>&, const std::optional<MemoryConfig>&)>(
            &ttnn::prelu),
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc",
        R"doc(PReLU supports the case where weight is a scalar or 1D list/array of size=1 or a 1D tensor :attr:`input_tensor_b` of size = the second dimension in :attr:`input_tensor_a`)doc");

    detail::bind_binary_composite<"outer">(
        mod,
        R"doc(Computes outer for :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{output\_tensor} = \mathrm{input\_tensor\_a} \text{ } \otimes \text{ } \mathrm{input\_tensor\_b})doc",
        &ttnn::outer,
        R"doc(BFLOAT16, FLOAT32)doc");

    detail::bind_polyval<"polyval">(
        mod,
        R"doc(Computes polyval of all elements of :attr:`input_tensor_a` with coefficients :attr:`coeffs` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{output\_tensor} = \sum_{i=0}^{n} (\mathrm{coeffs}_i) (\mathrm{input\_tensor}^i)
        )doc",
        &ttnn::polyval,
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");

    detail::bind_binary_overload_operation<"fmod">(
        mod,
        R"doc(Performs an eltwise-fmod operation.)doc",
        R"doc(\mathrm{{output\_tensor}} = \verb|fmod|(\mathrm{{input\_tensor\_a,input\_tensor\_b}}))doc",
        static_cast<Tensor (*)(
            const Tensor&, float, const std::optional<MemoryConfig>&, const std::optional<CoreRangeSet>&)>(&ttnn::fmod),
        static_cast<Tensor (*)(
            const Tensor&, const Tensor&, const std::optional<MemoryConfig>&, const std::optional<CoreRangeSet>&)>(
            &ttnn::fmod),
        R"doc(BFLOAT16, FLOAT32, INT32)doc");

    detail::bind_binary_overload_operation<"remainder">(
        mod,
        R"doc(Performs an eltwise-modulus operation.)doc",
        R"doc(\mathrm{{output\_tensor}} = \verb|remainder|(\mathrm{{input\_tensor\_a,input\_tensor\_b}}))doc",
        static_cast<Tensor (*)(
            const Tensor&, float, const std::optional<MemoryConfig>&, const std::optional<CoreRangeSet>&)>(
            &ttnn::remainder),
        static_cast<Tensor (*)(
            const Tensor&, const Tensor&, const std::optional<MemoryConfig>&, const std::optional<CoreRangeSet>&)>(
            &ttnn::remainder),
        R"doc(BFLOAT16, FLOAT32, INT32)doc");

    detail::bind_inplace_operation<"gt_">(
        mod,
        R"doc(Performs Greater than in-place operation on :attr:`input_a` and :attr:`input_b` and returns the tensor with the same layout as :attr:`input_tensor`)doc",
        R"doc(\mathrm{{input\_tensor\_a}} > \mathrm{{input\_tensor\_b}})doc",
        &detail::inplace_binding_tensor_scalar<static_cast<detail::InplaceScalarFn>(&ttnn::gt_)>,
        &detail::inplace_binding_tensor_tensor<static_cast<detail::InplaceTensorFn>(&ttnn::gt_)>,
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");

    detail::bind_inplace_operation<"ge_">(
        mod,
        R"doc(Performs Greater than or equal to in-place operation on :attr:`input_a` and :attr:`input_b` and returns the tensor with the same layout as :attr:`input_tensor`)doc",
        R"doc(\mathrm{{input\_tensor\_a}} >= \mathrm{{input\_tensor\_b}})doc",
        &detail::inplace_binding_tensor_scalar<static_cast<detail::InplaceScalarFn>(&ttnn::ge_)>,
        &detail::inplace_binding_tensor_tensor<static_cast<detail::InplaceTensorFn>(&ttnn::ge_)>,
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");

    detail::bind_inplace_operation<"lt_">(
        mod,
        R"doc(Performs Less than in-place operation on :attr:`input_a` and :attr:`input_b` and returns the tensor with the same layout as :attr:`input_tensor`)doc",
        R"doc(\mathrm{{input\_tensor\_a}} < \mathrm{{input\_tensor\_b}})doc",
        &detail::inplace_binding_tensor_scalar<static_cast<detail::InplaceScalarFn>(&ttnn::lt_)>,
        &detail::inplace_binding_tensor_tensor<static_cast<detail::InplaceTensorFn>(&ttnn::lt_)>,
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");

    detail::bind_inplace_operation<"le_">(
        mod,
        R"doc(Performs Less than or equal to in-place operation on :attr:`input_a` and :attr:`input_b` and returns the tensor with the same layout as :attr:`input_tensor`)doc",
        R"doc(\mathrm{{input\_tensor\_a}} <= \mathrm{{input\_tensor\_b}})doc",
        &detail::inplace_binding_tensor_scalar<static_cast<detail::InplaceScalarFn>(&ttnn::le_)>,
        &detail::inplace_binding_tensor_tensor<static_cast<detail::InplaceTensorFn>(&ttnn::le_)>,
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");

    detail::bind_inplace_operation<"eq_">(
        mod,
        R"doc(Performs Equal to in-place operation on :attr:`input_a` and :attr:`input_b` and returns the tensor with the same layout as :attr:`input_tensor`)doc",
        R"doc(\mathrm{{input\_tensor\_a}} == \mathrm{{input\_tensor\_b}})doc",
        &detail::inplace_binding_tensor_scalar<static_cast<detail::InplaceScalarFn>(&ttnn::eq_)>,
        &detail::inplace_binding_tensor_tensor<static_cast<detail::InplaceTensorFn>(&ttnn::eq_)>,
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");

    detail::bind_inplace_operation<"ne_">(
        mod,
        R"doc(Performs Not equal to in-place operation on :attr:`input_a` and :attr:`input_b` and returns the tensor with the same layout as :attr:`input_tensor`)doc",
        R"doc(\mathrm{{input\_tensor\_a}}\: != \mathrm{{input\_tensor\_b}})doc",
        &detail::inplace_binding_tensor_scalar<static_cast<detail::InplaceScalarFn>(&ttnn::ne_)>,
        &detail::inplace_binding_tensor_tensor<static_cast<detail::InplaceTensorFn>(&ttnn::ne_)>,
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");

    detail::bind_inplace_operation<"ldexp_">(
        mod,
        R"doc(Performs ldexp in-place operation on :attr:`input_a` and :attr:`input_b` and returns the tensor with the same layout as :attr:`input_tensor`)doc",
        R"doc(\verb|ldexp|(\mathrm{{input\_tensor\_a,input\_tensor\_b}}))doc",
        &detail::inplace_binding_tensor_scalar<static_cast<detail::InplaceScalarFn>(&ttnn::ldexp_)>,
        &detail::inplace_binding_tensor_tensor<static_cast<detail::InplaceTensorFn>(&ttnn::ldexp_)>,
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");

    detail::bind_inplace_operation<"logaddexp_">(
        mod,
        R"doc(Performs logaddexp in-place operation on :attr:`input_a` and :attr:`input_b` and returns the tensor with the same layout as :attr:`input_tensor`)doc",
        R"doc(\verb|logaddexp|(\mathrm{{input\_tensor\_a,input\_tensor\_b}}))doc",
        &detail::inplace_binding_tensor_scalar<static_cast<detail::InplaceScalarFn>(&ttnn::logaddexp_)>,
        &detail::inplace_binding_tensor_tensor<static_cast<detail::InplaceTensorFn>(&ttnn::logaddexp_)>,
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");

    detail::bind_inplace_operation<"logaddexp2_">(
        mod,
        R"doc(Performs logaddexp2 in-place operation on :attr:`input_a` and :attr:`input_b` and returns the tensor with the same layout as :attr:`input_tensor`)doc",
        R"doc(\verb|logaddexp2|(\mathrm{{input\_tensor\_a,input\_tensor\_b}}))doc",
        &detail::inplace_binding_tensor_scalar<static_cast<detail::InplaceScalarFn>(&ttnn::logaddexp2_)>,
        &detail::inplace_binding_tensor_tensor<static_cast<detail::InplaceTensorFn>(&ttnn::logaddexp2_)>,
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");

    detail::bind_inplace_operation<"squared_difference_">(
        mod,
        R"doc(Performs squared_difference in-place operation on :attr:`input_a` and :attr:`input_b` and returns the tensor with the same layout as :attr:`input_tensor`)doc",
        R"doc(\verb|squared_difference|(\mathrm{{input\_tensor\_a,input\_tensor\_b}}))doc",
        &detail::inplace_binding_tensor_scalar<static_cast<detail::InplaceScalarFn>(&ttnn::squared_difference_)>,
        &detail::inplace_binding_tensor_tensor<static_cast<detail::InplaceTensorFn>(&ttnn::squared_difference_)>,
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32, INT32, UINT32, UINT16)doc");

    detail::bind_inplace_operation_with_fast_approx<"multiply_">(
        mod,
        R"doc(Performs in-place multiplication operation on :attr:`input_a` and :attr:`input_b` and returns the tensor with the same layout as :attr:`input_tensor`)doc",
        R"doc(\verb|multiply|(\mathrm{{input\_tensor\_a,input\_tensor\_b}}))doc",
        &detail::multiply_inplace_fast_approx_tensor_scalar,
        &detail::multiply_inplace_fast_approx_tensor_tensor,
        R"doc(BFLOAT16, FLOAT32, UINT16)doc",
        R"doc(
        When :attr:`fast_and_approximate_mode` is `True` for bfloat16 datatype, the operation uses FPU implementation for better performance.
        When :attr:`fast_and_approximate_mode` is `False` for bfloat16 datatype, the operation uses SFPU with the result rounded to nearest even (RNE).
        The operation is not supported for INT32 inputs since the outputs are returned as FLOAT32.
        )doc");
    detail::bind_inplace_operation_with_fast_approx<"divide_">(
        mod,
        R"doc(Performs in-place division operation on :attr:`input_a` and :attr:`input_b` and returns the tensor with the same layout as :attr:`input_tensor`)doc",
        R"doc(\verb|divide|(\mathrm{{input\_tensor\_a,input\_tensor\_b}}))doc",
        &detail::divide_inplace_fast_approx_tensor_scalar,
        &detail::divide_inplace_fast_approx_tensor_tensor,
        R"doc(BFLOAT16, FLOAT32, UINT16)doc",
        R"doc(
        When :attr:`fast_and_approximate_mode` is `True`, the operation uses FPU+SFPU implementation for better performance.
        When :attr:`fast_and_approximate_mode` is `False` (default), the operation uses SFPU implementation for better accuracy.
        The operation is not supported for INT32 inputs since the outputs are returned as FLOAT32.
        )doc");

    detail::bind_inplace_operation<"rsub_">(
        mod,
        R"doc(Subtracts :attr:`input_a` from :attr:`input_b` in-place and returns the tensor with the same layout as :attr:`input_tensor`)doc",
        R"doc(\mathrm{{input\_tensor\_b}} - \mathrm{{input\_tensor\_a}})doc",
        &detail::inplace_binding_tensor_scalar<static_cast<detail::InplaceScalarFn>(&ttnn::rsub_)>,
        &detail::inplace_binding_tensor_tensor<static_cast<detail::InplaceTensorFn>(&ttnn::rsub_)>,
        R"doc(FLOAT32, BFLOAT16, BFLOAT8_B, INT32, UINT32, UINT16)doc");

    detail::bind_inplace_operation<"bias_gelu_">(
        mod,
        R"doc(Performs bias_gelu in-place operation on :attr:`input_a` and :attr:`input_b` and returns the tensor with the same layout as :attr:`input_tensor`)doc",
        R"doc(\verb|bias_gelu|(\mathrm{{input\_tensor\_a,input\_tensor\_b}}))doc",
        &detail::inplace_binding_tensor_scalar<static_cast<detail::InplaceScalarFn>(&ttnn::bias_gelu_)>,
        &detail::inplace_binding_tensor_tensor<static_cast<detail::InplaceTensorFn>(&ttnn::bias_gelu_)>,
        R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");

    detail::bind_power(
        mod,
        R"doc(When :attr:`exponent` is a Tensor, supported dtypes are: BFLOAT16, FLOAT32. Both input tensors should be of same dtype.)doc");
}

}  // namespace ttnn::operations::binary
