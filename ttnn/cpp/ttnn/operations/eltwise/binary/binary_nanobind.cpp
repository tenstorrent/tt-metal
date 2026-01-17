// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "binary_nanobind.hpp"

#include <array>
#include <string>
#include <optional>

#include <fmt/format.h>
#include <nanobind/nanobind.h>
#include <nanobind/operators.h>  // testing
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/variant.h>

#include <ttnn-nanobind/small_vector_caster.hpp>

#include "ttnn/decorators.hpp"  // testing
#include "ttnn-nanobind/decorators.hpp"
#include "ttnn-nanobind/export_enum.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/binary/binary_composite.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::binary {

namespace detail {

// Common broadcasting and performance documentation for binary operations
constexpr auto BINARY_BROADCAST_DOC = R"doc(
        Binary elementwise operations, C=op(A,B), support input tensors A and B in row major and tile layout, in interleaved or sharded format (height, width or block sharded), in DRAM or L1. A and B are completely independent, and can have different tensor specs.

        Broadcast of A and B operands is supported up to dimension 5 (DNCHW). Any dimensions of size 1 in either A or B will be expanded to match the other input, and data will be duplicated along that dimension. For example, if the shape of A is [2,1,1,32] and B is [1,16,8,1], the output shape will be [2,16,8,32]. The size of dimensions higher than 5 must match between A and B.

        The output C also supports row major and tile layout, interleaved or sharded format (height, width or block sharded), in DRAM or L1. The tensor spec of C is independent of A and B, and can be explicitly set using the optional output tensor input; if not provided, the operation will attempt a best decision at an appropriate tensor spec. The dimensions of C, or equivalently the optional output tensor, must match the broadcast-matched size of A and B.

        Performance considerations:
        Elementwise operations operate natively in tile format, tiled tensors are preferred as an input, and row-major tensors are tilized and untilized during the operation.
        L1 sharded layout is preferred, with no broadcast and matching tensor specs for A, B and C.
)doc";

template <typename binary_operation_t>
void bind_primitive_binary_operation(
    nb::module_& mod, const binary_operation_t& operation, const std::string& description) {
    auto doc = fmt::format(
        R"doc(
        {2}

        {3}

        Args:
            * :attr:`input_tensor_a`
            * :attr:`input_tensor_b` (ttnn.Tensor or Number): the tensor or number to add to :attr:`input_tensor_a`.

        Keyword args:
            * :attr:`memory_config` (Optional[ttnn.MemoryConfig]): memory config for the output tensor
            * :attr:`dtype` (Optional[ttnn.DataType]): data type for the output tensor
            * :attr:`output_tensor` (Optional[ttnn.Tensor]): preallocated output tensor
            * :attr:`activations` (Optional[List[str]]): list of activation functions to apply to the output tensor

        Example:

            >>> tensor1 = ttnn.to_device(ttnn.from_torch(torch.tensor(([[1, 2], [3, 4]]), dtype=torch.bfloat16)), device)
            >>> tensor2 = ttnn.to_device(ttnn.from_torch(torch.tensor(([[1, 2], [3, 4]]), dtype=torch.bfloat16)), device)
            >>> output = {1}(tensor1, tensor2)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        description,
        BINARY_BROADCAST_DOC);

    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const binary_operation_t& self,
               const ttnn::Tensor& input_tensor_a,
               const ttnn::Tensor& input_tensor_b,
               BinaryOpType binary_op_type,
               const std::optional<const DataType>& dtype,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor,
               const std::optional<unary::EltwiseFusedActivations>& activations,
               const std::optional<unary::EltwiseUnaryWithParam>& input_tensor_a_activation) -> ttnn::Tensor {
                return self(
                    input_tensor_a,
                    input_tensor_b,
                    binary_op_type,
                    dtype,
                    memory_config,
                    output_tensor,
                    activations,
                    input_tensor_a_activation);
            },
            nb::arg("input_tensor_a"),
            nb::arg("input_tensor_b"),
            nb::arg("binary_op_type"),
            nb::kw_only(),
            nb::arg("dtype") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none(),
            nb::arg("activations") = nb::none(),
            nb::arg("input_tensor_a_activation") = nb::none()});
}

template <typename binary_operation_t>
void bind_binary_operation(
    nb::module_& mod,
    const binary_operation_t& operation,
    const std::string& description,
    const std::string& math,
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
        math,
        info,
        supported_dtype,
        note,
        BINARY_BROADCAST_DOC);

    bind_registered_operation(
        mod,
        operation,
        doc,
        // tensor and scalar
        ttnn::nanobind_overload_t{
            [](const binary_operation_t& self,
               const ttnn::Tensor& input_tensor_a,
               const float scalar,
               const std::optional<const DataType>& dtype,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor,
               const ttnn::SmallVector<unary::EltwiseUnaryWithParam>& activations,
               const ttnn::SmallVector<unary::EltwiseUnaryWithParam>& input_tensor_a_activations,
               const ttnn::SmallVector<unary::EltwiseUnaryWithParam>& input_tensor_b_activations,
               const std::optional<bool>& use_legacy,
               const std::optional<CoreRangeSet>& sub_core_grids) -> ttnn::Tensor {
                return self(
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
            },
            nb::arg("input_tensor_a"),
            nb::arg("input_tensor_b"),
            nb::kw_only(),
            nb::arg("dtype") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none(),
            nb::arg("activations") = nb::cast(ttnn::SmallVector<unary::EltwiseUnaryWithParam>()),
            nb::arg("input_tensor_a_activations") = nb::cast(ttnn::SmallVector<unary::EltwiseUnaryWithParam>()),
            nb::arg("input_tensor_b_activations") = nb::cast(ttnn::SmallVector<unary::EltwiseUnaryWithParam>()),
            nb::arg("use_legacy") = nb::none(),
            nb::arg("sub_core_grids") = nb::none()},

        // tensor and tensor
        ttnn::nanobind_overload_t{
            [](const binary_operation_t& self,
               const ttnn::Tensor& input_tensor_a,
               const ttnn::Tensor& input_tensor_b,
               const std::optional<const DataType>& dtype,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor,
               const ttnn::SmallVector<unary::EltwiseUnaryWithParam>& activations,
               const ttnn::SmallVector<unary::EltwiseUnaryWithParam>& input_tensor_a_activations,
               const ttnn::SmallVector<unary::EltwiseUnaryWithParam>& input_tensor_b_activations,
               const std::optional<bool>& use_legacy,
               const std::optional<CoreRangeSet>& sub_core_grids) -> ttnn::Tensor {
                return self(
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
            },
            nb::arg("input_tensor_a"),
            nb::arg("input_tensor_b"),
            nb::kw_only(),
            nb::arg("dtype") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none(),
            nb::arg("activations") = nb::cast(ttnn::SmallVector<unary::EltwiseUnaryWithParam>()),
            nb::arg("input_tensor_a_activations") = nb::cast(ttnn::SmallVector<unary::EltwiseUnaryWithParam>()),
            nb::arg("input_tensor_b_activations") = nb::cast(ttnn::SmallVector<unary::EltwiseUnaryWithParam>()),
            nb::arg("use_legacy") = nb::none(),
            nb::arg("sub_core_grids") = nb::none()});
}

template <typename binary_operation_t>
void bind_binary_gcd_lcm_operation(
    nb::module_& mod,
    const binary_operation_t& operation,
    const std::string& description,
    const std::string& math,
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& supported_rank = "2, 3, 4",
    const std::string& example_tensor1 =
        "ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)",
    const std::string& example_tensor2 =
        "ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)",
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

        {9}

        Note:
            Supported dtypes, layouts, and ranks:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
                 - Ranks
               * - {4}
                 - TILE
                 - {5}

            {8}
        )doc",

        operation.base_name(),
        operation.python_fully_qualified_name(),
        description,
        math,
        supported_dtype,
        supported_rank,
        example_tensor1,
        example_tensor2,
        note,
        BINARY_BROADCAST_DOC);

    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const binary_operation_t& self,
               const ttnn::Tensor& input_tensor_a,
               const ttnn::Tensor& input_tensor_b,
               const std::optional<const DataType>& dtype,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor,
               const ttnn::SmallVector<unary::EltwiseUnaryWithParam>& activations,
               const ttnn::SmallVector<unary::EltwiseUnaryWithParam>& input_tensor_a_activations,
               const ttnn::SmallVector<unary::EltwiseUnaryWithParam>& input_tensor_b_activations,
               const std::optional<bool>& use_legacy) -> ttnn::Tensor {
                return self(
                    input_tensor_a,
                    input_tensor_b,
                    dtype,
                    memory_config,
                    output_tensor,
                    activations,
                    input_tensor_a_activations,
                    input_tensor_b_activations,
                    use_legacy);
            },
            nb::arg("input_tensor_a"),
            nb::arg("input_tensor_b"),
            nb::kw_only(),
            nb::arg("dtype") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none(),
            nb::arg("activations") = nb::cast(ttnn::SmallVector<unary::EltwiseUnaryWithParam>()),
            nb::arg("input_tensor_a_activations") = nb::cast(ttnn::SmallVector<unary::EltwiseUnaryWithParam>()),
            nb::arg("input_tensor_b_activations") = nb::cast(ttnn::SmallVector<unary::EltwiseUnaryWithParam>()),
            nb::arg("use_legacy") = nb::none()});
}

template <typename binary_operation_t>
void bind_binary_unary_max_operation(
    nb::module_& mod,
    const binary_operation_t& operation,
    const std::string& description,
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
        note,
        BINARY_BROADCAST_DOC);

    bind_registered_operation(
        mod,
        operation,
        doc,
        // tensor and scalar
        ttnn::nanobind_overload_t{
            [](const binary_operation_t& self,
               const ttnn::Tensor& input_tensor_a,
               unary::ScalarVariant scalar,
               const std::optional<const DataType>& dtype,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor,
               const ttnn::SmallVector<unary::EltwiseUnaryWithParam>& activations,
               const ttnn::SmallVector<unary::EltwiseUnaryWithParam>& input_tensor_a_activations,
               const ttnn::SmallVector<unary::EltwiseUnaryWithParam>& input_tensor_b_activations,
               const std::optional<bool>& use_legacy) -> ttnn::Tensor {
                return self(
                    input_tensor_a,
                    scalar,
                    dtype,
                    memory_config,
                    output_tensor,
                    activations,
                    input_tensor_a_activations,
                    input_tensor_b_activations,
                    use_legacy);
            },
            nb::arg("input_tensor_a"),
            nb::arg("input_b"),
            nb::kw_only(),
            nb::arg("dtype") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none(),
            nb::arg("activations") = nb::cast(ttnn::SmallVector<unary::EltwiseUnaryWithParam>()),
            nb::arg("input_tensor_a_activations") = nb::cast(ttnn::SmallVector<unary::EltwiseUnaryWithParam>()),
            nb::arg("input_tensor_b_activations") = nb::cast(ttnn::SmallVector<unary::EltwiseUnaryWithParam>()),
            nb::arg("use_legacy") = nb::none(),
        },

        // tensor and tensor
        ttnn::nanobind_overload_t{
            [](const binary_operation_t& self,
               const ttnn::Tensor& input_tensor_a,
               const ttnn::Tensor& input_tensor_b,
               const std::optional<const DataType>& dtype,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor,
               const ttnn::SmallVector<unary::EltwiseUnaryWithParam>& activations,
               const ttnn::SmallVector<unary::EltwiseUnaryWithParam>& input_tensor_a_activations,
               const ttnn::SmallVector<unary::EltwiseUnaryWithParam>& input_tensor_b_activations,
               const std::optional<bool>& use_legacy) -> ttnn::Tensor {
                return self(
                    input_tensor_a,
                    input_tensor_b,
                    dtype,
                    memory_config,
                    output_tensor,
                    activations,
                    input_tensor_a_activations,
                    input_tensor_b_activations,
                    use_legacy);
            },
            nb::arg("input_tensor_a"),
            nb::arg("input_tensor_b"),
            nb::kw_only(),
            nb::arg("dtype") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none(),
            nb::arg("activations") = nb::cast(ttnn::SmallVector<unary::EltwiseUnaryWithParam>()),
            nb::arg("input_tensor_a_activations") = nb::cast(ttnn::SmallVector<unary::EltwiseUnaryWithParam>()),
            nb::arg("input_tensor_b_activations") = nb::cast(ttnn::SmallVector<unary::EltwiseUnaryWithParam>()),
            nb::arg("use_legacy") = nb::none()});
}

template <typename binary_operation_t>
void bind_binary_unary_operation(
    nb::module_& mod,
    const binary_operation_t& operation,
    const std::string& description,
    const std::string& math,
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
        math,
        info,
        supported_dtype,
        note,
        BINARY_BROADCAST_DOC);

    bind_registered_operation(
        mod,
        operation,
        doc,
        // tensor and scalar
        ttnn::nanobind_overload_t{
            [](const binary_operation_t& self,
               const ttnn::Tensor& input_tensor_a,
               const float scalar,
               const std::optional<const DataType>& dtype,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor,
               const ttnn::SmallVector<unary::EltwiseUnaryWithParam>& activations,
               const ttnn::SmallVector<unary::EltwiseUnaryWithParam>& input_tensor_a_activations,
               const ttnn::SmallVector<unary::EltwiseUnaryWithParam>& input_tensor_b_activations,
               const std::optional<bool>& use_legacy) -> ttnn::Tensor {
                return self(
                    input_tensor_a,
                    scalar,
                    dtype,
                    memory_config,
                    output_tensor,
                    activations,
                    input_tensor_a_activations,
                    input_tensor_b_activations,
                    use_legacy);
            },
            nb::arg("input_tensor_a"),
            nb::arg("input_b"),
            nb::kw_only(),
            nb::arg("dtype") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none(),
            nb::arg("activations") = nb::cast(ttnn::SmallVector<unary::EltwiseUnaryWithParam>()),
            nb::arg("input_tensor_a_activations") = nb::cast(ttnn::SmallVector<unary::EltwiseUnaryWithParam>()),
            nb::arg("input_tensor_b_activations") = nb::cast(ttnn::SmallVector<unary::EltwiseUnaryWithParam>()),
            nb::arg("use_legacy") = nb::none()},

        // tensor and tensor
        ttnn::nanobind_overload_t{
            [](const binary_operation_t& self,
               const ttnn::Tensor& input_tensor_a,
               const ttnn::Tensor& input_tensor_b,
               const std::optional<const DataType>& dtype,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor,
               const ttnn::SmallVector<unary::EltwiseUnaryWithParam>& activations,
               const ttnn::SmallVector<unary::EltwiseUnaryWithParam>& input_tensor_a_activations,
               const ttnn::SmallVector<unary::EltwiseUnaryWithParam>& input_tensor_b_activations,
               const std::optional<bool>& use_legacy) -> ttnn::Tensor {
                return self(
                    input_tensor_a,
                    input_tensor_b,
                    dtype,
                    memory_config,
                    output_tensor,
                    activations,
                    input_tensor_a_activations,
                    input_tensor_b_activations,
                    use_legacy);
            },
            nb::arg("input_tensor_a"),
            nb::arg("input_tensor_b"),
            nb::kw_only(),
            nb::arg("dtype") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none(),
            nb::arg("activations") = ttnn::SmallVector<unary::EltwiseUnaryWithParam>(),
            nb::arg("input_tensor_a_activations") = ttnn::SmallVector<unary::EltwiseUnaryWithParam>(),
            nb::arg("input_tensor_b_activations") = ttnn::SmallVector<unary::EltwiseUnaryWithParam>(),
            nb::arg("use_legacy") = nb::none()});
}

template <typename binary_operation_t>
void bind_binary_with_float_param(
    nb::module_& mod,
    const binary_operation_t& operation,
    const std::string& description,
    const std::string& math,
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
        description,
        math,
        supported_dtype,
        note,
        BINARY_BROADCAST_DOC);

    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const binary_operation_t& self,
               const Tensor& input_tensor_a,
               const Tensor& input_tensor_b,
               float alpha,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor) -> ttnn::Tensor {
                return self(input_tensor_a, input_tensor_b, alpha, memory_config, output_tensor);
            },
            nb::arg("input_tensor_a"),
            nb::arg("input_tensor_b"),
            nb::arg("alpha") = 1.0f,
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none()});
}

template <typename binary_operation_t>
void bind_bitwise_binary_ops_operation(
    nb::module_& mod,
    const binary_operation_t& operation,
    const std::string& description,
    const std::string& math,
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
        math,
        info,
        supported_dtype,
        note,
        BINARY_BROADCAST_DOC);

    bind_registered_operation(
        mod,
        operation,
        doc,
        // tensor and scalar
        ttnn::nanobind_overload_t{
            [](const binary_operation_t& self,
               const ttnn::Tensor& input_tensor_a,
               const int32_t scalar,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor,
               const ttnn::SmallVector<unary::EltwiseUnaryWithParam>& activations,
               const ttnn::SmallVector<unary::EltwiseUnaryWithParam>& input_tensor_a_activations,
               const ttnn::SmallVector<unary::EltwiseUnaryWithParam>& input_tensor_b_activations,
               const std::optional<bool>& use_legacy) -> ttnn::Tensor {
                return self(
                    input_tensor_a,
                    scalar,
                    memory_config,
                    output_tensor,
                    activations,
                    input_tensor_a_activations,
                    input_tensor_b_activations,
                    use_legacy);
            },
            nb::arg("input_tensor_a"),
            nb::arg("input_b"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none(),
            nb::arg("activations") = nb::cast(ttnn::SmallVector<unary::EltwiseUnaryWithParam>()),
            nb::arg("input_tensor_a_activations") = nb::cast(ttnn::SmallVector<unary::EltwiseUnaryWithParam>()),
            nb::arg("input_tensor_b_activations") = nb::cast(ttnn::SmallVector<unary::EltwiseUnaryWithParam>()),
            nb::arg("use_legacy") = nb::none()},

        // tensor and tensor
        ttnn::nanobind_overload_t{
            [](const binary_operation_t& self,
               const ttnn::Tensor& input_tensor_a,
               const ttnn::Tensor& input_tensor_b,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor,
               const ttnn::SmallVector<unary::EltwiseUnaryWithParam>& activations,
               const ttnn::SmallVector<unary::EltwiseUnaryWithParam>& input_tensor_a_activations,
               const ttnn::SmallVector<unary::EltwiseUnaryWithParam>& input_tensor_b_activations,
               const std::optional<bool>& use_legacy) -> ttnn::Tensor {
                return self(
                    input_tensor_a,
                    input_tensor_b,
                    memory_config,
                    output_tensor,
                    activations,
                    input_tensor_a_activations,
                    input_tensor_b_activations,
                    use_legacy);
            },
            nb::arg("input_tensor_a"),
            nb::arg("input_tensor_b"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none(),
            nb::arg("activations") = nb::cast(ttnn::SmallVector<unary::EltwiseUnaryWithParam>()),
            nb::arg("input_tensor_a_activations") = nb::cast(ttnn::SmallVector<unary::EltwiseUnaryWithParam>()),
            nb::arg("input_tensor_b_activations") = nb::cast(ttnn::SmallVector<unary::EltwiseUnaryWithParam>()),
            nb::arg("use_legacy") = nb::none(),
        });
}

template <typename binary_operation_t>
void bind_logical_binary_ops_operation(
    nb::module_& mod,
    const binary_operation_t& operation,
    const std::string& description,
    const std::string& math,
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
        math,
        info,
        supported_dtype,
        note,
        BINARY_BROADCAST_DOC);

    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const binary_operation_t& self,
               const ttnn::Tensor& input_tensor_a,
               const ttnn::Tensor& input_tensor_b,
               const std::optional<const DataType>& dtype,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor,
               const std::optional<bool>& use_legacy) -> ttnn::Tensor {
                return self(
                    input_tensor_a,
                    input_tensor_b,
                    dtype,
                    memory_config,
                    output_tensor,
                    ttnn::SmallVector<unary::EltwiseUnaryWithParam>(),
                    ttnn::SmallVector<unary::EltwiseUnaryWithParam>(),
                    ttnn::SmallVector<unary::EltwiseUnaryWithParam>(),
                    use_legacy);
            },
            nb::arg("input_tensor_a"),
            nb::arg("input_tensor_b"),
            nb::kw_only(),
            nb::arg("dtype") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none(),
            nb::arg("use_legacy") = nb::none()});
}

template <typename binary_operation_t>
void bind_binary_composite(
    nb::module_& mod,
    const binary_operation_t& operation,
    const std::string& description,
    const std::string& math,
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& supported_rank = "2, 3, 4",
    const std::string& example_tensor1 =
        "ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)",
    const std::string& example_tensor2 =
        "ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)",
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

        {9}

        Note:
            Supported dtypes, layouts, and ranks:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
                 - Ranks
               * - {4}
                 - TILE
                 - {5}

            {8}
        )doc",

        operation.base_name(),
        operation.python_fully_qualified_name(),
        description,
        math,
        supported_dtype,
        supported_rank,
        example_tensor1,
        example_tensor2,
        note,
        BINARY_BROADCAST_DOC);

    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const binary_operation_t& self,
               const Tensor& input_tensor_a,
               const Tensor& input_tensor_b,
               const std::optional<MemoryConfig>& memory_config) {
                return self(input_tensor_a, input_tensor_b, memory_config);
            },
            nb::arg("input_tensor_a"),
            nb::arg("input_tensor_b"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()});
}

template <typename binary_operation_t>
void bind_binary_composite_with_rtol_atol(
    nb::module_& mod, const binary_operation_t& operation, const std::string& description, const std::string& math) {
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
        description,
        math,
        BINARY_BROADCAST_DOC);

    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const binary_operation_t& self,
               const Tensor& input_tensor_a,
               const Tensor& input_tensor_b,
               float rtol,
               float atol,
               const bool equal_nan,
               const std::optional<MemoryConfig>& memory_config) {
                return self(input_tensor_a, input_tensor_b, rtol, atol, equal_nan, memory_config);
            },
            nb::arg("input_tensor_a"),
            nb::arg("input_tensor_b"),
            nb::kw_only(),
            nb::arg("rtol") = 1e-05f,
            nb::arg("atol") = 1e-08f,
            nb::arg("equal_nan") = false,
            nb::arg("memory_config") = nb::none()});
}

// https://nanobind.readthedocs.io/en/latest/api_extra.html

template <typename binary_operation_t>
void bind_binary_composite_overload(
    nb::module_& mod,
    const binary_operation_t& operation,
    const std::string& description,
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& supported_rank = "2, 3, 4",
    const std::string& example_tensor1 =
        "ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)",
    const std::string& example_tensor2 =
        "ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)",
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

        {8}

        Note:
            Supported dtypes, layouts, and ranks:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
                 - Ranks
               * - {3}
                 - TILE
                 - {4}

            {7}
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        description,
        supported_dtype,
        supported_rank,
        example_tensor1,
        example_tensor2,
        note,
        BINARY_BROADCAST_DOC);

    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const binary_operation_t& self,
               const Tensor& input_tensor_a,
               const Tensor& input_tensor_b,
               const std::optional<MemoryConfig>& memory_config) {
                return self(input_tensor_a, input_tensor_b, memory_config);
            },
            nb::arg("input_tensor_a"),
            nb::arg("input_tensor_b"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()},

        ttnn::nanobind_overload_t{
            [](const binary_operation_t& self,
               const Tensor& input_tensor_a,
               float value,
               const std::optional<MemoryConfig>& memory_config) { return self(input_tensor_a, value, memory_config); },
            nb::arg("input_tensor_a"),
            nb::arg("value"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()});
}

template <typename binary_operation_t>
void bind_prelu(
    nb::module_& mod,
    const binary_operation_t& operation,
    const std::string& description,
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& supported_rank = "2, 3, 4",
    const std::string& example_tensor1 =
        "ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)",
    const std::string& example_tensor2 =
        "ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)",
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

        {8}

        Note:
            Supported dtypes, layouts, and ranks:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
                 - Ranks
               * - {3}
                 - TILE
                 - {4}

            {7}
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        description,
        supported_dtype,
        supported_rank,
        example_tensor1,
        example_tensor2,
        note,
        BINARY_BROADCAST_DOC);

    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const binary_operation_t& self,
               const Tensor& input_tensor_a,
               const Tensor& input_tensor_b,
               const std::optional<MemoryConfig>& memory_config) {
                return self(input_tensor_a, input_tensor_b, memory_config);
            },
            nb::arg("input_tensor_a"),
            nb::arg("weight"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()},

        ttnn::nanobind_overload_t{
            [](const binary_operation_t& self,
               const Tensor& input_tensor_a,
               float value,
               const std::optional<MemoryConfig>& memory_config) { return self(input_tensor_a, value, memory_config); },
            nb::arg("input_tensor_a"),
            nb::arg("weight"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()},

        ttnn::nanobind_overload_t{
            [](const binary_operation_t& self,
               const Tensor& input_tensor_a,
               const std::array<float, 1>& weight,
               const std::optional<MemoryConfig>& memory_config) {
                return self(input_tensor_a, weight, memory_config);
            },
            nb::arg("input_tensor_a"),
            nb::arg("weight"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()});
}

template <typename binary_operation_t>
void bind_div(
    nb::module_& mod,
    const binary_operation_t& operation,
    const std::string& description,
    const std::string& math,
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
        description,
        math,
        supported_dtype,
        note,
        BINARY_BROADCAST_DOC);

    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const binary_operation_t& self,
               const Tensor& input_tensor_a,
               const Tensor& input_tensor_b,
               bool fast_and_approximate_mode,
               const std::optional<std::string>& rounding_mode,
               const std::optional<const DataType>& dtype,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor,
               const ttnn::SmallVector<unary::EltwiseUnaryWithParam>& activations,
               const ttnn::SmallVector<unary::EltwiseUnaryWithParam>& input_tensor_a_activations,
               const ttnn::SmallVector<unary::EltwiseUnaryWithParam>& input_tensor_b_activations,
               const std::optional<bool>& use_legacy,
               const std::optional<CoreRangeSet>& sub_core_grids) -> ttnn::Tensor {
                return self(
                    input_tensor_a,
                    input_tensor_b,
                    fast_and_approximate_mode,
                    rounding_mode,
                    dtype,
                    memory_config,
                    output_tensor,
                    activations,
                    input_tensor_a_activations,
                    input_tensor_b_activations,
                    use_legacy,
                    sub_core_grids);
            },
            nb::arg("input_tensor_a"),
            nb::arg("input_tensor_b"),
            nb::kw_only(),
            nb::arg("fast_and_approximate_mode") = false,
            nb::arg("rounding_mode") = nb::none(),
            nb::arg("dtype") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none(),
            nb::arg("activations") = nb::cast(ttnn::SmallVector<unary::EltwiseUnaryWithParam>()),
            nb::arg("input_tensor_a_activations") = nb::cast(ttnn::SmallVector<unary::EltwiseUnaryWithParam>()),
            nb::arg("input_tensor_b_activations") = nb::cast(ttnn::SmallVector<unary::EltwiseUnaryWithParam>()),
            nb::arg("use_legacy") = nb::none(),
            nb::arg("sub_core_grids") = nb::none(),
        },

        ttnn::nanobind_overload_t{
            [](const binary_operation_t& self,
               const Tensor& input_tensor_a,
               float value,
               bool fast_and_approximate_mode,
               const std::optional<std::string>& rounding_mode,
               const std::optional<const DataType>& dtype,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor,
               const ttnn::SmallVector<unary::EltwiseUnaryWithParam>& activations,
               const ttnn::SmallVector<unary::EltwiseUnaryWithParam>& input_tensor_a_activations,
               const ttnn::SmallVector<unary::EltwiseUnaryWithParam>& input_tensor_b_activations,
               const std::optional<bool>& use_legacy,
               const std::optional<CoreRangeSet>& sub_core_grids) -> ttnn::Tensor {
                return self(
                    input_tensor_a,
                    value,
                    fast_and_approximate_mode,
                    rounding_mode,
                    dtype,
                    memory_config,
                    output_tensor,
                    activations,
                    input_tensor_a_activations,
                    input_tensor_b_activations,
                    use_legacy,
                    sub_core_grids);
            },
            nb::arg("input_tensor_a"),
            nb::arg("value"),
            nb::kw_only(),
            nb::arg("fast_and_approximate_mode") = false,
            nb::arg("rounding_mode") = nb::none(),
            nb::arg("dtype") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none(),
            nb::arg("activations") = nb::cast(ttnn::SmallVector<unary::EltwiseUnaryWithParam>()),
            nb::arg("input_tensor_a_activations") = nb::cast(ttnn::SmallVector<unary::EltwiseUnaryWithParam>()),
            nb::arg("input_tensor_b_activations") = nb::cast(ttnn::SmallVector<unary::EltwiseUnaryWithParam>()),
            nb::arg("use_legacy") = nb::none(),
            nb::arg("sub_core_grids") = nb::none(),
        });
}

template <typename binary_operation_t>
void bind_binary_operation_with_fast_approx(
    nb::module_& mod,
    const binary_operation_t& operation,
    const std::string& description,
    const std::string& math,
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
        description,
        math,
        supported_dtype,
        note,
        BINARY_BROADCAST_DOC);

    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const binary_operation_t& self,
               const Tensor& input_tensor_a,
               const Tensor& input_tensor_b,
               bool fast_and_approximate_mode,
               const std::optional<const DataType>& dtype,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor,
               const ttnn::SmallVector<unary::EltwiseUnaryWithParam>& activations,
               const ttnn::SmallVector<unary::EltwiseUnaryWithParam>& input_tensor_a_activations,
               const ttnn::SmallVector<unary::EltwiseUnaryWithParam>& input_tensor_b_activations,
               const std::optional<bool>& use_legacy,
               const std::optional<CoreRangeSet>& sub_core_grids) -> ttnn::Tensor {
                return self(
                    input_tensor_a,
                    input_tensor_b,
                    dtype,
                    memory_config,
                    output_tensor,
                    activations,
                    input_tensor_a_activations,
                    input_tensor_b_activations,
                    use_legacy,
                    fast_and_approximate_mode,
                    sub_core_grids);
            },
            nb::arg("input_tensor_a"),
            nb::arg("input_tensor_b"),
            nb::kw_only(),
            nb::arg("fast_and_approximate_mode") = false,
            nb::arg("dtype") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none(),
            nb::arg("activations") = nb::cast(ttnn::SmallVector<unary::EltwiseUnaryWithParam>()),
            nb::arg("input_tensor_a_activations") = nb::cast(ttnn::SmallVector<unary::EltwiseUnaryWithParam>()),
            nb::arg("input_tensor_b_activations") = nb::cast(ttnn::SmallVector<unary::EltwiseUnaryWithParam>()),
            nb::arg("use_legacy") = nb::none(),
            nb::arg("sub_core_grids") = nb::none(),
        },

        ttnn::nanobind_overload_t{
            [](const binary_operation_t& self,
               const Tensor& input_tensor_a,
               float value,
               bool fast_and_approximate_mode,
               const std::optional<const DataType>& dtype,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor,
               const ttnn::SmallVector<unary::EltwiseUnaryWithParam>& activations,
               const ttnn::SmallVector<unary::EltwiseUnaryWithParam>& input_tensor_a_activations,
               const ttnn::SmallVector<unary::EltwiseUnaryWithParam>& input_tensor_b_activations,
               const std::optional<bool>& use_legacy,
               const std::optional<CoreRangeSet>& sub_core_grids) -> ttnn::Tensor {
                return self(
                    input_tensor_a,
                    value,
                    dtype,
                    memory_config,
                    output_tensor,
                    activations,
                    input_tensor_a_activations,
                    input_tensor_b_activations,
                    use_legacy,
                    fast_and_approximate_mode,
                    sub_core_grids);
            },
            nb::arg("input_tensor_a"),
            nb::arg("value"),
            nb::kw_only(),
            nb::arg("fast_and_approximate_mode") = false,
            nb::arg("dtype") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none(),
            nb::arg("activations") = nb::cast(ttnn::SmallVector<unary::EltwiseUnaryWithParam>()),
            nb::arg("input_tensor_a_activations") = nb::cast(ttnn::SmallVector<unary::EltwiseUnaryWithParam>()),
            nb::arg("input_tensor_b_activations") = nb::cast(ttnn::SmallVector<unary::EltwiseUnaryWithParam>()),
            nb::arg("use_legacy") = nb::none(),
            nb::arg("sub_core_grids") = nb::none()});
}

template <typename binary_operation_t>
void bind_polyval(
    nb::module_& mod,
    const binary_operation_t& operation,
    const std::string& description,
    const std::string& math,
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
        description,
        math,
        supported_dtype,
        note,
        BINARY_BROADCAST_DOC);

    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const binary_operation_t& self,
               const Tensor& input_tensor_a,
               const std::vector<float>& coeffs,
               const std::optional<MemoryConfig>& memory_config) {
                return self(input_tensor_a, coeffs, memory_config);
            },
            nb::arg("input_tensor_a"),
            nb::arg("coeffs"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()});
}

template <typename binary_operation_t>
void bind_binary_overload_operation(
    nb::module_& mod,
    const binary_operation_t& operation,
    const std::string& description,
    const std::string& math,
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
        description,
        math,
        supported_dtype,
        note,
        BINARY_BROADCAST_DOC);

    bind_registered_operation(
        mod,
        operation,
        doc,

        // tensor and scalar
        ttnn::nanobind_overload_t{
            [](const binary_operation_t& self,
               const Tensor& input_tensor,
               float scalar,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<CoreRangeSet>& sub_core_grids) {
                return self(input_tensor, scalar, memory_config, sub_core_grids);
            },
            nb::arg("input_tensor"),
            nb::arg("scalar"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("sub_core_grids") = nb::none()},

        // tensor and tensor
        ttnn::nanobind_overload_t{
            [](const binary_operation_t& self,
               const Tensor& input_tensor_a,
               const Tensor& input_tensor_b,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<CoreRangeSet>& sub_core_grids) {
                return self(input_tensor_a, input_tensor_b, memory_config, sub_core_grids);
            },
            nb::arg("input_a"),
            nb::arg("input_b"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("sub_core_grids") = nb::none()});
}

template <typename binary_operation_t>
void bind_inplace_operation(
    nb::module_& mod,
    const binary_operation_t& operation,
    const std::string& description,
    const std::string& math,
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
        description,
        math,
        supported_dtype,
        note,
        BINARY_BROADCAST_DOC);

    bind_registered_operation(
        mod,
        operation,
        doc,

        // tensor and scalar
        ttnn::nanobind_overload_t{
            [](const binary_operation_t& self,
               const Tensor& input_tensor,
               const float scalar,
               const ttnn::SmallVector<unary::EltwiseUnaryWithParam>& activations,
               const ttnn::SmallVector<unary::EltwiseUnaryWithParam>& input_tensor_a_activations,
               const ttnn::SmallVector<unary::EltwiseUnaryWithParam>& input_tensor_b_activations,
               const std::optional<bool>& use_legacy,
               const std::optional<CoreRangeSet>& sub_core_grids) {
                return self(
                    input_tensor,
                    scalar,
                    activations,
                    input_tensor_a_activations,
                    input_tensor_b_activations,
                    use_legacy,
                    sub_core_grids);
            },
            nb::arg("input_a"),
            nb::arg("input_b"),
            nb::arg("activations") = nb::cast(ttnn::SmallVector<unary::EltwiseUnaryWithParam>()),
            nb::arg("input_tensor_a_activations") = nb::cast(ttnn::SmallVector<unary::EltwiseUnaryWithParam>()),
            nb::arg("input_tensor_b_activations") = nb::cast(ttnn::SmallVector<unary::EltwiseUnaryWithParam>()),
            nb::arg("use_legacy") = nb::none(),
            nb::arg("sub_core_grids") = nb::none(),
        },

        // tensor and tensor
        ttnn::nanobind_overload_t{
            [](const binary_operation_t& self,
               const Tensor& input_tensor_a,
               const Tensor& input_tensor_b,
               const ttnn::SmallVector<unary::EltwiseUnaryWithParam>& activations,
               const ttnn::SmallVector<unary::EltwiseUnaryWithParam>& input_tensor_a_activations,
               const ttnn::SmallVector<unary::EltwiseUnaryWithParam>& input_tensor_b_activations,
               const std::optional<bool>& use_legacy,
               const std::optional<CoreRangeSet>& sub_core_grids) {
                return self(
                    input_tensor_a,
                    input_tensor_b,
                    activations,
                    input_tensor_a_activations,
                    input_tensor_b_activations,
                    use_legacy,
                    sub_core_grids);
            },
            nb::arg("input_a"),
            nb::arg("input_b"),
            nb::kw_only(),
            nb::arg("activations") = nb::cast(ttnn::SmallVector<unary::EltwiseUnaryWithParam>()),
            nb::arg("input_tensor_a_activations") = nb::cast(ttnn::SmallVector<unary::EltwiseUnaryWithParam>()),
            nb::arg("input_tensor_b_activations") = nb::cast(ttnn::SmallVector<unary::EltwiseUnaryWithParam>()),
            nb::arg("use_legacy") = nb::none(),
            nb::arg("sub_core_grids") = nb::none(),
        });
}

template <typename binary_operation_t>
void bind_inplace_operation_with_fast_approx(
    nb::module_& mod,
    const binary_operation_t& operation,
    const std::string& description,
    const std::string& math,
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
        description,
        math,
        supported_dtype,
        note,
        BINARY_BROADCAST_DOC);

    bind_registered_operation(
        mod,
        operation,
        doc,
        // tensor and tensor
        ttnn::nanobind_overload_t{
            [](const binary_operation_t& self,
               const Tensor& input_tensor_a,
               const Tensor& input_tensor_b,
               bool fast_and_approximate_mode,
               const ttnn::SmallVector<unary::EltwiseUnaryWithParam>& activations,
               const ttnn::SmallVector<unary::EltwiseUnaryWithParam>& input_tensor_a_activations,
               const ttnn::SmallVector<unary::EltwiseUnaryWithParam>& input_tensor_b_activations,
               const std::optional<bool>& use_legacy,
               const std::optional<CoreRangeSet>& sub_core_grids) {
                return self(
                    input_tensor_a,
                    input_tensor_b,
                    activations,
                    input_tensor_a_activations,
                    input_tensor_b_activations,
                    use_legacy,
                    fast_and_approximate_mode,
                    sub_core_grids);
            },
            nb::arg("input_a"),
            nb::arg("input_b"),
            nb::kw_only(),
            nb::arg("fast_and_approximate_mode") = false,
            nb::arg("activations") = nb::cast(ttnn::SmallVector<unary::EltwiseUnaryWithParam>()),
            nb::arg("input_tensor_a_activations") = nb::cast(ttnn::SmallVector<unary::EltwiseUnaryWithParam>()),
            nb::arg("input_tensor_b_activations") = nb::cast(ttnn::SmallVector<unary::EltwiseUnaryWithParam>()),
            nb::arg("use_legacy") = nb::none(),
            nb::arg("sub_core_grids") = nb::none(),
        },

        // tensor and scalar
        ttnn::nanobind_overload_t{
            [](const binary_operation_t& self,
               const Tensor& input_tensor_a,
               float value,
               bool fast_and_approximate_mode,
               const ttnn::SmallVector<unary::EltwiseUnaryWithParam>& activations,
               const ttnn::SmallVector<unary::EltwiseUnaryWithParam>& input_tensor_a_activations,
               const ttnn::SmallVector<unary::EltwiseUnaryWithParam>& input_tensor_b_activations,
               const std::optional<bool>& use_legacy,
               const std::optional<CoreRangeSet>& sub_core_grids) {
                return self(
                    input_tensor_a,
                    value,
                    activations,
                    input_tensor_a_activations,
                    input_tensor_b_activations,
                    use_legacy,
                    fast_and_approximate_mode,
                    sub_core_grids);
            },
            nb::arg("input_a"),
            nb::arg("value"),
            nb::kw_only(),
            nb::arg("fast_and_approximate_mode") = false,
            nb::arg("activations") = nb::cast(ttnn::SmallVector<unary::EltwiseUnaryWithParam>()),
            nb::arg("input_tensor_a_activations") = nb::cast(ttnn::SmallVector<unary::EltwiseUnaryWithParam>()),
            nb::arg("input_tensor_b_activations") = nb::cast(ttnn::SmallVector<unary::EltwiseUnaryWithParam>()),
            nb::arg("use_legacy") = nb::none(),
            nb::arg("sub_core_grids") = nb::none(),
        });
}

template <typename binary_operation_t>
void bind_logical_inplace_operation(
    nb::module_& mod,
    const binary_operation_t& operation,
    const std::string& description,
    const std::string& math,
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

        Returns:
            ttnn.Tensor: the output tensor.

        {6}

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
        description,
        math,
        supported_dtype,
        note,
        BINARY_BROADCAST_DOC);

    bind_registered_operation(
        mod,
        operation,
        doc,

        ttnn::nanobind_overload_t{
            [](const binary_operation_t& self,
               const Tensor& input_tensor_a,
               const Tensor& input_tensor_b,
               const ttnn::SmallVector<unary::EltwiseUnaryWithParam>& activations,
               const ttnn::SmallVector<unary::EltwiseUnaryWithParam>& input_tensor_a_activations,
               const ttnn::SmallVector<unary::EltwiseUnaryWithParam>& input_tensor_b_activations,
               const std::optional<bool>& use_legacy,
               const std::optional<CoreRangeSet>& sub_core_grids) {
                return self(
                    input_tensor_a,
                    input_tensor_b,
                    activations,
                    input_tensor_a_activations,
                    input_tensor_b_activations,
                    use_legacy,
                    sub_core_grids);
            },
            nb::arg("input_a"),
            nb::arg("input_b"),
            nb::arg("activations") = nb::cast(ttnn::SmallVector<unary::EltwiseUnaryWithParam>()),
            nb::arg("input_tensor_a_activations") = nb::cast(ttnn::SmallVector<unary::EltwiseUnaryWithParam>()),
            nb::arg("input_tensor_b_activations") = nb::cast(ttnn::SmallVector<unary::EltwiseUnaryWithParam>()),
            nb::arg("use_legacy") = nb::none(),
            nb::arg("sub_core_grids") = nb::none(),
        });
}

template <typename binary_operation_t>
void bind_binary_inplace_operation(
    nb::module_& mod, const binary_operation_t& operation, const std::string& description, const std::string& math) {
    auto doc = fmt::format(
        R"doc(
            {2}

            {4}

            .. math::
                {3}

            Args:
                * :attr:`input_a` (ttnn.Tensor)
                * :attr:`input_b` (ttnn.Tensor or Number)
            Keyword args:
                * :attr:`activations` (Optional[List[str]]): list of activation functions to apply to the output tensor
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        description,
        math,
        BINARY_BROADCAST_DOC);

    bind_registered_operation(
        mod,
        operation,
        doc,

        // tensor and scalar
        ttnn::nanobind_overload_t{
            [](const binary_operation_t& self,
               const Tensor& input_tensor,
               const float scalar,
               const ttnn::SmallVector<unary::EltwiseUnaryWithParam>& activations,
               const ttnn::SmallVector<unary::EltwiseUnaryWithParam>& input_tensor_a_activations,
               const ttnn::SmallVector<unary::EltwiseUnaryWithParam>& input_tensor_b_activations,
               const std::optional<bool>& use_legacy,
               const std::optional<CoreRangeSet>& sub_core_grids) {
                return self(
                    input_tensor,
                    scalar,
                    activations,
                    input_tensor_a_activations,
                    input_tensor_b_activations,
                    use_legacy,
                    sub_core_grids);
            },
            nb::arg("input_a"),
            nb::arg("input_b"),
            nb::kw_only(),
            nb::arg("activations") = nb::cast(ttnn::SmallVector<unary::EltwiseUnaryWithParam>()),
            nb::arg("input_tensor_a_activations") = nb::cast(ttnn::SmallVector<unary::EltwiseUnaryWithParam>()),
            nb::arg("input_tensor_b_activations") = nb::cast(ttnn::SmallVector<unary::EltwiseUnaryWithParam>()),
            nb::arg("use_legacy") = nb::none(),
            nb::arg("sub_core_grids") = nb::none(),
        },

        // tensor and tensor
        ttnn::nanobind_overload_t{
            [](const binary_operation_t& self,
               const Tensor& input_tensor_a,
               const Tensor& input_tensor_b,
               const ttnn::SmallVector<unary::EltwiseUnaryWithParam>& activations,
               const ttnn::SmallVector<unary::EltwiseUnaryWithParam>& input_tensor_a_activations,
               const ttnn::SmallVector<unary::EltwiseUnaryWithParam>& input_tensor_b_activations,
               const std::optional<bool>& use_legacy,
               const std::optional<CoreRangeSet>& sub_core_grids) {
                return self(
                    input_tensor_a,
                    input_tensor_b,
                    activations,
                    input_tensor_a_activations,
                    input_tensor_b_activations,
                    use_legacy,
                    sub_core_grids);
            },
            nb::arg("input_a"),
            nb::arg("input_b"),
            nb::kw_only(),
            nb::arg("activations") = nb::cast(ttnn::SmallVector<unary::EltwiseUnaryWithParam>()),
            nb::arg("input_tensor_a_activations") = nb::cast(ttnn::SmallVector<unary::EltwiseUnaryWithParam>()),
            nb::arg("input_tensor_b_activations") = nb::cast(ttnn::SmallVector<unary::EltwiseUnaryWithParam>()),
            nb::arg("use_legacy") = nb::none(),
            nb::arg("sub_core_grids") = nb::none(),
        });
}

template <typename binary_operation_t>
void bind_power(nb::module_& mod, const binary_operation_t& /*operation*/, const std::string& note = "") {
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
            Supported dtypes, layouts, and ranks:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
                 - Ranks
               * - BFLOAT16, BFLOAT8_B
                 - TILE
                 - 2, 3, 4

            {2}
        )doc",
        ttnn::pow.base_name(),
        ttnn::pow.python_fully_qualified_name(),
        note,
        BINARY_BROADCAST_DOC);

    bind_registered_operation(
        mod,
        ttnn::pow,
        doc,
        // integer exponent
        ttnn::nanobind_overload_t{
            [](const binary_operation_t& self,
               const Tensor& input_tensor,
               uint32_t exponent,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<Tensor>& output_tensor) -> ttnn::Tensor {
                return self(input_tensor, exponent, memory_config, output_tensor);
            },
            nb::arg("input_tensor"),
            nb::arg("exponent"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none(),
        },

        // float exponent
        ttnn::nanobind_overload_t{
            [](const binary_operation_t& self,
               const Tensor& input_tensor,
               float exponent,
               const std::optional<MemoryConfig>& memory_config,
               std::optional<Tensor> output_tensor) -> ttnn::Tensor {
                return self(input_tensor, exponent, memory_config, output_tensor);
            },
            nb::arg("input_tensor"),
            nb::arg("exponent"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none()},

        // tensor exponent
        ttnn::nanobind_overload_t{
            [](const binary_operation_t& self,
               const Tensor& input_tensor,
               const Tensor& exponent,
               const std::optional<const DataType>& dtype,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor,
               const ttnn::SmallVector<unary::EltwiseUnaryWithParam>& activations,
               const ttnn::SmallVector<unary::EltwiseUnaryWithParam>& input_tensor_a_activations,
               const ttnn::SmallVector<unary::EltwiseUnaryWithParam>& input_tensor_b_activations,
               const std::optional<bool>& use_legacy) -> ttnn::Tensor {
                return self(
                    input_tensor,
                    exponent,
                    dtype,
                    memory_config,
                    output_tensor,
                    activations,
                    input_tensor_a_activations,
                    input_tensor_b_activations,
                    use_legacy);
            },
            nb::arg("input_tensor"),
            nb::arg("exponent"),
            nb::kw_only(),
            nb::arg("dtype") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none(),
            nb::arg("activations") = nb::cast(ttnn::SmallVector<unary::EltwiseUnaryWithParam>()),
            nb::arg("input_tensor_a_activations") = nb::cast(ttnn::SmallVector<unary::EltwiseUnaryWithParam>()),
            nb::arg("input_tensor_b_activations") = nb::cast(ttnn::SmallVector<unary::EltwiseUnaryWithParam>()),
            nb::arg("use_legacy") = nb::none(),
        },

        // scalar input - tensor exponent
        ttnn::nanobind_overload_t{
            [](const binary_operation_t& self,
               float input,
               const Tensor& exponent,
               const std::optional<const DataType>& dtype,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor,
               const ttnn::SmallVector<unary::EltwiseUnaryWithParam>& activations,
               const ttnn::SmallVector<unary::EltwiseUnaryWithParam>& input_tensor_a_activations,
               const ttnn::SmallVector<unary::EltwiseUnaryWithParam>& input_tensor_b_activations,
               const std::optional<bool>& use_legacy) -> ttnn::Tensor {
                return self(
                    input,
                    exponent,
                    dtype,
                    memory_config,
                    output_tensor,
                    activations,
                    input_tensor_a_activations,
                    input_tensor_b_activations,
                    use_legacy);
            },
            nb::arg("input"),
            nb::arg("exponent"),
            nb::kw_only(),
            nb::arg("dtype") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none(),
            nb::arg("activations") = nb::cast(ttnn::SmallVector<unary::EltwiseUnaryWithParam>()),
            nb::arg("input_tensor_a_activations") = nb::cast(ttnn::SmallVector<unary::EltwiseUnaryWithParam>()),
            nb::arg("input_tensor_b_activations") = nb::cast(ttnn::SmallVector<unary::EltwiseUnaryWithParam>()),
            nb::arg("use_legacy") = nb::none()});
}
}  // namespace detail

void py_module(nb::module_& mod) {
    export_enum<BinaryOpType>(mod, "BinaryOpType");

    detail::bind_binary_operation(
        mod,
        ttnn::add,
        R"doc(Adds :attr:`input_tensor_a` to :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{output\_tensor}}_i = \mathrm{{input\_tensor\_a}}_i + \mathrm{{input\_tensor\_b}}_i)doc",
        R"doc(: :code:`'None'` | :code:`'relu'`. )doc",
        R"doc(BFLOAT16, BFLOAT8_B, INT32, UINT32 (range: [0, 4294967295]), UINT16 (range: [0, 65535]))doc");

    detail::bind_binary_inplace_operation(
        mod,
        ttnn::add_,
        R"doc(Adds :attr:`input_tensor_a` to :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a` in-place)doc",
        R"doc(\mathrm{{input\_tensor\_a}}_i + \mathrm{{input\_tensor\_b}}_i)doc");

    detail::bind_binary_operation(
        mod,
        ttnn::subtract,
        R"doc(Subtracts :attr:`input_tensor_b` from :attr:`input_tensor_a` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{output\_tensor}}_i = \mathrm{{input\_tensor\_a}}_i - \mathrm{{input\_tensor\_b}}_i)doc",
        R"doc(: :code:`'None'` | :code:`'relu'`. )doc",
        R"doc(BFLOAT16, BFLOAT8_B, INT32, UINT16 (range: 0 - 65535), UINT32 (range: 0 - 4294967295))doc");

    detail::bind_binary_inplace_operation(
        mod,
        ttnn::subtract_,
        R"doc(Subtracts :attr:`input_tensor_b` from :attr:`input_tensor_a` and returns the tensor with the same layout as :attr:`input_tensor_a` in-place)doc",
        R"doc(\mathrm{{input\_tensor\_a}}_i - \mathrm{{input\_tensor\_b}}_i)doc");

    detail::bind_binary_operation(
        mod,
        ttnn::multiply,
        R"doc(Multiplies :attr:`input_tensor_a` by :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{output\_tensor}}_i = \mathrm{{input\_tensor\_a}}_i * \mathrm{{input\_tensor\_b}}_i)doc",
        R"doc(: :code:`'None'` | :code:`'relu'`. )doc",
        R"doc(BFLOAT16, BFLOAT8_B, UINT16 (range: 0 - 65535), INT32, UINT32)doc");

    detail::bind_binary_inplace_operation(
        mod,
        ttnn::multiply_,
        R"doc(Multiplies :attr:`input_tensor_a` by :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a` in-place)doc",
        R"doc(\mathrm{{input\_tensor\_a}}_i \times \mathrm{{input\_tensor\_b}}_i)doc");

    detail::bind_binary_operation(
        mod,
        ttnn::eq,
        R"doc(Compares if :attr:`input_tensor_a` is equal to :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{output\_tensor}}_i = (\mathrm{{input\_tensor\_a}}_i == \mathrm{{input\_tensor\_b}}_i))doc",
        ". ",
        R"doc(Float32, BFLOAT16, BFLOAT8_B, INT32, UINT32, UINT16)doc");

    detail::bind_binary_operation(
        mod,
        ttnn::ne,
        R"doc(Compares if :attr:`input_tensor_a` is not equal to :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{output\_tensor}}_i = (\mathrm{{input\_tensor\_a}}_i != \mathrm{{input\_tensor\_b}}_i))doc",
        ". ",
        R"doc(Float32, BFLOAT16, BFLOAT8_B, INT32, UINT32, UINT16)doc");

    detail::bind_binary_operation(
        mod,
        ttnn::lt,
        R"doc(Compares if :attr:`input_tensor_a` is less than :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{output\_tensor}}_i = (\mathrm{{input\_tensor\_a}}_i < \mathrm{{input\_tensor\_b}}_i))doc",
        ". ",
        R"doc(Float32, BFLOAT16, BFLOAT8_B, INT32)doc",
        "INT32 supported only for tensor-tensor.");

    detail::bind_binary_operation(
        mod,
        ttnn::le,
        R"doc(Compares if :attr:`input_tensor_a` is less than or equal to :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{output\_tensor}}_i = (\mathrm{{input\_tensor\_a}}_i <= \mathrm{{input\_tensor\_b}}_i))doc",
        ". ",
        R"doc(Float32, BFLOAT16, BFLOAT8_B, INT32)doc",
        "INT32 supported only for tensor-tensor.");

    detail::bind_binary_operation(
        mod,
        ttnn::gt,
        R"doc(Compares if :attr:`input_tensor_a` is greater than :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{output\_tensor}}_i = (\mathrm{{input\_tensor\_a}}_i > \mathrm{{input\_tensor\_b}}_i))doc",
        ". ",
        R"doc(Float32, BFLOAT16, BFLOAT8_B, INT32)doc",
        "INT32 supported only for tensor-tensor.");

    detail::bind_binary_operation(
        mod,
        ttnn::ge,
        R"doc(Compares if :attr:`input_tensor_a` is greater than or equal to :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{output\_tensor}}_i = (\mathrm{{input\_tensor\_a}}_i >= \mathrm{{input\_tensor\_b}}_i))doc",
        ". ",
        R"doc(Float32, BFLOAT16, BFLOAT8_B, INT32)doc",
        "INT32 supported only for tensor-tensor.");

    detail::bind_binary_operation(
        mod,
        ttnn::logical_and,
        R"doc(Computes logical AND of :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{output\_tensor}}_i = \mathrm{{input\_tensor\_a}}_i \, \& \, \mathrm{{input\_tensor\_b}}_i)doc",
        ". ",
        R"doc(BFLOAT16, BFLOAT8_B, INT32, UINT16)doc",
        "INT32 for tensor-scalar is supported only when use_legacy= False.");

    detail::bind_binary_operation(
        mod,
        ttnn::logical_or,
        R"doc(Computes logical OR of :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{output\_tensor}}_i = \mathrm{{input\_tensor\_a}}_i \, | \, \mathrm{{input\_tensor\_b}}_i)doc",
        ". ",
        R"doc(BFLOAT16, BFLOAT8_B, INT32, UINT32, UINT16)doc");

    detail::bind_binary_operation(
        mod,
        ttnn::ldexp,
        R"doc(Computes ldexp of :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{output\_tensor}} = \verb|ldexp|(\mathrm{{input\_tensor\_a,input\_tensor\_b}}))doc",
        ". ",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_binary_operation(
        mod,
        ttnn::logaddexp,
        R"doc(Computes logaddexp of :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{output\_tensor}} = \verb|logaddexp|(\mathrm{{input\_tensor\_a,input\_tensor\_b}}))doc",
        ". ",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_binary_operation(
        mod,
        ttnn::logaddexp2,
        R"doc(Computes logaddexp2 of :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{output\_tensor}} = \verb|logaddexp2|(\mathrm{{input\_tensor\_a,input\_tensor\_b}}))doc",
        ". ",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_binary_operation(
        mod,
        ttnn::squared_difference,
        R"doc(Computes squared difference of :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{output\_tensor}} = \verb|squared_difference|(\mathrm{{input\_tensor\_a,input\_tensor\_b}}))doc",
        ". ",
        R"doc(BFLOAT16, BFLOAT8_B, INT32, UINT32, UINT16)doc");

    detail::bind_binary_operation(
        mod,
        ttnn::bias_gelu,
        R"doc(Computes bias_gelu of :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{output\_tensor}} = \verb|bias_gelu|(\mathrm{{input\_tensor\_a,input\_tensor\_b}}))doc",
        ". ",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_binary_operation(
        mod,
        ttnn::xlogy,
        R"doc(Computes xlogy :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{output\_tensor}_i = \mathrm{input\_tensor\_a}_i \cdot \log(\mathrm{input\_tensor\_b}_i)
        )doc");

    detail::bind_binary_unary_operation(
        mod,
        ttnn::rsub,
        R"doc(Subtracts :attr:`input_tensor_a` from :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{output\_tensor}}_i = \mathrm{{input\_tensor\_b}}_i - \mathrm{{input\_tensor\_a}}_i)doc",
        ". ",
        R"doc(FLOAT32,BFLOAT16, BFLOAT8_B, INT32, UINT32, UINT16)doc");

    detail::bind_bitwise_binary_ops_operation(
        mod,
        ttnn::bitwise_and,
        R"doc(Perform bitwise_and operation on :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{output\_tensor}}_i = \verb|bitwise_and|(\mathrm{{input\_tensor\_a, input\_tensor\_b}}))doc",
        ". ",
        R"doc(INT32, UINT16 (range: 0 - 65535), UINT32)doc");

    detail::bind_bitwise_binary_ops_operation(
        mod,
        ttnn::bitwise_or,
        R"doc(Perform bitwise_or operation on :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{output\_tensor}}_i = \verb|bitwise_or|(\mathrm{{input\_tensor\_a, input\_tensor\_b}}))doc",
        ". ",
        R"doc(INT32, UINT16 (range: 0 - 65535), UINT32)doc");

    detail::bind_bitwise_binary_ops_operation(
        mod,
        ttnn::bitwise_xor,
        R"doc(Perform bitwise_xor operation on :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{output\_tensor}}_i = \verb|bitwise_xor|(\mathrm{{input\_tensor\_a, input\_tensor\_b}}))doc",
        ". ",
        R"doc(INT32, UINT16 (range: 0 - 65535), UINT32)doc");

    detail::bind_bitwise_binary_ops_operation(
        mod,
        ttnn::bitwise_left_shift,
        R"doc(Perform bitwise_left_shift operation on :attr:`input_tensor_a` by :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`. :attr:`input_tensor_b` has shift_bits which are integers within range (0, 31))doc",
        R"doc(\mathrm{{output\_tensor}}_i = \verb|bitwise_and|(\mathrm{{input\_tensor\_a, input\_tensor\_b}}))doc",
        ". ",
        R"doc(INT32, UINT32, UINT16)doc");

    detail::bind_bitwise_binary_ops_operation(
        mod,
        ttnn::bitwise_right_shift,
        R"doc(Perform bitwise_right_shift operation on :attr:`input_tensor_a` by :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`. :attr:`input_tensor_b` has shift_bits which are integers within range (0, 31))doc",
        R"doc(\mathrm{{output\_tensor}}_i = \verb|bitwise_and|(\mathrm{{input\_tensor\_a, input\_tensor\_b}}))doc",
        ". ",
        R"doc(INT32, UINT32, UINT16)doc");

    detail::bind_bitwise_binary_ops_operation(
        mod,
        ttnn::logical_left_shift,
        R"doc(Perform logical_left_shift operation on :attr:`input_tensor_a` by :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`. :attr:`input_tensor_b` has shift_bits which are integers within range (0, 31). Equivalent to multiplying by 2^shift_amt.)doc",
        R"doc(\mathrm{{output\_tensor}}_i = \verb|logical_left_shift|(\mathrm{{input\_tensor\_a, input\_tensor\_b}}))doc",
        ". ",
        R"doc(INT32, UINT32)doc");

    detail::bind_logical_binary_ops_operation(
        mod,
        ttnn::logical_right_shift,
        R"doc(Perform logical_right_shift operation on :attr:`input_tensor_a` by :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`. :attr:`input_tensor_b` has shift_bits which are integers within range (0, 31). Logical right shift fills vacated bits with zeros. Equivalent to integer division by 2^shift_amt.)doc",
        R"doc(\mathrm{{output\_tensor}}_i = \verb|logical_right_shift|(\mathrm{{input\_tensor\_a, input\_tensor\_b}}))doc",
        ". ",
        R"doc(INT32, UINT32)doc");

    detail::bind_binary_composite(
        mod,
        ttnn::hypot,
        R"doc(Computes hypot :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{output\_tensor}_i = \sqrt{(\mathrm{input\_tensor\_a}_i^2 + \mathrm{input\_tensor\_b}_i^2)})doc",
        R"doc(FLOAT32, BFLOAT16, BFLOAT8_B)doc");

    detail::bind_binary_composite(
        mod,
        ttnn::nextafter,
        R"doc(Computes nextafter :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{output\_tensor}_i = \begin{cases} \mathrm{next\_float}(\mathrm{input\_tensor\_a}_i, \mathrm{input\_tensor\_b}_i), & \text{if } \mathrm{input\_tensor\_a}_i \neq \mathrm{input\_tensor\_b}_i \\ \mathrm{input\_tensor\_a}_i, & \text{if } \mathrm{input\_tensor\_a}_i = \mathrm{input\_tensor\_b}_i \end{cases}
        )doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_binary_unary_max_operation(
        mod,
        ttnn::minimum,
        R"doc(Computes minimum for :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc");

    detail::bind_binary_composite(
        mod,
        ttnn::atan2,
        R"doc(Computes atan2 :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{output\_tensor}_i = \arctan\left(\frac{\mathrm{input\_tensor\_a}_i}{\mathrm{input\_tensor\_b}_i}\right)
        )doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc",
        R"doc(2, 3, 4)doc",
        R"doc(ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device))doc",
        R"doc(ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device))doc",
        R"doc(Input arguments for the atan2 function are in the format (y, x))doc");

    detail::bind_binary_operation(
        mod,
        ttnn::logical_xor,
        R"doc(Compute logical_xor :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{output\_tensor}_i = (\mathrm{input\_tensor\_a}_i \land \lnot \mathrm{input\_tensor\_b}_i) \lor (\lnot \mathrm{input\_tensor\_a}_i \land \mathrm{input\_tensor\_b}_i))doc",
        ".",
        R"doc(BFLOAT16, BFLOAT8_B, INT32, UINT32, UINT16)doc");

    detail::bind_logical_inplace_operation(
        mod,
        ttnn::logical_or_,
        R"doc(Computes inplace logical OR of :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{input\_tensor\_a}}_i | \mathrm{{input\_tensor\_b}}_i)doc",
        R"doc(BFLOAT16, BFLOAT8_B, INT32, UINT32, UINT16)doc");

    detail::bind_logical_inplace_operation(
        mod,
        ttnn::logical_xor_,
        R"doc(Computes inplace logical XOR of :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{input\_tensor\_a}_i \land \lnot \mathrm{input\_tensor\_b}_i) \lor (\lnot \mathrm{input\_tensor\_a}_i \land \mathrm{input\_tensor\_b}_i)doc",
        R"doc(BFLOAT16, BFLOAT8_B, INT32, UINT32, UINT16)doc");

    detail::bind_logical_inplace_operation(
        mod,
        ttnn::logical_and_,
        R"doc(Computes inplace logical AND of :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{input\_tensor\_a}}_i \& \mathrm{{input\_tensor\_b}}_i)doc",
        R"doc(BFLOAT16, BFLOAT8_B, INT32, UINT32, UINT16)doc");

    detail::bind_binary_gcd_lcm_operation(
        mod,
        ttnn::gcd,
        R"doc(Computes Greatest common divisor of :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`.
        [supported range [-2147483647, 2147483648]].)doc",
        R"doc(\mathrm{output\_tensor}_i = \verb|gcd|\left(\mathrm{input\_tensor\_a}_i , \mathrm{input\_tensor\_b}_i\right)
        )doc",
        R"doc(INT32)doc",
        R"doc(2, 3, 4)doc",
        R"doc(ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.int32), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device))doc",
        R"doc(ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.int32), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device))doc");

    detail::bind_binary_gcd_lcm_operation(
        mod,
        ttnn::lcm,
        R"doc(Computes Least common multiple of :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`.
        [supported range [-32767, 32768]].)doc",
        R"doc(\mathrm{output\_tensor}_i = \verb|lcm|\left(\mathrm{input\_tensor\_a}_i , \mathrm{input\_tensor\_b}_i\right)
        )doc",
        R"doc(INT32)doc",
        R"doc(2, 3, 4)doc",
        R"doc(ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.int32), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device))doc",
        R"doc(ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.int32), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device))doc");

    detail::bind_binary_with_float_param(
        mod,
        ttnn::addalpha,
        R"doc(Computes addalpha for :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{output\_tensor}} = \mathrm{{input\_tensor\_a\ + input\_tensor\_b\ * \alpha}})doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_binary_with_float_param(
        mod,
        ttnn::subalpha,
        R"doc(Computes subalpha for :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{output\_tensor}} = \mathrm{{input\_tensor\_a\ - input\_tensor\_b\ * \alpha}})doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_binary_composite_with_rtol_atol(
        mod,
        ttnn::isclose,
        R"doc(Computes isclose for :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{output\_tensor} = \begin{cases} 1, & \text{if } |\mathrm{input\_tensor\_a} - \mathrm{input\_tensor\_b}| \leq (\mathrm{atol} + \mathrm{rtol} \times |\mathrm{input\_tensor\_b}|) \\ 0, & \text{otherwise} \end{cases}
        )doc");

    detail::bind_div(
        mod,
        ttnn::div,
        R"doc(Divides :attr:`input_tensor_a` by :attr:`input_tensor_b` and returns a tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{output}_i = \begin{cases} \mathrm{\left(\frac{\mathrm{input\_tensor\_a}_i}{\mathrm{input\_tensor\_b}_i}\right)}, & \text{if } \mathrm{round\_mode} = \mathrm{None} \\ \mathrm{\text{floor}\left(\frac{\mathrm{input\_tensor\_a}_i}{\mathrm{input\_tensor\_b}_i}\right)}, & \text{if } \mathrm{round\_mode} = \mathrm{floor} \\ \mathrm{\text{trunc}\left(\frac{\mathrm{input\_tensor\_a}_i}{\mathrm{input\_tensor\_b}_i}\right)}, & \text{if } \mathrm{round\_mode} = \mathrm{trunc} \end{cases}
        )doc",
        R"doc(BFLOAT16, FLOAT32, INT32, UINT16)doc",
        R"doc(
        With INT32 inputs, rounding_mode `None` produces a FLOAT32 output, while `floor` and `trunc` produce an INT32 output.
        When :attr:`fast_and_approximate_mode` is `True`, operation assumes that :attr:`input_tensor_b` is not zero for fast approximation.
        When :attr:`fast_and_approximate_mode` is `False` (default), operation properly handles division by zero (accurate mode).
        )doc");

    detail::bind_binary_composite_overload(
        mod,
        ttnn::div_no_nan,
        R"doc(Computes div_no_nan for :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc");

    detail::bind_binary_composite_overload(
        mod,
        ttnn::floor_div,
        R"doc(Computes floor division for :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc");

    detail::bind_binary_unary_max_operation(
        mod,
        ttnn::maximum,
        R"doc(Computes maximum for :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(Supported range for :attr:`input_tensor_b` when its of scalar type is [-16777216, 16777216])doc");

    detail::bind_prelu(
        mod,
        ttnn::prelu,
        R"doc(Perform an eltwise-prelu operation.)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc",
        R"doc(2, 3, 4, 5)doc",
        R"doc(ttnn.from_torch(torch.rand([1, 2, 32, 32], dtype=torch.bfloat16), device=device))doc",
        R"doc(ttnn.from_torch(torch.tensor([1, 2], dtype=torch.bfloat16), device=device))doc",
        R"doc(PReLU supports the case where weight is a scalar or 1D list/array of size=1 or a 1D tensor :attr:`input_tensor_b` of size = the second dimension in :attr:`input_tensor_a`)doc");

    detail::bind_binary_composite(
        mod,
        ttnn::outer,
        R"doc(Computes outer for :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{output\_tensor} = \mathrm{input\_tensor\_a} \text{ } \otimes \text{ } \mathrm{input\_tensor\_b})doc",
        R"doc(BFLOAT16)doc",
        R"doc(4)doc",
        R"doc(ttnn.from_torch(torch.rand([1, 1, 32, 1], dtype=torch.bfloat16), device=device))doc",
        R"doc(ttnn.from_torch(torch.rand([1, 1, 1, 32], dtype=torch.bfloat16), device=device))doc");

    detail::bind_polyval(
        mod,
        ttnn::polyval,
        R"doc(Computes polyval of all elements of :attr:`input_tensor_a` with coefficients :attr:`coeffs` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{output\_tensor} = \sum_{i=0}^{n} (\mathrm{coeffs}_i) (\mathrm{input\_tensor}^i)
        )doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_binary_overload_operation(
        mod,
        ttnn::fmod,
        R"doc(Performs an eltwise-fmod operation.)doc",
        R"doc(\mathrm{{output\_tensor}} = \verb|fmod|(\mathrm{{input\_tensor\_a,input\_tensor\_b}}))doc",
        R"doc(BFLOAT16, FLOAT32)doc");

    detail::bind_binary_overload_operation(
        mod,
        ttnn::remainder,
        R"doc(Performs an eltwise-modulus operation.)doc",
        R"doc(\mathrm{{output\_tensor}} = \verb|remainder|(\mathrm{{input\_tensor\_a,input\_tensor\_b}}))doc",
        R"doc(BFLOAT16)doc");

    detail::bind_inplace_operation(
        mod,
        ttnn::gt_,
        R"doc(Performs Greater than in-place operation on :attr:`input_a` and :attr:`input_b` and returns the tensor with the same layout as :attr:`input_tensor`)doc",
        R"doc(\mathrm{{input\_tensor\_a}} > \mathrm{{input\_tensor\_b}})doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_inplace_operation(
        mod,
        ttnn::ge_,
        R"doc(Performs Greater than or equal to in-place operation on :attr:`input_a` and :attr:`input_b` and returns the tensor with the same layout as :attr:`input_tensor`)doc",
        R"doc(\mathrm{{input\_tensor\_a}} >= \mathrm{{input\_tensor\_b}})doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_inplace_operation(
        mod,
        ttnn::lt_,
        R"doc(Performs Less than in-place operation on :attr:`input_a` and :attr:`input_b` and returns the tensor with the same layout as :attr:`input_tensor`)doc",
        R"doc(\mathrm{{input\_tensor\_a}} < \mathrm{{input\_tensor\_b}})doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_inplace_operation(
        mod,
        ttnn::le_,
        R"doc(Performs Less than or equal to in-place operation on :attr:`input_a` and :attr:`input_b` and returns the tensor with the same layout as :attr:`input_tensor`)doc",
        R"doc(\mathrm{{input\_tensor\_a}} <= \mathrm{{input\_tensor\_b}})doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_inplace_operation(
        mod,
        ttnn::eq_,
        R"doc(Performs Equal to in-place operation on :attr:`input_a` and :attr:`input_b` and returns the tensor with the same layout as :attr:`input_tensor`)doc",
        R"doc(\mathrm{{input\_tensor\_a}} == \mathrm{{input\_tensor\_b}})doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_inplace_operation(
        mod,
        ttnn::ne_,
        R"doc(Performs Not equal to in-place operation on :attr:`input_a` and :attr:`input_b` and returns the tensor with the same layout as :attr:`input_tensor`)doc",
        R"doc(\mathrm{{input\_tensor\_a}}\: != \mathrm{{input\_tensor\_b}})doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_inplace_operation(
        mod,
        ttnn::ldexp_,
        R"doc(Performs ldexp in-place operation on :attr:`input_a` and :attr:`input_b` and returns the tensor with the same layout as :attr:`input_tensor`)doc",
        R"doc(\verb|ldexp|(\mathrm{{input\_tensor\_a,input\_tensor\_b}}))doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_inplace_operation(
        mod,
        ttnn::logaddexp_,
        R"doc(Performs logaddexp in-place operation on :attr:`input_a` and :attr:`input_b` and returns the tensor with the same layout as :attr:`input_tensor`)doc",
        R"doc(\verb|logaddexp|(\mathrm{{input\_tensor\_a,input\_tensor\_b}}))doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_inplace_operation(
        mod,
        ttnn::logaddexp2_,
        R"doc(Performs logaddexp2 in-place operation on :attr:`input_a` and :attr:`input_b` and returns the tensor with the same layout as :attr:`input_tensor`)doc",
        R"doc(\verb|logaddexp2|(\mathrm{{input\_tensor\_a,input\_tensor\_b}}))doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_inplace_operation(
        mod,
        ttnn::squared_difference_,
        R"doc(Performs squared_difference in-place operation on :attr:`input_a` and :attr:`input_b` and returns the tensor with the same layout as :attr:`input_tensor`)doc",
        R"doc(\verb|squared_difference|(\mathrm{{input\_tensor\_a,input\_tensor\_b}}))doc",
        R"doc(BFLOAT16, BFLOAT8_B, INT32, UINT32, UINT16)doc");

    detail::bind_inplace_operation_with_fast_approx(
        mod,
        ttnn::divide_,
        R"doc(Performs division in-place operation on :attr:`input_a` and :attr:`input_b` and returns the tensor with the same layout as :attr:`input_tensor`)doc",
        R"doc(\verb|divide|(\mathrm{{input\_tensor\_a,input\_tensor\_b}}))doc",
        R"doc(BFLOAT16, FLOAT32, UINT16)doc",
        R"doc(
        When :attr:`fast_and_approximate_mode` is `True`, the operation uses FPU+SFPU implementation for better performance.
        When :attr:`fast_and_approximate_mode` is `False` (default), the operation uses SFPU implementation for better accuracy.
        The operation is not supported for INT32 inputs since the outputs are returned as FLOAT32.
        )doc");

    detail::bind_inplace_operation(
        mod,
        ttnn::rsub_,
        R"doc(Subtracts :attr:`input_a` from :attr:`input_b` in-place and returns the tensor with the same layout as :attr:`input_tensor`)doc",
        R"doc(\mathrm{{input\_tensor\_b}} - \mathrm{{input\_tensor\_a}})doc",
        R"doc(FLOAT32, BFLOAT16, BFLOAT8_B, INT32, UINT32, UINT16)doc");

    detail::bind_inplace_operation(
        mod,
        ttnn::bias_gelu_,
        R"doc(Performs bias_gelu in-place operation on :attr:`input_a` and :attr:`input_b` and returns the tensor with the same layout as :attr:`input_tensor`)doc",
        R"doc(\verb|bias_gelu|(\mathrm{{input\_tensor\_a,input\_tensor\_b}}))doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_power(
        mod,
        ttnn::pow,
        R"doc(When :attr:`exponent` is a Tensor, supported dtypes are: BFLOAT16, FLOAT32. Both input tensors should be of same dtype.)doc");
}

}  // namespace ttnn::operations::binary
