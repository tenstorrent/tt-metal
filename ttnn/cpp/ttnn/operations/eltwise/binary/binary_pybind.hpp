// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/binary/binary_composite.hpp"
#include "ttnn/types.hpp"
#include "ttnn/cpp/pybind11/export_enum.hpp"

namespace py = pybind11;


namespace ttnn {
namespace operations {
namespace binary {

namespace detail {


template <typename binary_operation_t>
void bind_primitive_binary_operation(py::module& module, const binary_operation_t& operation, const std::string& description) {
    auto doc = fmt::format(
        R"doc(
        {2}

        Supports broadcasting (except with scalar)

        Args:
            * :attr:`input_tensor_a`
            * :attr:`input_tensor_b` (ttnn.Tensor or Number): the tensor or number to add to :attr:`input_tensor_a`.

        Keyword args:
            * :attr:`memory_config` (Optional[ttnn.MemoryConfig]): memory config for the output tensor
            * :attr:`dtype` (Optional[ttnn.DataType]): data type for the output tensor
            * :attr:`output_tensor` (Optional[ttnn.Tensor]): preallocated output tensor
            * :attr:`activations` (Optional[List[str]]): list of activation functions to apply to the output tensor
            * :attr:`queue_id` (Optional[uint8]): command queue id

        Example:

            >>> tensor1 = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device)
            >>> tensor2 = ttnn.to_device(ttnn.from_torch(torch.tensor((0, 1), dtype=torch.bfloat16)), device)
            >>> output = {1}(tensor1, tensor2)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        description);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const binary_operation_t& self,
               const ttnn::Tensor& input_tensor_a,
               const ttnn::Tensor& input_tensor_b,
               BinaryOpType binary_op_type,
               const std::optional<const DataType>& dtype,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor,
               const std::optional<unary::FusedActivations>& activations,
               const std::optional<unary::UnaryWithParam>& input_tensor_a_activation) -> ttnn::Tensor {
                return self(input_tensor_a, input_tensor_b, binary_op_type, dtype, memory_config, output_tensor, activations, input_tensor_a_activation);
            },
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::arg("binary_op_type"),
            py::kw_only(),
            py::arg("dtype") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("output_tensor") = std::nullopt,
            py::arg("activations") = std::nullopt,
            py::arg("input_tensor_a_activation") = std::nullopt});
}


template <typename binary_operation_t>
void bind_binary_operation(py::module& module, const binary_operation_t& operation, const std::string& description) {
    auto doc = fmt::format(
        R"doc(
        {2}

        Supports broadcasting.

        Args:
            * :attr:`input_tensor_a`
            * :attr:`input_tensor_b` (ttnn.Tensor or Number): the tensor or number to add to :attr:`input_tensor_a`.

        Keyword args:
            * :attr:`memory_config` (Optional[ttnn.MemoryConfig]): memory config for the output tensor
            * :attr:`dtype` (Optional[ttnn.DataType]): data type for the output tensor
            * :attr:`output_tensor` (Optional[ttnn.Tensor]): preallocated output tensor
            * :attr:`activations` (Optional[List[str]]): list of activation functions to apply to the output tensor
            * :attr:`queue_id` (Optional[uint8]): command queue id

        Example:

            >>> tensor1 = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device)
            >>> tensor2 = ttnn.to_device(ttnn.from_torch(torch.tensor((0, 1), dtype=torch.bfloat16)), device)
            >>> output = {1}(tensor1, tensor2)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        description);

    bind_registered_operation(
        module,
        operation,
        doc,
        // tensor and scalar
        ttnn::pybind_overload_t{
            [](const binary_operation_t& self,
               const ttnn::Tensor& input_tensor_a,
               const float scalar,
               const std::optional<const DataType>& dtype,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor,
               const std::optional<unary::FusedActivations>& activations,
               const std::optional<unary::UnaryWithParam>& input_tensor_a_activation,
               const uint8_t& queue_id) -> ttnn::Tensor {
                return self(queue_id, input_tensor_a, scalar, dtype, memory_config, output_tensor, activations, input_tensor_a_activation);
            },
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::kw_only(),
            py::arg("dtype") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("output_tensor") = std::nullopt,
            py::arg("activations") = std::nullopt,
            py::arg("input_tensor_a_activation") = std::nullopt,
            py::arg("queue_id") = 0},

        // tensor and tensor
        ttnn::pybind_overload_t{
            [](const binary_operation_t& self,
               const ttnn::Tensor& input_tensor_a,
               const ttnn::Tensor& input_tensor_b,
               const std::optional<const DataType>& dtype,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor,
               const std::optional<unary::FusedActivations>& activations,
               const std::optional<unary::UnaryWithParam>& input_tensor_a_activation,
               uint8_t queue_id) -> ttnn::Tensor {
                return self(queue_id, input_tensor_a, input_tensor_b, dtype, memory_config, output_tensor, activations, input_tensor_a_activation);
            },
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::kw_only(),
            py::arg("dtype") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("output_tensor") = std::nullopt,
            py::arg("activations") = std::nullopt,
            py::arg("input_tensor_a_activation") = std::nullopt,
            py::arg("queue_id") = 0});
}

template <typename binary_operation_t>
void bind_binary_composite(py::module& module, const binary_operation_t& operation, const std::string& description) {
    auto doc = fmt::format(
        R"doc(
            {2}

            Args:
                * :attr:`input_tensor_a`
                * :attr:`input_tensor_b` (ttnn.Tensor or Number): the tensor or number to add to :attr:`input_tensor_a`.

            Keyword Args:
                * :attr:`memory_config` (Optional[ttnn.MemoryConfig]): Memory configuration for the operation.

            Example:

                >>> tensor1 = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device)
                >>> tensor2 = ttnn.to_device(ttnn.from_torch(torch.tensor((0, 1), dtype=torch.bfloat16)), device)
                >>> output = {1}(tensor1, tensor2)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        description);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const binary_operation_t& self,
               const Tensor& input_tensor_a,
               const Tensor& input_tensor_b,
               const std::optional<MemoryConfig>& memory_config) {
                    return self(input_tensor_a, input_tensor_b, memory_config);
                },
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt});
}

template <typename binary_operation_t>
void bind_binary_composite_with_alpha(py::module& module, const binary_operation_t& operation, const std::string& description) {
    auto doc = fmt::format(
        R"doc(
            {2}

            Args:
                * :attr:`input_tensor_a`
                * :attr:`input_tensor_b` (ttnn.Tensor or Number): the tensor or number to add to :attr:`input_tensor_a`.
                * :attr:`alpha`

            Keyword Args:
                * :attr:`memory_config` (Optional[ttnn.MemoryConfig]): Memory configuration for the operation.

            Example:

                >>> tensor1 = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device)
                >>> tensor2 = ttnn.to_device(ttnn.from_torch(torch.tensor((0, 1), dtype=torch.bfloat16)), device)
                >>> output = {1}(tensor1, tensor2, alpha)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        description);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const binary_operation_t& self,
               const Tensor& input_tensor_a,
               const Tensor& input_tensor_b,
               float alpha,
               const std::optional<MemoryConfig>& memory_config) {
                    return self(input_tensor_a, input_tensor_b, alpha, memory_config);
                },
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::arg("alpha") = 1.0f,
            py::kw_only(),
            py::arg("memory_config") = std::nullopt});
}

template <typename binary_operation_t>
void bind_binary_composite_with_rtol_atol(py::module& module, const binary_operation_t& operation, const std::string& description) {
    auto doc = fmt::format(
        R"doc(
            {2}

            Args:
                * :attr:`input_tensor_a`
                * :attr:`input_tensor_b` (ttnn.Tensor or Number): the tensor or number to add to :attr:`input_tensor_a`.
                * :attr:`rtol`
                * :attr:`atol`
                * :attr:`equal_nan`

            Keyword Args:
                * :attr:`memory_config` (Optional[ttnn.MemoryConfig]): Memory configuration for the operation.

            Example:

                >>> tensor1 = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device)
                >>> tensor2 = ttnn.to_device(ttnn.from_torch(torch.tensor((0, 1), dtype=torch.bfloat16)), device)
                >>> output = {1}(tensor1, tensor2, rtol, atol, equal_nan)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        description);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const binary_operation_t& self,
               const Tensor& input_tensor_a,
               const Tensor& input_tensor_b,
               float rtol,
               float atol,
               const bool equal_nan,
               const std::optional<MemoryConfig>& memory_config) {
                    return self(input_tensor_a, input_tensor_b, rtol, atol, equal_nan, memory_config);
                },
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::kw_only(),
            py::arg("rtol") = 1e-05f,
            py::arg("atol") = 1e-08f,
            py::arg("equal_nan") = false,
            py::arg("memory_config") = std::nullopt});
}

template <typename binary_operation_t>
void bind_div_like_ops(py::module& module, const binary_operation_t& operation, const std::string& description) {
    auto doc = fmt::format(
        R"doc(
            {2}

            Args:
                * :attr:`input_tensor_a`
                * :attr:`input_tensor_b` (ttnn.Tensor or Number)

            Keyword Args:
                * :attr:`memory_config` (Optional[ttnn.MemoryConfig]): Memory configuration for the operation.

            Example:

                >>> tensor1 = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device)
                >>> tensor2 = ttnn.to_device(ttnn.from_torch(torch.tensor((0, 1), dtype=torch.bfloat16)), device)
                >>> output = {1}(tensor1, tensor2/scalar)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        description);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const binary_operation_t& self,
               const Tensor& input_tensor_a,
               const Tensor& input_tensor_b,
               const std::optional<MemoryConfig>& memory_config) {
                    return self(input_tensor_a, input_tensor_b, memory_config);
                },
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt},

        ttnn::pybind_overload_t{
            [](const binary_operation_t& self,
               const Tensor& input_tensor_a,
               float value,
               const std::optional<MemoryConfig>& memory_config) {
                    return self(input_tensor_a, value, memory_config);
                },
            py::arg("input_tensor_a"),
            py::arg("value"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt});
}

template <typename binary_operation_t>
void bind_div(py::module& module, const binary_operation_t& operation, const std::string& description) {
    auto doc = fmt::format(
        R"doc(
            {2}

            Args:
                * :attr:`input_tensor_a`
                * :attr:`input_tensor_b` (ttnn.Tensor or Number)
                * :attr:`accurate_mode`: ``false`` if input_tensor_b is non-zero, else ``true``.
                * :attr:`round_mode`

            Keyword Args:
                * :attr:`accurate_mode`: ``false`` if input_tensor_b is non-zero, else ``true`` (Only if the input tensor is not ComplexTensor)
                * :attr:`round_mode` : (Only if the input tensor is not ComplexTensor)
                * :attr:`memory_config` (Optional[ttnn.MemoryConfig]): Memory configuration for the operation.

            Example:
                >>> tensor1 = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device)
                >>> tensor2 = ttnn.to_device(ttnn.from_torch(torch.tensor((0, 1), dtype=torch.bfloat16)), device)
                >>> output = {1}(tensor1, tensor2/scalar)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        description);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const binary_operation_t& self,
               const Tensor& input_tensor_a,
               const Tensor& input_tensor_b,
               bool accurate_mode,
               const std::string& round_mode,
               const std::optional<MemoryConfig>& memory_config) {
                    return self(input_tensor_a, input_tensor_b, accurate_mode, round_mode, memory_config);
                },
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::kw_only(),
            py::arg("accurate_mode") = false,
            py::arg("round_mode") = "None",
            py::arg("memory_config") = std::nullopt},

        ttnn::pybind_overload_t{
            [](const binary_operation_t& self,
               const Tensor& input_tensor_a,
               float value,
               bool accurate_mode,
               const std::string& round_mode,
               const std::optional<MemoryConfig>& memory_config) {
                    return self(input_tensor_a, value, accurate_mode, round_mode, memory_config);
                },
            py::arg("input_tensor_a"),
            py::arg("value"),
            py::kw_only(),
            py::arg("accurate_mode") = false,
            py::arg("round_mode") = "None",
            py::arg("memory_config") = std::nullopt});
}

template <typename binary_operation_t>
void bind_polyval(py::module& module, const binary_operation_t& operation, const std::string& description) {
    auto doc = fmt::format(
        R"doc(
            {2}

            Args:
                * :attr:`input_tensor_a`
                * :attr:`coeffs` (Vector of floats)

            Keyword Args:
                * :attr:`memory_config` (Optional[ttnn.MemoryConfig]): Memory configuration for the operation.

            Example:
                >>> tensor1 = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device)
                >>> coeffs = (1, 2, 3, 4)
                >>> output = {1}(tensor1, coeffs)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        description);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const binary_operation_t& self,
            const Tensor& input_tensor_a,
            const std::vector<float>& coeffs,
            const std::optional<MemoryConfig>& memory_config) {
                    return self(input_tensor_a, coeffs, memory_config);
                },
            py::arg("input_tensor_a"),
            py::arg("coeffs"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt});
}

template <typename binary_operation_t>
void bind_binary_overload_operation(py::module& module, const binary_operation_t& operation, const std::string& description) {
    auto doc = fmt::format(
        R"doc(
            {2}

            Args:
                * :attr:`input_a` (ttnn.Tensor)
                * :attr:`input_b` (ttnn.Tensor or Number)

            Keyword Args:
                * :attr:`memory_config` (Optional[ttnn.MemoryConfig]): Memory configuration for the operation.

            Example::
                >>> tensor = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
                >>> output = {1}(tensor1, tensor2)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        description);

    bind_registered_operation(
        module,
        operation,
        doc,

        //tensor and scalar
        ttnn::pybind_overload_t{
            [](const binary_operation_t& self,
            const Tensor& input_tensor,
            float scalar,
            const std::optional<MemoryConfig>& memory_config) {
                return self(input_tensor, scalar, memory_config); },
            py::arg("input_tensor"),
            py::arg("scalar"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt},

        //tensor and tensor
        ttnn::pybind_overload_t{
            [](const binary_operation_t& self,
            const Tensor& input_tensor_a,
            const Tensor& input_tensor_b,
            const std::optional<MemoryConfig>& memory_config) {
                return self(input_tensor_a, input_tensor_b, memory_config); },
            py::arg("input_a"),
            py::arg("input_b"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt});
}

template <typename binary_operation_t>
void bind_inplace_operation(py::module& module, const binary_operation_t& operation, const std::string& description) {
    auto doc = fmt::format(
        R"doc(
            {2}

            Args:
                * :attr:`input_a` (ttnn.Tensor)
                * :attr:`input_b` (ttnn.Tensor or Number)

            Example::
                >>> tensor = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
                >>> output = {1}(tensor1, tensor2)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        description);

    bind_registered_operation(
        module,
        operation,
        doc,

        //tensor and scalar
        ttnn::pybind_overload_t{
            [](const binary_operation_t& self,
            const Tensor& input_tensor,
            const float scalar) {
                return self(input_tensor, scalar); },
            py::arg("input_a"),
            py::arg("input_b")},

        //tensor and tensor
        ttnn::pybind_overload_t{
            [](const binary_operation_t& self,
            const Tensor& input_tensor_a,
            const Tensor& input_tensor_b) {
                return self(input_tensor_a, input_tensor_b); },
            py::arg("input_a"),
            py::arg("input_b")});
}

template <typename binary_operation_t>
void bind_logical_inplace_operation(py::module& module, const binary_operation_t& operation, const std::string& description) {
    auto doc = fmt::format(
        R"doc(
            {2}

            Args:
                * :attr:`input_a` (ttnn.Tensor)
                * :attr:`input_b` (ttnn.Tensor)

            Example::
                >>> tensor1 = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
                >>> tensor2 = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
                >>> output = {1}(tensor1, tensor2)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        description);

    bind_registered_operation(
        module,
        operation,
        doc,

        ttnn::pybind_overload_t{
            [](const binary_operation_t& self,
            const Tensor& input_tensor_a,
            const Tensor& input_tensor_b) {
                return self(input_tensor_a, input_tensor_b); },
            py::arg("input_a"),
            py::arg("input_b")});
}

template <typename binary_operation_t>
void bind_binary_inplace_operation(py::module& module, const binary_operation_t& operation, const std::string& description) {
    auto doc = fmt::format(
        R"doc(
            {2}

            Args:
                * :attr:`input_a` (ttnn.Tensor)
                * :attr:`input_b` (ttnn.Tensor or Number)
            Keyword args:
            * :attr:`activations` (Optional[List[str]]): list of activation functions to apply to the output tensor
            Example::
                >>> tensor = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
                >>> output = {1}(tensor1, tensor2)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        description);

    bind_registered_operation(
        module,
        operation,
        doc,

        //tensor and scalar
        ttnn::pybind_overload_t{
            [](const binary_operation_t& self,
            const Tensor& input_tensor,
            const float scalar,
            const std::optional<unary::FusedActivations>& activations,
            const std::optional<unary::UnaryWithParam>& input_tensor_a_activation) {
                return self(input_tensor, scalar, activations, input_tensor_a_activation); },
            py::arg("input_a"),
            py::arg("input_b"),
            py::kw_only(),
            py::arg("activations") = std::nullopt,
            py::arg("input_tensor_a_activation") = std::nullopt},

        //tensor and tensor
        ttnn::pybind_overload_t{
            [](const binary_operation_t& self,
            const Tensor& input_tensor_a,
            const Tensor& input_tensor_b,
            const std::optional<unary::FusedActivations>& activations,
            const std::optional<unary::UnaryWithParam>& input_tensor_a_activation) {
                return self(input_tensor_a, input_tensor_b, activations, input_tensor_a_activation); },
            py::arg("input_a"),
            py::arg("input_b"),
            py::kw_only(),
            py::arg("activations") = std::nullopt,
            py::arg("input_tensor_a_activation") = std::nullopt});
}
}  // namespace detail

void py_module(py::module& module) {
    export_enum<BinaryOpType>(module, "BinaryOpType");

    detail::bind_binary_operation(
        module,
        ttnn::add,
        R"doc(Adds :attr:`input_tensor_a` to :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`
        .. math:: \mathrm{{ input\_tensor\_a }}_i + \mathrm{{ input\_tensor\_b }}_i)doc");

    detail::bind_binary_inplace_operation(
        module,
        ttnn::add_,
        R"doc(Adds :attr:`input_tensor_a` to :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a` in-place
        .. math:: \mathrm{{input\_tensor\_a}}_i + \mathrm{{input\_tensor\_b}}_i)doc");

    detail::bind_binary_operation(
        module,
        ttnn::subtract,
        R"doc(Subtracts :attr:`input_tensor_b` from :attr:`input_tensor_a` and returns the tensor with the same layout as :attr:`input_tensor_a`
        .. math:: \mathrm{{ input\_tensor\_a }}_i - \mathrm{{ input\_tensor\_b }}_i)doc");

    detail::bind_binary_inplace_operation(
        module,
        ttnn::subtract_,
        R"doc(Subtracts :attr:`input_tensor_b` from :attr:`input_tensor_a` and returns the tensor with the same layout as :attr:`input_tensor_a` in-place
        .. math:: \mathrm{{input\_tensor\_a}}_i - \mathrm{{input\_tensor\_b}}_i)doc");

    detail::bind_binary_operation(
        module,
        ttnn::multiply,
        R"doc(Multiplies :attr:`input_tensor_a` by :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`
        .. math:: \mathrm{{ input\_tensor\_a }}_i \times \mathrm{{ input\_tensor\_b }}_i)doc");

    detail::bind_binary_inplace_operation(
        module,
        ttnn::multiply_,
        R"doc(Multiplies :attr:`input_tensor_a` by :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a` in-place
        .. math:: \mathrm{{input\_tensor\_a}}_i \times \mathrm{{input\_tensor\_b}}_i)doc");

    detail::bind_binary_operation(
        module,
        ttnn::eq,
        R"doc(Compares if :attr:`input_tensor_a` is equal to :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`
        .. math:: \mathrm{{input\_tensor\_a}}_i == \mathrm{{input\_tensor\_b}}_i)doc");

    detail::bind_binary_operation(
        module,
        ttnn::ne,
        R"doc(Compares if :attr:`input_tensor_a` is not equal to :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`
        .. math:: \mathrm{{input\_tensor\_a}}_i != \mathrm{{input\_tensor\_b}}_i)doc");

    detail::bind_binary_operation(
        module,
        ttnn::lt,
        R"doc(Compares if :attr:`input_tensor_a` is less than :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`
        .. math:: \mathrm{{input\_tensor\_a}}_i < \mathrm{{input\_tensor\_b}}_i)doc");

    detail::bind_binary_operation(
        module,
        ttnn::le,
        R"doc(MCompares if :attr:`input_tensor_a` is less than or equal to :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`
        .. math:: \mathrm{{input\_tensor\_a}}_i <= \mathrm{{input\_tensor\_b}}_i)doc");

    detail::bind_binary_operation(
        module,
        ttnn::gt,
        R"doc(Compares if :attr:`input_tensor_a` is greater than :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`
        .. math:: \mathrm{{input\_tensor\_a}}_i > \mathrm{{input\_tensor\_b}}_i)doc");

    detail::bind_binary_operation(
        module,
        ttnn::ge,
        R"doc(Compares if :attr:`input_tensor_a` is greater than or equal to :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`
        .. math:: \mathrm{{input\_tensor\_a}}_i >= \mathrm{{input\_tensor\_b}}_i)doc");

    detail::bind_binary_operation(
        module,
        ttnn::logical_and,
        R"doc(Compute logical AND of :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`
        .. math:: \mathrm{{input\_tensor\_a}}_i && \mathrm{{input\_tensor\_b}}_i)doc");

    detail::bind_binary_operation(
        module,
        ttnn::logical_or,
        R"doc(Compute logical OR of :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`
        .. math:: \mathrm{{input\_tensor\_a}}_i || \mathrm{{input\_tensor\_b}}_i)doc");

    detail::bind_binary_operation(
        module,
        ttnn::ldexp,
        R"doc(Compute ldexp of :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`
        .. math:: \mathrm{{input\_tensor\_a}}_i || \mathrm{{input\_tensor\_b}}_i)doc");

    detail::bind_binary_operation(
        module,
        ttnn::logaddexp,
        R"doc(Compute logaddexp of :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`
        .. math:: \mathrm{{input\_tensor\_a}}_i || \mathrm{{input\_tensor\_b}}_i)doc");

    detail::bind_binary_operation(
        module,
        ttnn::logaddexp2,
        R"doc(Compute logaddexp2 of :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`
        .. math:: \mathrm{{input\_tensor\_a}}_i || \mathrm{{input\_tensor\_b}}_i)doc");

    detail::bind_binary_operation(
        module,
        ttnn::squared_difference,
        R"doc(Compute squared difference of :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`
        .. math:: \mathrm{{input\_tensor\_a}}_i || \mathrm{{input\_tensor\_b}}_i)doc");

    detail::bind_binary_operation(
        module,
        ttnn::bias_gelu,
        R"doc(Compute bias_gelu of :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`
        .. math:: \mathrm{{input\_tensor\_a}}_i || \mathrm{{input\_tensor\_b}}_i)doc");

    detail::bind_binary_operation(
        module,
        ttnn::divide,
        R"doc(Divides :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`
        .. math:: \mathrm{{input\_tensor\_a}}_i || \mathrm{{input\_tensor\_b}}_i)doc");

    auto prim_module = module.def_submodule("prim", "Primitive binary operations");

    detail::bind_primitive_binary_operation(
        prim_module,
        ttnn::prim::binary,
        R"doc(Applied binary operation on :attr:`input_tensor_a` to :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc");

    // new imported
    detail::bind_binary_composite(
        module,
        ttnn::hypot,
        R"doc(compute Hypot :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`
        .. math:: \mathrm{{input\_tensor\_a}}_i || \mathrm{{input\_tensor\_b}}_i)doc");

    detail::bind_binary_composite(
        module,
        ttnn::xlogy,
        R"doc(Compute xlogy :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`
        .. math:: \mathrm{{input\_tensor\_a}}_i || \mathrm{{input\_tensor\_b}}_i)doc");

    detail::bind_binary_composite(
        module,
        ttnn::nextafter,
        R"doc(Compute nextafter :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`
        .. math:: \mathrm{{input\_tensor\_a}}_i || \mathrm{{input\_tensor\_b}}_i)doc");

    detail::bind_binary_composite(
        module,
        ttnn::minimum,
        R"doc(Compute minimum :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`
        .. math:: \mathrm{{input\_tensor\_a}}_i || \mathrm{{input\_tensor\_b}}_i)doc");

    detail::bind_binary_composite(
        module,
        ttnn::maximum,
        R"doc(Compute maximum :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`
        .. math:: \mathrm{{input\_tensor\_a}}_i || \mathrm{{input\_tensor\_b}}_i)doc");

    detail::bind_binary_composite(
        module,
        ttnn::atan2,
        R"doc(Compute atan2 :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`
        .. math:: \mathrm{{input\_tensor\_a}}_i || \mathrm{{input\_tensor\_b}}_i)doc");

    detail::bind_binary_composite(
        module,
        ttnn::logical_xor,
        R"doc(Compute logical_xor :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`
        .. math:: \mathrm{{input\_tensor\_a}}_i || \mathrm{{input\_tensor\_b}}_i)doc");

    detail::bind_logical_inplace_operation(
        module,
        ttnn::logical_or_,
        R"doc(Compute inplace logical OR of :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`
        .. math:: \mathrm{{input\_tensor\_a}}_i || \mathrm{{input\_tensor\_b}}_i)doc");

    detail::bind_binary_composite(
        module,
        ttnn::logical_xor_,
        R"doc(Compute inplace logical XOR of :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`
        .. math:: \mathrm{{input\_tensor\_a}}_i || \mathrm{{input\_tensor\_b}}_i)doc");

    detail::bind_logical_inplace_operation(
        module,
        ttnn::logical_and_,
        R"doc(Compute inplace logical AND of :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`
        .. math:: \mathrm{{input\_tensor\_a}}_i || \mathrm{{input\_tensor\_b}}_i)doc");

    detail::bind_binary_composite_with_alpha(
        module,
        ttnn::addalpha,
        R"doc(Compute addalpha :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`
        .. math:: \mathrm{{input\_tensor\_a}}_i || \mathrm{{input\_tensor\_b}}_i)doc");

    detail::bind_binary_composite_with_alpha(
        module,
        ttnn::subalpha,
        R"doc(Compute subalpha :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`
        .. math:: \mathrm{{input\_tensor\_a}}_i || \mathrm{{input\_tensor\_b}}_i)doc");

    detail::bind_binary_composite_with_rtol_atol(
        module,
        ttnn::isclose,
        R"doc(Compute isclose :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`
        .. math:: \mathrm{{input\_tensor\_a}}_i || \mathrm{{input\_tensor\_b}}_i)doc");

    detail::bind_div(
        module,
        ttnn::div,
        R"doc(Compute div :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`
        .. math:: \mathrm{{input\_tensor\_a}}_i || \mathrm{{input\_tensor\_b}}_i)doc");

    detail::bind_div_like_ops(
        module,
        ttnn::div_no_nan,
        R"doc(Compute div_no_nan :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`
        .. math:: \mathrm{{input\_tensor\_a}}_i || \mathrm{{input\_tensor\_b}}_i)doc");

    detail::bind_div_like_ops(
        module,
        ttnn::floor_div,
        R"doc(Compute floor division :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`
        .. math:: \mathrm{{input\_tensor\_a}}_i || \mathrm{{input\_tensor\_b}}_i)doc");

    detail::bind_binary_composite(
        module,
        ttnn::scatter,
        R"doc(Compute scatter :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`
        .. math:: \mathrm{{input\_tensor\_a}}_i || \mathrm{{input\_tensor\_b}}_i)doc");

    detail::bind_binary_composite(
        module,
        ttnn::outer,
        R"doc(Compute outer :attr:`input_tensor_a` and :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`
        .. math:: \mathrm{{input\_tensor\_a}}_i || \mathrm{{input\_tensor\_b}}_i)doc");

    detail::bind_polyval(
        module,
        ttnn::polyval,
        R"doc(Compute polyval of all elements of :attr:`input_tensor_a` with coefficient :attr:`coeffs` and returns the tensor with the same layout as :attr:`input_tensor_a`
        .. math:: \mathrm{{input\_tensor\_a}}_i || \mathrm{{coeffs}}_i)doc");

    detail::bind_binary_overload_operation(
        module,
        ttnn::fmod,
        R"doc(Perform an eltwise-fmod operation. Formula : a - a.div(b, rounding_mode=trunc) * b . Support provided only for WH_B0.)doc");

    detail::bind_binary_overload_operation(
        module,
        ttnn::remainder,
        R"doc(Perform an eltwise-modulus operation a - a.div(b, rounding_mode=floor) * b.", "Support provided only for WH_B0.)doc");

    detail::bind_inplace_operation(
        module,
        ttnn::gt_,
        R"doc(Perform Greater than in-place operation on :attr:`input_a` and :attr:`input_b` and returns the tensor with the same layout as :attr:`input_tensor`
        .. math:: \mathrm{{input\_a}}_i > \mathrm{{input\_b}}_i)doc");

    detail::bind_inplace_operation(
        module,
        ttnn::ge_,
        R"doc(Perform Greater than or equal to in-place operation on :attr:`input_a` and :attr:`input_b` and returns the tensor with the same layout as :attr:`input_tensor`
        .. math:: \mathrm{{input\_a}}_i >= \mathrm{{input\_b}}_i)doc");

    detail::bind_inplace_operation(
        module,
        ttnn::lt_,
        R"doc(Perform Less than in-place operation on :attr:`input_a` and :attr:`input_b` and returns the tensor with the same layout as :attr:`input_tensor`
        .. math:: \mathrm{{input\_a}}_i < \mathrm{{input\_b}}_i)doc");

    detail::bind_inplace_operation(
        module,
        ttnn::le_,
        R"doc(Perform Less than or equal to in-place operation on :attr:`input_a` and :attr:`input_b` and returns the tensor with the same layout as :attr:`input_tensor`
        .. math:: \mathrm{{input\_a}}_i <= \mathrm{{input\_b}}_i)doc");

    detail::bind_inplace_operation(
        module,
        ttnn::eq_,
        R"doc(Perform Equal to in-place operation on :attr:`input_a` and :attr:`input_b` and returns the tensor with the same layout as :attr:`input_tensor`
        .. math:: \mathrm{{input\_a}}_i = \mathrm{{input\_b}}_i)doc");

    detail::bind_inplace_operation(
        module,
        ttnn::ne_,
        R"doc(Perform Not equal to in-place operation on :attr:`input_a` and :attr:`input_b` and returns the tensor with the same layout as :attr:`input_tensor`
        .. math:: \mathrm{{input\_a}}_i != \mathrm{{input\_b}}_i)doc");

}

}  // namespace binary
}  // namespace operations
}  // namespace ttnn
