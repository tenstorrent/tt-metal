// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/eltwise/unary/unary_composite.hpp"
#include "ttnn/operations/eltwise/complex_unary/complex_unary.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operations/eltwise/complex/complex.hpp"
#include "unary.hpp"

namespace py = pybind11;

namespace ttnn {
namespace operations {
namespace unary {

namespace detail {

template <typename unary_operation_t>
void bind_unary_operation(py::module& module, const unary_operation_t& operation, const std::string& math, const std::string& info_doc = "" ) {
    auto doc = fmt::format(
        R"doc(
        Applies {0} to :attr:`input_tensor` element-wise.

        .. math::
            {2}

        Args:
            input_tensor (ttnn.Tensor): the input tensor.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): preallocated output tensor. Defaults to `None`.
            queue_id (int, optional): command queue id. Defaults to `0`.

        Returns:
            ttnn.Tensor: the output tensor.

        Note:
            {3}

        Example:
            >>> tensor = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
            >>> output = {1}(tensor)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        math,
        info_doc);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const unary_operation_t& self,
               const Tensor& input_tensor,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor,
               const uint8_t& queue_id) -> ttnn::Tensor {
                    return self(queue_id, input_tensor, memory_config, output_tensor);
                },
            py::arg("input_tensor"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("output_tensor") = std::nullopt,
            py::arg("queue_id") = 0});
}


template <typename unary_operation_t>
void bind_unary_operation_overload_complex(py::module& module, const unary_operation_t& operation, const std::string& info_doc = "" ) {
    auto doc = fmt::format(
        R"doc(
        Applies {0} to :attr:`input_tensor` element-wise.

        {2}

        .. math::
            \mathrm{{output\_tensor}}_i = {0}(\mathrm{{input\_tensor}}_i)

        Args:
            input_tensor (ttnn.Tensor): the input tensor.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): preallocated output tensor. Defaults to `None`.
            queue_id (int, optional): command queue id. Defaults to `0`.

        Returns:
            ttnn.Tensor: the output tensor.

        Example:
            >>> tensor = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
            >>> output = {1}(tensor)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        info_doc);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const unary_operation_t& self,
               const Tensor& input_tensor,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor,
               const uint8_t& queue_id) -> ttnn::Tensor {
                    return self(queue_id, input_tensor, memory_config, output_tensor);
                },
            py::arg("input_tensor"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("output_tensor") = std::nullopt,
            py::arg("queue_id") = 0},

        ttnn::pybind_overload_t{
            [](const unary_operation_t& self,
               const ComplexTensor& input_tensor,
               const ttnn::MemoryConfig& memory_config) -> Tensor {
                return self(input_tensor, memory_config);
            },
            py::arg("input_tensor"),
            py::kw_only(),
            py::arg("memory_config")});
}

template <typename unary_operation_t>
void bind_unary_operation_overload_complex_return_complex(py::module& module, const unary_operation_t& operation, const std::string& info_doc = "" ) {
    auto doc = fmt::format(
        R"doc(
        Applies {0} to :attr:`input_tensor` element-wise.

        {2}

        .. math::
            \mathrm{{output\_tensor}}_i = {0}(\mathrm{{input\_tensor}}_i)

        Args:
            input_tensor (ttnn.Tensor): the input tensor.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): preallocated output tensor. Defaults to `None`.
            queue_id (int, optional): command queue id. Defaults to `0`.

        Returns:
            ttnn.Tensor: the output tensor.

        Example:
            >>> tensor = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
            >>> output = {1}(tensor)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        info_doc);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const unary_operation_t& self,
               const Tensor& input_tensor,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor,
               const uint8_t& queue_id) -> ttnn::Tensor {
                    return self(queue_id, input_tensor, memory_config, output_tensor);
                },
            py::arg("input_tensor"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("output_tensor") = std::nullopt,
            py::arg("queue_id") = 0},

        ttnn::pybind_overload_t{
            [](const unary_operation_t& self,
               const ComplexTensor& input_tensor,
               const ttnn::MemoryConfig& memory_config) -> ComplexTensor {
                return self(input_tensor, memory_config);
            },
            py::arg("input_tensor"),
            py::kw_only(),
            py::arg("memory_config")});
}

template <typename unary_operation_t>
void bind_unary_operation_with_fast_and_approximate_mode(py::module& module, const unary_operation_t& operation, const std::string& info_doc = "" ) {
    auto doc = fmt::format(
        R"doc(
        Applies {0} to :attr:`input_tensor` element-wise.

        {2}

        .. math::
            \mathrm{{output\_tensor}}_i = {0}(\mathrm{{input\_tensor}}_i)

        Args:
            input_tensor (ttnn.Tensor): the input tensor.

        Keyword Args:
            fast_and_approximate_mode (bool): Use the fast and approximate mode.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): preallocated output tensor. Defaults to `None`.
            queue_id (int, optional): command queue id. Defaults to `0`.

        Returns:
            ttnn.Tensor: the output tensor.

        Example:
            >>> tensor = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
            >>> output = {1}(tensor, fast_and_approximate_mode=true)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        info_doc);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const unary_operation_t& self,
               const Tensor& input_tensor,
               const bool parameter,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor,
               const uint8_t& queue_id)  -> ttnn::Tensor {
                return self(queue_id, input_tensor, parameter, memory_config, output_tensor);
            },
            py::arg("input_tensor"),
            py::kw_only(),
            py::arg("fast_and_approximate_mode") = false,
            py::arg("memory_config") = std::nullopt,
            py::arg("output_tensor") = std::nullopt,
            py::arg("queue_id") = 0});
}

template <typename unary_operation_t>
void bind_unary_operation_with_float_parameter(
    py::module& module,
    const unary_operation_t& operation,
    const std::string& parameter_name,
    const std::string& parameter_doc,
    const std::string& info_doc) {
    auto doc = fmt::format(
        R"doc(
        Applies {0} to :attr:`input_tensor` element-wise.

        {4}

        .. math::
            \mathrm{{output\_tensor}}_i = \verb|{0}|(\mathrm{{input\_tensor}}_i)

        Args:
            input_tensor (ttnn.Tensor): the input tensor.

        Keyword Args:
            {2} (float): {3}.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): preallocated output tensor. Defaults to `None`.
            queue_id (int, optional): command queue id. Defaults to `0`.

        Returns:
            ttnn.Tensor: the output tensor.

        Example:
            >>> tensor = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
            >>> output = {1}(tensor, {2})
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        parameter_name,
        parameter_doc,
        info_doc);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const unary_operation_t& self,
               const Tensor& input_tensor,
               const float parameter,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor,
               const uint8_t& queue_id) {
                return self(queue_id, input_tensor, parameter, memory_config, output_tensor);
            },
            py::arg("input_tensor"),
            py::arg(parameter_name.c_str()),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("output_tensor") = std::nullopt,
            py::arg("queue_id") = 0});
}

template <typename unary_operation_t>
void bind_unary_operation_with_integer_parameter(
    py::module& module,
    const unary_operation_t& operation,
    const std::string& parameter_name,
    const std::string& parameter_doc,
    const std::string& info_doc) {

    auto doc = fmt::format(
        R"doc(
        Applies {0} to :attr:`input_tensor` element-wise.

        {4}

        .. math::
            \mathrm{{output\_tensor}}_i = \verb|{0}|(\mathrm{{input\_tensor}}_i)

        Args:
            input_tensor (ttnn.Tensor): the input tensor.

        Keyword Args:
            {2} (int): {3}.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): preallocated output tensor. Defaults to `None`.
            queue_id (int, optional): command queue id. Defaults to `0`.

        Returns:
            ttnn.Tensor: the output tensor.

        Example:
            >>> tensor = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
            >>> output = {1}(tensor, {2})
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        parameter_name,
        parameter_doc,
        info_doc);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const unary_operation_t& self,
               const Tensor& input_tensor,
               int parameter,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor,
               const uint8_t& queue_id) {
                return self(queue_id, input_tensor, parameter, memory_config, output_tensor);
            },
            py::arg("input_tensor"),
            py::arg(parameter_name.c_str()),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("output_tensor") = std::nullopt,
            py::arg("queue_id") = 0});
}


template <typename unary_operation_t>
void bind_unary_operation_with_dim_parameter(
    py::module& module,
    const unary_operation_t& operation,
    const std::string& parameter_name,
    const std::string& parameter_doc,
    const std::string& info_doc) {

    auto doc = fmt::format(
        R"doc(
        Applies {0} to :attr:`input_tensor` element-wise.

        {4}

        .. math::
            \mathrm{{output\_tensor}}_i = {0}(\mathrm{{input\_tensor}}_i)

        Args:
            input_tensor (ttnn.Tensor): the input tensor.

        Keyword Args:
            {2} (int): {3}.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): preallocated output tensor. Defaults to `None`.
            queue_id (int, optional): command queue id. Defaults to `0`.

        Returns:
            ttnn.Tensor: the output tensor.

        Example:
            >>> tensor = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
            >>> output = {1}(tensor, {2})
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        parameter_name,
        parameter_doc,
        info_doc);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const unary_operation_t& self,
               const Tensor& input_tensor,
               int dim,
               const std::optional<MemoryConfig>& memory_config) {
                return self(input_tensor, dim, memory_config);
            },
            py::arg("input_tensor"),
            py::arg("dim") = -1,
            py::kw_only(),
            py::arg("memory_config") = std::nullopt});
}

template <typename unary_operation_t>
void bind_unary_rdiv(py::module& module, const unary_operation_t& operation, const std::string& parameter_name_a, const std::string& parameter_a_doc, const std::string& parameter_name_b, const std::string& parameter_b_doc, const std::string parameter_b_value, const std::string& description) {
    auto doc = fmt::format(
        R"doc(
        {7}

        Args:
            input_tensor (ttnn.Tensor): the input tensor.
            {2} (int): {3}.

        Keyword Args:
            {4} (string): {5}. Defaults to `{6}`.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): preallocated output tensor. Defaults to `None`.
            queue_id (int, optional): command queue id. Defaults to `0`.

        Returns:
            ttnn.Tensor: the output tensor.

        Example:
            >>> tensor = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
            >>> output = {1}(tensor, {2}, {4} = {6})
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        parameter_name_a,
        parameter_a_doc,
        parameter_name_b,
        parameter_b_doc,
        parameter_b_value,
        description);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const unary_operation_t& self,
               const ttnn::Tensor& input_tensor,
               float parameter_a,
               const std::string& parameter_b,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor,
               const uint8_t& queue_id) {
                return self(queue_id, input_tensor, parameter_a, parameter_b, memory_config, output_tensor);
            },
            py::arg("input_tensor"),
            py::arg(parameter_name_a.c_str()),
            py::kw_only(),
            py::arg(parameter_name_b.c_str()) = parameter_b_value,
            py::arg("memory_config") = std::nullopt,
            py::arg("output_tensor") = std::nullopt,
            py::arg("queue_id") = 0});
}

template <typename unary_operation_t>
void bind_softplus(py::module& module, const unary_operation_t& operation) {
    auto doc = fmt::format(
        R"doc(
        Applies {0} to :attr:`input_tensor` element-wise.

        .. math::
            \mathrm{{output\_tensor}}_i = {0}(\mathrm{{input\_tensor}}_i)

        Args:
            input_tensor (ttnn.Tensor): the input tensor.

        Keyword Args:
            beta (float): Scales the input before applying the Softplus function. By modifying :attr:`beta`, you can adjust the steepness of the function. A higher :attr:`beta` value makes the function steeper, approaching a hard threshold like the ReLU function for large values of :attr:`beta`.
            threshold (float): Used to switch to a linear function for large values to improve numerical stability. This avoids issues with floating-point representation for very large values.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): preallocated output tensor. Defaults to `None`.
            queue_id (int, optional): command queue id. Defaults to `0`.

        Returns:
            ttnn.Tensor: the output tensor.

        Example:
            >>> tensor = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
            >>> output = {1}(tensor, beta=1.0, threshold=20.0)
        )doc",
        ttnn::softplus.base_name(),
        ttnn::softplus.python_fully_qualified_name());

    bind_registered_operation(
        module,
        ttnn::softplus,
        doc,
        ttnn::pybind_overload_t{
            [](const unary_operation_t& self,
               const Tensor& input,
               const float beta,
               const float threshold,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<Tensor>& output_tensor,
               const uint8_t queue_id) {
                return self(queue_id, input, beta, threshold, memory_config, output_tensor);
            },
            py::arg("input_tensor"),
            py::kw_only(),
            py::arg("beta") = 1.0f,
            py::arg("threshold") = 20.0f,
            py::arg("memory_config") = std::nullopt,
            py::arg("output_tensor") = std::nullopt,
            py::arg("queue_id") = 0});
}

template <typename unary_operation_t>
void bind_sigmoid_accurate(py::module& module, const unary_operation_t& operation) {
    auto doc = fmt::format(
        R"doc(
        Applies {0} to :attr:`input_tensor` element-wise.

        .. math::
            \mathrm{{output\_tensor}}_i = \verb|{0}|(\mathrm{{input\_tensor}}_i)

        Args:
            input_tensor (ttnn.Tensor): the input tensor.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): preallocated output tensor. Defaults to `None`.
            queue_id (int, optional): command queue id. Defaults to `0`.

        Returns:
            ttnn.Tensor: the output tensor.

        Example:
            >>> tensor = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
            >>> output = {1}(tensor)
        )doc",
        ttnn::sigmoid_accurate.base_name(),
        ttnn::sigmoid_accurate.python_fully_qualified_name());

    bind_registered_operation(
        module,
        ttnn::sigmoid_accurate,
        doc,
        ttnn::pybind_overload_t{
            [](const unary_operation_t& self,
               const Tensor& input_tensor,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<Tensor>& output_tensor,
               const uint8_t queue_id) -> ttnn::Tensor {
                return self(queue_id, input_tensor, memory_config, output_tensor);
            },
            py::arg("input_tensor"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("output_tensor") = std::nullopt,
            py::arg("queue_id") = 0});
}

template <typename unary_operation_t>
void bind_unary_chain(py::module& module, const unary_operation_t& operation) {
    auto doc = fmt::format(
        R"doc(
        Applies {0} to :attr:`input_tensor` element-wise.

        .. math::
            \mathrm{{output\_tensor}}_i = {0}(\mathrm{{input\_tensor}}_i)

        Args:
            input_tensor (ttnn.Tensor): the input tensor.

        Keyword Args:
            ops_chain (list[ttnn.UnaryWithParam]): List of unary ops to be chained.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): preallocated output tensor. Defaults to `None`.
            queue_id (int, optional): command queue id. Defaults to `0`.

        Returns:
            ttnn.Tensor: the output tensor.

        Example:

            >>> tensor = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
            >>> output = {1}(tensor, ops_chain)
        )doc",
        ttnn::unary_chain.base_name(),
        ttnn::unary_chain.python_fully_qualified_name());

    bind_registered_operation(
        module,
        ttnn::unary_chain,
        doc,
        ttnn::pybind_overload_t{
            [](const unary_operation_t& self,
               const Tensor& input_tensor,
               const FusedActivations& ops_chain,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<Tensor>& output_tensor,
               const uint8_t queue_id) {
                return self(queue_id, input_tensor, ops_chain, memory_config, output_tensor);
            },
            py::arg("input_tensor"),
            py::arg("ops_chain"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("output_tensor") = std::nullopt,
            py::arg("queue_id") = 0});
}

template <typename unary_operation_t>
void bind_identity(py::module& module, const unary_operation_t& operation) {
    auto doc = fmt::format(
        R"doc(
        Returns a copy of same tensor :attr:`input_tensor`; useful for profiling the SFPU.
        This shouldn't normally be used; users should normally use clone operation instead for same functionality as this would be lower performance.

        .. math::
            \mathrm{{output\_tensor}}_i = {0}(\mathrm{{input\_tensor}}_i)

        Args:
            input_tensor (ttnn.Tensor): the input tensor.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): preallocated output tensor. Defaults to `None`.
            queue_id (int, optional): command queue id. Defaults to `0`.

        Returns:
            ttnn.Tensor: the output tensor.

        Example:
            >>> tensor = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
            >>> output = {1}(tensor)
        )doc",
        ttnn::identity.base_name(),
        ttnn::identity.python_fully_qualified_name());

    bind_registered_operation(
        module,
        ttnn::identity,
        doc,
        ttnn::pybind_overload_t{
            [](const unary_operation_t& self,
               const Tensor& input_tensor,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<Tensor>& output_tensor,
               const uint8_t queue_id) {
                return self(queue_id, input_tensor, memory_config, output_tensor);
            },
            py::arg("input_tensor"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("output_tensor") = std::nullopt,
            py::arg("queue_id") = 0});
}

template <typename unary_operation_t>
void bind_power(py::module& module, const unary_operation_t& operation) {
    auto doc = fmt::format(
        R"doc(
        Applies {0} to :attr:`input_tensor` element-wise.

        .. math::
            \mathrm{{output\_tensor}}_i = {0}(\mathrm{{input\_tensor}}_i)

        Args:
            input_tensor (ttnn.Tensor): the input tensor.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): Preallocated output tensor. Defaults to `None`.
            queue_id (int, optional): command queue id. Defaults to `0`.
            exponent (float,int, optional): exponent is an integer. Defaults to `>0`.

        Returns:
            ttnn.Tensor: the output tensor.

        Example:
            >>> tensor = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
            >>> output = {1}(tensor, exponent)
        )doc",
        ttnn::pow.base_name(),
        ttnn::pow.python_fully_qualified_name());

    bind_registered_operation(
        module,
        ttnn::pow,
        doc,
        // integer exponent
        ttnn::pybind_overload_t{
            [](const unary_operation_t& self,
               const Tensor& input_tensor,
               uint32_t exponent,
               const std::optional<MemoryConfig>& memory_config,
               std::optional<Tensor> output_tensor,
               const uint8_t queue_id) -> ttnn::Tensor {
                return self(queue_id, input_tensor, exponent, memory_config, output_tensor);
            },
            py::arg("input_tensor"),
            py::arg("exponent"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("output_tensor") = std::nullopt,
            py::arg("queue_id") = 0},

            // float exponent
            ttnn::pybind_overload_t{
                [](const unary_operation_t& self,
                const Tensor& input_tensor,
                float exponent,
                const std::optional<MemoryConfig>& memory_config,
                std::optional<Tensor> output_tensor,
                const uint8_t queue_id) -> ttnn::Tensor {
                    return self(queue_id, input_tensor, exponent, memory_config, output_tensor);
                },
                py::arg("input_tensor"),
                py::arg("exponent"),
                py::kw_only(),
                py::arg("memory_config") = std::nullopt,
                py::arg("output_tensor") = std::nullopt,
                py::arg("queue_id") = 0}
            );
}

template <typename unary_operation_t>
void bind_unary_composite(py::module& module, const unary_operation_t& operation, const std::string& description,const std::string& info_doc = "", const std::string& range = "") {
    auto doc = fmt::format(
        R"doc(
        {2}

        Args:
            input_tensor (ttnn.Tensor): the input tensor. {4}

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

        Note:
            {3}

        Example:
            >>> tensor = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
            >>> output = {1}(tensor)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        description,
        info_doc,
        range);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const unary_operation_t& self,
               const Tensor& input_tensor,
               const std::optional<MemoryConfig>& memory_config) {
                    return self(input_tensor, memory_config);
                },
            py::arg("input_tensor"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt});
}

//OpHandler_1int
template <typename unary_operation_t>
void bind_unary_composite_int_with_default(py::module& module, const unary_operation_t& operation, const std::string& parameter_name_a, const std::string& parameter_a_doc, int32_t parameter_a_value, const std::string& description) {
    auto doc = fmt::format(
        R"doc(
        {5}

        Args:
            input_tensor (ttnn.Tensor): the input tensor.

        Keyword args:
            {2} (float): {3}. Defaults to `{4}`.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

        Example:
            >>> tensor = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
            >>> output = {1}(tensor, {2} = {4})
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        parameter_name_a,
        parameter_a_doc,
        parameter_a_value,
        description);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const unary_operation_t& self,
               const ttnn::Tensor& input_tensor,
               int32_t parameter_a,
               const std::optional<MemoryConfig>& memory_config)  {
                return self(input_tensor, parameter_a, memory_config);
            },
            py::arg("input_tensor"),
            py::kw_only(),
            py::arg(parameter_name_a.c_str()) = parameter_a_value,
            py::arg("memory_config") = std::nullopt});
}

//OpHandler_two_float_with_default
template <typename unary_operation_t>
void bind_unary_composite_floats_with_default(py::module& module, const unary_operation_t& operation, const std::string& parameter_name_a, const std::string& parameter_a_doc, float parameter_a_value, const std::string& parameter_name_b, const std::string& parameter_b_doc, float parameter_b_value, const std::string& description) {
    auto doc = fmt::format(
        R"doc(
        {8}

        Args:
            input_tensor (ttnn.Tensor): the input tensor.

        Keyword args:
            {2} (float): {3}. Defaults to `{4}`.
            {5} (float): {6}. Defaults to `{7}`.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

        Example:
            >>> tensor = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
            >>> output = {1}(tensor, {2} = {4}, {5} = {7})
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        parameter_name_a,
        parameter_a_doc,
        parameter_a_value,
        parameter_name_b,
        parameter_b_doc,
        parameter_b_value,
        description);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const unary_operation_t& self,
               const ttnn::Tensor& input_tensor,
               float parameter_a,
               float parameter_b,
               const std::optional<MemoryConfig>& memory_config)  {
                return self(input_tensor, parameter_a, parameter_b, memory_config);
            },
            py::arg("input_tensor"),
            py::kw_only(),
            py::arg(parameter_name_a.c_str()) = parameter_a_value,
            py::arg(parameter_name_b.c_str()) = parameter_b_value,
            py::arg("memory_config") = std::nullopt});
}
//OpHandler_two_float_with_default
template <typename unary_operation_t>
void bind_unary_composite_int(py::module& module, const unary_operation_t& operation, const std::string& parameter_name_a, const std::string& parameter_a_doc, const std::string& description) {
    auto doc = fmt::format(
        R"doc(
        {4}

        Args:
            input_tensor (ttnn.Tensor): the input tensor.

        Keyword args:
            {2} (int): {3}.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

        Example:
            >>> tensor = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
            >>> output = {1}(tensor, {2})
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        parameter_name_a,
        parameter_a_doc,
        description);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const unary_operation_t& self,
               const ttnn::Tensor& input_tensor,
               int32_t parameter_a,
               const std::optional<MemoryConfig>& memory_config)  {
                return self(input_tensor, parameter_a, memory_config);
            },
            py::arg("input_tensor"),
            py::arg(parameter_name_a.c_str()),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt});
}

//OpHandler_two_float_with_default
template <typename unary_operation_t>
void bind_unary_composite_floats(py::module& module, const unary_operation_t& operation, const std::string& parameter_name_a, const std::string& parameter_a_doc,  const std::string& parameter_name_b, const std::string& parameter_b_doc, const std::string& description) {
    auto doc = fmt::format(
        R"doc(
        {6}

        Args:
            input_tensor (ttnn.Tensor): the input tensor.

        Keyword args:
            {2} (float): {3}.
            {4} (float): {5}.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

        Example:
            >>> tensor = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
            >>> output = {1}(tensor, {2}, {4})
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        parameter_name_a,
        parameter_a_doc,
        parameter_name_b,
        parameter_b_doc,
        description);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const unary_operation_t& self,
               const ttnn::Tensor& input_tensor,
               float parameter_a,
               float parameter_b,
               const std::optional<MemoryConfig>& memory_config)  {
                return self(input_tensor, parameter_a, parameter_b, memory_config);
            },
            py::arg("input_tensor"),
            py::arg(parameter_name_a.c_str()),
            py::arg(parameter_name_b.c_str()),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt});
}

template <typename unary_operation_t>
void bind_unary_composite_operation(py::module& module, const unary_operation_t& operation, const std::string& description) {
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
            queue_id (int, optional): command queue id. Defaults to `0`.

        Returns:
            ttnn.Tensor: the output tensor.

        Example:
            >>> tensor = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
            >>> output = {1}(tensor)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        description);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const unary_operation_t& self,
               const Tensor& input_tensor,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor,
               const uint8_t& queue_id) {
                    return self(queue_id, input_tensor, memory_config, output_tensor);
                },
            py::arg("input_tensor"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("output_tensor") = std::nullopt,
            py::arg("queue_id") = ttnn::DefaultQueueId});
}


//OpHandler_float_with_default
template <typename unary_operation_t>
void bind_unary_composite_float_with_default(py::module& module, const unary_operation_t& operation, const std::string& parameter_name_a, const std::string& parameter_a_doc, float parameter_a_value, const std::string& description) {
    auto doc = fmt::format(
        R"doc(
        {5}

        Args:
            input_tensor (ttnn.Tensor): the input tensor.

        Keyword args:
            {2} (float): {3}. Defaults to `{4}`.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

        Supported dtypes and layouts:

        +----------------------------+---------------------------------+-------------------+
        |     Dtypes                 |         Layouts                 |     Ranks         |
        +----------------------------+---------------------------------+-------------------+
        |    BFLOAT16                |       TILE                      |      2, 3, 4      |
        +----------------------------+---------------------------------+-------------------+

        Example:
            >>> tensor = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
            >>> output = {1}(tensor, {2} = {4})
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        parameter_name_a,
        parameter_a_doc,
        parameter_a_value,
        description);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const unary_operation_t& self,
               const ttnn::Tensor& input_tensor,
               float parameter_a,
               const std::optional<MemoryConfig>& memory_config)  {
                return self(input_tensor, parameter_a, memory_config);
            },
            py::arg("input_tensor"),
            py::kw_only(),
            py::arg(parameter_name_a.c_str()) = parameter_a_value,
            py::arg("memory_config") = std::nullopt});
}

template <typename unary_operation_t>
void bind_unary_composite_float(py::module& module, const unary_operation_t& operation, const std::string& parameter_name_a, const std::string& parameter_a_doc, const std::string& description) {
    auto doc = fmt::format(
        R"doc(
        {4}

        Args:
            input_tensor (ttnn.Tensor): the input tensor.
            {2}

        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

        Example:
            >>> tensor = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
            >>> output = {1}(tensor, {2})
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        parameter_name_a,
        parameter_a_doc,
        description);
    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const unary_operation_t& self,
               const ttnn::Tensor& input_tensor,
               float parameter_a,
               const std::optional<MemoryConfig>& memory_config)  {
                return self(input_tensor, parameter_a, memory_config);
            },
            py::arg("input_tensor"),
            py::arg(parameter_name_a.c_str()),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt});
}

template <typename unary_operation_t>
void bind_unary_operation_with_scale_and_shift(py::module& module, const unary_operation_t& operation) {
    auto doc = fmt::format(
        R"doc(
        Applies {0} to :attr:`input_tensor` element-wise.

        .. math::
            \mathrm{{output\_tensor}}_i = {0}(\mathrm{{input\_tensor}}_i)

        Args:
            input_tensor (ttnn.Tensor): the input tensor.
            scale (float)
            shift (float)

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): preallocated output tensor. Defaults to `None`.
            queue_id (int, optional): command queue id. Defaults to `0`.

        Returns:
            ttnn.Tensor: the output tensor.

        Example::

            >>> tensor = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
            >>> output = {1}(tensor)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name());

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const unary_operation_t& self,
               const Tensor& input_tensor,
               float scale,
               float shift,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor,
               const uint8_t& queue_id) {
                return self(queue_id, input_tensor, scale, shift, memory_config, output_tensor);
            },
            py::arg("input_tensor"),
            py::arg("scale")=1.0f/6.0f,
            py::arg("shift")=0.5f,
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("output_tensor") = std::nullopt,
            py::arg("queue_id") = 0});
}

template <typename unary_operation_t>
void bind_unary_operation_with_low_and_high(py::module& module, const unary_operation_t& operation) {
    auto doc = fmt::format(
        R"doc(
        Applies {0} to :attr:`input_tensor` element-wise.

        .. math::
            \mathrm{{output\_tensor}}_i = {0}(\mathrm{{input\_tensor}}_i)

        Args:
            input_tensor (ttnn.Tensor): the input tensor.
            low (float)
            high (float)

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): preallocated output tensor. Defaults to `None`.
            queue_id (int, optional): command queue id. Defaults to `0`.

        Returns:
            ttnn.Tensor: the output tensor.

        Example::

            >>> tensor = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
            >>> output = {1}(tensor)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name());

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const unary_operation_t& self,
               const Tensor& input_tensor,
               float low,
               float high,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor,
               const uint8_t& queue_id) {
                return self(queue_id, input_tensor, low, high, memory_config, output_tensor);
            },
            py::arg("input_tensor"),
            py::arg("low") = -1.0f,
            py::arg("high") = 1.0f,
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("output_tensor") = std::nullopt,
            py::arg("queue_id") = 0});
}

template <typename unary_operation_t>
void bind_unary_operation_with_diag(py::module& module, const unary_operation_t& operation) {
    auto doc = fmt::format(
        R"doc(
        Applies {0} to :attr:`input_tensor` element-wise.

        .. math::
            \mathrm{{output\_tensor}}_i = {0}(\mathrm{{input\_tensor}}_i)

        Args:
            input_tensor (ttnn.Tensor): the input tensor.
            diag (int)

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): preallocated output tensor. Defaults to `None`.
            queue_id (int, optional): command queue id. Defaults to `0`.

        Example::

            >>> tensor = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
            >>> output = {1}(tensor)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name());

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const unary_operation_t& self,
               const Tensor& input_tensor,
               int32_t diag,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor,
               const uint8_t& queue_id) {
                return self(queue_id, input_tensor, diag, memory_config, output_tensor); },
            py::arg("input_tensor"),
            py::arg("diag") = 0,
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("output_tensor") = std::nullopt,
            py::arg("queue_id") = 0});
}


template <typename unary_operation_t>
void bind_dropout(py::module& module, const unary_operation_t& operation) {
    auto doc = fmt::format(
        R"doc({0}(input_tensor: ttnn.Tensor, *, seed: uint32_t, probability: float, scale: float, memory_config: Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor

            Applies {0} to :attr:`input_tensor` element-wise.

            .. math::
                {0}(\\mathrm{{input\\_tensor}}_i)

            Args:
                * :attr:`input_tensor`

            Keyword Args:
                * :attr:`seed` (uint32_t): seed used for RNG
                * :attr:`probability` (float): Dropout probability. In average total_elems * probability elements will be zero out.
                * :attr:`scale` (float): Scales output tensor. In general scale == 1.0/(1.0-probability)
                * :attr:`memory_config` (Optional[ttnn.MemoryConfig]): Memory configuration for the operation.
                * :attr:`output_tensor` (Optional[ttnn.Tensor]): preallocated output tensor
                * :attr:`queue_id` (Optional[uint8]): command queue id

            Example:

                >>> tensor = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
                >>> output = {1}(tensor, seed=42, probability=0.2, scale= 1.0/(1.0 - probability))
        )doc",
        ttnn::dropout.base_name(),
        ttnn::dropout.python_fully_qualified_name());

    bind_registered_operation(
        module,
        ttnn::dropout,
        doc,
        ttnn::pybind_overload_t{
            [](const unary_operation_t& self,
               const Tensor& input,
               const uint32_t seed,
               const float probability,
               const float scale,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<Tensor>& output_tensor,
               const uint8_t queue_id) {
                return self(queue_id, input, seed, probability, scale, memory_config, output_tensor);
            },
            py::arg("input_tensor"),
            py::kw_only(),
            py::arg("seed"),
            py::arg("probability"),
            py::arg("scale"),
            py::arg("memory_config") = std::nullopt,
            py::arg("output_tensor") = std::nullopt,
            py::arg("queue_id") = 0});
}

}  // namespace detail

void py_module(py::module& module) {
    detail::bind_unary_operation_overload_complex(module, ttnn::abs);
    detail::bind_unary_operation(module, ttnn::acos, R"doc(\mathrm{{output\_tensor}}_i = acos(\mathrm{{input\_tensor}}_i))doc");
    detail::bind_unary_operation(module, ttnn::asin, R"doc(\mathrm{{output\_tensor}}_i = asin(\mathrm{{input\_tensor}}_i))doc");
    detail::bind_unary_operation(module, ttnn::atan, R"doc(\mathrm{{output\_tensor}}_i = atan(\mathrm{{input\_tensor}}_i))doc");
    detail::bind_unary_operation(module, ttnn::cos, R"doc(\mathrm{{output\_tensor}}_i = cos(\mathrm{{input\_tensor}}_i))doc");
    detail::bind_unary_operation(module, ttnn::erfinv, R"doc(\mathrm{{output\_tensor}}_i = erfinv(\mathrm{{input\_tensor}}_i))doc");
    detail::bind_unary_operation(module, ttnn::exp2, R"doc(\mathrm{{output\_tensor}}_i = exp2(\mathrm{{input\_tensor}}_i))doc",
    R"doc(Supported dtypes, layouts, and ranks:

        +----------------------------+---------------------------------+-------------------+
        |     Dtypes                 |         Layouts                 |     Ranks         |
        +----------------------------+---------------------------------+-------------------+
        |    BFLOAT16, BFLOAT8_B     |          TILE                   |      2, 3, 4      |
        +----------------------------+---------------------------------+-------------------+)doc");
    detail::bind_unary_operation(module, ttnn::expm1, R"doc(\mathrm{{output\_tensor}}_i = expm1(\mathrm{{input\_tensor}}_i))doc",
    R"doc(Supported dtypes, layouts, and ranks:

        +----------------------------+---------------------------------+-------------------+
        |     Dtypes                 |         Layouts                 |     Ranks         |
        +----------------------------+---------------------------------+-------------------+
        |    BFLOAT16, BFLOAT8_B     |          TILE                   |      2, 3, 4      |
        +----------------------------+---------------------------------+-------------------+)doc");
    detail::bind_unary_operation(module, ttnn::eqz, R"doc(\mathrm{{output\_tensor}}_i = (\mathrm{{input\_tensor_i\ == 0}}))doc");
    detail::bind_unary_operation(module, ttnn::floor, R"doc(\mathrm{{output\_tensor}}_i = floor(\mathrm{{input\_tensor}}_i))doc", "Available for Wormhole_B0 only");
    detail::bind_unary_operation(module, ttnn::ceil, R"doc(\mathrm{{output\_tensor}}_i = ceil(\mathrm{{input\_tensor}}_i))doc", "Available for Wormhole_B0 only");
    detail::bind_unary_operation(module, ttnn::gez, R"doc(\mathrm{{output\_tensor}}_i = (\mathrm{{input\_tensor_i\ >= 0}}))doc");
    detail::bind_unary_operation(module, ttnn::gtz, R"doc(\mathrm{{output\_tensor}}_i= (\mathrm{{input\_tensor_i\ > 0}}))doc");
    detail::bind_unary_operation(module, ttnn::i0, R"doc(\mathrm{{output\_tensor}}_i = i0(\mathrm{{input\_tensor}}_i))doc");
    detail::bind_unary_operation(module, ttnn::isfinite, R"doc(\mathrm{{output\_tensor}}_i = isfinite(\mathrm{{input\_tensor}}_i))doc");
    detail::bind_unary_operation(module, ttnn::isinf, R"doc(\mathrm{{output\_tensor}}_i = isinf(\mathrm{{input\_tensor}}_i))doc");
    detail::bind_unary_operation(module, ttnn::isnan, R"doc(\mathrm{{output\_tensor}}_i = isnan(\mathrm{{input\_tensor}}_i))doc");
    detail::bind_unary_operation(module, ttnn::isneginf, R"doc(\mathrm{{output\_tensor}}_i = isneginf(\mathrm{{input\_tensor}}_i))doc");
    detail::bind_unary_operation(module, ttnn::isposinf, R"doc(\mathrm{{output\_tensor}}_i = isposinf(\mathrm{{input\_tensor}}_i))doc");
    detail::bind_unary_operation(module, ttnn::lez, R"doc(\mathrm{{output\_tensor}}_i = (\mathrm{{input\_tensor_i\ <= 0}}))doc");
    detail::bind_unary_operation(module, ttnn::log, R"doc(\mathrm{{output\_tensor}}_i = log(\mathrm{{input\_tensor}}_i))doc");
    detail::bind_unary_operation(module, ttnn::log10, R"doc(\mathrm{{output\_tensor}}_i = log10(\mathrm{{input\_tensor}}_i))doc",
    R"doc(Supported dtypes, layouts, and ranks:

        +----------------------------+---------------------------------+-------------------+
        |     Dtypes                 |         Layouts                 |     Ranks         |
        +----------------------------+---------------------------------+-------------------+
        |    BFLOAT16                |          TILE                   |      2, 3, 4      |
        +----------------------------+---------------------------------+-------------------+)doc");
    detail::bind_unary_operation(module, ttnn::log2, R"doc(\mathrm{{output\_tensor}}_i = log2(\mathrm{{input\_tensor}}_i))doc",
    R"doc(Supported dtypes, layouts, and ranks:

        +----------------------------+---------------------------------+-------------------+
        |     Dtypes                 |         Layouts                 |     Ranks         |
        +----------------------------+---------------------------------+-------------------+
        |    BFLOAT16                |          TILE                   |      2, 3, 4      |
        +----------------------------+---------------------------------+-------------------+)doc");
    detail::bind_unary_operation(module, ttnn::logical_not, R"doc(\mathrm{{output\_tensor}}_i = \mathrm{{!input\_tensor_i}})doc", R"doc(Supports bfloat16 dtype and both TILE and ROW_MAJOR layout)doc");
    detail::bind_unary_operation(module, ttnn::ltz, R"doc(\mathrm{{output\_tensor}}_i = (\mathrm{{input\_tensor_i\ < 0}}))doc");
    detail::bind_unary_operation(module, ttnn::neg, R"doc(\mathrm{{output\_tensor}}_i = neg(\mathrm{{input\_tensor}}_i))doc");
    detail::bind_unary_operation(module, ttnn::nez, R"doc(\mathrm{{output\_tensor}}_i = (\mathrm{{input\_tensor_i\ != 0}}))doc");
    detail::bind_unary_operation_overload_complex_return_complex(module, ttnn::reciprocal);
    detail::bind_unary_operation(module, ttnn::relu, R"doc(\mathrm{{output\_tensor}}_i = relu(\mathrm{{input\_tensor}}_i))doc");
    detail::bind_unary_operation(module, ttnn::relu6, R"doc(\mathrm{{output\_tensor}}_i = relu6(\mathrm{{input\_tensor}}_i))doc");
    detail::bind_unary_operation(module, ttnn::sigmoid, R"doc(\mathrm{{output\_tensor}}_i = sigmoid(\mathrm{{input\_tensor}}_i))doc");
    detail::bind_unary_operation(module, ttnn::sign, R"doc(\mathrm{{output\_tensor}}_i = sign(\mathrm{{input\_tensor}}_i))doc");
    detail::bind_unary_operation(module, ttnn::signbit, R"doc(\mathrm{{output\_tensor}}_i = signbit(\mathrm{{input\_tensor}}_i))doc");
    detail::bind_unary_operation(module, ttnn::silu, R"doc(\mathrm{{output\_tensor}}_i = silu(\mathrm{{input\_tensor}}_i))doc");
    detail::bind_unary_operation(module, ttnn::sin, R"doc(\mathrm{{output\_tensor}}_i = sin(\mathrm{{input\_tensor}}_i))doc");
    detail::bind_unary_operation(module, ttnn::sqrt, R"doc(\mathrm{{output\_tensor}}_i = sqrt(\mathrm{{input\_tensor}}_i))doc");
    detail::bind_unary_operation(module, ttnn::square, R"doc(\mathrm{{output\_tensor}}_i = square(\mathrm{{input\_tensor}}_i))doc");
    detail::bind_unary_operation(module, ttnn::tan, R"doc(\mathrm{{output\_tensor}}_i = tan(\mathrm{{input\_tensor}}_i))doc");
    detail::bind_unary_operation(module, ttnn::tanh, R"doc(\mathrm{{output\_tensor}}_i = tanh(\mathrm{{input\_tensor}}_i))doc");
    detail::bind_unary_operation(module, ttnn::log_sigmoid, R"doc(\mathrm{{output\_tensor}}_i = \verb|log_sigmoid|(\mathrm{{input\_tensor}}_i))doc",
    R"doc(Supported dtypes, layouts, and ranks:

        +----------------------------+---------------------------------+-------------------+
        |     Dtypes                 |         Layouts                 |     Ranks         |
        +----------------------------+---------------------------------+-------------------+
        |    BFLOAT16                |          TILE                   |      2, 3, 4      |
        +----------------------------+---------------------------------+-------------------+)doc");

    //  Unaries with fast_and_approximate_mode
    detail::bind_unary_operation_with_fast_and_approximate_mode(module, ttnn::exp,
    R"doc(Supported dtypes, layouts, and ranks:

        +----------------------------+---------------------------------+-------------------+
        |     Dtypes                 |         Layouts                 |     Ranks         |
        +----------------------------+---------------------------------+-------------------+
        |    BFLOAT16, BFLOAT8_B     |          TILE                   |      2, 3, 4      |
        +----------------------------+---------------------------------+-------------------+)doc");
    detail::bind_unary_operation_with_fast_and_approximate_mode(module, ttnn::erf);
    detail::bind_unary_operation_with_fast_and_approximate_mode(module, ttnn::erfc);
    detail::bind_unary_operation_with_fast_and_approximate_mode(module, ttnn::gelu);
    detail::bind_unary_operation_with_fast_and_approximate_mode(module, ttnn::rsqrt);

    // Unaries with float parameter
    detail::bind_unary_operation_with_float_parameter(module, ttnn::elu, "alpha", "The alpha parameter for the ELU function", "");
    detail::bind_unary_operation_with_float_parameter(module, ttnn::rsub, "value", "subtrahent value which is actually calculated as minuend", "Returns tensor with respective elements of the input tensor subtracted from the value.");
    detail::bind_unary_operation_with_float_parameter(module, ttnn::heaviside, "value", "The value parameter for the Heaviside function", "");
    detail::bind_unary_operation_with_float_parameter(module, ttnn::leaky_relu, "slope", "The slope parameter for the Leaky ReLU function", "");
    detail::bind_unary_operation_with_float_parameter(module, ttnn::relu_max, "upper_limit", "The max value for ReLU function", "This function caps off the input to a max value and a min value of 0");
    detail::bind_unary_operation_with_float_parameter(module, ttnn::relu_min, "lower_limit", "The min value for ReLU function", "This will carry out ReLU operation at min value instead of the standard 0");

    // Unaries with integer parameter
    detail::bind_unary_operation_with_integer_parameter(module, ttnn::bitwise_left_shift, "shift_bits", "integer within range (0, 31)", "Input tensor needs to be of INT32 dtype. Support provided for Wormhole_B0 only");
    detail::bind_unary_operation_with_integer_parameter(module, ttnn::bitwise_right_shift, "shift_bits", "integer within range (0, 31)", "Input tensor needs to be of INT32 dtype. Support provided for Wormhole_B0 only");
    detail::bind_unary_operation_with_integer_parameter(module, ttnn::bitwise_and, "value", "scalar value", "Input tensor needs to be positive, INT32 dtype. Support provided only for Wormhole_B0.");
    detail::bind_unary_operation_with_integer_parameter(module, ttnn::bitwise_or, "value", "scalar value", "Input tensor needs to be positive, INT32 dtype. Support provided only for Wormhole_B0.");
    detail::bind_unary_operation_with_integer_parameter(module, ttnn::bitwise_xor, "value", "scalar value", "Input tensor needs to be positive, INT32 dtype. Support provided only for Wormhole_B0.");
    detail::bind_unary_operation_with_integer_parameter(module, ttnn::bitwise_not, "value", "scalar value", "Input tensor needs to be in the range [-2147483647, 2147483647], INT32 dtype. Support provided only for Wormhole_B0.");


    // Unary ops with dim parameter
    detail::bind_unary_operation_with_dim_parameter(module, ttnn::glu, "dim", "Dimenstion to split input tensor. Supported dimension -1 or 3", "Split the tensor into two, apply glu function on second tensor followed by mul op with first tensor");
    detail::bind_unary_operation_with_dim_parameter(module, ttnn::reglu, "dim", "Dimenstion to split input tensor. Supported dimension -1 or 3", "Split the tensor into two, apply relu function on second tensor followed by mul op with first tensor");
    detail::bind_unary_operation_with_dim_parameter(module, ttnn::geglu, "dim", "Dimenstion to split input tensor. Supported dimension -1 or 3", "Split the tensor into two, apply gelu function on second tensor followed by mul op with first tensor");
    detail::bind_unary_operation_with_dim_parameter(module, ttnn::swiglu, "dim", "Dimenstion to split input tensor. Supported dimension -1 or 3", "Split the tensor into two, apply silu function on second tensor followed by mul op with first tensor");

    // Other unaries (unary chain operations)
    detail::bind_softplus(module, ttnn::softplus);
    detail::bind_dropout(module, ttnn::dropout);
    detail::bind_sigmoid_accurate(module, ttnn::sigmoid_accurate);
    detail::bind_unary_chain(module, ttnn::unary_chain);
    detail::bind_identity(module, ttnn::identity);
    detail::bind_power(module, ttnn::pow);

// it is only supporetd for range -9 to 9
// not supported for grayskull
    // unary composite imported into ttnn
    detail::bind_unary_composite(module, ttnn::deg2rad, R"doc(Performs deg2rad function on :attr:`input_tensor`.)doc");
    detail::bind_unary_composite(module, ttnn::rad2deg, R"doc(Performs rad2deg function on :attr:`input_tensor`.)doc");
    detail::bind_unary_composite(module, ttnn::tanhshrink, R"doc(Performs tanhshrink function on :attr:`input_tensor`.)doc");
    detail::bind_unary_composite(module, ttnn::acosh, R"doc(Performs acosh function on :attr:`input_tensor`.)doc");
    detail::bind_unary_composite(module, ttnn::asinh, R"doc(Performs asinh function on :attr:`input_tensor`.)doc");
    detail::bind_unary_composite(module, ttnn::atanh, R"doc(Performs atanh function on :attr:`input_tensor`.)doc");
    detail::bind_unary_composite(module, ttnn::cbrt, R"doc(Performs cbrt function on :attr:`input_tensor`.)doc");
    detail::bind_unary_composite(module, ttnn::cosh, R"doc(Performs cosh function on :attr:`input_tensor`.)doc", "[supported range -9 to 9]");
    detail::bind_unary_composite(module, ttnn::digamma, R"doc(Performs digamma function on :attr:`input_tensor`.)doc", "[supported for value greater than 0]");
    detail::bind_unary_composite(module, ttnn::lgamma, R"doc(Performs lgamma function on :attr:`input_tensor`.)doc", "[supported for value greater than 0]");
    detail::bind_unary_composite(module, ttnn::log1p, R"doc(Performs log1p function on :attr:`input_tensor`.)doc",
    R"doc(Supported dtypes, layouts, and ranks:

        +----------------------------+---------------------------------+-------------------+
        |     Dtypes                 |         Layouts                 |     Ranks         |
        +----------------------------+---------------------------------+-------------------+
        |    BFLOAT16                |          TILE                   |      2, 3, 4      |
        +----------------------------+---------------------------------+-------------------+)doc", "[supported range -1 to 1]");
    detail::bind_unary_composite(module, ttnn::mish, R"doc(Performs mish function on :attr:`input_tensor`, not supported for grayskull.)doc");
    detail::bind_unary_composite(module, ttnn::multigammaln, R"doc(Performs multigammaln function on :attr:`input_tensor`.)doc", "[supported range 1.6 to inf]");
    detail::bind_unary_composite(module, ttnn::sinh, R"doc(Performs sinh function on :attr:`input_tensor`.)doc", "[supported range -88 to 88]");
    detail::bind_unary_composite(module, ttnn::softsign, R"doc(Performs softsign function on :attr:`input_tensor`.)doc");
    detail::bind_unary_composite(module, ttnn::swish, R"doc(Performs swish function on :attr:`input_tensor`.)doc");
    detail::bind_unary_composite(module, ttnn::var_hw, R"doc(Performs var_hw function on :attr:`input_tensor`.)doc");
    detail::bind_unary_composite(module, ttnn::std_hw, R"doc(Performs std_hw function on :attr:`input_tensor`.)doc");
    detail::bind_unary_composite(module, ttnn::normalize_hw, R"doc(Performs normalize_hw function on :attr:`input_tensor`.)doc");
    detail::bind_unary_composite(module, ttnn::logical_not_, R"doc(Performs logical_not inplace function on :attr:`input_tensor`.)doc");
    detail::bind_unary_composite(module, ttnn::normalize_global, R"doc(Performs normalize_global function on :attr:`input_tensor`.)doc");
    detail::bind_unary_composite(module, ttnn::frac, R"doc(Performs frac function on :attr:`input_tensor`.)doc");
    detail::bind_unary_composite_operation(module, ttnn::trunc, R"doc(Not supported for grayskull.)doc");

    detail::bind_unary_composite_floats_with_default(
        module,
        ttnn::hardswish,
        "scale", "Scale value", 1.0f/6.0f,
        "shift", "Shift value", 0.5f,
        R"doc(Performs hardswish function on :attr:`input_tensor`, :attr:`scale`, :attr:`shift`.)doc");
    detail::bind_unary_composite_floats_with_default(
        module,
        ttnn::hardsigmoid,
        "scale", "Scale value", 1.0f/6.0f,
        "shift", "Shift value", 0.5f,
        R"doc(Performs hardsigmoid function on :attr:`input_tensor`, :attr:`scale`, :attr:`shift`.)doc");
    detail::bind_unary_composite_floats_with_default(
        module,
        ttnn::hardtanh,
        "min", "min value", -1.0f,
        "max", "max value", 1.0f,
        R"doc(Performs hardtanh function on :attr:`input_tensor`, :attr:`min`, :attr:`max`.)doc");
    detail::bind_unary_composite_floats(
        module,
        ttnn::clip,
        "low", "Low value",
        "high", "High value",
        R"doc(Performs clip function on :attr:`input_tensor`, :attr:`low`, :attr:`high`.)doc");
    detail::bind_unary_composite_floats(
        module,
        ttnn::clamp,
        "low", "Low value",
        "high", "High value",
        R"doc(Performs clamp function on :attr:`input_tensor`, :attr:`low`, :attr:`high`.)doc");
    detail::bind_unary_composite_floats_with_default(
        module,
        ttnn::selu,
        "scale", "Scale value", 1.0507,
        "alpha", "Alpha value", 1.67326,
        R"doc(Performs selu function on :attr:`input_tensor`, :attr:`scale`, :attr:`alpha`.)doc");
    detail::bind_unary_composite_floats(
        module,
        ttnn::threshold,
        "threshold", "Threshold value",
        "value", "Value value",
        R"doc(Performs threshold function on :attr:`input_tensor`, :attr:`threshold`, :attr:`value`.)doc");
    detail::bind_unary_composite_int_with_default(
        module,
        ttnn::tril,
        "diagonal", "diagonal value", 0,
        R"doc(Performs tril function on :attr:`input_tensor`, :attr:`diagonal`.)doc");
    detail::bind_unary_composite_int_with_default(
        module,
        ttnn::triu,
        "diagonal", "diagonal value", 0,
        R"doc(Performs triu function on :attr:`input_tensor`, :attr:`diagonal`.)doc");
    detail::bind_unary_composite_int_with_default(
        module,
        ttnn::round,
        "decimals", "decimals value", 0,
        R"doc(Performs round function on :attr:`input_tensor`, not supported for grayskull, :attr:`decimals`.)doc");
    detail::bind_unary_composite_int(
        module,
        ttnn::polygamma,
        "k", "k value",
        R"doc(Performs polygamma function on :attr:`input_tensor`, :attr:`decimals`. it is supported for range 1 to 10 only)doc");

    // unary composite with float imported into ttnn
    detail::bind_unary_composite_float_with_default(
        module,
        ttnn::hardshrink,
        "lambd", "lambd value", 0.5f,
        R"doc(Performs hardshrink function on :attr:`input_tensor`, :attr:`lambd`.)doc");
    detail::bind_unary_composite_float_with_default(
        module,
        ttnn::softshrink,
        "lambd", "lambd value", 0.5f,
        R"doc(Performs softhrink function on :attr:`input_tensor`, :attr:`lambd`.)doc");
    detail::bind_unary_composite_float_with_default(
        module,
        ttnn::celu,
        "alpha", "alpha value", 1.0f,
        R"doc(Performs celu function on :attr:`input_tensor`, :attr:`alpha`.)doc");
    detail::bind_unary_composite_float_with_default(
        module,
        ttnn::logit,
        "eps", "eps", 0.0f,
        R"doc(Performs logit function on :attr:`input_tensor`, :attr:`eps`. Not available for Wormhole_B0.)doc");
    detail::bind_unary_composite_float(
        module,
        ttnn::rpow,
        "exponent", "exponent value",
        R"doc(Performs rpow function on :attr:`input_tensor`, :attr:`exponent`.)doc");

    detail::bind_unary_rdiv(
    module,
    ttnn::rdiv,
    "value", "denominator value which is actually calculated as numerator float value >= 0",
    "round_mode", "rounding_mode value", "None",
    R"doc(Performs the element-wise division of a scalar ``value`` by a tensor ``input`` and rounds the result using round_mode. Support provided only for Wormhole_B0.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.)doc");

}

}  // namespace unary
}  // namespace operations
}  // namespace ttnn
