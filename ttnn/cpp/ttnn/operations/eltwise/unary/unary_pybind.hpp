// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/eltwise/unary/unary_composite.hpp"
#include "ttnn/types.hpp"

namespace py = pybind11;

namespace ttnn {
namespace operations {
namespace unary {

namespace detail {

template <typename unary_operation_t>
void bind_unary_operation(py::module& module, const unary_operation_t& operation, const std::string& info_doc = "" ) {
    auto doc = fmt::format(
        R"doc({0}(input_tensor: ttnn.Tensor, *, memory_config: Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor

            Applies {0} to :attr:`input_tensor` element-wise.

            {2}

            .. math::
                {0}(\\mathrm{{input\\_tensor}}_i)

            Args:
                * :attr:`input_tensor`

            Keyword Args:
                * :attr:`memory_config` (Optional[ttnn.MemoryConfig]): Memory configuration for the operation.
                * :attr:`output_tensor` (Optional[ttnn.Tensor]): preallocated output tensor
                * :attr:`queue_id` (Optional[uint8]): command queue id

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
            py::arg("queue_id") = 0});
}


template <typename unary_operation_t>
void bind_unary_operation_with_fast_and_approximate_mode(py::module& module, const unary_operation_t& operation) {
    auto doc = fmt::format(
        R"doc({0}(input_tensor: ttnn.Tensor, *, fast_and_approximate_mode: bool = False, memory_config: Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor

            Applies {0} to :attr:`input_tensor` element-wise.

            .. math::
                {0}(\\mathrm{{input\\_tensor}}_i)

            Args:
                * :attr:`input_tensor`

            Keyword Args:
                * :attr:`fast_and_approximate_mode` (bool): "Use fast and approximate mode".
                * :attr:`memory_config` (Optional[ttnn.MemoryConfig]): Memory configuration for the operation.
                * :attr:`output_tensor` (Optional[ttnn.Tensor]): preallocated output tensor
                * :attr:`queue_id` (Optional[uint8]): command queue id

            Example:

                >>> tensor = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
                >>> output = {1}(tensor, fast_and_approximate_mode=true)
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
        R"doc({0}(input_tensor: ttnn.Tensor, *, {2}: float, memory_config: Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor

            Applies {0} to :attr:`input_tensor` element-wise.

            {4}

            .. math::
                {0}(\\mathrm{{input\\_tensor}}_i)

            Args:
                * :attr:`input_tensor`

            Keyword Args:
                * :attr:`{2}` (float): {3}.
                * :attr:`memory_config` (Optional[ttnn.MemoryConfig]): Memory configuration for the operation.
                * :attr:`output_tensor` (Optional[ttnn.Tensor]): preallocated output tensor
                * :attr:`queue_id` (Optional[uint8]): command queue id

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
        R"doc({0}(input_tensor: ttnn.Tensor, *, {2}: int32_t, memory_config: Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor

            Applies {0} to :attr:`input_tensor` element-wise.

            {4}

            .. math::
                {0}(\\mathrm{{input\\_tensor}}_i)

            Args:
                * :attr:`input_tensor`

            Keyword Args:
                * :attr:`{2}` (int32_t): {3}.
                * :attr:`memory_config` (Optional[ttnn.MemoryConfig]): Memory configuration for the operation.
                * :attr:`output_tensor` (Optional[ttnn.Tensor]): preallocated output tensor
                * :attr:`queue_id` (Optional[uint8]): command queue id

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
void bind_softplus(py::module& module, const unary_operation_t& operation) {
    auto doc = fmt::format(
        R"doc({0}(input_tensor: ttnn.Tensor, *, beta: float = 1.0, threshold: float = 20.0, memory_config: Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor

            Applies {0} to :attr:`input_tensor` element-wise.

            .. math::
                {0}(\\mathrm{{input\\_tensor}}_i)

            Args:
                * :attr:`input_tensor`

            Keyword Args:
                * :attr:`beta` (float): Scales the input before applying the Softplus function. By modifying beta, you can adjust the steepness of the function. A higher beta value makes the function steeper, approaching a hard threshold like the ReLU function for large values of beta
                * :attr:`threshold` (float): Used to switch to a linear function for large values to improve numerical stability. This avoids issues with floating-point representation for very large values
                * :attr:`memory_config` (Optional[ttnn.MemoryConfig]): Memory configuration for the operation.
                * :attr:`output_tensor` (Optional[ttnn.Tensor]): preallocated output tensor
                * :attr:`queue_id` (Optional[uint8]): command queue id

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
        R"doc({0}(input_tensor: ttnn.Tensor, memory_config: Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor

            Applies {0} to :attr:`input_tensor` element-wise.

            .. math::
                {0}(\\mathrm{{input\\_tensor}}_i)

            Args:
                * :attr:`input_tensor`

            Keyword Args:
                * :attr:`memory_config` (Optional[ttnn.MemoryConfig]): Memory configuration for the operation.
                * :attr:`output_tensor` (Optional[ttnn.Tensor]): preallocated output tensor
                * :attr:`queue_id` (Optional[uint8]): command queue id

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
        R"doc({0}(input_tensor: ttnn.Tensor, ops_chain: std::vector<UnaryWithParam>, memory_config: Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor

            Applies {0} to :attr:`input_tensor` element-wise.

            .. math::
                {0}(\\mathrm{{input\\_tensor}}_i)

            Args:
                * :attr:`input_tensor`

            Keyword Args:
                * :attr:`ops_chain` (vector): Vector of unary ops chain
                * :attr:`memory_config` (Optional[ttnn.MemoryConfig]): Memory configuration for the operation.
                * :attr:`output_tensor` (Optional[ttnn.Tensor]): preallocated output tensor
                * :attr:`queue_id` (Optional[uint8]): command queue id

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
               const std::vector<UnaryWithParam>& ops_chain,
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
        R"doc({0}(input_tensor: ttnn.Tensor, memory_config: Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor

            Returns a copy of same tensor ``input``; useful for profiling the SFPU.
            this shouldn't normally be used; users should normally use clone operation instead for same functionality as this would be lower performance.

            .. math::
                {0}(\\mathrm{{input\\_tensor}}_i)

            Args:
                * :attr:`input_tensor`

            Keyword Args:
                * :attr:`memory_config` (Optional[ttnn.MemoryConfig]): Memory configuration for the operation.
                * :attr:`output_tensor` (Optional[ttnn.Tensor]): preallocated output tensor
                * :attr:`queue_id` (Optional[uint8]): command queue id

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
        R"doc({0}(input_tensor: ttnn.Tensor, exponent:  Union[float, int], memory_config: Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor

            Applies {0} to :attr:`input_tensor` element-wise.

            .. math::
                {0}(\\mathrm{{input\\_tensor}}_i)

            Args:
                * :attr:`input_tensor`

            Keyword Args:
                * :attr:`exponent` ( Union[float, int]): exponent is an integer >= 0
                * :attr:`memory_config` (Optional[ttnn.MemoryConfig]): Memory configuration for the operation.
                * :attr:`output_tensor` (Optional[ttnn.Tensor]): preallocated output tensor
                * :attr:`queue_id` (Optional[uint8]): command queue id

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
               const std::optional<Tensor>& output_tensor,
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
                const std::optional<Tensor>& output_tensor,
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
void bind_unary_composite_operation(py::module& module, const unary_operation_t& operation) {
    auto doc = fmt::format(
        R"doc({0}(input_tensor: ttnn.Tensor, *, memory_config: Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor

            Applies {0} to :attr:`input_tensor` element-wise.

            .. math::
                {0}(\\mathrm{{input\\_tensor}}_i)

            Args:
                * :attr:`input_tensor`

            Keyword Args:
                * :attr:`memory_config` (Optional[ttnn.MemoryConfig]): Memory configuration for the operation.
                * :attr:`output_tensor` (Optional[ttnn.Tensor]): preallocated output tensor
                * :attr:`queue_id` (Optional[uint8]): command queue id

            Example:

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
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor,
               const uint8_t& queue_id) {
                    return self(queue_id, input_tensor, memory_config, output_tensor);
                },
            py::arg("input_tensor"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("output_tensor") = std::nullopt,
            py::arg("queue_id") = 0});
}


template <typename unary_operation_t>
void bind_unary_operation_with_scale_and_shift(py::module& module, const unary_operation_t& operation) {
    auto doc = fmt::format(
        R"doc({0}(input_tensor: ttnn.Tensor, scale, shift, *, memory_config: Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor

            Applies {0} to :attr:`input_tensor` element-wise.

            .. math::
                {0}(\\mathrm{{input\\_tensor}}_i)

            Args:
                * :attr:`input_tensor`
                * :attr:`scale`
                * :attr:`shift`

            Keyword Args:
                * :attr:`memory_config` (Optional[ttnn.MemoryConfig]): Memory configuration for the operation.
                * :attr:`output_tensor` (Optional[ttnn.Tensor]): preallocated output tensor
                * :attr:`queue_id` (Optional[uint8]): command queue id

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
        R"doc({0}(input_tensor: ttnn.Tensor, low, high, *, memory_config: Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor

            Applies {0} to :attr:`input_tensor` element-wise.

            .. math::
                {0}(\\mathrm{{input\\_tensor}}_i)

            Args:
                * :attr:`input_tensor`
                * :attr:`low`
                * :attr:`high`

            Keyword Args:
                * :attr:`memory_config` (Optional[ttnn.MemoryConfig]): Memory configuration for the operation.
                * :attr:`output_tensor` (Optional[ttnn.Tensor]): preallocated output tensor
                * :attr:`queue_id` (Optional[uint8]): command queue id

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
        R"doc({0}(input_tensor: ttnn.Tensor, diag, *, memory_config: Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor

            Applies {0} to :attr:`input_tensor` element-wise.

            .. math::
                {0}(\\mathrm{{input\\_tensor}}_i)

            Args:
                * :attr:`input_tensor`
                * :attr:`diag`

            Keyword Args:
                * :attr:`memory_config` (Optional[ttnn.MemoryConfig]): Memory configuration for the operation.
                * :attr:`output_tensor` (Optional[ttnn.Tensor]): preallocated output tensor
                * :attr:`queue_id` (Optional[uint8]): command queue id

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

}  // namespace detail

void py_module(py::module& module) {
    detail::bind_unary_operation(module, ttnn::abs);
    detail::bind_unary_operation(module, ttnn::acos);
    detail::bind_unary_operation(module, ttnn::asin);
    detail::bind_unary_operation(module, ttnn::atan);
    detail::bind_unary_operation(module, ttnn::cos);
    detail::bind_unary_operation(module, ttnn::erfinv);
    detail::bind_unary_operation(module, ttnn::exp2);
    detail::bind_unary_operation(module, ttnn::expm1);
    detail::bind_unary_operation(module, ttnn::eqz);
    detail::bind_unary_operation(module, ttnn::floor, "Available for Wormhole_B0 only");
    detail::bind_unary_operation(module, ttnn::ceil, "Available for Wormhole_B0 only");
    detail::bind_unary_operation(module, ttnn::gez);
    detail::bind_unary_operation(module, ttnn::gtz);
    detail::bind_unary_operation(module, ttnn::i0);
    detail::bind_unary_operation(module, ttnn::isfinite);
    detail::bind_unary_operation(module, ttnn::isinf);
    detail::bind_unary_operation(module, ttnn::isnan);
    detail::bind_unary_operation(module, ttnn::isneginf);
    detail::bind_unary_operation(module, ttnn::isposinf);
    detail::bind_unary_operation(module, ttnn::lez);
    detail::bind_unary_operation(module, ttnn::log);
    detail::bind_unary_operation(module, ttnn::log10);
    detail::bind_unary_operation(module, ttnn::log2);
    detail::bind_unary_operation(module, ttnn::logical_not);
    detail::bind_unary_operation(module, ttnn::ltz);
    detail::bind_unary_operation(module, ttnn::neg);
    detail::bind_unary_operation(module, ttnn::nez);
    detail::bind_unary_operation(module, ttnn::reciprocal);
    detail::bind_unary_operation(module, ttnn::relu);
    detail::bind_unary_operation(module, ttnn::relu6);
    detail::bind_unary_operation(module, ttnn::sigmoid);
    detail::bind_unary_operation(module, ttnn::sign);
    detail::bind_unary_operation(module, ttnn::signbit);
    detail::bind_unary_operation(module, ttnn::silu);
    detail::bind_unary_operation(module, ttnn::sin);
    detail::bind_unary_operation(module, ttnn::sqrt);
    detail::bind_unary_operation(module, ttnn::square);
    detail::bind_unary_operation(module, ttnn::tan);
    detail::bind_unary_operation(module, ttnn::tanh);
    detail::bind_unary_operation(module, ttnn::log_sigmoid);

    //  Unaries with fast_and_approximate_mode
    detail::bind_unary_operation_with_fast_and_approximate_mode(module, ttnn::exp);
    detail::bind_unary_operation_with_fast_and_approximate_mode(module, ttnn::erf);
    detail::bind_unary_operation_with_fast_and_approximate_mode(module, ttnn::erfc);
    detail::bind_unary_operation_with_fast_and_approximate_mode(module, ttnn::gelu);
    detail::bind_unary_operation_with_fast_and_approximate_mode(module, ttnn::rsqrt);

    // Unaries with float parameter
    detail::bind_unary_operation_with_float_parameter(module, ttnn::elu, "alpha", "The alpha parameter for the ELU function", "");
    detail::bind_unary_operation_with_float_parameter(module, ttnn::rsub, "value", "subtrahent value which is actually calculated as minuend", "Returns tensor with respective elements of the input tensor subtracted from the value.");
    detail::bind_unary_operation_with_float_parameter(module, ttnn::rdiv, "value", "denominator value which is actually calculated as numerator float value >= 0 ", "Returns tensor  with scalar value divided by each of respective elements of the input tensor.");
    detail::bind_unary_operation_with_float_parameter(module, ttnn::heaviside, "value", "The value parameter for the Heaviside function", "");
    detail::bind_unary_operation_with_float_parameter(module, ttnn::leaky_relu, "slope", "The slope parameter for the Leaky ReLU function", "");
    detail::bind_unary_operation_with_float_parameter(module, ttnn::relu_max, "upper_limit", "The max value for ReLU function", "This function caps off the input to a max value and a min value of 0");
    detail::bind_unary_operation_with_float_parameter(module, ttnn::relu_min, "lower_limit", "The min value for ReLU function", "This will carry out ReLU operation at min value instead of the standard 0");
    detail::bind_unary_operation_with_float_parameter(module, ttnn::remainder, "value", "Perform an eltwise-modulus operation a - a.div(b, rounding_mode=floor) * b.", "Support provided only for WH_B0.");
    detail::bind_unary_operation_with_float_parameter(module, ttnn::fmod, "value", "scalar value", "Perform an eltwise-fmod operation. Formula : a - a.div(b, rounding_mode=trunc) * b . Support provided only for WH_B0.");
    // detail::bind_unary_operation_with_float_parameter(module, ttnn::prelu, "weight", "The weight parameter for the PReLU function");

    // Unaries with integer parameter
    detail::bind_unary_operation_with_integer_parameter(module, ttnn::bitwise_left_shift, "shift_bits", "integer within range (0, 31)", "Input tensor needs to be of INT32 dtype. Support provided for Wormhole_B0 only");
    detail::bind_unary_operation_with_integer_parameter(module, ttnn::bitwise_right_shift, "shift_bits", "integer within range (0, 31)", "Input tensor needs to be of INT32 dtype. Support provided for Wormhole_B0 only");
    detail::bind_unary_operation_with_integer_parameter(module, ttnn::bitwise_and, "value", "scalar value", "Input tensor needs to be positive, INT32 dtype. Support provided only for Wormhole_B0.");
    detail::bind_unary_operation_with_integer_parameter(module, ttnn::bitwise_or, "value", "scalar value", "Input tensor needs to be positive, INT32 dtype. Support provided only for Wormhole_B0.");
    detail::bind_unary_operation_with_integer_parameter(module, ttnn::bitwise_xor, "value", "scalar value", "Input tensor needs to be positive, INT32 dtype. Support provided only for Wormhole_B0.");
    detail::bind_unary_operation_with_integer_parameter(module, ttnn::bitwise_not, "value", "scalar value", "Input tensor needs to be in the range [-2147483647, 2147483647], INT32 dtype. Support provided only for Wormhole_B0.");


    // Other unaries (unary chain operations)
    detail::bind_softplus(module, ttnn::softplus);
    detail::bind_sigmoid_accurate(module, ttnn::sigmoid_accurate);
    detail::bind_unary_chain(module, ttnn::unary_chain);
    detail::bind_identity(module, ttnn::identity);
    detail::bind_power(module, ttnn::pow);

    // Other Unary composite
    detail::bind_unary_composite_operation(module, ttnn::rad2deg);
    detail::bind_unary_composite_operation(module, ttnn::deg2rad);

    detail::bind_unary_composite_operation(module, ttnn::acosh);
    detail::bind_unary_composite_operation(module, ttnn::asinh);
    detail::bind_unary_composite_operation(module, ttnn::atanh);
    detail::bind_unary_composite_operation(module, ttnn::cbrt);
    detail::bind_unary_composite_operation(module, ttnn::cosh);
    detail::bind_unary_composite_operation(module, ttnn::digamma);
    detail::bind_unary_operation_with_scale_and_shift(module, ttnn::hardswish);
    detail::bind_unary_operation_with_scale_and_shift(module, ttnn::hardsigmoid);
    detail::bind_unary_operation_with_low_and_high(module, ttnn::hardtanh);
    detail::bind_unary_composite_operation(module, ttnn::lgamma);
    detail::bind_unary_composite_operation(module, ttnn::log1p);
    detail::bind_unary_composite_operation(module, ttnn::mish);
    detail::bind_unary_composite_operation(module, ttnn::multigammaln);
    detail::bind_unary_composite_operation(module, ttnn::sinh);
    detail::bind_unary_composite_operation(module, ttnn::softsign);
    detail::bind_unary_composite_operation(module, ttnn::swish);
    detail::bind_unary_composite_operation(module, ttnn::tanhshrink);
    detail::bind_unary_operation_with_diag(module, ttnn::tril);
    detail::bind_unary_operation_with_diag(module, ttnn::triu);
}

}  // namespace unary
}  // namespace operations
}  // namespace ttnn
