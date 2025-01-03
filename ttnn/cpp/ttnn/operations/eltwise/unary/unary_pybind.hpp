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
void bind_unary_composite_optional_floats_with_default(
    py::module& module,
    const unary_operation_t& operation,
    const std::string& parameter_name_a,
    const std::string& parameter_a_doc,
    std::optional<float> parameter_a_value,
    const std::string& parameter_name_b,
    const std::string& parameter_b_doc,
    std::optional<float> parameter_b_value,
    const std::string& description,
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& note="") {
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

        Example:
            >>> input_tensor = ttnn.from_torch(torch.tensor([[1, 2], [3,4]], dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            >>> min_tensor = ttnn.from_torch(torch.tensor([[0, 2], [0,4]], dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            >>> max_tensor = ttnn.from_torch(torch.tensor([[1, 2], [3,4]], dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            >>> output = {1}(input_tensor, min_tensor, max_tensor)

            >>> input_tensor = ttnn.from_torch(torch.tensor([[1, 2], [3,4]], dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            >>> output = {1}(input_tensor, min = 2, max = 9)
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
        supported_dtype, note);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const unary_operation_t& self,
               const ttnn::Tensor& input_tensor,
               std::optional<Tensor> parameter_a,
               std::optional<Tensor> parameter_b,
               const std::optional<MemoryConfig>& memory_config)  {
                return self(input_tensor, parameter_a, parameter_b, memory_config);
            },
            py::arg("input_tensor"),
            py::arg(parameter_name_a.c_str()) = parameter_a_value,
            py::arg(parameter_name_b.c_str()) = parameter_b_value,
            py::kw_only(),
            py::arg("memory_config") = std::nullopt},

        ttnn::pybind_overload_t{
            [](const unary_operation_t& self,
               const ttnn::Tensor& input_tensor,
               std::optional<float> parameter_a,
               std::optional<float> parameter_b,
               const std::optional<MemoryConfig>& memory_config)  {
                return self(input_tensor, parameter_a, parameter_b, memory_config);
            },
            py::arg("input_tensor"),
            py::arg(parameter_name_a.c_str()) = parameter_a_value,
            py::arg(parameter_name_b.c_str()) = parameter_b_value,
            py::kw_only(),
            py::arg("memory_config") = std::nullopt});
}

template <typename unary_operation_t>
void bind_unary_operation(
    py::module& module,
    const unary_operation_t& operation,
    const std::string& math,
    const std::string& supported_dtype ="BFLOAT16",
    const std::string& note = "",
    const std::string& example_tensor = "torch.rand([2, 2], dtype=torch.bfloat16)") {
    auto doc = fmt::format(
        R"doc(
        Applies {0} to :attr:`input_tensor` element-wise.

        .. math::
            {2}

        Args:
            input_tensor (ttnn.Tensor): the input tensor.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the operation. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): preallocated output tensor. Defaults to `None`.
            queue_id (int, optional): command queue id. Defaults to `0`.

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

        Example:
            >>> tensor = ttnn.from_torch({5}, layout=ttnn.TILE_LAYOUT, device=device)
            >>> output = {1}(tensor)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        math,
        supported_dtype,
        note,
        example_tensor);

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
void bind_unary_operation_overload_complex(
    py::module& module,
    const unary_operation_t& operation,
    const std::string& math,
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& note = "" ) {
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
            queue_id (int, optional): command queue id. Defaults to `0`.

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

        Example:
            >>> tensor = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> output = {1}(tensor)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        math,
        supported_dtype,
        note);

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
void bind_unary_operation_overload_complex_return_complex(
    py::module& module,
    const unary_operation_t& operation,
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& info_doc = "" ) {
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
            queue_id (int, optional): command queue id. Defaults to `0`.

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

        Example:
            >>> tensor = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> output = {1}(tensor)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        supported_dtype,
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
void bind_unary_operation_with_fast_and_approximate_mode(py::module& module, const unary_operation_t& operation, const std::string& supported_dtype = "BFLOAT16", const std::string& info_doc = "" ) {
    auto doc = fmt::format(
        R"doc(
        Applies {0} to :attr:`input_tensor` element-wise.

        .. math::
            \mathrm{{output\_tensor}}_i = {0}(\mathrm{{input\_tensor}}_i)

        Args:
            input_tensor (ttnn.Tensor): the input tensor.

        Keyword Args:
            fast_and_approximate_mode (bool, optional): Use the fast and approximate mode. Defaults to `False`.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): preallocated output tensor. Defaults to `None`.
            queue_id (int, optional): command queue id. Defaults to `0`.

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

        Example:
            >>> tensor = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            >>> output = {1}(tensor, fast_and_approximate_mode=True)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        supported_dtype,
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
            queue_id (int, optional): command queue id. Defaults to `0`.

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

        Example:
            >>> tensor = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            >>> {2} = 3
            >>> output = {1}(tensor, {2})
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        parameter_name,
        parameter_doc,
        info_doc,
        supported_dtype,
        note);

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
void bind_unary_operation_with_dim_parameter(
    py::module& module,
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

        Example:
            >>> tensor = ttnn.from_torch(torch.rand([1, 1, 32, 64], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> {2} = 3
            >>> output = {1}(tensor, {2})
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        parameter_name,
        parameter_doc,
        info_doc,
        supported_dtype,
        note);

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
void bind_unary_rdiv(
    py::module& module,
    const unary_operation_t& operation,
    const std::string& parameter_name_a,
    const std::string& parameter_a_doc,
    const std::string& parameter_name_b,
    const std::string& parameter_b_doc,
    const std::string parameter_b_value,
    const std::string& description,
    const std::string& supported_dtype = "BFLOAT16",
    const std::string& note = "") {
    auto doc = fmt::format(
        R"doc(
        {7}

        Args:
            input_tensor (ttnn.Tensor): the input tensor.
            {2} (int): {3}.

        Keyword Args:
            {4} (string): {5}. Can be  None, "trunc", "floor". Defaults to `None`.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): preallocated output tensor. Defaults to `None`.
            queue_id (int, optional): command queue id. Defaults to `0`.

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

        Example:
            >>> tensor = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            >>> {2} = 2
            >>> output = {1}(tensor, {2}, {4} = None)
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
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const unary_operation_t& self,
               const ttnn::Tensor& input_tensor,
               float parameter_a,
               const std::optional<std::string> parameter_b,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor,
               const uint8_t& queue_id) {
                return self(queue_id, input_tensor, parameter_a, parameter_b, memory_config, output_tensor);
            },
            py::arg("input_tensor"),
            py::arg(parameter_name_a.c_str()),
            py::kw_only(),
            py::arg(parameter_name_b.c_str()) = std::nullopt,
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
            beta (float, optional): Scales the input before applying the Softplus function. By modifying :attr:`beta`, you can adjust the steepness of the function. A higher :attr:`beta` value makes the function steeper, approaching a hard threshold like the ReLU function for large values of :attr:`beta`. Defaults to `1`.
            threshold (float, optional): Used to switch to a linear function for large values to improve numerical stability. This avoids issues with floating-point representation for very large values. Defaults to `20`.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): preallocated output tensor. Defaults to `None`.
            queue_id (int, optional): command queue id. Defaults to `0`.

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

        Example:
            >>> tensor = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            >>> output = {1}(tensor, beta = 1.0, threshold = 20.0)
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
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the operation. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): preallocated output tensor. Defaults to `None`.
            queue_id (int, optional): command queue id. Defaults to `0`.

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

        Example:
            >>> tensor = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
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
            \mathrm{{output\_tensor}}_i = \verb|{0}|(\mathrm{{input\_tensor}}_i)

        Args:
            input_tensor (ttnn.Tensor): the input tensor.
            ops_chain (list[ttnn.UnaryWithParam]): list of unary ops to be chained.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the operation. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): preallocated output tensor. Defaults to `None`.
            queue_id (int, optional): command queue id. Defaults to `0`.

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

        Example:

            >>> tensor = ttnn.from_torch(torch.randn([32, 32], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> ops_chain = [ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU), ttnn.UnaryWithParam(ttnn.UnaryOpType.EXP, False), ttnn.UnaryWithParam(ttnn.UnaryOpType.POWER, 2)]
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
        Returns a copy of the :attr:`input_tensor`; useful for profiling the SFPU.
        This shouldn't normally be used. Users should normally use clone operation instead for the same functionality since this results in lower performance.

        .. math::
            \mathrm{{output\_tensor}}_i = \verb|{0}|(\mathrm{{input\_tensor}}_i)

        Args:
            input_tensor (ttnn.Tensor): the input tensor.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the operation. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): preallocated output tensor. Defaults to `None`.
            queue_id (int, optional): command queue id. Defaults to `0`.

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

        Example:
            >>> tensor = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.float16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
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
void bind_unary_composite(
    py::module& module,
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

        Example:
            >>> tensor = ttnn.from_torch({8}, layout=ttnn.TILE_LAYOUT, device=device)
            >>> output = {1}(tensor)
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
void bind_unary_composite_int_with_default(
    py::module& module,
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

        Example:
            >>> tensor = ttnn.from_torch(torch.rand([2, 2], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> output = {1}(tensor, {2} = {4})
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
void bind_unary_composite_floats_with_default(
    py::module& module,
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

        Example:
            >>> tensor = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
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
        supported_dtype,
        info_doc);

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

template <typename unary_operation_t>
void bind_hardtanh(
    py::module& module,
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
            {2} (float): {3}. Defaults to `{4}`.
            {5} (float): {6}. Defaults to `{7}`.

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
               * - {8}
                 - TILE
                 - 2, 3, 4

            {9}

        Example:
            >>> tensor = ttnn.from_torch(input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            >>> min = 2
            >>> max = 8
            >>> output = {1}(tensor, min, max)
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
            py::arg(parameter_name_a.c_str()) = parameter_a_value,
            py::arg(parameter_name_b.c_str()) = parameter_b_value,
            py::kw_only(),
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

        Example:
            >>> tensor = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            >>> output = {1}(tensor, 3)
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
void bind_unary_composite_threshold(
    py::module& module,
    const unary_operation_t& operation,
    const std::string& parameter_name_a,
    const std::string& parameter_a_doc,
    const std::string& parameter_name_b,
    const std::string& parameter_b_doc,
    const std::string& description) {
    auto doc = fmt::format(
        R"doc(
        {6}

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

        Example:
            >>> tensor = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            >>> {2} = 1.0
            >>> {4} = 10.0
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
void bind_unary_composite_trunc(py::module& module, const unary_operation_t& operation, const std::string& description) {
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

        Example:
            >>> tensor = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
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
void bind_unary_composite_float_with_default(
    py::module& module,
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

        Example:
            >>> tensor = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            >>> output = {1}(tensor, {2} = 5)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        parameter_name_a,
        parameter_a_doc,
        parameter_a_value,
        supported_dtype,
        info_doc);

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
void bind_unary_composite_rpow(
    py::module& module,
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

        Example:
            >>> tensor = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            >>> exponent = 2
            >>> output = {1}(tensor, exponent)
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



}  // namespace detail

void py_module(py::module& module) {
    detail::bind_unary_operation_overload_complex(module, ttnn::abs, R"doc(\mathrm{{output\_tensor}}_i = \verb|abs|(\mathrm{{input\_tensor}}_i))doc", R"doc(BFLOAT16, BFLOAT8_B)doc");
    detail::bind_unary_operation(module, ttnn::acos, R"doc(\mathrm{{output\_tensor}}_i = \verb|acos|(\mathrm{{input\_tensor}}_i))doc", R"doc(BFLOAT16, BFLOAT8_B)doc");
    detail::bind_unary_operation(module, ttnn::asin, R"doc(\mathrm{{output\_tensor}}_i = \verb|asin|(\mathrm{{input\_tensor}}_i))doc", R"doc(BFLOAT16, BFLOAT8_B)doc");
    detail::bind_unary_operation(module, ttnn::atan, R"doc(\mathrm{{output\_tensor}}_i = \verb|atan|(\mathrm{{input\_tensor}}_i))doc", R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_unary_operation(module, ttnn::cos, R"doc(\mathrm{{output\_tensor}}_i = \verb|cos|(\mathrm{{input\_tensor}}_i))doc", R"doc(BFLOAT16, BFLOAT8_B)doc");
    detail::bind_unary_operation(module, ttnn::erfinv, R"doc(\mathrm{{output\_tensor}}_i = \verb|erfinv|(\mathrm{{input\_tensor}}_i))doc", R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_unary_operation(module, ttnn::exp2, R"doc(\mathrm{{output\_tensor}}_i = \verb|exp2|(\mathrm{{input\_tensor}}_i))doc", R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_unary_operation(module, ttnn::expm1, R"doc(\mathrm{{output\_tensor}}_i = \verb|expm1|(\mathrm{{input\_tensor}}_i))doc", R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_unary_operation(module, ttnn::eqz, R"doc(\mathrm{{output\_tensor}}_i = (\mathrm{{input\_tensor_i\ == 0}}))doc");
    detail::bind_unary_operation(module, ttnn::floor, R"doc(\mathrm{{output\_tensor}}_i = \verb|floor|(\mathrm{{input\_tensor}}_i))doc", R"doc(BFLOAT16, BFLOAT8_B)doc", R"doc(Supported only for Wormhole_B0.)doc");
    detail::bind_unary_operation(module, ttnn::ceil, R"doc(\mathrm{{output\_tensor}}_i = \verb|ceil|(\mathrm{{input\_tensor}}_i))doc", R"doc(BFLOAT16, BFLOAT8_B)doc", R"doc(Supported only for Wormhole_B0.)doc");
    detail::bind_unary_operation(module, ttnn::gez, R"doc(\mathrm{{output\_tensor}}_i = (\mathrm{{input\_tensor_i\ >= 0}}))doc", R"doc(BFLOAT16, BFLOAT8_B)doc");
    detail::bind_unary_operation(module, ttnn::gtz, R"doc(\mathrm{{output\_tensor}}_i= (\mathrm{{input\_tensor_i\ > 0}}))doc", R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_unary_operation(module, ttnn::i0, R"doc(\mathrm{{output\_tensor}}_i = \verb|i0|(\mathrm{{input\_tensor}}_i))doc", R"doc(BFLOAT16, BFLOAT8_B)doc");
    detail::bind_unary_operation(module, ttnn::i1, R"doc(\mathrm{{output\_tensor}}_i = \verb|i1|(\mathrm{{input\_tensor}}_i))doc", R"doc(BFLOAT16, BFLOAT8_B)doc");
    detail::bind_unary_operation(module, ttnn::isfinite, R"doc(\mathrm{{output\_tensor}}_i = \verb|isfinite|(\mathrm{{input\_tensor}}_i))doc", R"doc(BFLOAT16, BFLOAT8_B)doc");
    detail::bind_unary_operation(module, ttnn::isinf, R"doc(\mathrm{{output\_tensor}}_i = \verb|isinf|(\mathrm{{input\_tensor}}_i))doc", R"doc(BFLOAT16, BFLOAT8_B)doc");
    detail::bind_unary_operation(module, ttnn::isnan, R"doc(\mathrm{{output\_tensor}}_i = \verb|isnan|(\mathrm{{input\_tensor}}_i))doc", R"doc(BFLOAT16, BFLOAT8_B)doc");
    detail::bind_unary_operation(module, ttnn::isneginf, R"doc(\mathrm{{output\_tensor}}_i = \verb|isneginf|(\mathrm{{input\_tensor}}_i))doc", R"doc(BFLOAT16, BFLOAT8_B)doc");
    detail::bind_unary_operation(module, ttnn::isposinf, R"doc(\mathrm{{output\_tensor}}_i = \verb|isposinf|(\mathrm{{input\_tensor}}_i))doc", R"doc(BFLOAT16, BFLOAT8_B)doc");
    detail::bind_unary_operation(module, ttnn::lez, R"doc(\mathrm{{output\_tensor}}_i = (\mathrm{{input\_tensor_i\ <= 0}}))doc", R"doc(BFLOAT16, BFLOAT8_B)doc");
    detail::bind_unary_operation(module, ttnn::log, R"doc(\mathrm{{output\_tensor}}_i = \verb|log|(\mathrm{{input\_tensor}}_i))doc", R"doc(BFLOAT16, BFLOAT8_B)doc");
    detail::bind_unary_operation(module, ttnn::log10, R"doc(\mathrm{{output\_tensor}}_i = \verb|log10|(\mathrm{{input\_tensor}}_i))doc", R"doc(BFLOAT16, BFLOAT8_B)doc", R"doc(BFLOAT8_B is only supported in WHB0.)doc");
    detail::bind_unary_operation(module, ttnn::log2, R"doc(\mathrm{{output\_tensor}}_i = \verb|log2|(\mathrm{{input\_tensor}}_i))doc", R"doc(BFLOAT16, BFLOAT8_B)doc", R"doc(BFLOAT8_B is only supported in WHB0.)doc");
    detail::bind_unary_operation(module, ttnn::logical_not, R"doc(\mathrm{{output\_tensor}}_i = \mathrm{{!input\_tensor_i}})doc", R"doc(BFLOAT16, BFLOAT8_B)doc");
    detail::bind_unary_operation(module, ttnn::ltz, R"doc(\mathrm{{output\_tensor}}_i = (\mathrm{{input\_tensor_i\ < 0}}))doc", R"doc(BFLOAT16, BFLOAT8_B)doc");
    detail::bind_unary_operation(module, ttnn::neg, R"doc(\mathrm{{output\_tensor}}_i = \verb|neg|(\mathrm{{input\_tensor}}_i))doc", R"doc(BFLOAT16, BFLOAT8_B)doc");
    detail::bind_unary_operation(module, ttnn::nez, R"doc(\mathrm{{output\_tensor}}_i = (\mathrm{{input\_tensor_i\ != 0}}))doc", R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_unary_operation_overload_complex_return_complex(module, ttnn::reciprocal, R"doc(BFLOAT16, BFLOAT8_B)doc", "bfloat8_b is supported only for non-zero inputs. Inputs containing zero may produce inaccurate results due to the characteristics of the block-FP format. In the block-FP format, 16 consecutive numbers share the same exponent, making it impossible to process them separately. This format isn't ideal when the numbers in the group are very different from each other");

    detail::bind_unary_operation(module, ttnn::relu, R"doc(\mathrm{{output\_tensor}}_i = \verb|relu|(\mathrm{{input\_tensor}}_i))doc", R"doc(BFLOAT16, BFLOAT8_B)doc");
    detail::bind_unary_operation(module, ttnn::relu6, R"doc(\mathrm{{output\_tensor}}_i = \verb|relu6|(\mathrm{{input\_tensor}}_i))doc", R"doc(BFLOAT16, BFLOAT8_B)doc");
    detail::bind_unary_operation(module, ttnn::sigmoid, R"doc(\mathrm{{output\_tensor}}_i = \verb|sigmoid|(\mathrm{{input\_tensor}}_i))doc", R"doc(BFLOAT16, BFLOAT8_B)doc");
    detail::bind_unary_operation(module, ttnn::sign, R"doc(\mathrm{{output\_tensor}}_i = \verb|sign|(\mathrm{{input\_tensor}}_i))doc", R"doc(BFLOAT16, BFLOAT8_B)doc");
    detail::bind_unary_operation(module, ttnn::signbit, R"doc(\mathrm{{output\_tensor}}_i = \verb|signbit|(\mathrm{{input\_tensor}}_i))doc");
    detail::bind_unary_operation(module, ttnn::silu, R"doc(\mathrm{{output\_tensor}}_i = \verb|silu|(\mathrm{{input\_tensor}}_i))doc", R"doc(BFLOAT16, BFLOAT8_B)doc");
    detail::bind_unary_operation(module, ttnn::sin, R"doc(\mathrm{{output\_tensor}}_i = \verb|sin|(\mathrm{{input\_tensor}}_i))doc", R"doc(BFLOAT16, BFLOAT8_B)doc");
    detail::bind_unary_operation(module, ttnn::sqrt, R"doc(\mathrm{{output\_tensor}}_i = \verb|sqrt|(\mathrm{{input\_tensor}}_i))doc", R"doc(BFLOAT16, BFLOAT8_B)doc");
    detail::bind_unary_operation(module, ttnn::square, R"doc(\mathrm{{output\_tensor}}_i = \verb|square|(\mathrm{{input\_tensor}}_i))doc", R"doc(BFLOAT16, BFLOAT8_B)doc");
    detail::bind_unary_operation(module, ttnn::tan, R"doc(\mathrm{{output\_tensor}}_i = \verb|tan|(\mathrm{{input\_tensor}}_i))doc", R"doc(BFLOAT16, BFLOAT8_B)doc" , "Supported input range is (-1.45, 1.45)");
    detail::bind_unary_operation(module, ttnn::tanh, R"doc(\mathrm{{output\_tensor}}_i = \verb|tanh|(\mathrm{{input\_tensor}}_i))doc", R"doc(BFLOAT16, BFLOAT8_B)doc");
    detail::bind_unary_operation(module, ttnn::log_sigmoid, R"doc(\mathrm{{output\_tensor}}_i = \verb|log_sigmoid|(\mathrm{{input\_tensor}}_i))doc", R"doc(BFLOAT16, BFLOAT8_B)doc");
    detail::bind_unary_operation(module, ttnn::bitwise_not, R"doc(\mathrm{{output\_tensor}}_i = \verb|bitwise_not|(\mathrm{{input\_tensor}}_i))doc", R"doc(INT32)doc",
    R"doc(Supported input range is [-2147483647, 2147483647]. Supported for Wormhole_B0 only.)doc",
    R"doc(torch.tensor([[1, 2], [3, 4]], dtype=torch.int32))doc");

    //  Unaries with fast_and_approximate_mode
    detail::bind_unary_operation_with_fast_and_approximate_mode(module, ttnn::exp, R"doc(BFLOAT16, BFLOAT8_B)doc");
    detail::bind_unary_operation_with_fast_and_approximate_mode(module, ttnn::erf, R"doc(BFLOAT16, BFLOAT8_B)doc");
    detail::bind_unary_operation_with_fast_and_approximate_mode(module, ttnn::erfc, R"doc(BFLOAT16, BFLOAT8_B)doc");
    detail::bind_unary_operation_with_fast_and_approximate_mode(module, ttnn::gelu, R"doc(BFLOAT16, BFLOAT8_B)doc");
    detail::bind_unary_operation_with_fast_and_approximate_mode(module, ttnn::rsqrt, R"doc(BFLOAT16, BFLOAT8_B)doc");

    // Unaries with float parameter
    detail::bind_unary_operation_with_float_parameter(module, ttnn::elu, "alpha", "The alpha parameter for the ELU function","",R"doc(BFLOAT16, BFLOAT8_B)doc");
    detail::bind_unary_operation_with_float_parameter(module, ttnn::heaviside, "value", "The value parameter for the Heaviside function", "", R"doc(BFLOAT16, BFLOAT8_B)doc");
    detail::bind_unary_operation_with_float_parameter(module, ttnn::leaky_relu, "negative_slope", "The slope parameter for the Leaky ReLU function", "",R"doc(BFLOAT16, BFLOAT8_B)doc");
    detail::bind_unary_operation_with_float_parameter(module, ttnn::fill, "fill_value", "The value to be filled in the output tensor",
        "This will create a tensor of same shape and dtype as input reference tensor with fill_value.", R"doc(BFLOAT16, BFLOAT8_B)doc", R"doc(Support provided for float32 dtypes in Wormhole_B0. System memory is not supported.)doc");
    detail::bind_unary_operation_with_float_parameter(module, ttnn::relu_max, "upper_limit", "The max value for ReLU function",
        "This function caps off the input to a max value and a min value of 0", R"doc(BFLOAT16, BFLOAT8_B)doc",
        R"doc(System memory is not supported.)doc");
    detail::bind_unary_operation_with_float_parameter(module, ttnn::relu_min, "lower_limit", "The min value for ReLU function",
        "This will carry out ReLU operation at min value instead of the standard 0", R"doc(BFLOAT16)doc",
        R"doc(System memory is not supported.)doc");

    // Unary ops with dim parameter
    detail::bind_unary_operation_with_dim_parameter(
        module,
        ttnn::glu,
        "dim",
        "Dimension to split input tensor. Supported only for last dimension (dim = -1 or 3)",
        "Split the tensor into two parts, apply the GLU function on the second tensor, and then perform multiplication with the first tensor.",
        R"doc(BFLOAT16, BFLOAT8_B)doc",
        R"doc(System memory is not supported.

           Last dimension of input tensor should be divisible by 64.

        )doc");

    detail::bind_unary_operation_with_dim_parameter(
        module,
        ttnn::reglu,
        "dim",
        "Dimension to split input tensor. Supported only for last dimension (dim = -1 or 3)",
        "Split the tensor into two parts, apply the ReLU function on the second tensor, and then perform multiplication with the first tensor.",
        R"doc(BFLOAT16, BFLOAT8_B)doc",
        R"doc(System memory is not supported.

           Last dimension of input tensor should be divisible by 64.

        )doc");

    detail::bind_unary_operation_with_dim_parameter(
        module,
        ttnn::geglu,
        "dim",
        "Dimension to split input tensor. Supported only for last dimension (dim = -1 or 3)",
        "Split the tensor into two parts, apply the GELU function on the second tensor, and then perform multiplication with the first tensor.",
        R"doc(BFLOAT16, BFLOAT8_B)doc",
        R"doc(System memory is not supported.

           Last dimension of input tensor should be divisible by 64.

        )doc");

    detail::bind_unary_operation_with_dim_parameter(
        module,
        ttnn::swiglu,
        "dim",
        "Dimension to split input tensor. Supported only for last dimension (dim = -1 or 3)",
        "Split the tensor into two parts, apply the SiLU function on the second tensor, and then perform multiplication with the first tensor.",
        R"doc(BFLOAT16, BFLOAT8_B)doc",
        R"doc(System memory is not supported.

           Last dimension of input tensor should be divisible by 64.

        )doc");

    // Other unaries (unary chain operations)
    detail::bind_softplus(module, ttnn::softplus);
    detail::bind_sigmoid_accurate(module, ttnn::sigmoid_accurate);
    detail::bind_unary_chain(module, ttnn::unary_chain);
    detail::bind_identity(module, ttnn::identity);

    // unary composite imported into ttnn
    detail::bind_unary_composite(module, ttnn::deg2rad, R"doc(Performs deg2rad function on :attr:`input_tensor`.)doc", "", R"doc(BFLOAT16, BFLOAT8_B)doc");
    detail::bind_unary_composite(module, ttnn::rad2deg, R"doc(Performs rad2deg function on :attr:`input_tensor`.)doc", "", R"doc(BFLOAT16, BFLOAT8_B)doc");
    detail::bind_unary_composite(module, ttnn::tanhshrink, R"doc(Performs tanhshrink function on :attr:`input_tensor`.)doc", "", R"doc(BFLOAT16, BFLOAT8_B)doc");
    detail::bind_unary_composite(module, ttnn::acosh, R"doc(Performs acosh function on :attr:`input_tensor`.)doc", "", R"doc(BFLOAT16)doc", R"doc(TILE)doc", R"doc(2, 3, 4)doc",
        R"doc(System memory is not supported.)doc");
    detail::bind_unary_composite(module, ttnn::asinh, R"doc(Performs asinh function on :attr:`input_tensor`.)doc", "", R"doc(BFLOAT16)doc", R"doc(TILE)doc", R"doc(2, 3, 4)doc",
        R"doc(System memory is not supported.)doc");
    detail::bind_unary_composite(module, ttnn::atanh, R"doc(Performs atanh function on :attr:`input_tensor`.)doc", "", R"doc(BFLOAT16)doc", R"doc(TILE)doc", R"doc(2, 3, 4)doc",
        R"doc(System memory is not supported.)doc");
    detail::bind_unary_composite(module, ttnn::cbrt, R"doc(Performs cbrt function on :attr:`input_tensor`.)doc");
    detail::bind_unary_composite(module, ttnn::cosh, R"doc(Performs cosh function on :attr:`input_tensor`.)doc", "[supported range -9 to 9]", R"doc(BFLOAT16, BFLOAT8_B)doc");
    detail::bind_unary_composite(module, ttnn::digamma, R"doc(Performs digamma function on :attr:`input_tensor`.)doc", "[supported for values greater than 0].",
        R"doc(BFLOAT16, BFLOAT8_B)doc", R"doc(TILE)doc", R"doc(2, 3, 4)doc", "", R"doc(torch.tensor([[2, 3], [4, 5]], dtype=torch.bfloat16))doc");
    detail::bind_unary_composite(module, ttnn::lgamma, R"doc(Performs lgamma function on :attr:`input_tensor`.)doc", "[supported for value greater than 0].", R"doc(BFLOAT16)doc");
    detail::bind_unary_composite(module, ttnn::log1p, R"doc(Performs log1p function on :attr:`input_tensor`.)doc", "[supported range -1 to 1].", R"doc(BFLOAT16, BFLOAT8_B)doc");
    detail::bind_unary_composite(module, ttnn::mish, R"doc(Performs mish function on :attr:`input_tensor`.)doc", "[supported range -20 to inf].", R"doc(BFLOAT16, BFLOAT8_B)doc",
        R"doc(TILE)doc", R"doc(2, 3, 4)doc", R"doc(Not supported on Grayskull.)doc");
    detail::bind_unary_composite(module, ttnn::multigammaln, R"doc(Performs multigammaln function on :attr:`input_tensor`.)doc", "[supported range 1.6 to inf].", R"doc(BFLOAT16)doc",
        R"doc(TILE)doc", R"doc(2, 3, 4)doc", "", R"doc(torch.tensor([[2, 3], [4, 5]], dtype=torch.bfloat16))doc");
    detail::bind_unary_composite(module, ttnn::sinh, R"doc(Performs sinh function on :attr:`input_tensor`.)doc", "[supported range -9 to 9].", R"doc(BFLOAT16, BFLOAT8_B)doc");
    detail::bind_unary_composite(module, ttnn::softsign, R"doc(Performs softsign function on :attr:`input_tensor`.)doc");
    detail::bind_unary_composite(module, ttnn::swish, R"doc(Performs swish function on :attr:`input_tensor`.)doc", "", R"doc(BFLOAT16, BFLOAT8_B)doc");
    detail::bind_unary_composite(module, ttnn::var_hw, R"doc(Performs var_hw function on :attr:`input_tensor`.)doc");
    detail::bind_unary_composite(module, ttnn::std_hw, R"doc(Performs std_hw function on :attr:`input_tensor`.)doc");
    detail::bind_unary_composite(module, ttnn::normalize_hw, R"doc(Performs normalize_hw function on :attr:`input_tensor`.)doc", "", R"doc(BFLOAT16)doc", R"doc(ROW_MAJOR, TILE)doc",
        R"doc(4)doc", "", R"doc(torch.rand([1, 1, 32, 32], dtype=torch.bfloat16))doc");
    detail::bind_unary_composite(module, ttnn::logical_not_, R"doc(Performs logical_not inplace function on :attr:`input_tensor`.)doc", "", R"doc(BFLOAT16, BFLOAT8_B)doc");
    detail::bind_unary_composite(module, ttnn::normalize_global, R"doc(Performs normalize_global function on :attr:`input_tensor`.)doc", "", R"doc(BFLOAT16)doc",
        R"doc(ROW_MAJOR, TILE)doc", R"doc(4)doc", "", R"doc(torch.rand([1, 1, 32, 32], dtype=torch.bfloat16))doc");
    detail::bind_unary_composite(module, ttnn::frac, R"doc(Performs frac function on :attr:`input_tensor`.)doc",  "", R"doc(BFLOAT16, BFLOAT8_B)doc");
    detail::bind_unary_composite_trunc(module, ttnn::trunc, R"doc(Not supported for grayskull.)doc");

    detail::bind_unary_composite_floats_with_default(
        module,
        ttnn::hardswish,
        "scale", "Scale value", 1.0f/6.0f,
        "shift", "Shift value", 0.5f);

    detail::bind_unary_composite_floats_with_default(
        module,
        ttnn::hardsigmoid,
        "scale", "Scale value", 1.0f/6.0f,
        "shift", "Shift value", 0.5f);

    detail::bind_hardtanh(
        module,
        ttnn::hardtanh,
        "min_val", "min value", -1.0f,
        "max_val", "max value", 1.0f);

    detail::bind_unary_composite_optional_floats_with_default(
        module,
        ttnn::clip,
        "min", "Minimum value", std::nullopt,
        "max", "Maximum value", std::nullopt,
        R"doc(Performs clip function on :attr:`input_tensor`, :attr:`min`, :attr:`max`. Only one of 'min' or 'max' value can be None.)doc");
    detail::bind_unary_composite_optional_floats_with_default(
        module,
        ttnn::clamp,
        "min", "Minimum value", std::nullopt,
        "max", "Maximum value", std::nullopt,
        R"doc(Performs clamp function on :attr:`input_tensor`, :attr:`min`, :attr:`max`. Only one of 'min' or 'max' value can be None.)doc");
    detail::bind_unary_composite_floats_with_default(
        module,
        ttnn::selu,
        "scale", "Scale value", 1.0507,
        "alpha", "Alpha value", 1.67326);
    detail::bind_unary_composite_threshold(
        module,
        ttnn::threshold,
        "threshold", "Threshold value",
        "value", "Replacing value",
        R"doc(Performs threshold function on :attr:`input_tensor`, :attr:`threshold`, :attr:`value`.)doc");
    detail::bind_unary_composite_int_with_default(
        module,
        ttnn::tril,
        "diagonal", "diagonal value", 0,
        R"doc(Performs tril function on :attr:`input_tensor`, :attr:`diagonal`.)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");
    detail::bind_unary_composite_int_with_default(
        module,
        ttnn::triu,
        "diagonal", "diagonal value", 0,
        R"doc(Performs triu function on :attr:`input_tensor`, :attr:`diagonal`.)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc");
    detail::bind_unary_composite_int_with_default(
        module,
        ttnn::round,
        "decimals", "decimals value", 0,
        R"doc(Performs round function on :attr:`input_tensor`, :attr:`decimals`.)doc",
        R"doc(BFLOAT16, BFLOAT8_B)doc",
        R"doc(Not supported on Grayskull.)doc");
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
            R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_unary_composite_float_with_default(
        module,
        ttnn::softshrink,
        "lambd", "lambd value", 0.5f,
            R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_unary_composite_float_with_default(
        module,
        ttnn::celu,
        "alpha", "alpha value", 1.0f, R"doc(BFLOAT16, BFLOAT8_B)doc");

    detail::bind_unary_composite_float_with_default(
        module,
        ttnn::logit,
        "eps", "eps", 0.0f,  R"doc(BFLOAT16)doc",
            R"doc(Not available for Wormhole_B0.)doc");

    detail::bind_unary_composite_rpow(
        module,
        ttnn::rpow,
        "exponent", "exponent value. Non-positive values are not supported.",
        R"doc(Performs rpow function on :attr:`input_tensor`, :attr:`exponent`.)doc",
        R"doc(Supported for input range upto 28)doc",
        R"doc(BFLOAT16)doc", R"doc(System memory is not supported.)doc");

    detail::bind_unary_rdiv(
    module,
    ttnn::rdiv,
    "value", "denominator value which is actually calculated as numerator float value >= 0",
    "round_mode", "rounding_mode value", "None",
    R"doc(Performs the element-wise division of a scalar ``value`` by a tensor ``input`` and rounds the result using round_mode. Support provided only for Wormhole_B0.

        Input tensor must have BFLOAT16 data type.

        Output tensor will have BFLOAT16 data type.)doc",

        R"doc(BFLOAT16)doc", R"doc(System memory is not supported.)doc");
}

}  // namespace unary
}  // namespace operations
}  // namespace ttnn
