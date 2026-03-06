// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ternary_backward_nanobind.hpp"

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/eltwise/ternary_backward/ternary_backward.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::ternary_backward {

void py_module(nb::module_& mod) {
    ttnn::bind_function<"addcmul_bw">(
        mod,
        R"doc(
        Performs backward operations for addcmul of :attr:`input_tensor_a`, :attr:`input_tensor_b` and :attr:`input_tensor_c` with given :attr:`grad_tensor`.


        Args:
            grad_tensor (ttnn.Tensor): the input gradient tensor.
            input_tensor_a (ttnn.Tensor): the input tensor.
            input_tensor_b (ttnn.Tensor): the input tensor.
            input_tensor_c (ttnn.Tensor): the input tensor.
            alpha (float): the alpha value.

        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the operation. Defaults to `None`.

        Returns:
            List of ttnn.Tensor: the output tensor.

        Note:
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - BFLOAT16, BFLOAT8_B
                 - TILE

        )doc",
        &ttnn::addcmul_bw,
        nb::arg("grad_tensor"),
        nb::arg("input_tensor_a"),
        nb::arg("input_tensor_b"),
        nb::arg("input_tensor_c"),
        nb::arg("alpha"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none());

    ttnn::bind_function<"addcdiv_bw">(
        mod,
        R"doc(
        Performs backward operations for addcdiv of :attr:`input_tensor_a`, :attr:`input_tensor_b` and :attr:`input_tensor_c` with given :attr:`grad_tensor`.


        Args:
            grad_tensor (ttnn.Tensor): the input gradient tensor.
            input_tensor_a (ttnn.Tensor): the input tensor.
            input_tensor_b (ttnn.Tensor): the input tensor.
            input_tensor_c (ttnn.Tensor): the input tensor.
            alpha (float): the alpha value.

        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the operation. Defaults to `None`.

        Returns:
            List of ttnn.Tensor: the output tensor.

        Note:
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - BFLOAT16
                 - TILE

            For more details about BFLOAT8_B, refer to the `BFLOAT8_B limitations <../tensor.html#limitation-of-bfloat8-b>`_.
        )doc",
        &ttnn::addcdiv_bw,
        nb::arg("grad_tensor"),
        nb::arg("input_tensor_a"),
        nb::arg("input_tensor_b"),
        nb::arg("input_tensor_c"),
        nb::arg("alpha"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none());

    ttnn::bind_function<"where_bw">(
        mod,
        R"doc(
        Performs backward operations for where of :attr:`input_tensor_a`, :attr:`input_tensor_b` and :attr:`input_tensor_c` with given :attr:`grad_tensor`.


        Args:
            grad_tensor (ttnn.Tensor): the input gradient tensor.
            input_tensor_a (ttnn.Tensor): the input tensor.
            input_tensor_b (ttnn.Tensor): the input tensor.
            input_tensor_c (ttnn.Tensor): the input tensor.

        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the operation. Defaults to `None`.
            are_required_outputs (List[bool], optional): list of required outputs. Defaults to `[True, True]`.
            input_a_grad (ttnn.Tensor, optional): preallocated output tensor. Defaults to `None`.
            input_b_grad (ttnn.Tensor, optional): preallocated output tensor. Defaults to `None`.

        Returns:
            List of ttnn.Tensor: the output tensor.


        Note:
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - BFLOAT16, BFLOAT8_B
                 - TILE
        )doc",
        &ttnn::where_bw,
        nb::arg("grad_tensor"),
        nb::arg("input_tensor_a"),
        nb::arg("input_tensor_b"),
        nb::arg("input_tensor_c"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("are_required_outputs") = nb::cast(std::vector<bool>{true, true}),
        nb::arg("input_a_grad") = nb::none(),
        nb::arg("input_b_grad") = nb::none());

    ttnn::bind_function<"lerp_bw">(
        mod,
        R"doc(
        Performs backward operations for lerp of :attr:`input_tensor_a`, :attr:`input_tensor_b` and :attr:`input_tensor_c` or :attr:`scalar` with given :attr:`grad_tensor`.


        Args:
            grad_tensor (ttnn.Tensor): the input gradient tensor.
            input_tensor_a (ttnn.Tensor): the input tensor.
            input_tensor_b (ttnn.Tensor): the input tensor.
            input_tensor_c (ttnn.Tensor or Number): the input tensor.


        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the operation. Defaults to `None`.


        Returns:
            List of ttnn.Tensor: the output tensor.


        Note:
            Supported dtypes and layouts:

            For Inputs : :attr:`input_tensor_a`, :attr:`input_tensor_b` and :attr:`input_tensor_c`

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - BFLOAT16
                 - TILE

            For Inputs : :attr:`input_tensor_a`, :attr:`input_tensor_b` and :attr:`scalar`

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - BFLOAT16, BFLOAT8_B
                 - TILE

            bfloat8_b/bfloat4_b is only supported on TILE_LAYOUT
        )doc",
        ttnn::overload_t(
            static_cast<std::vector<ttnn::Tensor> (*)(
                const ttnn::Tensor&,
                const ttnn::Tensor&,
                const ttnn::Tensor&,
                const ttnn::Tensor&,
                const std::optional<ttnn::MemoryConfig>&)>(&ttnn::lerp_bw),
            nb::arg("grad_tensor"),
            nb::arg("input_tensor_a"),
            nb::arg("input_tensor_b"),
            nb::arg("input_tensor_c"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()),
        ttnn::overload_t(
            static_cast<std::vector<ttnn::Tensor> (*)(
                const ttnn::Tensor&,
                const ttnn::Tensor&,
                const ttnn::Tensor&,
                float,
                const std::optional<ttnn::MemoryConfig>&)>(&ttnn::lerp_bw),
            nb::arg("grad_tensor"),
            nb::arg("input_tensor_a"),
            nb::arg("input_tensor_b"),
            nb::arg("scalar"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()));
}

}  // namespace ttnn::operations::ternary_backward
