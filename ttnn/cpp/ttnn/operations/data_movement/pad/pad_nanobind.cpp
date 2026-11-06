// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "pad_nanobind.hpp"

#include <array>
#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/small_vector_caster.hpp"
#include "ttnn-nanobind/bind_function.hpp"

#include "pad.hpp"
#include "pad_force.hpp"

namespace ttnn::operations::data_movement::detail {

void bind_pad(nb::module_& mod) {
    const auto* doc = R"doc(
        Returns a padded tensor, with a specified value at the specified location. If the input tensor is on host, the pad will be performed on host, and if its on device it will be performed on device.
        Any rank of tensor is supported. For rank > 4, leading dimensions are padded via a reshape path before the 4D device kernel; tile layout still does not support front padding on device.

        Args:
            * :attr:`input_tensor`: (ttnn.Tensor): the input tensor.
            * :attr:`padding`: (list[Tuple[int,int]]): padding to apply. Each element of padding should be a tuple of 2 integers, with the first integer specifying the number of values to add before the tensor and the second integer specifying the number of values to add after the tensor. Mutually exclusive to output_tensor_shape and input_tensor_start.
            * :attr:`value`: (Union[float,int]): value to pad with.

        Keyword Args:
            * :attr:`use_multicore`: (Optional[bool]) switch to use multicore implementation
            * :attr:`memory_config`: (Optional[ttnn.MemoryConfig]): Memory configuration for the operation. Defaults to `None`.
            * :attr:`sub_core_grids`: (Optional[ttnn.CoreRangeSet]): Sub core grids to run the operation on. Defaults to `None`.

        Returns:
            List of ttnn.Tensor: the output tensor.
    )doc";

    ttnn::bind_function<"pad">(
        mod,
        doc,
        ttnn::overload_t(
            nb::overload_cast<
                const ttnn::Tensor&,
                const ttsl::SmallVector<std::array<uint32_t, 2>>&,
                float,
                bool,
                const std::optional<MemoryConfig>&,
                const std::optional<CoreRangeSet>&>(&ttnn::pad),
            nb::arg("input_tensor"),
            nb::arg("padding"),
            nb::arg("value"),
            nb::kw_only(),
            nb::arg("use_multicore") = true,
            nb::arg("memory_config") = nb::none(),
            nb::arg("sub_core_grids") = nb::none()),
        ttnn::overload_t(
            nb::overload_cast<
                const ttnn::Tensor&,
                const ttnn::Array4D&,
                const ttnn::Array4D&,
                float,
                bool,
                const std::optional<MemoryConfig>&,
                const std::optional<CoreRangeSet>&>(&ttnn::pad),
            nb::arg("input_tensor"),
            nb::arg("output_padded_shape"),
            nb::arg("input_tensor_start"),
            nb::arg("value"),
            nb::kw_only(),
            nb::arg("use_multicore") = true,
            nb::arg("memory_config") = nb::none(),
            nb::arg("sub_core_grids") = nb::none()));

    // Bound with a plain def rather than ttnn::bind_function: the latter tags the callable for
    // auto_register_ttnn_cpp_operations, which would republish these as ttnn.* operations. They are
    // meant to stay reachable only via this private module. See pad_force.hpp.
    mod.def(
        "pad_force_native",
        &pad_force_native,
        nb::arg("input_tensor"),
        nb::arg("padding"),
        nb::arg("value"),
        nb::kw_only(),
        nb::arg("use_multicore") = true,
        nb::arg("memory_config") = nb::none(),
        nb::arg("sub_core_grids") = nb::none(),
        nb::call_guard<nb::gil_scoped_release>(),
        R"doc(
            Verification only: runs the native pad implementation unconditionally. Not part of the ttnn
            API; use ttnn.pad, which selects an implementation on its own.

            Args:
                input_tensor (ttnn.Tensor): the input tensor.
                padding (list[Tuple[int,int]]): (before, after) elements to add per dimension.
                value (Union[float,int]): value to pad with.

            Keyword Args:
                use_multicore (bool, optional): switch to use the multicore implementation. Defaults to `True`.
                memory_config (ttnn.MemoryConfig, optional): memory configuration for the output.
                    Defaults to `None`, which derives it from the input.
                sub_core_grids (ttnn.CoreRangeSet, optional): sub core grids to run the operation on.
                    Defaults to `None`.

            Returns:
                ttnn.Tensor: the output tensor.
        )doc");

    mod.def(
        "pad_force_codegen",
        &pad_force_codegen,
        nb::arg("input_tensor"),
        nb::arg("padding"),
        nb::arg("value"),
        nb::kw_only(),
        nb::arg("use_multicore") = true,
        nb::arg("memory_config") = nb::none(),
        nb::arg("sub_core_grids") = nb::none(),
        nb::call_guard<nb::gil_scoped_release>(),
        R"doc(
            Verification only: runs the codegen pad implementation, raising for a case outside its
            support scope rather than falling back to native, and ignoring the perf demotions. A pad
            that adds nothing returns the input. Not part of the ttnn API; use ttnn.pad, which selects
            an implementation on its own.

            Args:
                input_tensor (ttnn.Tensor): the input tensor.
                padding (list[Tuple[int,int]]): (before, after) elements to add per dimension.
                value (Union[float,int]): value to pad with.

            Keyword Args:
                use_multicore (bool, optional): switch to use the multicore implementation. Defaults to `True`.
                memory_config (ttnn.MemoryConfig, optional): memory configuration for the output.
                    Defaults to `None`, which derives it from the input.
                sub_core_grids (ttnn.CoreRangeSet, optional): sub core grids to run the operation on.
                    Defaults to `None`.

            Returns:
                ttnn.Tensor: the output tensor.
        )doc");
}
}  // namespace ttnn::operations::data_movement::detail
