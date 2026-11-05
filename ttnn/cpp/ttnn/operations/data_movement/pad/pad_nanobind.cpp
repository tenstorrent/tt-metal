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
        Returns a padded tensor, with a specified value at the specified location. If the input tensor is on host, the pad will be performed on host, and if its on device it will be performed on device. Padding on host needs ROW_MAJOR layout.
        Any rank of tensor is supported. For rank > 4, leading dimensions are padded via a reshape path before the 4D device kernel; on host, a tensor of rank > 4 can be padded only in its last 3 dimensions.

        A second overload pads to an explicit shape instead: ``ttnn.pad(input_tensor, output_padded_shape, input_tensor_start, value)``,
        where ``output_padded_shape`` is the 4D output shape and ``input_tensor_start`` is the 4D position of the input inside it. The input must have rank 4.

        Args:
            input_tensor (ttnn.Tensor): the input tensor.
            padding (List[Tuple[int, int]]): padding to apply. Each element is a tuple of 2 integers: the number of values to add before and after the tensor in that dimension. A list shorter than the tensor's rank applies to the trailing dimensions.
            value (float): value to pad with. Integer values are converted to float32, so ``INT32`` and ``UINT32`` values with a magnitude above 2^24 may be rounded.

        Keyword Args:
            use_multicore (bool, optional): switch to use the multicore implementation. Defaults to `True`.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`, which uses the input tensor's memory configuration.
            sub_core_grids (ttnn.CoreRangeSet, optional): Sub core grids to run the operation on. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

        Note:
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - BFLOAT16, FLOAT32, INT32, UINT32, UINT16
                 - TILE, ROW_MAJOR
               * - BFLOAT8_B
                 - TILE

            TILE layout does not support front padding: every "before" value must be 0. Depending on the execution
            path, ``BFLOAT8_B`` input may be padded through a ``BFLOAT16`` typecast.

        Memory Support:
            - Interleaved: DRAM and L1
            - Sharded: HEIGHT_SHARDED, WIDTH_SHARDED and BLOCK_SHARDED in L1; ROW_MAJOR HEIGHT_SHARDED and WIDTH_SHARDED inputs are also accepted in DRAM
            - ND sharded: L1
            - ``sub_core_grids`` requires interleaved input and output, and ``use_multicore=True``
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
