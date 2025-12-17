// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "tilize.hpp"
#include "ttnn-nanobind/decorators.hpp"

namespace ttnn::operations::data_movement::detail {

void bind_tilize(nb::module_& mod) {
    const auto* doc = R"doc(
        Changes data layout of input tensor to TILE.

        Input tensor must be on TT accelerator device, in ROW_MAJOR layout, and have BFLOAT16 data type.

        Output tensor will be on TT accelerator device, in TILE layout, and have BFLOAT16 data type.

        Args:
            input_tensor (ttnn.Tensor): the input tensor.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            dtype (data type, optional): Data type of the output tensor. Defaults to `None`.
            use_multicore (bool, optional): Whether to use multicore. Defaults to `True`.
            use_low_perf (bool, optional): Use a low performance version that uses less memory. USE ONLY IF ABSOLUTELY NEEDED IN MODELS. Defaults to `False`.
            sub_core_grids (CoreRangeSet, optional): Used to restrict tilize to a set of cores, Defaults to using the entire device

        Returns:
            ttnn.Tensor: the output tensor.
    )doc";

    using OperationType = decltype(ttnn::tilize);
    ttnn::bind_registered_operation(
        mod,
        ttnn::tilize,
        doc,
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const std::optional<MemoryConfig>& memory_config,
               std::optional<DataType> output_dtype,
               bool use_multicore,
               bool use_low_perf,
               const std::optional<CoreRangeSet>& sub_core_grids) {
                return self(input_tensor, memory_config, output_dtype, use_multicore, use_low_perf, sub_core_grids);
            },
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("dtype") = nb::none(),
            nb::arg("use_multicore") = true,
            nb::arg("use_low_perf") = false,
            nb::arg("sub_core_grids") = nb::none()});
}
}  // namespace ttnn::operations::data_movement::detail
