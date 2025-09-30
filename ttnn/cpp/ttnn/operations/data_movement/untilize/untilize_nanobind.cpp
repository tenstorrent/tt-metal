// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "untilize.hpp"
#include "ttnn-nanobind/decorators.hpp"

namespace ttnn::operations::data_movement::detail {

void bind_untilize(nb::module_& mod) {
    auto doc =
        R"doc(
            Changes data layout of input tensor to ROW_MAJOR.

            Input tensor must be on TT accelerator device, in TILE layout, and have BFLOAT16 data type.

            Output tensor will be on TT accelerator device, in ROW_MAJOR layout, and have BFLOAT16 data type.

            Args:
                input_tensor (ttnn.Tensor): the input tensor.

            Keyword Args:

                memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
                use_multicore (bool, optional): Whether to use multicore. Defaults to `True`.
                use_pack_untilize (bool, optional): Whether to use pack untilize. Defaults to `True`.
                sub_core_grids (ttnn.CoreRangeSet, optional): Sub core grids. Defaults to `None`.

            Returns:
                List of ttnn.Tensor: the output tensor.

        )doc";

    using OperationType = decltype(ttnn::untilize);
    ttnn::bind_registered_operation(
        mod,
        ttnn::untilize,
        doc,
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const std::optional<MemoryConfig>& memory_config,
               bool use_multicore,
               bool use_pack_untilize,
               const std::optional<CoreRangeSet>&& sub_core_grids) {
                return self(input_tensor, memory_config, use_multicore, use_pack_untilize, sub_core_grids);
            },
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("use_multicore") = true,
            nb::arg("use_pack_untilize") = true,
            nb::arg("sub_core_grids") = nb::none()});
}
}  // namespace ttnn::operations::data_movement::detail
