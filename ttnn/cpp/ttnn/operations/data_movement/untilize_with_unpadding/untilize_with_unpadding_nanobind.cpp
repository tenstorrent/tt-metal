// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_with_unpadding_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "untilize_with_unpadding.hpp"

namespace ttnn::operations::data_movement::detail {

void bind_untilize_with_unpadding(nb::module_& mod) {
    const auto* doc =
        R"doc(
        Changes data layout of input tensor to ROW_MAJOR and unpads/removes elements from the tensor.

        Input tensor must be on TT accelerator device, in TILE layout, and have BFLOAT16 data type.

        Output tensor will be on TT accelerator device, in ROW_MAJOR layout, and have BFLOAT16 data type.

        Args:
            input_tensor (ttnn.Tensor): the input tensor
            output_tensor_end (shape): End indices of input tensor in output tensor.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            use_multicore (bool, optional): Whether to use multicore. Defaults to `True`.
            use_pack_untilize (bool, optional): Whether to use pack untilize. Defaults to `True`.
            sub_core_grids (ttnn.CoreRangeSet, optional): Sub core grids. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.
    )doc";

    ttnn::bind_function<"untilize_with_unpadding">(
        mod,
        doc,
        ttnn::overload_t(
            &ttnn::untilize_with_unpadding,
            nb::arg("input_tensor"),
            nb::arg("output_tensor_end"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("use_multicore") = true,
            nb::arg("use_pack_untilize") = true,
            nb::arg("sub_core_grids") = nb::none()));
}

}  // namespace ttnn::operations::data_movement::detail
