// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "indexed_fill_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "indexed_fill.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::data_movement::detail {

void bind_indexed_fill(nb::module_& mod) {
    const auto* doc =
        R"doc(
            Replaces batch of input in input_b denoted by batch_ids into input_a.

            Args:
                batch_id (ttnn.Tensor): the input tensor.
                input_tensor_a (ttnn.Tensor): the input tensor.
                input_tensor_b (ttnn.Tensor): the input tensor.

            Keyword Args:
                memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
                dim (int, optional): Dimension value. Defaults to `0`.

            Returns:
                ttnn.Tensor: the output tensor.
        )doc";

    ttnn::bind_function<"indexed_fill">(
        mod,
        doc,
        ttnn::overload_t(
            &ttnn::indexed_fill,
            nb::arg("batch_id").noconvert(),
            nb::arg("input_tensor_a").noconvert(),
            nb::arg("input_tensor_b").noconvert(),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("dim") = 0));
}

}  // namespace ttnn::operations::data_movement::detail
