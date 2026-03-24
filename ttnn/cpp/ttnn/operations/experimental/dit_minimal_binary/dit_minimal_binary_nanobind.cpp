// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "dit_minimal_binary_nanobind.hpp"

#include <optional>
#include <string>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/dit_minimal_binary/dit_minimal_binary.hpp"

namespace ttnn::operations::experimental::dit_minimal_binary::detail {

void bind_dit_minimal_binary(nb::module_& mod) {
    const auto* doc =
        R"doc(
        Element-wise binary operation on row-major DRAM tensors, tilizing internally for the compute phase.

        Both inputs must have identical shape, ROW_MAJOR_LAYOUT, and DRAM interleaved memory.
        Only bfloat16 and float32 dtypes are supported. No broadcast.

        Args:
            input_a (ttnn.Tensor): First input tensor.
            input_b (ttnn.Tensor): Second input tensor.

        Keyword Args:
            op (str): Operation to perform — "add" (default) or "mul".
            memory_config (ttnn.MemoryConfig, optional): Output memory config. Defaults to input memory config.

        Returns:
            ttnn.Tensor: Output in ROW_MAJOR_LAYOUT, same dtype as inputs.
        )doc";

    ttnn::bind_function<"dit_minimal_binary", "ttnn.experimental.">(
        mod,
        doc,
        ttnn::overload_t(
            &ttnn::operations::experimental::DitMinimalRmBinaryOperation::invoke,
            nb::arg("input_a").noconvert(),
            nb::arg("input_b").noconvert(),
            nb::arg("op") = "add",
            nb::arg("memory_config") = nb::none()));
}

}  // namespace ttnn::operations::experimental::dit_minimal_binary::detail
