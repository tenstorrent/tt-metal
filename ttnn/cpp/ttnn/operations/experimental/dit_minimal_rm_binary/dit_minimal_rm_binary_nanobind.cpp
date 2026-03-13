// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "dit_minimal_rm_binary_nanobind.hpp"

#include <optional>
#include <string>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/experimental/dit_minimal_rm_binary/dit_minimal_rm_binary.hpp"

namespace ttnn::operations::experimental::dit_minimal_rm_binary::detail {

void bind_dit_minimal_rm_binary(nb::module_& mod) {
    const auto* doc =
        R"doc(
        Element-wise binary operation on row-major DRAM tensors without tilize/untilize.

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

    using OperationType = decltype(ttnn::dit_minimal_rm_binary);
    bind_registered_operation(
        mod,
        ttnn::dit_minimal_rm_binary,
        doc,
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_a,
               const ttnn::Tensor& input_b,
               const std::string& op,
               const std::optional<tt::tt_metal::MemoryConfig>& memory_config) {
                return self(input_a, input_b, op, memory_config);
            },
            nb::arg("input_a").noconvert(),
            nb::arg("input_b").noconvert(),
            nb::arg("op") = "add",
            nb::arg("memory_config") = nb::none()});
}

}  // namespace ttnn::operations::experimental::dit_minimal_rm_binary::detail
