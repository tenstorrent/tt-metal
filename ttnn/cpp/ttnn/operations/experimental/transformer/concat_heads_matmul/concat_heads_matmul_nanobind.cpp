// SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "concat_heads_matmul_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "concat_heads_matmul.hpp"

namespace ttnn::operations::experimental::transformer {

void bind_concat_heads_matmul(nb::module_& mod) {
    ttnn::bind_function<"concat_heads_matmul", "ttnn.experimental.">(
        mod,
        R"doc(
        Fused concat-heads + output-projection matmul in a single dispatch. attn
        [1, num_heads, seq, head_dim] is consumed directly as the matmul in0 (K = num_heads*head_dim);
        for seq <= 1 tile this equals nlp_concat_heads(attn) @ weight (PCC 1.0). weight is
        [num_heads*head_dim, N]. Returns [1, 1, seq, N].

        Args:
            attn (ttnn.Tensor): [1, num_heads, seq, head_dim], TILE, interleaved.
            weight (ttnn.Tensor): [num_heads*head_dim, N], TILE, interleaved.

        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): Defaults to `None`.
            output_dtype (ttnn.DataType, optional): Defaults to bfloat16.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): Defaults to `None`.

        Returns:
            ttnn.Tensor: the projected output [1, 1, seq, N].
        )doc",
        &ttnn::experimental::concat_heads_matmul,
        nb::arg("attn"),
        nb::arg("weight"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("output_dtype") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("program_config") = nb::none());
}

}  // namespace ttnn::operations::experimental::transformer
