// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "gdn_gates_nanobind.hpp"
#include "gdn_gates.hpp"

#include <nanobind/stl/tuple.h>

#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::experimental::kda::gdn_gates::detail {

void bind_gdn_gates(nb::module_& mod) {
    ttnn::bind_function<"gdn_gates", "ttnn.experimental.">(
        mod,
        R"doc(
        Gated-DeltaNet gates beta and g from the a and b columns of the [g|a|b] projection output.

            beta = bf16(sigmoid(b)) * beta_scale
            g    = a_neg * bf16(softplus(bf16(a + dt_bias)))

        Bit-identical to the chain ttnn.multiply(b, beta_scale, input_tensor_a_activations=[SIGMOID]),
        ttnn.add(a, dt_bias, activations=[SOFTPLUS(1, 20)]), ttnn.multiply(a_neg, sp) followed by a
        typecast to FLOAT32, in one op.

        Args:
            gab (ttnn.Tensor): ``[1, 1, T, W] or [1, T, W]`` interleaved TILE BFLOAT16 (T tile aligned). The a and b
                heads sit in the first ``num_heads`` columns of the tile at columns ``a_col_offset`` and
                ``b_col_offset`` (multiples of 32).
            dt_bias (ttnn.Tensor): ``[1, 1, 1, num_heads]`` interleaved TILE BFLOAT16 (row broadcast).
            a_neg (ttnn.Tensor): ``[1, 1, 1, num_heads]`` interleaved TILE BFLOAT16 (row broadcast), -exp(A_log).

        Keyword Args:
            a_col_offset (int), b_col_offset (int), num_heads (int, at most 32).
            beta_scale (float): Scale applied after the sigmoid (1.0 or 2.0). Defaults to 1.0.
            memory_config (ttnn.MemoryConfig, optional): Interleaved output memory config. Defaults to DRAM.

        Returns:
            tuple(ttnn.Tensor, ttnn.Tensor): (beta, g), each ``[1, 1, T, num_heads]`` (``[1, T, num_heads]`` for a rank-3 gab) TILE FLOAT32.
        )doc",
        &ttnn::experimental::gdn_gates,
        nb::arg("gab").noconvert(),
        nb::arg("dt_bias").noconvert(),
        nb::arg("a_neg").noconvert(),
        nb::kw_only(),
        nb::arg("a_col_offset"),
        nb::arg("b_col_offset"),
        nb::arg("num_heads"),
        nb::arg("beta_scale") = 1.0f,
        nb::arg("memory_config") = nb::none());
}

}  // namespace ttnn::operations::experimental::kda::gdn_gates::detail
