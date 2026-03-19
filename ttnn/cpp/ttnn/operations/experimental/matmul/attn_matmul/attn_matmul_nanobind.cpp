// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "attn_matmul_nanobind.hpp"

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"

#include "ttnn/operations/experimental/matmul/attn_matmul/attn_matmul.hpp"

namespace ttnn::operations::experimental::matmul::detail {

void bind_attn_matmul(nb::module_& mod) {
    ttnn::bind_function<"attn_matmul", "ttnn.experimental.">(
        mod,
        R"doc(
            Performs a special pre-softmax matmul with [q_len, q_heads, batch, head_dim] and [batch, kv_heads, head_dim, kv_len]. q_len and kv_heads must be 1 and an intermediate value of [q_heads, batch, batch, kv_len] is produced (only on device cores). Batch dim from Z and Y is combined by taking the 1st, 2nd, ..., and 32nd row of Y from the batches in Z. Final output tensor is [1, q_heads, batch, kv_len]. In PyTorch, this is equivalent to: torch.matmul(A.transpose(0, 2), B).transpose(0, 2). Similar concept for post-softmax matmul.
        )doc",
        &ttnn::experimental::attn_matmul,
        nb::arg("input_tensor_a").noconvert(),
        nb::arg("input_tensor_b").noconvert(),
        nb::kw_only(),
        nb::arg("compute_with_storage_grid_size").noconvert(),
        nb::arg("dtype").noconvert() = nb::none(),
        nb::arg("compute_kernel_config").noconvert() = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("output_tensor") = nb::none());
}

void bind_attn_matmul_from_cache(nb::module_& mod) {
    ttnn::bind_function<"attn_matmul_from_cache", "ttnn.experimental.">(
        mod,
        R"doc(
            Performs the same matmul as attn_matmul, but fuses additional functionality for reading in in1. For in1, read num_tokens (rounded up to 32) from full cache along in1.padded_shape()[2] (num_tokens must be > 0 and <= max_cache_len). For example, 64 tokens will be read for 32 < token_idx <= 64. Additional option to apply transpose_hw to in1 for pre-attention matmul with transpose_hw=true. For post-attention matmul, transpose_hw should be false.
        )doc",
        &ttnn::experimental::attn_matmul_from_cache,
        nb::arg("input_tensor_a").noconvert(),
        nb::arg("input_tensor_b").noconvert(),
        nb::kw_only(),
        nb::arg("num_tokens").noconvert(),
        nb::arg("transpose_hw").noconvert(),
        nb::arg("compute_with_storage_grid_size").noconvert(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("dtype") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("output_tensor") = nb::none());
}

}  // namespace ttnn::operations::experimental::matmul::detail
