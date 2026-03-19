// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "group_attn_matmul_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/experimental/matmul/group_attn_matmul/group_attn_matmul.hpp"

namespace ttnn::operations::experimental::matmul::detail {

void bind_group_attn_matmul(nb::module_& mod) {
    ttnn::bind_function<"group_attn_matmul", "ttnn.experimental.">(
        mod,
        R"doc(
            Performs a special pre-softmax matmul with [q_len, q_heads, batch, head_dim] and [batch, kv_heads, head_dim, kv_len]. q_len and kv_heads must be 1 and an intermediate value of [q_heads, batch, batch, kv_len] is produced (only on device cores). Batch dim from Z and Y is combined by taking the 1st, 2nd, ..., and 32nd row of Y from the batches in Z. Final output tensor is [1, q_heads, batch, kv_len]. In PyTorch, this is equivalent to: torch.matmul(A.transpose(0, 2), B).transpose(0, 2). Similar concept for post-softmax matmul.
        )doc",
        &ttnn::experimental::group_attn_matmul,
        nb::arg("input_tensor_a").noconvert(),
        nb::arg("input_tensor_b").noconvert(),
        nb::kw_only(),
        nb::arg("compute_with_storage_grid_size").noconvert(),
        nb::arg("memory_config").noconvert() = tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        nb::arg("dtype").noconvert() = nb::none(),
        nb::arg("compute_kernel_config").noconvert() = nb::none(),
        nb::arg("optional_output_tensor").noconvert() = nb::none());
}

}  // namespace ttnn::operations::experimental::matmul::detail
