// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "group_attn_matmul_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/experimental/matmul/group_attn_matmul/group_attn_matmul.hpp"

namespace ttnn::operations::experimental::matmul::detail {

void bind_group_attn_matmul(nb::module_& mod) {
    using OperationType = decltype(ttnn::experimental::group_attn_matmul);
    ttnn::bind_registered_operation(
        mod,
        ttnn::experimental::group_attn_matmul,
        R"doc(
            Performs a special pre-softmax matmul with [q_len, q_heads, batch, head_dim] and [batch, kv_heads, head_dim, kv_len]. q_len and kv_heads must be 1 and an intermediate value of [q_heads, batch, batch, kv_len] is produced (only on device cores). Batch dim from Z and Y is combined by taking the 1st, 2nd, ..., and 32nd row of Y from the batches in Z. Final output tensor is [1, q_heads, batch, kv_len]. In PyTorch, this is equivalent to: torch.matmul(A.transpose(0, 2), B).transpose(0, 2). Similar concept for post-softmax matmul.
        )doc",
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const Tensor& input_tensor_a,
               const Tensor& input_tensor_b,
               const CoreCoord& compute_with_storage_grid_size,
               const std::optional<MemoryConfig>& memory_config,
               std::optional<const DataType> output_dtype,
               std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
               std::optional<Tensor> optional_output_tensor) {
                return self(
                    input_tensor_a,
                    input_tensor_b,
                    compute_with_storage_grid_size,
                    memory_config,
                    output_dtype,
                    compute_kernel_config,
                    optional_output_tensor);
            },
            nb::arg().noconvert(),
            nb::arg().noconvert(),
            nb::kw_only(),
            nb::arg("compute_with_storage_grid_size").noconvert(),
            nb::arg("memory_config").noconvert() = tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
            nb::arg("dtype").noconvert() = nb::none(),
            nb::arg("compute_kernel_config").noconvert() = nb::none(),
            nb::arg("optional_output_tensor").noconvert() = nb::none()});
}

}  // namespace ttnn::operations::experimental::matmul::detail
