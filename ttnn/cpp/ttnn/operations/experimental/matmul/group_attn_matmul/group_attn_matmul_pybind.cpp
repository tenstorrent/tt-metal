// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/pybind11/decorators.hpp"

#include "ttnn/operations/experimental/matmul/group_attn_matmul/group_attn_matmul_pybind.hpp"
#include "ttnn/operations/experimental/matmul/group_attn_matmul/group_attn_matmul.hpp"

namespace ttnn::operations::experimental::matmul::detail {

void bind_group_attn_matmul(pybind11::module& module) {
    using OperationType = decltype(ttnn::experimental::group_attn_matmul);
    ttnn::bind_registered_operation(
        module,
        ttnn::experimental::group_attn_matmul,
        R"doc(
            Performs a special pre-softmax matmul with [q_len, q_heads, batch, head_dim] and [batch, kv_heads, head_dim, kv_len]. q_len and kv_heads must be 1 and an intermediate value of [q_heads, batch, batch, kv_len] is produced (only on device cores). Batch dim from Z and Y is combined by taking the 1st, 2nd, ..., and 32nd row of Y from the batches in Z. Final output tensor is [1, q_heads, batch, kv_len]. In PyTorch, this is equivalent to: torch.matmul(A.transpose(0, 2), B).transpose(0, 2). Similar concept for post-softmax matmul.
        )doc",
        ttnn::pybind_overload_t{[](const OperationType& self,
                                   const Tensor& input_tensor_a,
                                   const Tensor& input_tensor_b,
                                   const CoreCoord& compute_with_storage_grid_size,
                                   const std::optional<MemoryConfig>& memory_config,
                                   std::optional<const DataType> output_dtype,
                                   std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
                                   std::optional<Tensor> optional_output_tensor,
                                   uint8_t queue_id) {
                                    return self(queue_id,
                                                input_tensor_a,
                                                input_tensor_b,
                                                compute_with_storage_grid_size,
                                                memory_config,
                                                output_dtype,
                                                compute_kernel_config,
                                                optional_output_tensor);
                                },
                                pybind11::arg().noconvert(),
                                pybind11::arg().noconvert(),
                                pybind11::kw_only(),
                                pybind11::arg("compute_with_storage_grid_size").noconvert(),
                                pybind11::arg("memory_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
                                pybind11::arg("dtype").noconvert() = std::nullopt,
                                pybind11::arg("compute_kernel_config").noconvert() = std::nullopt,
                                pybind11::arg("optional_output_tensor").noconvert() = std::nullopt,
                                pybind11::arg("queue_id").noconvert() = 0});
}

}  // namespace ttnn::operations::experimental::matmul::detail
