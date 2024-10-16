// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/pybind11/decorators.hpp"

#include "ttnn/operations/experimental/matmul/attn_matmul/attn_matmul_pybind.hpp"
#include "ttnn/operations/experimental/matmul/attn_matmul/attn_matmul.hpp"

namespace ttnn::operations::experimental::matmul::detail {

void bind_attn_matmul(pybind11::module& module) {
    using OperationType = decltype(ttnn::experimental::attn_matmul);
    ttnn::bind_registered_operation(
        module,
        ttnn::experimental::attn_matmul,
        R"doc(
            Performs a special pre-softmax matmul with [q_len, q_heads, batch, head_dim] and [batch, kv_heads, head_dim, kv_len]. q_len and kv_heads must be 1 and an intermediate value of [q_heads, batch, batch, kv_len] is produced (only on device cores). Batch dim from Z and Y is combined by taking the 1st, 2nd, ..., and 32nd row of Y from the batches in Z. Final output tensor is [1, q_heads, batch, kv_len]. In PyTorch, this is equivalent to: torch.matmul(A.transpose(0, 2), B).transpose(0, 2). Similar concept for post-softmax matmul.
        )doc",
        ttnn::pybind_overload_t{[](const OperationType& self,
                                   const Tensor& input_tensor_a,
                                   const Tensor& input_tensor_b,
                                   const CoreCoord& compute_with_storage_grid_size,
                                   std::optional<const DataType> output_dtype,
                                   std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
                                   const std::optional<MemoryConfig>& memory_config,
                                   std::optional<Tensor> optional_output_tensor,
                                   uint8_t queue_id) {
                                    return self(queue_id,
                                                input_tensor_a,
                                                input_tensor_b,
                                                compute_with_storage_grid_size,
                                                output_dtype,
                                                compute_kernel_config,
                                                memory_config,
                                                optional_output_tensor);
                                },
                                pybind11::arg("input_tensor_a").noconvert(),
                                pybind11::arg("input_tensor_b").noconvert(),
                                pybind11::kw_only(),
                                pybind11::arg("compute_with_storage_grid_size").noconvert(),
                                pybind11::arg("dtype").noconvert() = std::nullopt,
                                pybind11::arg("compute_kernel_config").noconvert() = std::nullopt,
                                pybind11::arg("memory_config") = std::nullopt,
                                pybind11::arg("output_tensor") = std::nullopt,
                                pybind11::arg("queue_id") = 0});
}

void bind_attn_matmul_from_cache(pybind11::module& module) {
    using OperationType = decltype(ttnn::experimental::attn_matmul_from_cache);
    ttnn::bind_registered_operation(
        module,
        ttnn::experimental::attn_matmul_from_cache,
        R"doc(
            Performs the same matmul as attn_matmul, but fuses additional functionality for reading in in1. For in1, read num_tokens (rounded up to 32) from full cache along in1.get_legacy_shape()[2] (num_tokens must be > 0 and <= max_cache_len). For example, 64 tokens will be read for 32 < token_idx <= 64. Additional option to apply transpose_hw to in1 for pre-attention matmul with transpose_hw=true. For post-attention matmul, transpose_hw should be false.
        )doc",
        ttnn::pybind_overload_t{[](const OperationType& self,
                                   const Tensor& input_tensor_a,
                                   const Tensor& input_tensor_b,
                                   const uint32_t num_tokens,
                                   const bool transpose_hw,
                                   const CoreCoord& compute_with_storage_grid_size,
                                   const std::optional<MemoryConfig>& memory_config,
                                   std::optional<const DataType> dtype,
                                   std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
                                   uint8_t queue_id) {
                                    return self(queue_id,
                                                input_tensor_a,
                                                input_tensor_b,
                                                num_tokens,
                                                transpose_hw,
                                                compute_with_storage_grid_size,
                                                memory_config,
                                                dtype,
                                                compute_kernel_config);
                                },
                                pybind11::arg("input_tensor_a").noconvert(),
                                pybind11::arg("input_tensor_b").noconvert(),
                                pybind11::kw_only(),
                                pybind11::arg("num_tokens").noconvert(),
                                pybind11::arg("transpose_hw").noconvert(),
                                pybind11::arg("compute_with_storage_grid_size").noconvert(),
                                pybind11::arg("memory_config") = std::nullopt,
                                pybind11::arg("dtype") = std::nullopt,
                                pybind11::arg("compute_kernel_config") = std::nullopt,
                                pybind11::arg("queue_id") = 0});
}

}  // namespace ttnn::operations::experimental::matmul::detail
