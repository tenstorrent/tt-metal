// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn-pybind/decorators.hpp"

#include "ttnn/operations/experimental/transformer/nlp_create_qkv_heads_decode/nlp_create_qkv_heads_decode.hpp"

namespace ttnn::operations::experimental::transformer::detail {

void bind_nlp_create_qkv_heads_decode(pybind11::module& module) {
    using OperationType = decltype(ttnn::experimental::nlp_create_qkv_heads_decode);
    ttnn::bind_registered_operation(
        module,
        ttnn::experimental::nlp_create_qkv_heads_decode,
        R"doc(
            Shuffles [1, S=1, B, head_dim * (num_heads + 2*num_kv_heads)] fused qkv matrix into Q, K, and V heads with shape [S, B, num_heads, head_dim] for Q and [S, B, num_kv_heads, head_dim] for K and V, where num_heads and num_kv_heads will be padded to nearest 32. Input must be sharded, B=32 and S=1. If ttnn pads B from some number < 32 to 32, this op respects the unpadded B.
            overlap_qk_coregrid is a boolean flag that determines whether the output Q and K heads are on same core grid. If true, then Q, K, and V heads are on the same core grid. If false, the Q and K heads are on non-overlapping core-grid useful for processing Q and K in parallel.
        )doc",
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const uint32_t num_q_heads,
               const std::optional<uint32_t> num_kv_heads,
               const std::optional<const bool> overlap_qk_coregrid,
               const std::optional<const Tensor>& batch_offset,
               const std::optional<const uint32_t> slice_size,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               std::optional<std::array<Tensor, 3>> optional_output_tensors,
               QueueId queue_id) {
                return self(
                    queue_id,
                    input_tensor,
                    num_q_heads,
                    num_kv_heads,
                    overlap_qk_coregrid,
                    batch_offset,
                    slice_size,
                    memory_config,
                    optional_output_tensors);
            },
            pybind11::arg("input_tensor").noconvert(),
            pybind11::kw_only(),
            pybind11::arg("num_heads").noconvert(),
            pybind11::arg("num_kv_heads").noconvert() = std::nullopt,
            pybind11::arg("overlap_qk_coregrid").noconvert() = true,
            pybind11::arg("batch_offset").noconvert() = std::nullopt,
            pybind11::arg("slice_size").noconvert() = std::nullopt,
            pybind11::arg("memory_config") = std::nullopt,
            pybind11::arg("output_tensors") = std::nullopt,
            pybind11::arg("queue_id") = DefaultQueueId});
}

}  // namespace ttnn::operations::experimental::transformer::detail
