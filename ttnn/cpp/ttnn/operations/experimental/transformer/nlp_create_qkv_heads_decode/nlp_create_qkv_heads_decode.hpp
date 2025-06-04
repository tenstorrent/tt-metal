// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/nlp_create_qkv_heads_decode_device_operation.hpp"
#include "ttnn/run_operation.hpp"

namespace ttnn {
namespace operations::experimental::transformer {

struct NLPCreateHeadsDecodeOperation {
    static std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> invoke(
        QueueId queue_id,
        const Tensor& input_tensor,
        const uint32_t num_heads,
        const std::optional<const uint32_t> num_kv_heads,
        const std::optional<const bool> overlap_qk_coregrid = true,
        const std::optional<const Tensor>& batch_offset = std::nullopt,
        const std::optional<const uint32_t> slice_size = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<std::array<Tensor, 3>> optional_output_tensors = std::nullopt);

    static std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> invoke(
        const Tensor& input_tensor,
        const uint32_t num_heads,
        const std::optional<const uint32_t> num_kv_heads,
        const std::optional<const bool> overlap_qk_coregrid = true,
        const std::optional<const Tensor>& batch_offset = std::nullopt,
        const std::optional<const uint32_t> slice_size = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<std::array<Tensor, 3>> optional_output_tensors = std::nullopt);
};

}  // namespace operations::experimental::transformer

namespace experimental {

constexpr auto nlp_create_qkv_heads_decode = ttnn::register_operation<
    "ttnn::experimental::nlp_create_qkv_heads_decode",
    ttnn::operations::experimental::transformer::NLPCreateHeadsDecodeOperation>();

}  // namespace experimental

}  // namespace ttnn
