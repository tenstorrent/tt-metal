#pragma once

#include <optional>

#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/validation.hpp"

namespace ttnn {
namespace operations {
namespace embedding {

/**
 * Validates the rank of input tensors for embedding operations.
 * Ensures that input_ids tensor has rank 2 and weight tensor has rank 2.
 * 
 * @param input_ids: Input tensor containing indices (must be rank 2)
 * @param weight: Weight tensor containing embeddings (must be rank 2)
 * @throws std::runtime_error if rank validation fails
 */
void validate_embedding_input_ranks(const ttnn::Tensor& input_ids, const ttnn::Tensor& weight);

struct ExecuteEmbedding {
    /**
     * Performs embedding lookup operation.
     * 
     * Args:
     *     input_ids (ttnn::Tensor): Input tensor of shape [batch_size, sequence_length] containing indices.
     *                               Must have rank 2 and dtype UINT32.
     *     weight (ttnn::Tensor): Weight tensor of shape [vocab_size, embedding_dim] containing embeddings.
     *                           Must have rank 2.
     *     output_mem_config (ttnn::MemoryConfig, optional): Memory configuration for output tensor.
     *     output_dtype (ttnn::DataType, optional): Data type for output tensor.
     * 
     * Returns:
     *     ttnn::Tensor: Output tensor of shape [batch_size, sequence_length, embedding_dim].
     * 
     * Raises:
     *     std::runtime_error: If input tensors don't meet rank requirements (both must be rank 2).
     * 
     * Example:
     *     >>> input_ids = ttnn.zeros([2, 10], dtype=ttnn.uint32, device=device)  # batch_size=2, seq_len=10
     *     >>> weight = ttnn.zeros([1000, 128], device=device)  # vocab_size=1000, embedding_dim=128
     *     >>> output = ttnn.embedding(input_ids, weight)
     *     >>> print(output.shape)  # [2, 10, 128]
     */
    static ttnn::Tensor operator()(
        const ttnn::Tensor& input_ids,
        const ttnn::Tensor& weight,
        const std::optional<ttnn::MemoryConfig>& output_mem_config = std::nullopt,
        const std::optional<ttnn::DataType>& output_dtype = std::nullopt);
};

}  // namespace embedding

constexpr auto embedding = ttnn::register_operation_with_auto_launch_op<
    "ttnn::embedding",
    operations::embedding::ExecuteEmbedding>();

}  // namespace operations

using operations::embedding;

}  // namespace ttnn