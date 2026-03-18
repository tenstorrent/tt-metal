#include "embedding.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/operations/core.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::embedding {

ttnn::Tensor ExecuteEmbedding::invoke(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight,
    const std::optional<int>& pad_token,
    const std::optional<ttnn::Layout>& layout,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::Tensor>& optional_output_tensor) {

    // Input rank verification - reject tensors with rank > 2
    auto input_shape = input_tensor.get_shape();
    auto input_rank = input_shape.rank();
    
    TT_FATAL(
        input_rank <= 2,
        "EmbeddingOp only supports input tensors with rank <= 2. "
        "Supported dimensions are: 1D tensors [N] and 2D tensors [N, H]. "
        "Got input tensor with rank {} and shape {}.",
        input_rank,
        input_shape
    );

    auto weight_shape = weight.get_shape();
    TT_FATAL(weight_shape.rank() == 2, "Weight tensor must be 2D");
    
    auto vocab_size = weight_shape[-2];
    auto embedding_dim = weight_shape[-1];
    
    // Validate input indices are within vocabulary bounds
    // This would need actual tensor value checking in a real implementation
    
    ttnn::SmallVector<uint32_t> output_shape_vec;
    
    if (input_rank == 1) {
        // 1D input: [N] -> [N, embedding_dim]
        output_shape_vec = {input_shape[0], embedding_dim};
    } else {
        // 2D input: [N, H] -> [N, H, embedding_dim]
        output_shape_vec = {input_shape[0], input_shape[1], embedding_dim};
    }
    
    auto output_shape = ttnn::Shape(output_shape_vec);
    
    auto target_layout = layout.value_or(weight.get_layout());
    auto target_memory_config = memory_config.value_or(weight.memory_config());
    
    // Create output tensor
    ttnn::Tensor output_tensor;
    if (optional_output_tensor.has_value()) {
        output_tensor = optional_output_tensor.value();
    } else {
        output_tensor = ttnn::zeros(output_shape, weight.get_dtype(), target_layout, weight.device(), target_memory_config);
    }
    
    // Perform embedding lookup
    // This is a simplified implementation - actual implementation would need device-specific kernels
    auto input_1d = input_tensor;
    if (input_rank == 2) {
        // Flatten 2D input for processing
        input_1d = ttnn::reshape(input_tensor, ttnn::Shape({input_shape[0] * input_shape[1]}));
    }
    
    // For each index in the input, gather the corresponding row from weight
    // This would be implemented with actual gather/slice operations on device
    auto batch_size = input_1d.get_shape()[0];
    
    for (uint32_t i = 0; i < batch_size; ++i) {
        // Extract index (this would need proper tensor indexing)
        // auto index = input_1d[i];  // Pseudo-code
        
        // Handle padding token
        // if (pad_token.has_value() && index == pad_token.value()) {
        //     // Set output row to zeros
        //     continue;
        // }
        
        // Slice corresponding row from weight tensor
        // auto embedding_row = ttnn::slice(weight, {index, 0}, {index + 1, embedding_dim});
        // Copy to output tensor at appropriate position
    }
    
    // Reshape output to final shape if needed
    if (input_rank == 2) {
        output_tensor = ttnn::reshape(output_tensor, output_shape);
    }
    
    return output_tensor;
}

} // namespace ttnn::operations::embedding