// SPDX-License-Identifier: MIT

#include "ttnn/operations/reduction/generic/generic_reductions_with_padding.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core.hpp"
#include "tt_metal/common/constants.hpp"

namespace ttnn {
namespace operations {
namespace reduction {

namespace {

bool supports_native_padding(const Tensor& input_tensor, ReduceOpMath reduce_op, const std::vector<int64_t>& dims) {
    // Check if tensor is on device and in tile layout
    if (!input_tensor.is_allocated() || input_tensor.get_layout() != Layout::TILE) {
        return false;
    }
    
    // Check if reduction operation supports native padding
    switch (reduce_op) {
        case ReduceOpMath::SUM:
        case ReduceOpMath::MEAN:
        case ReduceOpMath::MAX:
        case ReduceOpMath::MIN:
            break;
        default:
            return false;
    }
    
    // Check if dimensions are supported for native padding
    const auto& shape = input_tensor.get_legacy_shape();
    for (int64_t dim : dims) {
        if (dim < 0) {
            dim += shape.rank();
        }
        // Only support reduction on last two dimensions for now
        if (dim < shape.rank() - 2) {
            return false;
        }
    }
    
    return true;
}

Tensor reduce_with_native_padding_impl(
    const Tensor& input_tensor,
    ReduceOpMath reduce_op,
    const std::vector<int64_t>& dims,
    bool keep_dim,
    const std::optional<Tensor>& output_tensor,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<DataType> output_dtype) {
    
    // Native padding kernel dispatch would go here
    // For now, this is a placeholder that would call the actual kernel
    
    // Create operation attributes
    ReduceOpParallelizationStrategy strategy = ReduceOpParallelizationStrategy::MULTI_CORE_H;
    
    return ttnn::prim::reduce(
        input_tensor,
        reduce_op,
        dims,
        keep_dim,
        output_tensor,
        memory_config.value_or(input_tensor.memory_config()),
        output_dtype,
        strategy,
        true  // use_native_padding flag
    );
}

}  // anonymous namespace

Tensor ttnn_reduce_with_native_padding(
    const Tensor& input_tensor,
    ReduceOpMath reduce_op,
    const std::vector<int64_t>& dims,
    bool keep_dim,
    const std::optional<Tensor>& output_tensor,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<DataType> output_dtype) {
    
    // Validate input parameters
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Input tensor must be on device");
    TT_FATAL(!dims.empty(), "Reduction dimensions cannot be empty");
    
    // Check if native padding is supported
    if (supports_native_padding(input_tensor, reduce_op, dims)) {
        return reduce_with_native_padding_impl(
            input_tensor,
            reduce_op,
            dims,
            keep_dim,
            output_tensor,
            memory_config,
            output_dtype
        );
    }
    
    // Fall back to original implementation with explicit padding
    return ttnn::prim::reduce(
        input_tensor,
        reduce_op,
        dims,
        keep_dim,
        output_tensor,
        memory_config.value_or(input_tensor.memory_config()),
        output_dtype
    );
}

Tensor sum_with_native_padding(
    const Tensor& input_tensor,
    const std::vector<int64_t>& dims,
    bool keep_dim,
    const std::optional<Tensor>& output_tensor,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<DataType> output_dtype) {
    
    return ttnn_reduce_with_native_padding(
        input_tensor,
        ReduceOpMath::SUM,
        dims,
        keep_dim,
        output_tensor,
        memory_config,
        output_dtype
    );
}

Tensor mean_with_native_padding(
    const Tensor& input_tensor,
    const std::vector<int64_t>& dims,
    bool keep_dim,
    const std::optional<Tensor>& output_tensor,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<DataType> output_dtype) {
    
    return ttnn_reduce_with_native_padding(
        input_tensor,
        ReduceOpMath::MEAN,
        dims,
        keep_dim,
        output_tensor,
        memory_config,
        output_dtype
    );
}

Tensor max_with_native_padding(
    const Tensor& input_tensor,
    const std::vector<int64_t>& dims,
    bool keep_dim,
    const std::optional<Tensor>& output_tensor,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<DataType> output_dtype) {
    
    return ttnn_reduce_with_native_padding(
        input_tensor,
        ReduceOpMath::MAX,
        dims,
        keep_dim,
        output_tensor,
        memory_config,
        output_dtype
    );
}

Tensor min_with_native_padding(
    const Tensor& input_tensor,
    const std::vector<int64_t>& dims,
    bool keep_dim,
    const std::optional<Tensor>& output_tensor,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<DataType> output_dtype) {
    
    return ttnn_reduce_with_native_padding(
        input_tensor,
        ReduceOpMath::MIN,
        dims,
        keep_dim,
        output_tensor,
        memory_config,
        output_dtype
    );
}

}  // namespace reduction
}  // namespace operations
}  // namespace ttnn