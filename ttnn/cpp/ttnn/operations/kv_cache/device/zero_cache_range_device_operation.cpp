// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "zero_cache_range_device_operation.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::prim {

void ZeroCacheRangeOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& cache_tensor = tensor_args.cache;
    TT_FATAL(cache_tensor.storage_type() == StorageType::DEVICE, "Cache tensor must be on device!");
    TT_FATAL(cache_tensor.buffer() != nullptr, "Cache tensor must be allocated in a buffer on device!");
    TT_FATAL(cache_tensor.layout() == Layout::TILE, "Cache tensor must be in TILE layout!");
    TT_FATAL(
        cache_tensor.memory_config().memory_layout() == TensorMemoryLayout::ND_SHARDED,
        "Cache tensor memory layout must be ND_SHARDED but got {}",
        cache_tensor.memory_config().memory_layout());
    TT_FATAL(
        cache_tensor.dtype() == DataType::FLOAT32 || cache_tensor.dtype() == DataType::BFLOAT16 ||
            cache_tensor.dtype() == DataType::BFLOAT8_B,
        "Cache tensor dtype must be FLOAT32, BFLOAT16, or BFLOAT8_B but got {}",
        cache_tensor.dtype());
    TT_FATAL(
        args.start_page < args.end_page,
        "start_page ({}) must be less than end_page ({})",
        args.start_page,
        args.end_page);
    TT_FATAL(
        args.end_page <= cache_tensor.buffer()->num_pages(),
        "end_page ({}) must be <= total pages in cache ({})",
        args.end_page,
        cache_tensor.buffer()->num_pages());
}

TensorSpec ZeroCacheRangeOperation::compute_output_specs(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    // In-place operation: cache tensor is the output.
    return tensor_args.cache.tensor_spec();
}

Tensor ZeroCacheRangeOperation::create_output_tensors(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    // In-place operation: cache tensor is the output.
    return tensor_args.cache;
}

tt::tt_metal::operation::Hash ZeroCacheRangeOperation::compute_program_hash(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    // Hash only on cache tensor spec, not page range (page range is runtime args only).
    return tt::tt_metal::operation::hash_operation<ZeroCacheRangeOperation>(std::vector<Tensor>{tensor_args.cache});
}

Tensor zero_cache_range(const Tensor& cache, const uint32_t start_page, const uint32_t end_page) {
    return ttnn::device_operation::launch<ZeroCacheRangeOperation>(
        ZeroCacheRangeParams{
            .start_page = start_page,
            .end_page = end_page,
        },
        ZeroCacheRangeInputs{.cache = cache});
}

}  // namespace ttnn::prim
