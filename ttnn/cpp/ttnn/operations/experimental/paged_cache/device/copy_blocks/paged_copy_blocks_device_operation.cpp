// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "paged_copy_blocks_device_operation.hpp"

#include "ttnn/device_operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

PagedCopyBlocksDeviceOperation::program_factory_t PagedCopyBlocksDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return PagedCopyBlocksProgramFactory{};
}

void PagedCopyBlocksDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    TT_FATAL(!in.caches.empty(), "paged_copy_blocks: caches must not be empty");
    const Tensor& first = in.caches.front();
    for (const Tensor& cache : in.caches) {
        TT_FATAL(
            cache.storage_type() == StorageType::DEVICE && cache.is_allocated(),
            "paged_copy_blocks: every cache must be an allocated device tensor");
        TT_FATAL(
            cache.tensor_spec() == first.tensor_spec() && cache.device() == first.device(),
            "paged_copy_blocks: every cache must have the shape, dtype, layout, memory config and device of the "
            "first; got shape {} dtype {} and shape {} dtype {}",
            cache.logical_shape(),
            cache.dtype(),
            first.logical_shape(),
            first.dtype());
    }
    TT_FATAL(
        first.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "paged_copy_blocks: caches must be interleaved, got {}",
        first.memory_config().memory_layout());
    TT_FATAL(
        first.padded_shape().rank() >= 3,
        "paged_copy_blocks: caches must have rank 3 or more with blocks on dim 0, got rank {}",
        first.padded_shape().rank());
    TT_FATAL(attrs.pages_per_block >= 1, "paged_copy_blocks: a block must hold at least one page");

    const Tensor& table = in.src_dst;
    TT_FATAL(
        table.storage_type() == StorageType::DEVICE && table.device() == first.device(),
        "paged_copy_blocks: src_dst must be on the device of the caches");
    TT_FATAL(
        table.dtype() == DataType::UINT32 || table.dtype() == DataType::INT32,
        "paged_copy_blocks: src_dst must be uint32 or int32, got {}",
        table.dtype());
    TT_FATAL(
        table.layout() == Layout::ROW_MAJOR, "paged_copy_blocks: src_dst must be ROW_MAJOR, got {}", table.layout());
    TT_FATAL(
        table.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "paged_copy_blocks: src_dst must be interleaved");
    const auto& table_shape = table.logical_shape();
    TT_FATAL(
        table_shape.rank() == 2 && table_shape[1] == 2 && table_shape[0] >= 1,
        "paged_copy_blocks: src_dst must be [num_pairs, 2], got {}",
        table_shape);
}

PagedCopyBlocksDeviceOperation::spec_return_value_t PagedCopyBlocksDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& in) {
    spec_return_value_t specs;
    specs.reserve(in.caches.size());
    for (const Tensor& cache : in.caches) {
        specs.push_back(cache.tensor_spec());
    }
    return specs;
}

PagedCopyBlocksDeviceOperation::tensor_return_value_t PagedCopyBlocksDeviceOperation::create_output_tensors(
    const operation_attributes_t&, const tensor_args_t& in) {
    return in.caches;
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::vector<Tensor> paged_copy_blocks(const std::vector<Tensor>& caches, const Tensor& src_dst) {
    using OperationType = ttnn::experimental::prim::PagedCopyBlocksDeviceOperation;
    TT_FATAL(!caches.empty(), "paged_copy_blocks: caches must not be empty");
    const Tensor& first = caches.front();
    TT_FATAL(first.is_allocated(), "paged_copy_blocks: every cache must be an allocated device tensor");
    const uint32_t blocks = first.padded_shape()[0];
    const uint32_t pages = first.buffer()->num_pages();
    TT_FATAL(
        blocks >= 1 && pages % blocks == 0, "paged_copy_blocks: {} pages do not split into {} blocks", pages, blocks);
    const auto attrs = OperationType::operation_attributes_t{.pages_per_block = pages / blocks};
    return ttnn::device_operation::launch<OperationType>(attrs, OperationType::tensor_args_t{caches, src_dst});
}

}  // namespace ttnn::prim
