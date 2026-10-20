// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "paged_pool_gather_device_operation.hpp"

#include <tt-metalium/constants.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::experimental::paged_pool_gather {

PagedPoolGatherDeviceOperation::program_factory_t PagedPoolGatherDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return PagedPoolGatherProgramFactory{};
}

void PagedPoolGatherDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    using namespace tt::constants;
    const auto& cache = in.cache;
    const auto& pt = in.page_table;
    const auto& ids = in.pool_ids;
    for (const auto* t : {&cache, &pt, &ids}) {
        TT_FATAL(
            t->storage_type() == StorageType::DEVICE && t->buffer() != nullptr,
            "paged_pool_gather: inputs must be allocated on device");
        TT_FATAL(t->device() == cache.device(), "paged_pool_gather: inputs must share a device");
        TT_FATAL(!t->is_sharded(), "paged_pool_gather: inputs must be interleaved");
    }
    TT_FATAL(
        cache.layout() == Layout::TILE && cache.dtype() == DataType::BFLOAT16,
        "paged_pool_gather: cache must be bf16 TILE");
    const auto& cs = cache.logical_shape();
    TT_FATAL(
        cs.rank() == 4 && cs[1] == 1 && cs[2] == TILE_HEIGHT && cs[3] % TILE_WIDTH == 0,
        "paged_pool_gather: cache must be [max_blocks, 1, 32, W] with W % 32 == 0, got {}",
        cs);
    TT_FATAL(
        pt.layout() == Layout::ROW_MAJOR && pt.dtype() == DataType::INT32 && pt.logical_shape().rank() == 2,
        "paged_pool_gather: page_table must be int32 ROW_MAJOR [B_table, blocks_per_seq]");
    TT_FATAL(
        ids.layout() == Layout::ROW_MAJOR && ids.dtype() == DataType::UINT32 && ids.logical_shape().rank() == 4 &&
            ids.logical_shape()[1] == 1 && ids.logical_shape()[2] == 1,
        "paged_pool_gather: pool_ids must be uint32 ROW_MAJOR [B, 1, 1, N], got {}",
        ids.logical_shape());
    const uint32_t B = ids.logical_shape()[0];
    TT_FATAL(
        B >= 1 && B <= static_cast<uint32_t>(pt.logical_shape()[0]),
        "paged_pool_gather: pool_ids batch {} must be within the page table rows {}",
        B,
        pt.logical_shape()[0]);
    const uint32_t P = attrs.pool_size;
    TT_FATAL(
        P >= 1 && P <= 16 && (16 % P) == 0,
        "paged_pool_gather: pool_size {} must be 1, 2, 4, 8 or 16 (a pool stays inside one tile face)",
        P);
    TT_FATAL(
        attrs.num_pools == static_cast<uint32_t>(ids.logical_shape()[3]) && attrs.num_pools >= 1,
        "paged_pool_gather: num_pools {} must equal the pool_ids width {}",
        attrs.num_pools,
        ids.logical_shape()[3]);
    TT_FATAL(
        attrs.out_blocks * TILE_HEIGHT >= attrs.num_pools * P,
        "paged_pool_gather: out_blocks {} too small for {} pools of {}",
        attrs.out_blocks,
        attrs.num_pools,
        P);
    TT_FATAL(
        attrs.output_mem_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "paged_pool_gather: the output must be interleaved");
}

PagedPoolGatherDeviceOperation::spec_return_value_t PagedPoolGatherDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    const auto& cs = in.cache.logical_shape();
    const uint32_t B = in.pool_ids.logical_shape()[0];
    ttnn::Shape shape({B * attrs.out_blocks, 1, cs[2], cs[3]});
    return TensorSpec(shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), attrs.output_mem_config));
}

PagedPoolGatherDeviceOperation::tensor_return_value_t PagedPoolGatherDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    return create_device_tensor(compute_output_specs(attrs, in), in.cache.device());
}

}  // namespace ttnn::operations::experimental::paged_pool_gather

namespace ttnn::experimental {

Tensor paged_pool_gather(
    const Tensor& cache,
    const Tensor& page_table,
    const Tensor& pool_ids,
    uint32_t pool_size,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config) {
    using OperationType = ttnn::operations::experimental::paged_pool_gather::PagedPoolGatherDeviceOperation;
    const uint32_t N = pool_ids.logical_shape()[-1];
    const uint32_t out_blocks = (N * pool_size + tt::constants::TILE_HEIGHT - 1) / tt::constants::TILE_HEIGHT;
    auto attrs = OperationType::operation_attributes_t{
        .pool_size = pool_size,
        .num_pools = N,
        .out_blocks = out_blocks,
        .output_mem_config = memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG),
    };
    auto tensor_args = OperationType::tensor_args_t{.cache = cache, .page_table = page_table, .pool_ids = pool_ids};
    return ttnn::device_operation::launch<OperationType>(attrs, tensor_args);
}

}  // namespace ttnn::experimental
