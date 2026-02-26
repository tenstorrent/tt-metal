// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "interleaved_to_sharded_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include <cstdint>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>

namespace ttnn::operations::onboarding {

InterleavedToShardedOperation::program_factory_t InterleavedToShardedOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return ProgramFactory{};
}

void InterleavedToShardedOperation::validate_on_program_cache_miss(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    TT_FATAL(tensor_args.input.layout() == Layout::TILE, "Input must be in TILE layout");
    TT_FATAL(
        tensor_args.input.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Input must be INTERLEAVED");
}

void InterleavedToShardedOperation::validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&) {
}

InterleavedToShardedOperation::spec_return_value_t InterleavedToShardedOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    auto shape = tensor_args.input.logical_shape();
    uint32_t M = shape[-2];
    uint32_t N = shape[-1];
    uint32_t Mt = M / tt::constants::TILE_HEIGHT;
    uint32_t Nt = N / tt::constants::TILE_WIDTH;
    auto grid_size = tensor_args.input.device()->compute_with_storage_grid_size();
    uint32_t total_cores = grid_size.x * grid_size.y;

    TensorMemoryLayout shard_strategy = attrs.shard_strategy;
    uint32_t shard_height;
    uint32_t shard_width;
    CoreRangeSet output_core_grid;

    if (shard_strategy == TensorMemoryLayout::HEIGHT_SHARDED) {
        uint32_t num_cores = std::min<uint32_t>(Mt, total_cores);
        shard_height = tt::round_up(tt::div_up(M, num_cores), tt::constants::TILE_HEIGHT);
        shard_width = N;
        output_core_grid = tt::tt_metal::num_cores_to_corerangeset(num_cores, grid_size, true);
    } else if (shard_strategy == TensorMemoryLayout::WIDTH_SHARDED) {
        uint32_t num_cores = std::min<uint32_t>(Nt, total_cores);
        shard_width = tt::round_up(tt::div_up(N, num_cores), tt::constants::TILE_WIDTH);
        shard_height = M;
        output_core_grid = tt::tt_metal::num_cores_to_corerangeset(num_cores, grid_size, true);
    } else {  // BLOCK_SHARDED
        uint32_t num_cores_y = std::min<uint32_t>(Mt, grid_size.y);
        uint32_t num_cores_x = std::min<uint32_t>(Nt, grid_size.x);
        shard_height = tt::round_up(tt::div_up(M, num_cores_y), tt::constants::TILE_HEIGHT);
        shard_width = tt::round_up(tt::div_up(N, num_cores_x), tt::constants::TILE_WIDTH);
        output_core_grid = CoreRangeSet(CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1}));
    }

    tt::tt_metal::ShardSpec shard_spec{output_core_grid, {shard_height, shard_width}};

    return TensorSpec(
        Shape(shape),
        tt::tt_metal::TensorLayout(
            tensor_args.input.dtype(),
            tt::tt_metal::PageConfig(tensor_args.input.layout()),
            MemoryConfig{shard_strategy, BufferType::L1, shard_spec}));
}

InterleavedToShardedOperation::tensor_return_value_t InterleavedToShardedOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(attrs, tensor_args), tensor_args.input.device());
}

}  // namespace ttnn::operations::onboarding

namespace ttnn::prim {

ttnn::Tensor onboarding_interleaved_to_sharded(const ttnn::Tensor& input, TensorMemoryLayout shard_strategy) {
    using OperationType = ttnn::operations::onboarding::InterleavedToShardedOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{shard_strategy}, OperationType::tensor_args_t{input});
}

}  // namespace ttnn::prim
