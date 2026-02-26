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
    (void)attrs;
    (void)tensor_args;
    // TODO: Implement compute_output_specs
    //
    // 1. Get M, N from input shape; compute Mt, Nt (tile counts)
    // 2. Get grid_size from the device
    // 3. Branch on attrs.shard_strategy:
    //    - HEIGHT_SHARDED: num_cores = min(Mt, total_cores), shard = {shard_height, N}
    //      Use num_cores_to_corerangeset for a 1D grid
    //    - WIDTH_SHARDED:  num_cores = min(Nt, total_cores), shard = {M, shard_width}
    //      Use num_cores_to_corerangeset for a 1D grid
    //    - BLOCK_SHARDED:  num_cores_y = min(Mt, grid_y), num_cores_x = min(Nt, grid_x)
    //      shard = {shard_height, shard_width}, use CoreRange for a 2D grid
    // 4. Round shard dimensions up to tile boundaries
    // 5. Build ShardSpec, MemoryConfig, and return TensorSpec
    TT_THROW("Not implemented");
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
