// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Exercise: Implement the ProgramFactory for interleaved-to-sharded

#include "interleaved_to_sharded_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/cb_utils.hpp"

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::operations::onboarding {

InterleavedToShardedOperation::ProgramFactory::cached_program_t InterleavedToShardedOperation::ProgramFactory::create(
    const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&) {
    // TODO: Implement ProgramFactory::create
    //
    // 1. Extract shard geometry from the output tensor:
    //    - shard_spec, all_cores, shard_height_tiles, shard_width_tiles, tiles_per_shard
    //    - Nt = input width in tiles
    //
    // 2. Create an output CB (CBIndex::c_16) backed by the sharded output buffer
    //    using create_cb() with output.buffer()
    //
    // 3. Create the reader kernel with compile-time args (TensorAccessorArgs)
    //    Kernel path: "ttnn/tutorials/onboarding/e09_sharding/Stage1/exercise_cpp/device/kernels/reader.cpp"
    //
    // 4. Determine num_cores_x for core-coordinate iteration:
    //    - BLOCK_SHARDED: grid width from the CoreRange
    //    - HEIGHT/WIDTH: device grid_size.x
    //
    // 5. Set runtime args per core: {src_addr, start_tile, shard_height_tiles, shard_width_tiles, Nt}
    //    start_tile depends on shard strategy:
    //      HEIGHT_SHARDED: core_i * shard_height_tiles * Nt
    //      WIDTH_SHARDED:  core_i * shard_width_tiles
    //      BLOCK_SHARDED:  gy * shard_height_tiles * Nt + gx * shard_width_tiles
    //
    // 6. Return {program, {reader_id, cb_out_handle, num_cores, num_cores_x}}
    TT_THROW("Not implemented");
}

void InterleavedToShardedOperation::ProgramFactory::override_runtime_arguments(
    cached_program_t&, const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&) {
    // TODO: Implement override_runtime_arguments
    // Update reader runtime args[0] (buffer address) for each core
    // Update the output CB's backing address via UpdateDynamicCircularBufferAddress
}

}  // namespace ttnn::operations::onboarding
