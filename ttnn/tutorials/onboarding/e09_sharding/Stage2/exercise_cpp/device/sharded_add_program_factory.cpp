// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Exercise: Implement the ProgramFactory for sharded elementwise add

#include "sharded_add_device_operation.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/bfloat16.hpp>
#include "ttnn/operations/cb_utils.hpp"

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::operations::onboarding {

ShardedAddOperation::ProgramFactory::cached_program_t ShardedAddOperation::ProgramFactory::create(
    const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&) {
    // TODO: Implement ProgramFactory::create
    //
    // 1. Extract shard geometry from input_a:
    //    - shard_spec, all_cores, shard_height_tiles, shard_width_tiles, tiles_per_shard
    //
    // 2. Create 3 CBs backed by sharded buffers:
    //    - cb_a   (c_0)  backed by input_a.buffer()
    //    - cb_b   (c_1)  backed by input_b.buffer()
    //    - cb_out (c_16) backed by output.buffer()
    //    All with tiles_per_shard pages.
    //
    // 3. Create 3 kernels with compile_args = {tiles_per_shard}:
    //    - reader  (DataMovement, RISCV_1) — signals tiles ready
    //    - compute (ComputeConfig) — add_tiles loop
    //    - writer  (DataMovement, RISCV_0) — waits for output
    //    Kernel paths:
    //    "ttnn/tutorials/onboarding/e09_sharding/Stage2/exercise_cpp/device/kernels/{reader,compute,writer}.cpp"
    //
    // 4. No per-core runtime args needed.
    //
    // 5. Return {program, {reader_id, compute_id, writer_id, cb_a_handle, cb_b_handle, cb_out_handle, num_cores}}
    TT_THROW("Not implemented");
}

void ShardedAddOperation::ProgramFactory::override_runtime_arguments(
    cached_program_t&, const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&) {
    // TODO: Implement override_runtime_arguments
    //
    // Update all 3 CB backing addresses (buffers may have moved):
    //   UpdateDynamicCircularBufferAddress(program, cb_a_handle, *input_a.buffer())
    //   UpdateDynamicCircularBufferAddress(program, cb_b_handle, *input_b.buffer())
    //   UpdateDynamicCircularBufferAddress(program, cb_out_handle, *output.buffer())
}

}  // namespace ttnn::operations::onboarding
