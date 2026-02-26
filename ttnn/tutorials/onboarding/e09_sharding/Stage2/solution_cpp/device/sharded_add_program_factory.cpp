// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "sharded_add_device_operation.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/bfloat16.hpp>
#include "ttnn/operations/cb_utils.hpp"

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::operations::onboarding {

ShardedAddOperation::ProgramFactory::cached_program_t ShardedAddOperation::ProgramFactory::create(
    const operation_attributes_t&, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    const auto& input_a = tensor_args.input_a;
    const auto& input_b = tensor_args.input_b;

    Program program{};

    // --- Shard geometry from input_a ---
    auto shard_spec = input_a.buffer()->shard_spec();
    auto all_cores = shard_spec.grid();
    uint32_t shard_height_tiles = shard_spec.shape()[0] / TILE_HEIGHT;
    uint32_t shard_width_tiles = shard_spec.shape()[1] / TILE_WIDTH;
    uint32_t tiles_per_shard = shard_height_tiles * shard_width_tiles;

    // --- Data format and tile size ---
    tt::DataFormat cb_format = datatype_to_dataformat_converter(input_a.dtype());
    uint32_t tile_size = tt::tile_size(cb_format);

    // --- 3 CBs backed by sharded buffers ---
    auto cb_a = CBIndex::c_0;
    auto cb_b = CBIndex::c_1;
    auto cb_out = CBIndex::c_16;

    auto [cb_a_index, cb_a_handle] =
        create_cb(cb_a, program, all_cores, tile_size, tiles_per_shard, cb_format, input_a.buffer());
    auto [cb_b_index, cb_b_handle] =
        create_cb(cb_b, program, all_cores, tile_size, tiles_per_shard, cb_format, input_b.buffer());
    auto [cb_out_index, cb_out_handle] =
        create_cb(cb_out, program, all_cores, tile_size, tiles_per_shard, cb_format, output.buffer());

    // --- 3 kernels with compile_args = {tiles_per_shard} ---
    std::vector<uint32_t> ct_args = {tiles_per_shard};

    auto reader_id = CreateKernel(
        program,
        "ttnn/tutorials/onboarding/e09_sharding/Stage2/solution_cpp/device/kernels/reader.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = ct_args});

    auto compute_id = CreateKernel(
        program,
        "ttnn/tutorials/onboarding/e09_sharding/Stage2/solution_cpp/device/kernels/compute.cpp",
        all_cores,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .compile_args = ct_args});

    auto writer_id = CreateKernel(
        program,
        "ttnn/tutorials/onboarding/e09_sharding/Stage2/solution_cpp/device/kernels/writer.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = ct_args});

    // No per-core runtime args needed — CBs are globally allocated and backed by sharded buffers

    return {
        std::move(program),
        {reader_id, compute_id, writer_id, cb_a_handle, cb_b_handle, cb_out_handle, all_cores.num_cores()}};
}

void ShardedAddOperation::ProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t&,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto& program = cached_program.program;

    // Buffers may have moved — update all 3 CB backing addresses
    UpdateDynamicCircularBufferAddress(
        program, cached_program.shared_variables.cb_a_handle, *tensor_args.input_a.buffer());
    UpdateDynamicCircularBufferAddress(
        program, cached_program.shared_variables.cb_b_handle, *tensor_args.input_b.buffer());
    UpdateDynamicCircularBufferAddress(program, cached_program.shared_variables.cb_out_handle, *output.buffer());
}

}  // namespace ttnn::operations::onboarding
