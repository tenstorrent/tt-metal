// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

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
    const operation_attributes_t&, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    const auto& input = tensor_args.input;

    Program program{};

    // --- Shard geometry from the output tensor ---
    auto shard_spec = output.buffer()->shard_spec();
    auto all_cores = shard_spec.grid();
    uint32_t shard_height_tiles = shard_spec.shape()[0] / TILE_HEIGHT;
    uint32_t shard_width_tiles = shard_spec.shape()[1] / TILE_WIDTH;
    uint32_t tiles_per_shard = shard_height_tiles * shard_width_tiles;

    // Input width in tiles (needed for tile-id arithmetic in the reader)
    uint32_t Nt = input.logical_shape()[-1] / TILE_WIDTH;

    // --- Data format and tile size ---
    tt::DataFormat cb_format = datatype_to_dataformat_converter(input.dtype());
    uint32_t tile_size = tt::tile_size(cb_format);

    // --- Output CB: backed by the sharded output buffer ---
    auto cb_out = CBIndex::c_16;
    auto [cb_out_index, cb_out_handle] =
        create_cb(cb_out, program, all_cores, tile_size, tiles_per_shard, cb_format, output.buffer());

    // --- Reader kernel ---
    // Compile-time args: TensorAccessorArgs for the interleaved input
    std::vector<uint32_t> reader_ct_args;
    TensorAccessorArgs(*input.buffer()).append_to(reader_ct_args);

    auto reader_id = CreateKernel(
        program,
        "ttnn/tutorials/onboarding/e09_sharding/Stage1/solution_cpp/device/kernels/reader.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = reader_ct_args});

    // --- Determine num_cores_x for core-coordinate iteration ---
    auto mem_layout = output.memory_config().memory_layout();
    uint32_t num_cores_x;
    if (mem_layout == TensorMemoryLayout::BLOCK_SHARDED) {
        auto& first_range = *all_cores.ranges().begin();
        num_cores_x = first_range.end_coord.x + 1;
    } else {
        num_cores_x = input.device()->compute_with_storage_grid_size().x;
    }

    // --- Runtime args per core ---
    // start_tile depends on shard strategy:
    //   HEIGHT_SHARDED: core_i * shard_height_tiles * Nt
    //   WIDTH_SHARDED:  core_i * shard_width_tiles
    //   BLOCK_SHARDED:  gy * shard_height_tiles * Nt + gx * shard_width_tiles
    uint32_t core_idx = 0;
    for (const auto& range : all_cores.ranges()) {
        for (const auto& core : range) {
            uint32_t start_tile;
            if (mem_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
                start_tile = core_idx * shard_height_tiles * Nt;
            } else if (mem_layout == TensorMemoryLayout::WIDTH_SHARDED) {
                start_tile = core_idx * shard_width_tiles;
            } else {  // BLOCK_SHARDED
                uint32_t gx = core_idx % num_cores_x;
                uint32_t gy = core_idx / num_cores_x;
                start_tile = gy * shard_height_tiles * Nt + gx * shard_width_tiles;
            }
            SetRuntimeArgs(
                program,
                reader_id,
                core,
                {input.buffer()->address(), start_tile, shard_height_tiles, shard_width_tiles, Nt});
            core_idx++;
        }
    }

    return {std::move(program), {reader_id, cb_out_handle, all_cores.num_cores(), num_cores_x}};
}

void InterleavedToShardedOperation::ProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t&,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto& program = cached_program.program;
    auto& num_cores = cached_program.shared_variables.num_cores;
    auto& num_cores_x = cached_program.shared_variables.num_cores_x;

    // Update reader runtime args with the (potentially new) input buffer address
    for (uint32_t i = 0; i < num_cores; i++) {
        CoreCoord core = {i % num_cores_x, i / num_cores_x};
        auto& reader_args = GetRuntimeArgs(program, cached_program.shared_variables.reader_id, core);
        reader_args[0] = tensor_args.input.buffer()->address();
    }

    // Update the output CB's backing address (output tensor may have moved)
    UpdateDynamicCircularBufferAddress(program, cached_program.shared_variables.cb_out_handle, *output.buffer());
}

}  // namespace ttnn::operations::onboarding
