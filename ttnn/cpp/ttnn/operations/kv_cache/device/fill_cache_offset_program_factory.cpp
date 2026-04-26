// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "fill_cache_offset_program_factory.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {

using namespace tt::constants;

FillCacheOffsetProgramFactory::cached_program_t FillCacheOffsetProgramFactory::create(
    const KvCacheParams& operation_attributes, const KvCacheInputs& tensor_args, Tensor& /*output_tensor*/) {
    const auto& cache_tensor = tensor_args.cache;
    const auto& input_tensor = tensor_args.input;
    const auto batch_idx = operation_attributes.batch_idx;
    const auto update_idx = operation_attributes.update_idx;
    const uint32_t sub_offset = update_idx % TILE_HEIGHT;
    Program program{};

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t single_tile_size = tt::tile_size(cb_data_format);

    uint32_t Wt = cache_tensor.padded_shape()[-1] / TILE_WIDTH;
    uint32_t input_Ht = input_tensor.padded_shape()[-2] / TILE_HEIGHT;
    uint32_t num_heads = input_tensor.padded_shape()[1];
    uint32_t cache_HtWt = cache_tensor.padded_shape()[-2] * Wt / TILE_HEIGHT;
    uint32_t cache_CHtWt = num_heads * cache_HtWt;
    uint32_t input_HtWt = input_Ht * Wt;
    uint32_t aligned_start_tile = update_idx / TILE_HEIGHT;

    uint32_t num_output_tiles_per_head = input_Ht + (sub_offset > 0 ? 1 : 0);
    uint32_t num_blocks_of_work = num_output_tiles_per_head * num_heads;

    uint32_t cache_batch_start = batch_idx * cache_CHtWt + aligned_start_tile * Wt;

    uint32_t elem_size = (input_tensor.dtype() == DataType::FLOAT32) ? 4 : 2;
    uint32_t face_row_bytes = 16 * elem_size;
    uint32_t face_bytes = 16 * 16 * elem_size;

    tt::tt_metal::IDevice* device = input_tensor.device();
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    bool row_major = true;
    uint32_t num_cores, num_blocks_per_core_group_1, num_blocks_per_core_group_2;
    CoreRangeSet all_cores, core_group_1, core_group_2;

    std::tie(
        num_cores, all_cores, core_group_1, core_group_2, num_blocks_per_core_group_1, num_blocks_per_core_group_2) =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_blocks_of_work, row_major);

    // CB 0: output tiles (double-buffered, Wt tiles per block)
    uint32_t cb_out_index = 0;
    uint32_t num_out_tiles = 2 * Wt;
    tt::tt_metal::CircularBufferConfig cb_out_config =
        tt::tt_metal::CircularBufferConfig(num_out_tiles * single_tile_size, {{cb_out_index, cb_data_format}})
            .set_page_size(cb_out_index, single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_out_config);

    // CB 1: scratch buffer for input tiles (2 tile rows = 2 * Wt tiles)
    uint32_t cb_scratch_index = 1;
    uint32_t num_scratch_tiles = 2 * Wt;
    tt::tt_metal::CircularBufferConfig cb_scratch_config =
        tt::tt_metal::CircularBufferConfig(num_scratch_tiles * single_tile_size, {{cb_scratch_index, cb_data_format}})
            .set_page_size(cb_scratch_index, single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_scratch_config);

    auto* src_buffer = input_tensor.buffer();
    auto* dst_buffer = cache_tensor.buffer();

    std::vector<uint32_t> reader_compile_time_args = {cb_out_index, cb_scratch_index, face_row_bytes, face_bytes};
    tt::tt_metal::TensorAccessorArgs(*dst_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(*src_buffer).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {cb_out_index};
    tt::tt_metal::TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/kv_cache/device/kernels/dataflow/reader_fill_cache_offset.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    tt::tt_metal::KernelHandle writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/kv_cache/device/kernels/dataflow/writer_fill_cache_offset.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    uint32_t g1_numcores = core_group_1.num_cores();
    const auto& cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, row_major);

    uint32_t block_start = 0;
    for (uint32_t i = 0; i < num_cores; i++) {
        const CoreCoord& core = cores.at(i);
        uint32_t num_blocks_per_core = (i < g1_numcores) ? num_blocks_per_core_group_1 : num_blocks_per_core_group_2;

        tt::tt_metal::SetRuntimeArgs(
            program,
            reader_kernel_id,
            core,
            {dst_buffer->address(),
             src_buffer->address(),
             Wt,
             sub_offset,
             num_output_tiles_per_head,
             input_Ht,
             cache_batch_start,
             cache_HtWt,
             input_HtWt,
             num_blocks_per_core,
             block_start});

        tt::tt_metal::SetRuntimeArgs(
            program,
            writer_kernel_id,
            core,
            {dst_buffer->address(),
             Wt,
             num_output_tiles_per_head,
             cache_batch_start,
             cache_HtWt,
             num_blocks_per_core,
             block_start});

        block_start += num_blocks_per_core;
    }

    return cached_program_t{
        std::move(program),
        shared_variables_t{
            .reader_kernel_id = reader_kernel_id,
            .writer_kernel_id = writer_kernel_id,
            .cores = cores,
            .g1_numcores = g1_numcores,
            .num_blocks_per_core_group_1 = num_blocks_per_core_group_1,
            .num_blocks_per_core_group_2 = num_blocks_per_core_group_2,
            .Wt = Wt,
            .cache_HtWt = cache_HtWt,
            .input_HtWt = input_HtWt,
            .input_Ht = input_Ht,
            .num_output_tiles_per_head = num_output_tiles_per_head,
        }};
}

void FillCacheOffsetProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const KvCacheParams& operation_attributes,
    const KvCacheInputs& tensor_args,
    Tensor& /*output_tensor*/) {
    auto& program = cached_program.program;
    const auto& reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    const auto& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;
    const auto& cores = cached_program.shared_variables.cores;
    const auto Wt = cached_program.shared_variables.Wt;
    const auto cache_HtWt = cached_program.shared_variables.cache_HtWt;

    const auto batch_idx = operation_attributes.batch_idx;
    const auto update_idx = operation_attributes.update_idx;
    const uint32_t sub_offset = update_idx % TILE_HEIGHT;
    const uint32_t aligned_start_tile = update_idx / TILE_HEIGHT;
    const uint32_t num_heads = tensor_args.input.padded_shape()[1];
    const uint32_t cache_CHtWt = num_heads * cache_HtWt;
    const uint32_t cache_batch_start = batch_idx * cache_CHtWt + aligned_start_tile * Wt;

    auto* src_buffer = tensor_args.input.buffer();
    auto* dst_buffer = tensor_args.cache.buffer();

    for (uint32_t i = 0; i < cores.size(); i++) {
        const CoreCoord& core = cores.at(i);

        {
            auto& runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
            runtime_args[0] = dst_buffer->address();
            runtime_args[1] = src_buffer->address();
            runtime_args[3] = sub_offset;
            runtime_args[6] = cache_batch_start;
        }

        {
            auto& runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
            runtime_args[0] = dst_buffer->address();
            runtime_args[3] = cache_batch_start;
        }
    }
}

}  // namespace ttnn::prim
