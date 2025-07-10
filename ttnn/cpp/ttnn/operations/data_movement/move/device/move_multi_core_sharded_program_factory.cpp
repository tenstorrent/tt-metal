// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/data_movement/move/device/move_multi_core_sharded_program_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/allocator.hpp>
#include <algorithm>

#include <tt-metalium/hal.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

// Sharded buffers are mapped to CBs. Move from top of src CB to dst CB
operation::ProgramWithCallbacks move_multi_core_sharded(const Tensor& input, Tensor& output) {
    tt::tt_metal::Program program{};

    tt::DataFormat cb_data_format = datatype_to_dataformat_converter(input.dtype());
    auto shard_spec = input.shard_spec().value();
    auto shard_shape = shard_spec.shape;
    auto shard_grid = shard_spec.grid;
    const auto& input_shape = input.logical_shape();
    auto input_dtype = input.dtype();
    auto input_layout = input.layout();
    TT_FATAL(
        input_layout == output.layout() && input_dtype == output.dtype() &&
            shard_shape == output.shard_spec().value().shape && input_shape == output.logical_shape(),
        "Error");
    const uint32_t src_cb_sharded = tt::CBIndex::c_0;
    const uint32_t dst_cb_sharded = tt::CBIndex::c_1;

    uint32_t total_size_bytes = input.buffer()->aligned_size_per_bank();
    uint32_t page_size_bytes = input.buffer()->aligned_page_size();

    CircularBufferConfig src_cb_sharded_config =
        CircularBufferConfig(total_size_bytes, {{src_cb_sharded, cb_data_format}})
            .set_page_size(src_cb_sharded, page_size_bytes);
    src_cb_sharded_config.set_globally_allocated_address(*input.buffer());
    auto src_sharded_cb = tt::tt_metal::CreateCircularBuffer(program, shard_grid, src_cb_sharded_config);

    CircularBufferConfig dst_cb_sharded_config =
        CircularBufferConfig(total_size_bytes, {{dst_cb_sharded, cb_data_format}})
            .set_page_size(dst_cb_sharded, page_size_bytes);
    dst_cb_sharded_config.set_globally_allocated_address(*output.buffer());
    auto dst_sharded_cb = tt::tt_metal::CreateCircularBuffer(program, shard_grid, dst_cb_sharded_config);

    auto input_buffer_address = input.buffer()->address();
    auto output_buffer_address = output.buffer()->address();

    uint32_t move_chunk_size_bytes = output_buffer_address - input_buffer_address;
    TT_FATAL(
        input.buffer()->alignment() == output.buffer()->alignment(),
        "Expected input buffer alignment ({} B) and output buffer alignment ({} B) to be equal",
        input.buffer()->alignment(),
        output.buffer()->alignment());
    TT_FATAL(
        move_chunk_size_bytes % input.buffer()->alignment() == 0,
        "Expected chunk size bytes to move to be {} byte aligned.",
        input.buffer()->alignment());
    uint32_t num_chunks = total_size_bytes / move_chunk_size_bytes;
    uint32_t remainder_chunk_size_bytes = total_size_bytes % move_chunk_size_bytes;

    std::vector<uint32_t> reader_compile_time_args = {src_cb_sharded, dst_cb_sharded};
    KernelHandle kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/move/device/kernels/dataflow/reader_unary_local_l1_copy_backwards.cpp",
        shard_grid,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::NOC_1, .compile_args = reader_compile_time_args});
    const std::array runtime_args = {total_size_bytes, num_chunks, move_chunk_size_bytes, remainder_chunk_size_bytes};
    SetRuntimeArgs(program, kernel_id, shard_grid, runtime_args);

    const auto& cores = corerange_to_cores(shard_grid, std::nullopt, true);
    auto override_runtime_args_callback = [shard_grid = shard_grid,
                                           kernel_id = kernel_id,
                                           src_sharded_cb = src_sharded_cb,
                                           dst_sharded_cb = dst_sharded_cb,
                                           total_size_bytes = total_size_bytes,
                                           cores = cores](
                                              const void* operation,
                                              Program& program,
                                              const std::vector<Tensor>& input_tensors,
                                              const std::vector<std::optional<const Tensor>>& optional_input_tensors,
                                              const std::vector<Tensor>& output_tensors) {
        auto src_buffer = input_tensors.at(0).buffer();
        auto dst_buffer = output_tensors.at(0).buffer();
        UpdateDynamicCircularBufferAddress(program, src_sharded_cb, *src_buffer);
        UpdateDynamicCircularBufferAddress(program, dst_sharded_cb, *dst_buffer);
        auto input_buffer_address = src_buffer->address();
        auto output_buffer_address = dst_buffer->address();
        uint32_t move_chunk_size_bytes = output_buffer_address - input_buffer_address;
        uint32_t num_chunks = total_size_bytes / move_chunk_size_bytes;
        uint32_t remainder_chunk_size_bytes = total_size_bytes % move_chunk_size_bytes;
        std::vector<uint32_t> new_runtime_args = {
            total_size_bytes, num_chunks, move_chunk_size_bytes, remainder_chunk_size_bytes};
        auto& runtime_args_by_core = GetRuntimeArgs(program, kernel_id);
        for (const auto& core : cores) {
            auto& runtime_args = runtime_args_by_core[core.x][core.y];
            std::copy(new_runtime_args.begin(), new_runtime_args.end(), runtime_args.data());
        }
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_args_callback};
}

}  // namespace ttnn::operations::data_movement
