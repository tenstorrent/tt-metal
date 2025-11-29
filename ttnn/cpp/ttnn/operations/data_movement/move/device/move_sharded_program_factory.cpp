// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "move_sharded_program_factory.hpp"

#include <math.h>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/hal.hpp>

namespace ttnn::operations::data_movement::move::program {

MoveShardedProgramFactory::cached_program_t MoveShardedProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt::constants;
    using namespace tt::tt_metal;
    const Tensor& input = tensor_args.input_tensor;
    tensor_return_value_t& output = tensor_return_value;

    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();
    MoveShardedProgramFactory::shared_variables_t shared_vars{};

    tt::DataFormat cb_data_format = datatype_to_dataformat_converter(input.dtype());
    const auto shard_spec = input.shard_spec().value();
    const auto shard_shape = shard_spec.shape;
    const auto shard_grid = shard_spec.grid;
    const auto& input_shape = input.logical_shape();
    const DataType input_dtype = input.dtype();
    const Layout input_layout = input.layout();
    TT_FATAL(
        input_layout == output.layout() && input_dtype == output.dtype() &&
            shard_shape == output.shard_spec().value().shape && input_shape == output.logical_shape(),
        "Error");
    const uint32_t src_cb_sharded = tt::CBIndex::c_0;
    const uint32_t dst_cb_sharded = tt::CBIndex::c_1;

    const uint32_t total_size_bytes = input.buffer()->aligned_size_per_bank();
    const uint32_t page_size_bytes = input.buffer()->aligned_page_size();

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

    shared_vars.src_sharded_cb = src_sharded_cb;
    shared_vars.dst_sharded_cb = dst_sharded_cb;
    shared_vars.total_size_bytes = total_size_bytes;
    shared_vars.cores = corerange_to_cores(shard_grid, std::nullopt, true);

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

    shared_vars.kernel_id = kernel_id;

    const std::array runtime_args = {total_size_bytes, num_chunks, move_chunk_size_bytes, remainder_chunk_size_bytes};
    tt::tt_metal::SetRuntimeArgs(program, kernel_id, shard_grid, runtime_args);

    return MoveShardedProgramFactory::cached_program_t{std::move(program), std::move(shared_vars)};
}

void MoveShardedProgramFactory::override_runtime_arguments(
    MoveShardedProgramFactory::cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt::tt_metal;

    auto& program = cached_program.program;
    const Tensor& input = tensor_args.input_tensor;
    tensor_return_value_t& output = tensor_return_value;

    Buffer* src_buffer = input.buffer();
    Buffer* dst_buffer = output.buffer();

    UpdateDynamicCircularBufferAddress(program, cached_program.shared_variables.src_sharded_cb, *src_buffer);
    UpdateDynamicCircularBufferAddress(program, cached_program.shared_variables.dst_sharded_cb, *dst_buffer);

    const uint32_t input_buffer_address = src_buffer->address();
    const uint32_t output_buffer_address = dst_buffer->address();
    const uint32_t move_chunk_size_bytes = output_buffer_address - input_buffer_address;
    const uint32_t num_chunks = cached_program.shared_variables.total_size_bytes / move_chunk_size_bytes;
    const uint32_t remainder_chunk_size_bytes =
        cached_program.shared_variables.total_size_bytes % move_chunk_size_bytes;

    std::vector<uint32_t> new_runtime_args = {
        cached_program.shared_variables.total_size_bytes,
        num_chunks,
        move_chunk_size_bytes,
        remainder_chunk_size_bytes};

    for (const auto& core : cached_program.shared_variables.cores) {
        SetRuntimeArgs(program, cached_program.shared_variables.kernel_id, core, new_runtime_args);
    }
}

}  // namespace ttnn::operations::data_movement::move::program
