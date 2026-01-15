// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/sharded/reshard/device/nd_reshard_program_factory_copy_pages.hpp"

#include <tt-metalium/tensor_accessor_args.hpp>
#include "tt-metalium/host_api.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement::reshard::program {

NdReshardCopyPagesFactory::cached_program_t NdReshardCopyPagesFactory::create(
    const reshard::ReshardParams& operation_attributes,
    const reshard::ReshardInputs& tensor_args,
    reshard::tensor_return_value_t& tensor_return_value) {
    const auto& input = tensor_args.input;
    auto& output = tensor_return_value;

    auto* input_buffer = input.buffer();
    auto* output_buffer = output.buffer();

    auto input_nd_shard_spec = input.memory_config().nd_shard_spec().value();

    const auto input_accessor_args = TensorAccessorArgs(*input_buffer);
    const auto output_accessor_args = TensorAccessorArgs(*output_buffer);

    auto aligned_page_size = input_buffer->aligned_page_size();

    // Create Program + Grid
    auto program = CreateProgram();
    auto grid_size = input.device()->compute_with_storage_grid_size();
    auto grid = CoreRangeSet({CoreRange(CoreCoord(0, 0), CoreCoord(grid_size.x - 1, grid_size.y - 1))});
    auto cores = corerange_to_cores(grid, std::nullopt, input_nd_shard_spec.orientation == ShardOrientation::ROW_MAJOR);

    // Create Circular Buffer
    const auto data_format = datatype_to_dataformat_converter(input.dtype());
    constexpr auto num_tiles_in_cb = 1;  // TODO: Try double buffering
    CBHandle cb_in0_idx = tt::CBIndex::c_0;
    auto c_in0_config = CircularBufferConfig(aligned_page_size * num_tiles_in_cb, {{cb_in0_idx, data_format}})
                            .set_page_size(cb_in0_idx, aligned_page_size);
    CreateCircularBuffer(program, grid, c_in0_config);

    // Prepare compile time arguments
    auto compile_time_args_reader = input_accessor_args.get_compile_time_args();
    compile_time_args_reader.push_back(cb_in0_idx);  // Circular buffer index
    compile_time_args_reader.push_back(aligned_page_size);

    auto compile_time_args_writer = output_accessor_args.get_compile_time_args();
    compile_time_args_writer.push_back(cb_in0_idx);
    compile_time_args_writer.push_back(aligned_page_size);

    // Create kernels
    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/sharded/reshard/device/kernels/nd_reshard_copy_pages_reader.cpp",
        grid,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = compile_time_args_reader,
        });

    KernelHandle writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/sharded/reshard/device/kernels/nd_reshard_copy_pages_writer.cpp",
        grid,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = compile_time_args_writer,
        });

    SetCommonRuntimeArgs(program, reader_kernel_id, {input.buffer()->address()});
    SetCommonRuntimeArgs(program, writer_kernel_id, {output.buffer()->address()});

    for (const auto& core : cores) {
        SetRuntimeArgs(program, reader_kernel_id, core, {0, 0});
        SetRuntimeArgs(program, writer_kernel_id, core, {0, 0});
    }

    auto cached_program = cached_program_t{
        std::move(program),
        {.reader_kernel_id = reader_kernel_id, .writer_kernel_id = writer_kernel_id, .grid = grid, .cores = cores}};

    // Set initial runtime arguments
    override_runtime_arguments(cached_program, operation_attributes, tensor_args, tensor_return_value);

    return cached_program;
}

void NdReshardCopyPagesFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const reshard::ReshardParams& /*operation_attributes*/,
    const reshard::ReshardInputs& tensor_args,
    reshard::tensor_return_value_t& tensor_return_value) {
    const auto& input = tensor_args.input;
    const auto& output = tensor_return_value;

    auto& program = cached_program.program;
    const auto& reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    const auto& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;
    const auto& cores = cached_program.shared_variables.cores;

    auto& common_runtime_args_reader = GetCommonRuntimeArgs(program, reader_kernel_id);
    auto& common_runtime_args_writer = GetCommonRuntimeArgs(program, writer_kernel_id);
    common_runtime_args_reader[0] = input.buffer()->address();
    common_runtime_args_writer[0] = output.buffer()->address();

    auto& runtime_args_by_core_reader = GetRuntimeArgs(program, reader_kernel_id);
    auto& runtime_args_by_core_writer = GetRuntimeArgs(program, writer_kernel_id);

    uint32_t start_page = 0;
    uint32_t num_dev_pages = input.buffer()->buffer_distribution_spec()->tensor_shape_in_pages().volume();
    uint32_t n_pages_per_core = num_dev_pages / cores.size();
    uint32_t remainder = num_dev_pages % cores.size();

    for (const auto& core : cores) {
        uint32_t num_pages_for_core = n_pages_per_core;
        if (remainder > 0) {
            num_pages_for_core++;
            remainder--;
        }
        runtime_args_by_core_reader[core.x][core.y][0] = start_page;
        runtime_args_by_core_reader[core.x][core.y][1] = start_page + num_pages_for_core;
        runtime_args_by_core_writer[core.x][core.y][0] = start_page;
        runtime_args_by_core_writer[core.x][core.y][1] = start_page + num_pages_for_core;
        start_page += num_pages_for_core;
    }
}

}  // namespace ttnn::operations::data_movement::reshard::program
