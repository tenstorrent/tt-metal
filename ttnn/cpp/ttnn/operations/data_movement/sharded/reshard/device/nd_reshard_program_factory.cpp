// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "nd_reshard_program_factory.hpp"

#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement::detail {
operation::ProgramWithCallbacks nd_reshard_multicore_local_reader(const Tensor& input, Tensor& output) {
    auto input_buffer = input.buffer();
    auto output_buffer = output.buffer();

    auto input_nd_shard_spec = input.memory_config().nd_shard_spec().value();
    auto output_nd_shard_spec = output.memory_config().nd_shard_spec().value();

    const auto input_accessor_args = TensorAccessorArgs(*input_buffer);
    const auto output_accessor_args = TensorAccessorArgs(*output_buffer);

    auto aligned_page_size = input_buffer->aligned_page_size();
    TT_ASSERT(aligned_page_size == output_buffer->aligned_page_size(), "Input and output page sizes must be the same");

    // Create Program + Grid
    auto program = CreateProgram();
    auto grid_size = input.device()->compute_with_storage_grid_size();

    // This implementation assumes that input and output grids are the same.
    auto cores = input_buffer->buffer_distribution_spec()->cores_with_data();
    auto grid = CoreRangeSet(cores);

    auto num_shards = input_buffer->buffer_distribution_spec()->num_shards();

    auto shard_id_stride = input_buffer->buffer_distribution_spec()->num_cores_with_data();

    // Prepare compile time arguments
    auto compile_time_args_reader = input_accessor_args.get_compile_time_args();
    output_accessor_args.append_to(compile_time_args_reader);
    compile_time_args_reader.push_back(aligned_page_size);

    // Create kernels
    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/sharded/reshard/device/kernels/nd_reshard_local_reader.cpp",
        grid,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = compile_time_args_reader,
        });

    // That common and unique runtime arguments to 0, and call overrtime_runtime_arguments callback
    SetCommonRuntimeArgs(
        program,
        reader_kernel_id,
        {input.buffer()->address(), output.buffer()->address(), num_shards, shard_id_stride});
    for (const auto& core : cores) {
        SetRuntimeArgs(program, reader_kernel_id, core, {0});
    }

    auto override_runtime_arguments_callback = [reader_kernel_id, grid, cores](
                                                   const void* operation,
                                                   Program& program,
                                                   const std::vector<Tensor>& input_tensors,
                                                   const std::vector<std::optional<const Tensor>>&,
                                                   const std::vector<Tensor>& output_tensors) {
        const auto& input = input_tensors.at(0);
        const auto& output = output_tensors.at(0);

        auto& common_runtime_args_reader = GetCommonRuntimeArgs(program, reader_kernel_id);
        common_runtime_args_reader[0] = input.buffer()->address();
        common_runtime_args_reader[1] = output.buffer()->address();
        common_runtime_args_reader[2] = input.buffer()->buffer_distribution_spec()->num_shards();
        common_runtime_args_reader[3] = input.buffer()->buffer_distribution_spec()->num_cores_with_data();

        auto& runtime_args_by_core_reader = GetRuntimeArgs(program, reader_kernel_id);

        auto start_shard_id = 0;

        for (const auto& core : cores) {
            runtime_args_by_core_reader[core.x][core.y][0] = start_shard_id;
            ++start_shard_id;
        }
    };

    override_runtime_arguments_callback(
        nullptr, program, {input}, std::vector<std::optional<const Tensor>>{}, {output});

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

operation::ProgramWithCallbacks nd_reshard_multi_core(const Tensor& input, Tensor& output) {
    return nd_reshard_multicore_local_reader(input, output);
}
}  // namespace ttnn::operations::data_movement::detail
