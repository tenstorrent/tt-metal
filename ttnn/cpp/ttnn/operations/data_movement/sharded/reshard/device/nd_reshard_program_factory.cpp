// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "nd_reshard_program_factory.hpp"

#include <tt-metalium/tensor_accessor_args.hpp>
#include "tt-metalium/host_api.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement::detail {
operation::ProgramWithCallbacks nd_reshard_multicore_copy_shard(const Tensor& input, Tensor& output, bool is_reader) {
    auto input_buffer = input.buffer();
    auto output_buffer = output.buffer();

    const auto input_accessor_args = TensorAccessorArgs(*input_buffer);
    const auto output_accessor_args = TensorAccessorArgs(*output_buffer);

    // Choose buffer and aligned page size based on is_reader flag
    auto buffer_for_distribution = is_reader ? input_buffer : output_buffer;
    auto aligned_page_size = buffer_for_distribution->aligned_page_size();
    auto other_aligned_page_size = is_reader ? output_buffer->aligned_page_size() : input_buffer->aligned_page_size();
    TT_ASSERT(aligned_page_size == other_aligned_page_size, "Input and output page sizes must be the same");

    // Create Program + Grid
    auto program = CreateProgram();

    // This implementation assumes that input and output grids are the same.
    auto cores = buffer_for_distribution->buffer_distribution_spec()->cores_with_data();
    auto grid = CoreRangeSet(cores);

    auto num_shards = buffer_for_distribution->buffer_distribution_spec()->num_shards();

    // num cores with data * 2 because we have two kernels
    auto shard_id_stride = buffer_for_distribution->buffer_distribution_spec()->num_cores_with_data() * 2;

    // Prepare compile time arguments
    auto compile_time_args_reader = input_accessor_args.get_compile_time_args();
    output_accessor_args.append_to(compile_time_args_reader);
    compile_time_args_reader.push_back(aligned_page_size);

    // Choose kernel path based on is_reader flag
    const char* kernel_path =
        is_reader ? "ttnn/cpp/ttnn/operations/data_movement/sharded/reshard/device/kernels/nd_reshard_local_reader.cpp"
                  : "ttnn/cpp/ttnn/operations/data_movement/sharded/reshard/device/kernels/nd_reshard_local_writer.cpp";

    // Create kernels
    KernelHandle brisc_kernel_id = CreateKernel(
        program,
        kernel_path,
        grid,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = compile_time_args_reader,
        });

    KernelHandle ncrisc_kernel_id = CreateKernel(
        program,
        kernel_path,
        grid,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = compile_time_args_reader,
        });

    std::vector<uint32_t> common_runtime_args = {
        input.buffer()->address(), output.buffer()->address(), num_shards, shard_id_stride};
    SetCommonRuntimeArgs(program, brisc_kernel_id, common_runtime_args);
    SetCommonRuntimeArgs(program, ncrisc_kernel_id, common_runtime_args);

    // That unique runtime arguments to 0, and call overrtime_runtime_arguments callback
    for (const auto& core : cores) {
        SetRuntimeArgs(program, brisc_kernel_id, core, {0});
        SetRuntimeArgs(program, ncrisc_kernel_id, core, {0});
    }

    auto override_runtime_arguments_callback = [brisc_kernel_id, ncrisc_kernel_id, grid, cores, is_reader](
                                                   const void* operation,
                                                   Program& program,
                                                   const std::vector<Tensor>& input_tensors,
                                                   const std::vector<std::optional<const Tensor>>&,
                                                   const std::vector<Tensor>& output_tensors) {
        const auto& input = input_tensors.at(0);
        const auto& output = output_tensors.at(0);

        // Choose buffer for distribution spec based on is_reader flag
        auto buffer_for_distribution = is_reader ? input.buffer() : output.buffer();
        auto num_shards = buffer_for_distribution->buffer_distribution_spec()->num_shards();
        auto shard_id_stride = buffer_for_distribution->buffer_distribution_spec()->num_cores_with_data() * 2;

        std::vector<uint32_t> common_runtime_args = {
            input.buffer()->address(), output.buffer()->address(), num_shards, shard_id_stride};
        auto& common_runtime_args_brisc = GetCommonRuntimeArgs(program, brisc_kernel_id);
        auto& common_runtime_args_ncrisc = GetCommonRuntimeArgs(program, ncrisc_kernel_id);
        for (size_t i = 0; i < common_runtime_args.size(); i++) {
            common_runtime_args_brisc[i] = common_runtime_args[i];
            common_runtime_args_ncrisc[i] = common_runtime_args[i];
        }

        auto& runtime_args_by_core_brisc = GetRuntimeArgs(program, brisc_kernel_id);
        auto& runtime_args_by_core_ncrisc = GetRuntimeArgs(program, ncrisc_kernel_id);

        auto start_shard_id = 0;

        // brisc copies shards [0, num_data_cores*2, num_data_cores*4, num_data_cores*6, ...]
        // ncrisc copies shards [num_data_cores, num_data_cores*3, num_data_cores*5, num_data_cores*7, ...]
        for (const auto& core : cores) {
            runtime_args_by_core_brisc[core.x][core.y][0] = start_shard_id;
            runtime_args_by_core_ncrisc[core.x][core.y][0] = start_shard_id + shard_id_stride / 2;
            ++start_shard_id;
        }
    };

    override_runtime_arguments_callback(
        nullptr, program, {input}, std::vector<std::optional<const Tensor>>{}, {output});

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

operation::ProgramWithCallbacks nd_reshard_multi_core(const Tensor& input, Tensor& output) {
    return nd_reshard_multicore_copy_shard(input, output, true);
}
}  // namespace ttnn::operations::data_movement::detail
