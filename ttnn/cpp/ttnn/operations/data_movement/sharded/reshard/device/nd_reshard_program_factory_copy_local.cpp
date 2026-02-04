// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/sharded/reshard/device/nd_reshard_program_factory_copy_local.hpp"

#include <tt-metalium/tensor_accessor_args.hpp>
#include "tt-metalium/host_api.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {

template <bool local_is_input>
NdReshardCopyLocalShardFactory<local_is_input>::cached_program_t NdReshardCopyLocalShardFactory<local_is_input>::create(
    const ReshardParams& /*operation_attributes*/, const ReshardInputs& tensor_args, Tensor& output_tensor) {
    const auto& input = tensor_args.input;
    auto& output = output_tensor;

    auto* input_buffer = input.buffer();
    auto* output_buffer = output.buffer();

    const auto input_accessor_args = TensorAccessorArgs(*input_buffer);
    const auto output_accessor_args = TensorAccessorArgs(*output_buffer);

    // Choose buffer and aligned page size based on is_reader flag
    auto* local_buffer = local_is_input ? input_buffer : output_buffer;
    auto aligned_page_size = local_buffer->aligned_page_size();
    auto other_aligned_page_size =
        local_is_input ? output_buffer->aligned_page_size() : input_buffer->aligned_page_size();

    // Create Program + Grid
    auto program = CreateProgram();

    // This implementation assumes that input and output grids are the same.
    auto cores_vec = local_buffer->buffer_distribution_spec()->cores_with_data();
    auto grid = CoreRangeSet(cores_vec);

    auto num_shards = local_buffer->buffer_distribution_spec()->num_shards();

    // num cores with data * 2 because we have two kernels
    auto shard_id_stride = local_buffer->buffer_distribution_spec()->num_cores_with_data() * 2;

    // Prepare compile time arguments
    auto logical_size = input.logical_shape();
    uint32_t logical_width = logical_size[-1] * input.element_size();
    uint32_t source_width = logical_width;
    uint32_t destination_width = logical_width;
    uint32_t base_page_size = aligned_page_size;

    if (input.memory_config().shard_spec().has_value() && output.memory_config().shard_spec().has_value()) {
        auto input_buffer_type = input.memory_config().memory_layout();
        auto output_buffer_type = output.memory_config().memory_layout();

        // for block sharded
        CoreCoord input_shard_grid = input_buffer->shard_spec().grid().ranges()[0].grid_size();
        uint32_t input_num_shard_cores = input_shard_grid.x;
        if (input_buffer->shard_spec().orientation() == ShardOrientation::COL_MAJOR) {
            input_num_shard_cores = input_shard_grid.y;
        }

        CoreCoord output_shard_grid = output_buffer->shard_spec().grid().ranges()[0].grid_size();
        uint32_t output_num_shard_cores = output_shard_grid.x;
        if (output_buffer->shard_spec().orientation() == ShardOrientation::COL_MAJOR) {
            output_num_shard_cores = output_shard_grid.y;
        }
        // for width sharded
        if (input_buffer_type == TensorMemoryLayout::WIDTH_SHARDED &&
            output_buffer_type == TensorMemoryLayout::WIDTH_SHARDED) {
            input_num_shard_cores = input_shard_grid.x == 1 ? input_shard_grid.y : input_shard_grid.x;
            output_num_shard_cores = output_shard_grid.x == 1 ? output_shard_grid.y : output_shard_grid.x;
        }

        source_width = input_buffer->shard_spec().shape()[1] * input.element_size() * input_num_shard_cores;
        destination_width = output_buffer->shard_spec().shape()[1] * output.element_size() * output_num_shard_cores;
        uint32_t input_page_size = input_buffer->page_size();
        uint32_t output_page_size = output_buffer->page_size();
        base_page_size = std::gcd(input_page_size, output_page_size);
    }
    auto compile_time_args_reader = input_accessor_args.get_compile_time_args();
    output_accessor_args.append_to(compile_time_args_reader);
    compile_time_args_reader.push_back(aligned_page_size);
    compile_time_args_reader.push_back(other_aligned_page_size);
    compile_time_args_reader.push_back(local_is_input);
    compile_time_args_reader.push_back(logical_width);
    compile_time_args_reader.push_back(source_width);
    compile_time_args_reader.push_back(destination_width);
    compile_time_args_reader.push_back(base_page_size);

    // Create kernels
    KernelHandle brisc_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/sharded/reshard/device/kernels/nd_reshard_copy_local_shards.cpp",
        grid,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = compile_time_args_reader,
        });

    KernelHandle ncrisc_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/sharded/reshard/device/kernels/nd_reshard_copy_local_shards.cpp",
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

    // Set unique runtime arguments to 0, and call
    for (const auto& core : cores_vec) {
        SetRuntimeArgs(program, brisc_kernel_id, core, {0});
        SetRuntimeArgs(program, ncrisc_kernel_id, core, {0});
    }

    auto cached_program = NdReshardCopyLocalShardFactory::cached_program_t{
        std::move(program),
        {.brisc_kernel_id = brisc_kernel_id, .ncrisc_kernel_id = ncrisc_kernel_id, .grid = grid, .cores = cores_vec}};

    // For now, directly set runtime args
    {
        const auto& input = tensor_args.input;
        const auto& output = output_tensor;
        auto* local_buffer = local_is_input ? input.buffer() : output.buffer();
        auto num_shards = local_buffer->buffer_distribution_spec()->num_shards();
        auto shard_id_stride = local_buffer->buffer_distribution_spec()->num_cores_with_data() * 2;

        std::vector<uint32_t> common_runtime_args = {
            input.buffer()->address(), output.buffer()->address(), num_shards, shard_id_stride};
        auto& common_runtime_args_brisc = GetCommonRuntimeArgs(cached_program.program, brisc_kernel_id);
        auto& common_runtime_args_ncrisc = GetCommonRuntimeArgs(cached_program.program, ncrisc_kernel_id);
        for (size_t i = 0; i < common_runtime_args.size(); i++) {
            common_runtime_args_brisc[i] = common_runtime_args[i];
            common_runtime_args_ncrisc[i] = common_runtime_args[i];
        }

        auto& runtime_args_by_core_brisc = GetRuntimeArgs(cached_program.program, brisc_kernel_id);
        auto& runtime_args_by_core_ncrisc = GetRuntimeArgs(cached_program.program, ncrisc_kernel_id);

        auto start_shard_id = 0;
        for (const auto& core : cores_vec) {
            runtime_args_by_core_brisc[core.x][core.y][0] = start_shard_id;
            runtime_args_by_core_ncrisc[core.x][core.y][0] = start_shard_id + shard_id_stride / 2;
            ++start_shard_id;
        }
    }

    return cached_program;
}

template <bool is_reader>
void NdReshardCopyLocalShardFactory<is_reader>::override_runtime_arguments(
    cached_program_t& cached_program,
    const ReshardParams& /*operation_attributes*/,
    const ReshardInputs& tensor_args,
    Tensor& output_tensor) {
    const auto& input = tensor_args.input;
    const auto& output = output_tensor;

    auto& program = cached_program.program;
    const auto& brisc_kernel_id = cached_program.shared_variables.brisc_kernel_id;
    const auto& ncrisc_kernel_id = cached_program.shared_variables.ncrisc_kernel_id;
    const auto& cores = cached_program.shared_variables.cores;

    // Choose buffer for distribution spec based on is_reader flag
    auto* local_buffer = is_reader ? input.buffer() : output.buffer();
    auto num_shards = local_buffer->buffer_distribution_spec()->num_shards();
    auto shard_id_stride = local_buffer->buffer_distribution_spec()->num_cores_with_data() * 2;

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
}

// Explicit template instantiations
template struct NdReshardCopyLocalShardFactory<true>;
template struct NdReshardCopyLocalShardFactory<false>;

}  // namespace ttnn::prim
