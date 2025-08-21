// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "nd_reshard_program_factory.hpp"

#include <tt-metalium/tensor_accessor_args.hpp>
#include "tt-metalium/host_api.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement::detail {
operation::ProgramWithCallbacks nd_reshard_multicore_copy_pages(const Tensor& input, Tensor& output) {
    auto input_buffer = input.buffer();
    auto output_buffer = output.buffer();

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

    // That common and unique runtime arguments to 0, and call overrtime_runtime_arguments callback
    SetCommonRuntimeArgs(program, reader_kernel_id, {input.buffer()->address()});
    SetCommonRuntimeArgs(program, writer_kernel_id, {output.buffer()->address()});
    for (const auto& core : cores) {
        SetRuntimeArgs(program, reader_kernel_id, core, {0, 0});
        SetRuntimeArgs(program, writer_kernel_id, core, {0, 0});
    }

    auto override_runtime_arguments_callback = [reader_kernel_id, writer_kernel_id, grid, cores](
                                                   const void* operation,
                                                   Program& program,
                                                   const std::vector<Tensor>& input_tensors,
                                                   const std::vector<std::optional<const Tensor>>&,
                                                   const std::vector<Tensor>& output_tensors) {
        const auto& input = input_tensors.at(0);
        const auto& output = output_tensors.at(0);

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
    };

    override_runtime_arguments_callback(
        nullptr, program, {input}, std::vector<std::optional<const Tensor>>{}, {output});

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

/*
    - This implementation runs program on grid from input (if is_reader is true) or output (if is_reader is false)
   sharded spec.
   - If is_reader is true, then all reads are local (cores are taken from input buffer distribution spec).
   - If is_reader is false, then all writes are local (cores are taken from output buffer distribution).
*/
operation::ProgramWithCallbacks nd_reshard_multicore_copy_local_shard(
    const Tensor& input, Tensor& output, bool is_reader) {
    auto input_buffer = input.buffer();
    auto output_buffer = output.buffer();

    const auto input_accessor_args = TensorAccessorArgs(*input_buffer);
    const auto output_accessor_args = TensorAccessorArgs(*output_buffer);

    // Choose buffer and aligned page size based on is_reader flag
    auto local_buffer = is_reader ? input_buffer : output_buffer;
    auto aligned_page_size = local_buffer->aligned_page_size();
    auto other_aligned_page_size = is_reader ? output_buffer->aligned_page_size() : input_buffer->aligned_page_size();

    // Create Program + Grid
    auto program = CreateProgram();

    // This implementation assumes that input and output grids are the same.
    auto cores = local_buffer->buffer_distribution_spec()->cores_with_data();
    auto grid = CoreRangeSet(cores);

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
        uint32_t num_shard_cores = input_buffer == local_buffer ? input_num_shard_cores : output_num_shard_cores;

        uint32_t sharded_width = local_buffer->shard_spec().shape()[1] * input.element_size() * num_shard_cores;
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
    compile_time_args_reader.push_back(is_reader);
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
        auto local_buffer = is_reader ? input.buffer() : output.buffer();
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
    };
    override_runtime_arguments_callback(
        nullptr, program, {input}, std::vector<std::optional<const Tensor>>{}, {output});

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

/*
   - Usually is_reader perform better for L1->L1 resharding, but if one of buffers is DRAM, then it's always better to
   take implementation such that accessing L1 banks are local to the core
   - For DRAM->DRAM resharding, simple copy of page by page is used.
*/
operation::ProgramWithCallbacks nd_reshard_multi_core(const Tensor& input, Tensor& output) {
    auto input_buffer_type = input.memory_config().buffer_type();
    auto output_buffer_type = output.memory_config().buffer_type();
    auto input_page_size = input.buffer()->page_size();
    auto output_page_size = output.buffer()->page_size();

    TT_FATAL(
        input_buffer_type == BufferType::DRAM || input_buffer_type == BufferType::L1,
        "Input buffer type must be DRAM or L1");
    TT_FATAL(
        output_buffer_type == BufferType::DRAM || output_buffer_type == BufferType::L1,
        "Output buffer type must be DRAM or L1");

    if (input_buffer_type == BufferType::DRAM && output_buffer_type == BufferType::DRAM) {
        return nd_reshard_multicore_copy_pages(input, output);
    }
    if (input_buffer_type == BufferType::L1 && output_buffer_type == BufferType::L1 &&
        input_page_size != output_page_size) {
        return nd_reshard_multicore_copy_local_shard(input, output, true);
    }
    if (input_buffer_type == BufferType::DRAM) {
        return nd_reshard_multicore_copy_local_shard(input, output, false);
    }
    return nd_reshard_multicore_copy_local_shard(input, output, true);
}
}  // namespace ttnn::operations::data_movement::detail
