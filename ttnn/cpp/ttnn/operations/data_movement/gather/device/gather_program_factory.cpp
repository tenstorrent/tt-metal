// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gather_program_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::prim {
// Single row - single core
GatherProgramFactorySingleRowSingleCore::cached_program_t GatherProgramFactorySingleRowSingleCore::create(
    const GatherParams& attributes, const GatherInputs& tensor_args, Tensor& output_tensor) {
    tt::tt_metal::Program program{};

    // Tensor config info
    const tt::DataFormat input_tensor_cb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(tensor_args.input_tensor.dtype());
    const tt::DataFormat input_index_tensor_cb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(tensor_args.input_index_tensor.dtype());
    const tt::DataFormat output_tensor_cb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(output_tensor.dtype());

    const uint32_t input_tensor_tile_size = tile_size(input_tensor_cb_data_format);
    const uint32_t input_index_tensor_tile_size = tile_size(input_index_tensor_cb_data_format);
    const uint32_t output_tensor_tile_size = tile_size(output_tensor_cb_data_format);

    auto* input_tensor_buffer = tensor_args.input_tensor.buffer();
    auto* input_index_tensor_buffer = tensor_args.input_index_tensor.buffer();
    auto* output_tensor_buffer = output_tensor.buffer();

    const bool input_tensor_is_dram = input_tensor_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    const bool input_index_tensor_is_dram = input_index_tensor_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    const bool output_tensor_is_dram = output_tensor_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;

    const auto tile_width = tensor_args.input_tensor.tensor_spec().tile().get_width();
    const auto tile_height = tensor_args.input_tensor.tensor_spec().tile().get_height();

    const auto input_index_shape = tensor_args.input_index_tensor.padded_shape();
    const auto input_shape = tensor_args.input_tensor.padded_shape();
    const uint32_t Ht = (input_index_shape[0] * input_index_shape[1] * input_index_shape[2]) / tile_height;
    const uint32_t Wt_input = input_shape[3] / tile_width;
    const uint32_t Wt_index = input_index_shape[3] / tile_width;

    // Calculate the number of cores available for computation
    auto* device = tensor_args.input_tensor.device();
    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t max_number_of_cores = compute_with_storage_grid_size.y * compute_with_storage_grid_size.x;

    // Create core grid
    CoreRangeSet core_grid =
        tt::tt_metal::num_cores_to_corerangeset(max_number_of_cores, compute_with_storage_grid_size, true);
    // Override core grid if sub_core_grids is provided in operation attributes
    if (attributes.sub_core_grids.has_value()) {
        core_grid = attributes.sub_core_grids.value();
    }

    const auto
        [total_number_of_cores,       // number of cores utilized
         core_range,                  // set of all cores used
         core_group_1,                // Primary core group
         core_group_2,                // Secondary core group
         num_tiles_per_core_group_1,  // Number of tiles each core in the primary group processes
         num_tiles_per_core_group_2   // Number of tiles each core in the secondary group processes
    ] = tt::tt_metal::split_work_to_cores(core_grid, Ht, true);
    const auto work_groups = {
        std::make_pair(core_group_1, num_tiles_per_core_group_1),
        std::make_pair(core_group_2, num_tiles_per_core_group_2)};
    const std::vector<CoreCoord>& cores = corerange_to_cores(core_range, total_number_of_cores, true);

    // Circular buffers
    constexpr uint32_t input_tensor_cb_index = tt::CBIndex::c_0;
    const tt::tt_metal::CircularBufferConfig input_tensor_cb_config =
        tt::tt_metal::CircularBufferConfig(
            Wt_input * input_tensor_tile_size, {{input_tensor_cb_index, input_tensor_cb_data_format}})
            .set_page_size(input_tensor_cb_index, input_tensor_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, core_range, input_tensor_cb_config);

    constexpr uint32_t input_index_tensor_cb_index = tt::CBIndex::c_1;
    const tt::tt_metal::CircularBufferConfig input_index_tensor_cb_config =
        tt::tt_metal::CircularBufferConfig(
            input_index_tensor_tile_size, {{input_index_tensor_cb_index, input_index_tensor_cb_data_format}})
            .set_page_size(input_index_tensor_cb_index, input_index_tensor_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, core_range, input_index_tensor_cb_config);

    constexpr uint32_t output_tensor_cb_index = tt::CBIndex::c_2;
    const tt::tt_metal::CircularBufferConfig output_tensor_cb_config =
        tt::tt_metal::CircularBufferConfig(
            output_tensor_tile_size, {{output_tensor_cb_index, output_tensor_cb_data_format}})
            .set_page_size(output_tensor_cb_index, output_tensor_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, core_range, output_tensor_cb_config);

    // Kernels
    std::vector<uint32_t> reader_compile_time_args = {
        input_tensor_cb_index,
        input_index_tensor_cb_index,
        output_tensor_cb_index,
        static_cast<uint32_t>(input_index_tensor_is_dram),
        Ht,
        Wt_input,
        Wt_index,
        total_number_of_cores,
        compute_with_storage_grid_size.x,
        compute_with_storage_grid_size.y};
    TensorAccessorArgs(*input_index_tensor_buffer).append_to(reader_compile_time_args);
    const std::string gather_reader_kernel_path =
        "ttnn/cpp/ttnn/operations/data_movement/gather/device/kernels/dataflow/"
        "gather_reader_single_row_single_core.cpp";
    tt::tt_metal::KernelHandle gather_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        gather_reader_kernel_path,
        core_range,
        tt::tt_metal::ReaderDataMovementConfig{reader_compile_time_args});

    std::vector<uint32_t> writer_compile_time_args = {
        input_tensor_cb_index,
        output_tensor_cb_index,
        static_cast<uint32_t>(input_tensor_is_dram),
        static_cast<uint32_t>(output_tensor_is_dram),
        Ht,
        Wt_input,
        Wt_index,
        total_number_of_cores,
        compute_with_storage_grid_size.x,
        compute_with_storage_grid_size.y};
    TensorAccessorArgs(*input_tensor_buffer).append_to(writer_compile_time_args);
    TensorAccessorArgs(*output_tensor_buffer).append_to(writer_compile_time_args);
    const std::string gather_writer_kernel_path =
        "ttnn/cpp/ttnn/operations/data_movement/gather/device/kernels/dataflow/"
        "gather_writer_single_row_single_core.cpp";
    tt::tt_metal::KernelHandle gather_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        gather_writer_kernel_path,
        core_range,
        tt::tt_metal::WriterDataMovementConfig{writer_compile_time_args});

    uint32_t id = 0;  // Offset for the next core in the group
    for (const auto& [group, work_per_core] : work_groups) {
        for (const auto& range : group.ranges()) {
            for (const auto& core : range) {
                SetRuntimeArgs(
                    program,
                    gather_reader_kernel_id,
                    core,
                    {input_index_tensor_buffer->address(), work_per_core, tile_width, tile_height, id});
                SetRuntimeArgs(
                    program,
                    gather_writer_kernel_id,
                    core,
                    {input_tensor_buffer->address(), output_tensor_buffer->address(), work_per_core, id});
                id++;
            }
        }
    }

    return {std::move(program), {gather_reader_kernel_id, gather_writer_kernel_id, cores}};
}

void GatherProgramFactorySingleRowSingleCore::override_runtime_arguments(
    cached_program_t& cached_program,
    const GatherParams& /*attributes*/,
    const GatherInputs& tensor_args,
    Tensor& output_tensor) {
    auto* input_tensor_buffer = tensor_args.input_tensor.buffer();
    auto* input_index_tensor_buffer = tensor_args.input_index_tensor.buffer();
    auto* output_tensor_buffer = output_tensor.buffer();

    for (const auto& core : cached_program.shared_variables.cores) {
        auto& gather_reader_runtime_args = tt::tt_metal::GetRuntimeArgs(
            cached_program.program, cached_program.shared_variables.gather_reader_kernel_id, core);
        gather_reader_runtime_args[0] = input_index_tensor_buffer->address();

        auto& gather_writer_runtime_args = tt::tt_metal::GetRuntimeArgs(
            cached_program.program, cached_program.shared_variables.gather_writer_kernel_id, core);
        gather_writer_runtime_args[0] = input_tensor_buffer->address();
        gather_writer_runtime_args[1] = output_tensor_buffer->address();
    }
}

// Single row - multi core
GatherProgramFactorySingleRowMultiCore::cached_program_t GatherProgramFactorySingleRowMultiCore::create(
    const GatherParams& attributes, const GatherInputs& tensor_args, Tensor& output_tensor) {
    tt::tt_metal::Program program{};

    // Tensor config info
    const tt::DataFormat input_tensor_cb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(tensor_args.input_tensor.dtype());
    const tt::DataFormat input_index_tensor_cb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(tensor_args.input_index_tensor.dtype());
    const tt::DataFormat output_tensor_cb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(output_tensor.dtype());

    const uint32_t input_tensor_tile_size = tile_size(input_tensor_cb_data_format);
    const uint32_t input_index_tensor_tile_size = tile_size(input_index_tensor_cb_data_format);
    const uint32_t output_tensor_tile_size = tile_size(output_tensor_cb_data_format);

    auto* const input_tensor_buffer = tensor_args.input_tensor.buffer();
    auto* const input_index_tensor_buffer = tensor_args.input_index_tensor.buffer();
    auto* const output_tensor_buffer = output_tensor.buffer();

    const bool input_tensor_is_dram = input_tensor_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    const bool input_index_tensor_is_dram = input_index_tensor_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    const bool output_tensor_is_dram = output_tensor_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;

    const auto tile_width = tensor_args.input_tensor.tensor_spec().tile().get_width();
    const auto tile_height = tensor_args.input_tensor.tensor_spec().tile().get_height();

    const auto input_index_shape = tensor_args.input_index_tensor.padded_shape();
    const auto input_shape = tensor_args.input_tensor.padded_shape();
    const uint32_t Ht = (input_index_shape[0] * input_index_shape[1] * input_index_shape[2]) / tile_height;
    const uint32_t Wt_input = input_shape[3] / tile_width;
    const uint32_t Wt_index = input_index_shape[3] / tile_width;

    // Calculate the number of cores available for computation
    auto* device = tensor_args.input_tensor.device();
    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t max_number_of_cores = compute_with_storage_grid_size.y * compute_with_storage_grid_size.x;

    // Create core grid
    CoreRangeSet core_grid =
        tt::tt_metal::num_cores_to_corerangeset(max_number_of_cores, compute_with_storage_grid_size, true);
    // Override core grid if sub_core_grids is provided in operation attributes
    if (attributes.sub_core_grids.has_value()) {
        core_grid = attributes.sub_core_grids.value();
    }

    const auto
        [total_number_of_cores,       // number of cores utilized
         core_range,                  // set of all cores used
         core_group_1,                // Primary core group
         core_group_2,                // Secondary core group
         num_tiles_per_core_group_1,  // Number of tiles each core in the primary group processes
         num_tiles_per_core_group_2   // Number of tiles each core in the secondary group processes
    ] = tt::tt_metal::split_work_to_cores(core_grid, Wt_index, true);
    const auto work_groups = {
        std::make_pair(core_group_1, num_tiles_per_core_group_1),
        std::make_pair(core_group_2, num_tiles_per_core_group_2)};
    const std::vector<CoreCoord>& cores = corerange_to_cores(core_range, total_number_of_cores, true);

    // Circular buffers
    constexpr uint32_t buffer_scale_factor = 2;
    constexpr uint32_t input_tensor_cb_index = tt::CBIndex::c_0;
    const tt::tt_metal::CircularBufferConfig input_tensor_cb_config =
        tt::tt_metal::CircularBufferConfig(
            buffer_scale_factor * input_tensor_tile_size, {{input_tensor_cb_index, input_tensor_cb_data_format}})
            .set_page_size(input_tensor_cb_index, input_tensor_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, core_range, input_tensor_cb_config);

    constexpr uint32_t input_index_tensor_cb_index = tt::CBIndex::c_1;
    const tt::tt_metal::CircularBufferConfig input_index_tensor_cb_config =
        tt::tt_metal::CircularBufferConfig(
            input_index_tensor_tile_size, {{input_index_tensor_cb_index, input_index_tensor_cb_data_format}})
            .set_page_size(input_index_tensor_cb_index, input_index_tensor_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, core_range, input_index_tensor_cb_config);

    constexpr uint32_t output_tensor_cb_index = tt::CBIndex::c_2;
    const tt::tt_metal::CircularBufferConfig output_tensor_cb_config =
        tt::tt_metal::CircularBufferConfig(
            output_tensor_tile_size, {{output_tensor_cb_index, output_tensor_cb_data_format}})
            .set_page_size(output_tensor_cb_index, output_tensor_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, core_range, output_tensor_cb_config);

    // Kernels
    std::vector<uint32_t> reader_compile_time_args = {
        input_tensor_cb_index,
        input_index_tensor_cb_index,
        output_tensor_cb_index,
        static_cast<uint32_t>(input_index_tensor_is_dram),
        Ht,
        Wt_input,
        Wt_index,
        total_number_of_cores,
        compute_with_storage_grid_size.x,
        compute_with_storage_grid_size.y};
    TensorAccessorArgs(*input_index_tensor_buffer).append_to(reader_compile_time_args);
    const std::string gather_reader_kernel_path =
        "ttnn/cpp/ttnn/operations/data_movement/gather/device/kernels/dataflow/gather_reader_single_row_multi_core.cpp";
    tt::tt_metal::KernelHandle gather_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        gather_reader_kernel_path,
        core_range,
        tt::tt_metal::ReaderDataMovementConfig{reader_compile_time_args});

    std::vector<uint32_t> writer_compile_time_args = {
        input_tensor_cb_index,
        output_tensor_cb_index,
        static_cast<uint32_t>(input_tensor_is_dram),
        static_cast<uint32_t>(output_tensor_is_dram),
        Ht,
        Wt_input,
        Wt_index,
        total_number_of_cores,
        compute_with_storage_grid_size.x,
        compute_with_storage_grid_size.y};
    TensorAccessorArgs(*input_tensor_buffer).append_to(writer_compile_time_args);
    TensorAccessorArgs(*output_tensor_buffer).append_to(writer_compile_time_args);
    const std::string gather_writer_kernel_path =
        "ttnn/cpp/ttnn/operations/data_movement/gather/device/kernels/dataflow/gather_writer_single_row_multi_core.cpp";
    tt::tt_metal::KernelHandle gather_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        gather_writer_kernel_path,
        core_range,
        tt::tt_metal::WriterDataMovementConfig{writer_compile_time_args});

    uint32_t id = 0;  // Offset for the next core in the group
    for (const auto& [group, work_per_core] : work_groups) {
        for (const auto& range : group.ranges()) {
            for (const auto& core : range) {
                SetRuntimeArgs(
                    program,
                    gather_reader_kernel_id,
                    core,
                    {input_index_tensor_buffer->address(), work_per_core, tile_width, tile_height, id});
                SetRuntimeArgs(
                    program,
                    gather_writer_kernel_id,
                    core,
                    {input_tensor_buffer->address(), output_tensor_buffer->address(), work_per_core, id});
                id++;
            }
        }
    }

    return {std::move(program), {gather_reader_kernel_id, gather_writer_kernel_id, cores}};
}

void GatherProgramFactorySingleRowMultiCore::override_runtime_arguments(
    cached_program_t& cached_program,
    const GatherParams& /*attributes*/,
    const GatherInputs& tensor_args,
    Tensor& output_tensor) {
    // Get tensor buffers
    auto* input_tensor_buffer = tensor_args.input_tensor.buffer();
    auto* input_index_tensor_buffer = tensor_args.input_index_tensor.buffer();
    auto* output_tensor_buffer = output_tensor.buffer();

    // Update runtime arguments for each core
    for (const auto& core : cached_program.shared_variables.cores) {
        auto& gather_reader_runtime_args = tt::tt_metal::GetRuntimeArgs(
            cached_program.program, cached_program.shared_variables.gather_reader_kernel_id, core);
        gather_reader_runtime_args[0] = input_index_tensor_buffer->address();

        auto& gather_writer_runtime_args = tt::tt_metal::GetRuntimeArgs(
            cached_program.program, cached_program.shared_variables.gather_writer_kernel_id, core);
        gather_writer_runtime_args[0] = input_tensor_buffer->address();
        gather_writer_runtime_args[1] = output_tensor_buffer->address();
    }
}
}  // namespace ttnn::prim
