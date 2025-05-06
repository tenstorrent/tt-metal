// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sort_program_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>

namespace ttnn::operations::experimental::reduction::sort::program {

SortProgramFactory::cached_program_t SortProgramFactory::create(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args, tensor_return_value_t& output_tensors) {
    // Program config
    tt::tt_metal::Program program{};

    const tt::DataFormat input_tensor_cb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(tensor_args.input_tensor.get_dtype());
    const tt::DataFormat value_tensor_cb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(output_tensors.at(0).get_dtype());
    const tt::DataFormat index_tensor_cb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(output_tensors.at(1).get_dtype());

    const uint32_t input_tensor_tile_size = tile_size(input_tensor_cb_data_format);
    const uint32_t value_tensor_tile_size = tile_size(value_tensor_cb_data_format);
    const uint32_t index_tensor_tile_size = tile_size(index_tensor_cb_data_format);

    auto input_buffer = tensor_args.input_tensor.buffer();
    auto value_buffer = output_tensors.at(0).buffer();
    auto index_buffer = output_tensors.at(1).buffer();

    const bool input_tensor_is_dram = input_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    const bool value_tensor_is_dram = value_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    const bool index_tensor_is_dram = index_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;

    const uint32_t num_input_tiles = tensor_args.input_tensor.volume() / tt::constants::TILE_HW;
    const uint32_t num_value_tiles = output_tensors.at(0).volume() / tt::constants::TILE_HW;

    const auto input_shape = tensor_args.input_tensor.get_padded_shape();
    const uint32_t Ht = (input_shape[0] * input_shape[1] * input_shape[2]) / tt::constants::TILE_HEIGHT;
    const uint32_t Wt = input_shape[3] / tt::constants::TILE_WIDTH;

    // Double buffering config
    constexpr uint32_t num_cb_unit = 2;                // Number of circular buffer units for double buffering
    constexpr uint32_t cb_in_units = 2 * num_cb_unit;  // Total number of circular buffer units

    // Calculate the number of cores available for computation
    auto device = tensor_args.input_tensor.device();
    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t total_number_of_cores = compute_with_storage_grid_size.y * compute_with_storage_grid_size.x;

    // Calculate the number of cores utilized based on the input tensor shape
    const uint32_t all_core_utilization_loop_count = Ht / total_number_of_cores;
    const uint32_t all_core_utilization_loop_residuum = Ht % total_number_of_cores;

    // Calculate core range
    /**
     * Calculates the core range based on the input tensor shape (Ht) and the total number of cores available
     * in the device's compute grid. The core range determines which cores will be utilized for computation.
     *
     * The calculation works as follows:
     * 1. If the height (Ht) of the input tensor is greater than or equal to the total number of cores,
     *    all cores in the compute grid are utilized. The core range is set to cover the entire grid.
     *
     * 2. If Ht is smaller than the total number of cores:
     *    - The number of rows (`core_grid_calculated_rows_number`) and columns (`core_grid_calculated_columns_number`)
     *      required to cover Ht are calculated based on the grid dimensions.
     *    - If both rows and columns are zero, only a single core is used.
     *    - If only rows are zero, the core range is set to cover the required number of columns in the first row.
     *    - Otherwise, the core range is set to cover the required rows, and if there are remaining columns,
     *      an additional range is added to cover those columns in the next row.
     *
     * The resulting core range is represented as a `CoreRangeSet`, which may consist of one or more `CoreRange`
     * objects depending on the configuration.
     */
    CoreRangeSet core_range;
    if (Ht >= total_number_of_cores) {
        core_range = CoreRangeSet(
            CoreRange({0, 0}, {compute_with_storage_grid_size.x - 1, compute_with_storage_grid_size.y - 1}));
    } else {
        const uint32_t core_grid_calculated_rows_number = Ht / compute_with_storage_grid_size.x;
        const uint32_t core_grid_calculated_columns_number = Ht % compute_with_storage_grid_size.x;

        if (core_grid_calculated_rows_number == 0 && core_grid_calculated_columns_number == 0) {
            core_range = CoreRangeSet(CoreCoord({0, 0}));
        } else if (core_grid_calculated_rows_number == 0) {
            core_range = CoreRangeSet(CoreRange({0, 0}, {core_grid_calculated_columns_number - 1, 0}));
        } else {
            core_range = CoreRangeSet(
                CoreRange({0, 0}, {compute_with_storage_grid_size.x - 1, core_grid_calculated_rows_number - 1}));
            if (core_grid_calculated_columns_number != 0) {
                const CoreRange additional_range(
                    {0, core_grid_calculated_rows_number},
                    {core_grid_calculated_columns_number, core_grid_calculated_rows_number});
                core_range = core_range.merge(CoreRangeSet(additional_range));
            }
        }
    }

    // Circular buffers
    constexpr uint32_t input_tensor_cb_index = tt::CBIndex::c_0;
    const tt::tt_metal::CircularBufferConfig input_tensor_cb_config =
        tt::tt_metal::CircularBufferConfig(
            cb_in_units * input_tensor_tile_size, {{input_tensor_cb_index, input_tensor_cb_data_format}})
            .set_page_size(input_tensor_cb_index, input_tensor_tile_size);
    auto cb_input_tensor = tt::tt_metal::CreateCircularBuffer(program, core_range, input_tensor_cb_config);

    constexpr uint32_t index_tensor_cb_index = tt::CBIndex::c_1;
    const tt::tt_metal::CircularBufferConfig index_tensor_cb_config =
        tt::tt_metal::CircularBufferConfig(
            cb_in_units * index_tensor_tile_size, {{index_tensor_cb_index, index_tensor_cb_data_format}})
            .set_page_size(index_tensor_cb_index, index_tensor_tile_size);
    auto cb_index_tensor = tt::tt_metal::CreateCircularBuffer(program, core_range, index_tensor_cb_config);

    constexpr uint32_t input_tensor_transposed_cb_index = tt::CBIndex::c_24;
    const tt::tt_metal::CircularBufferConfig input_tensor_transposed_cb_config =
        tt::tt_metal::CircularBufferConfig(
            Wt * input_tensor_tile_size, {{input_tensor_transposed_cb_index, input_tensor_cb_data_format}})
            .set_page_size(input_tensor_transposed_cb_index, input_tensor_tile_size);
    auto cb_input_tensor_transposed =
        tt::tt_metal::CreateCircularBuffer(program, core_range, input_tensor_transposed_cb_config);

    constexpr uint32_t index_tensor_transposed_cb_index = tt::CBIndex::c_25;
    const tt::tt_metal::CircularBufferConfig index_tensor_transposed_cb_config =
        tt::tt_metal::CircularBufferConfig(
            Wt * index_tensor_tile_size, {{index_tensor_transposed_cb_index, index_tensor_cb_data_format}})
            .set_page_size(index_tensor_transposed_cb_index, index_tensor_tile_size);
    auto cb_index_tensor_transposed =
        tt::tt_metal::CreateCircularBuffer(program, core_range, index_tensor_transposed_cb_config);

    constexpr uint32_t value_tensor_cb_index = tt::CBIndex::c_16;
    const tt::tt_metal::CircularBufferConfig value_tensor_cb_config =
        tt::tt_metal::CircularBufferConfig(
            num_cb_unit * value_tensor_tile_size, {{value_tensor_cb_index, value_tensor_cb_data_format}})
            .set_page_size(value_tensor_cb_index, index_tensor_tile_size);
    auto cb_value_tensor = tt::tt_metal::CreateCircularBuffer(program, core_range, value_tensor_cb_config);

    constexpr uint32_t index_tensor_output_cb_index = tt::CBIndex::c_17;
    const tt::tt_metal::CircularBufferConfig index_tensor_output_cb_config =
        tt::tt_metal::CircularBufferConfig(
            num_cb_unit * index_tensor_tile_size, {{index_tensor_output_cb_index, index_tensor_cb_data_format}})
            .set_page_size(index_tensor_output_cb_index, index_tensor_tile_size);
    auto cb_index_tensor_output =
        tt::tt_metal::CreateCircularBuffer(program, core_range, index_tensor_output_cb_config);

    // Kernels
    const std::vector<uint32_t> reader_compile_time_args = {
        input_tensor_cb_index,
        index_tensor_output_cb_index,
        static_cast<uint32_t>(input_tensor_is_dram),
        static_cast<uint32_t>(index_tensor_is_dram),
        Wt,
        Ht,
        total_number_of_cores,
        compute_with_storage_grid_size.x,
        compute_with_storage_grid_size.y};
    const std::string reader_kernel_path =
        "ttnn/cpp/ttnn/operations/experimental/reduction/sort/device/kernels/dataflow/reader.cpp";
    tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program, reader_kernel_path, core_range, tt::tt_metal::ReaderDataMovementConfig{reader_compile_time_args});
    SetRuntimeArgs(
        program,
        reader_kernel_id,
        core_range,
        {input_buffer->address(),
         index_buffer->address(),
         all_core_utilization_loop_count ? all_core_utilization_loop_count : 1});

    const std::vector<uint32_t> writer_compile_time_args = {
        value_tensor_cb_index,
        index_tensor_cb_index,
        static_cast<uint32_t>(value_tensor_is_dram),
        Wt,
        Ht,
        total_number_of_cores,
        compute_with_storage_grid_size.x,
        compute_with_storage_grid_size.y};
    const std::string writer_kernel_path =
        "ttnn/cpp/ttnn/operations/experimental/reduction/sort/device/kernels/dataflow/writer.cpp";
    tt::tt_metal::KernelHandle writer_kernel_id = tt::tt_metal::CreateKernel(
        program, writer_kernel_path, core_range, tt::tt_metal::WriterDataMovementConfig{writer_compile_time_args});
    SetRuntimeArgs(
        program,
        writer_kernel_id,
        core_range,
        {value_buffer->address(), all_core_utilization_loop_count ? all_core_utilization_loop_count : 1});

    const std::vector<uint32_t> compute_compile_time_args = {
        input_tensor_cb_index,
        index_tensor_cb_index,
        input_tensor_transposed_cb_index,
        index_tensor_transposed_cb_index,
        value_tensor_cb_index,
        index_tensor_output_cb_index,
        Wt,
        static_cast<uint32_t>(attributes.descending),
        static_cast<uint32_t>(attributes.stable),
        compute_with_storage_grid_size.x,
        compute_with_storage_grid_size.y};
    const std::string compute_kernel_path =
        "ttnn/cpp/ttnn/operations/experimental/reduction/sort/device/kernels/compute/sort.cpp";
    tt::tt_metal::KernelHandle compute_kernel_id = tt::tt_metal::CreateKernel(
        program,
        compute_kernel_path,
        core_range,
        tt::tt_metal::ComputeConfig{.compile_args = compute_compile_time_args});
    SetRuntimeArgs(
        program,
        compute_kernel_id,
        core_range,
        {all_core_utilization_loop_count ? all_core_utilization_loop_count : 1});

    if (all_core_utilization_loop_residuum != 0 && all_core_utilization_loop_count != 0) {
        uint32_t residuum_count = 0;
        for (uint32_t core_y = 0; core_y < compute_with_storage_grid_size.y; core_y++) {
            for (uint32_t core_x = 0; core_x < compute_with_storage_grid_size.x; core_x++) {
                const uint32_t new_loop_count = all_core_utilization_loop_count + 1;
                const CoreCoord core = {core_x, core_y};

                SetRuntimeArgs(
                    program,
                    reader_kernel_id,
                    core,
                    {input_buffer->address(), index_buffer->address(), new_loop_count});

                SetRuntimeArgs(program, writer_kernel_id, core, {value_buffer->address(), new_loop_count});

                SetRuntimeArgs(program, compute_kernel_id, core, {new_loop_count});

                residuum_count++;
                if (residuum_count >= all_core_utilization_loop_residuum) {
                    core_y = compute_with_storage_grid_size.y;  // Break outer loop
                    break;
                }
            }  // core_x loop
        }  // core_y loop
    }

    return {
        std::move(program), {reader_kernel_id, compute_kernel_id, writer_kernel_id, compute_with_storage_grid_size}};
}

void SortProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensors) {
    auto input_tensor_buffer = tensor_args.input_tensor.buffer();
    auto value_tensor_buffer = output_tensors.at(0).buffer();
    auto index_tensor_buffer = output_tensors.at(1).buffer();

    const auto input_shape = tensor_args.input_tensor.get_padded_shape();
    const uint32_t Ht = (input_shape[0] * input_shape[1] * input_shape[2]) / tt::constants::TILE_HEIGHT;
    const uint32_t total_number_of_cores =
        cached_program.shared_variables.storage_grid_size.x * cached_program.shared_variables.storage_grid_size.y;

    // Calculate the number of cores utilized based on the input tensor shape
    const uint32_t all_core_utilization_loop_count = Ht / total_number_of_cores;
    const uint32_t all_core_utilization_loop_residuum = Ht % total_number_of_cores;

    uint32_t residuum_count = 0;
    for (uint32_t core_y = 0; core_y < cached_program.shared_variables.storage_grid_size.y; core_y++) {
        for (uint32_t core_x = 0; core_x < cached_program.shared_variables.storage_grid_size.x; core_x++) {
            const CoreCoord core = {core_x, core_y};
            auto& reader_runtime_args =
                GetRuntimeArgs(cached_program.program, cached_program.shared_variables.reader_kernel_id, core);
            reader_runtime_args[0] = input_tensor_buffer->address();

            auto& writer_runtime_args =
                GetRuntimeArgs(cached_program.program, cached_program.shared_variables.writer_kernel_id, core);
            writer_runtime_args[0] = value_tensor_buffer->address();
            writer_runtime_args[1] = index_tensor_buffer->address();

            if (all_core_utilization_loop_count < 1 && all_core_utilization_loop_residuum != 0) {
                residuum_count++;
                if (residuum_count >= all_core_utilization_loop_residuum) {
                    core_y = cached_program.shared_variables.storage_grid_size.y;  // Break outer loop
                    break;
                }
            }
        }
    }
}

}  // namespace ttnn::operations::experimental::reduction::sort::program
