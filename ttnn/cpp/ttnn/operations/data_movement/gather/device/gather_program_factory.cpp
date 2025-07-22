// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gather_program_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>

namespace ttnn::operations::data_movement::gather::program {
// Single row - single core
GatherProgramFactorySingleRowSingleCore::cached_program_t GatherProgramFactorySingleRowSingleCore::create(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args, tensor_return_value_t& output_tensor) {
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

    auto input_tensor_buffer = tensor_args.input_tensor.buffer();
    auto input_index_tensor_buffer = tensor_args.input_index_tensor.buffer();
    auto output_tensor_buffer = output_tensor.buffer();

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
    const std::vector<uint32_t> reader_compile_time_args = {
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
    const std::string gather_reader_kernel_path =
        "ttnn/cpp/ttnn/operations/data_movement/gather/device/kernels/dataflow/"
        "gather_reader_single_row_single_core.cpp";
    tt::tt_metal::KernelHandle gather_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        gather_reader_kernel_path,
        core_range,
        tt::tt_metal::ReaderDataMovementConfig{reader_compile_time_args});
    SetRuntimeArgs(
        program,
        gather_reader_kernel_id,
        core_range,
        {input_index_tensor_buffer->address(),
         all_core_utilization_loop_count ? all_core_utilization_loop_count : 1,
         tile_width,
         tile_height});

    const std::vector<uint32_t> writer_compile_time_args = {
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
    const std::string gather_writer_kernel_path =
        "ttnn/cpp/ttnn/operations/data_movement/gather/device/kernels/dataflow/"
        "gather_writer_single_row_single_core.cpp";
    tt::tt_metal::KernelHandle gather_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        gather_writer_kernel_path,
        core_range,
        tt::tt_metal::WriterDataMovementConfig{writer_compile_time_args});
    SetRuntimeArgs(
        program,
        gather_writer_kernel_id,
        core_range,
        {input_tensor_buffer->address(),
         output_tensor_buffer->address(),
         all_core_utilization_loop_count ? all_core_utilization_loop_count : 1});

    // Adjust runtime arguments if cores are used more than once
    if (all_core_utilization_loop_residuum != 0 && all_core_utilization_loop_count != 0) {
        uint32_t residuum_count = 0;
        for (uint32_t core_y = 0; core_y < compute_with_storage_grid_size.y; core_y++) {
            for (uint32_t core_x = 0; core_x < compute_with_storage_grid_size.x; core_x++) {
                const uint32_t new_loop_count = all_core_utilization_loop_count + 1;
                const CoreCoord core = {core_x, core_y};

                SetRuntimeArgs(
                    program,
                    gather_reader_kernel_id,
                    core,
                    {input_index_tensor_buffer->address(), new_loop_count, tile_width, tile_height});

                SetRuntimeArgs(
                    program,
                    gather_writer_kernel_id,
                    core,
                    {input_tensor_buffer->address(), output_tensor_buffer->address(), new_loop_count});

                residuum_count++;
                if (residuum_count >= all_core_utilization_loop_residuum) {
                    core_y = compute_with_storage_grid_size.y;  // Break outer loop
                    break;
                }
            }  // core_x loop
        }  // core_y loop
    }  // if all_core_utilization_loop_residuum != 0 && all_core_utilization_loop_count != 0

    return {std::move(program), {gather_reader_kernel_id, gather_writer_kernel_id, compute_with_storage_grid_size}};
}

void GatherProgramFactorySingleRowSingleCore::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    auto input_tensor_buffer = tensor_args.input_tensor.buffer();
    auto input_index_tensor_buffer = tensor_args.input_tensor.buffer();
    auto output_tensor_buffer = output_tensor.buffer();

    const auto input_index_shape = tensor_args.input_index_tensor.padded_shape();
    const uint32_t Ht =
        (input_index_shape[0] * input_index_shape[1] * input_index_shape[2]) / tt::constants::TILE_HEIGHT;
    const uint32_t total_number_of_cores =
        cached_program.shared_variables.storage_grid_size.x * cached_program.shared_variables.storage_grid_size.y;

    // Calculate the number of cores utilized based on the input tensor shape
    const uint32_t all_core_utilization_loop_count = Ht / total_number_of_cores;
    const uint32_t all_core_utilization_loop_residuum = Ht % total_number_of_cores;

    uint32_t residuum_count = 0;
    for (uint32_t core_y = 0; core_y < cached_program.shared_variables.storage_grid_size.y; core_y++) {
        for (uint32_t core_x = 0; core_x < cached_program.shared_variables.storage_grid_size.x; core_x++) {
            const CoreCoord core = {core_x, core_y};
            auto& gather_reader_runtime_args =
                GetRuntimeArgs(cached_program.program, cached_program.shared_variables.gather_reader_kernel_id, core);
            gather_reader_runtime_args[0] = input_index_tensor_buffer->address();

            auto& gather_writer_runtime_args =
                GetRuntimeArgs(cached_program.program, cached_program.shared_variables.gather_writer_kernel_id, core);
            gather_writer_runtime_args[0] = input_tensor_buffer->address();
            gather_writer_runtime_args[1] = output_tensor_buffer->address();

            if (all_core_utilization_loop_count < 1 && all_core_utilization_loop_residuum != 0) {
                residuum_count++;
                if (residuum_count >= all_core_utilization_loop_residuum) {
                    core_y = cached_program.shared_variables.storage_grid_size.y;  // Break outer loop
                    break;
                }
            }
        }  // core_x loop
    }  // core_y loop
}

// Single row - multi core
GatherProgramFactorySingleRowMultiCore::cached_program_t GatherProgramFactorySingleRowMultiCore::create(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args, tensor_return_value_t& output_tensor) {
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

    const auto input_tensor_buffer = tensor_args.input_tensor.buffer();
    const auto input_index_tensor_buffer = tensor_args.input_index_tensor.buffer();
    const auto output_tensor_buffer = output_tensor.buffer();

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
    auto device = tensor_args.input_tensor.device();
    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t total_number_of_cores = compute_with_storage_grid_size.y * compute_with_storage_grid_size.x;

    // Calculate how many iterations of outer loop - index loop we need
    const auto all_core_utilization_loop_count = Wt_index / total_number_of_cores;
    const auto all_core_utilization_loop_residuum = Wt_index % total_number_of_cores;

    // Calculate core range
    /**
     * Calculates the core range based on the input tensor shape (Wt_index) and the total number of cores available
     * in the device's compute grid. The core range determines which cores will be utilized for computation.
     *
     * The calculation works as follows:
     * 1. If the width (Wt_index) of the input index tensor is greater than or equal to the total number of cores,
     *    all cores in the compute grid are utilized. The core range is set to cover the entire grid.
     *
     * 2. If Wt_index is smaller than the total number of cores:
     *    - The number of rows (`core_grid_calculated_rows_number`) and columns
     * (`core_grid_calculated_columns_number`) required to cover Wt_index are calculated based on the grid dimensions.
     *    - If both rows and columns are zero, only a single core is used.
     *    - If only rows are zero, the core range is set to cover the required number of columns in the first row.
     *    - Otherwise, the core range is set to cover the required rows, and if there are remaining columns,
     *      an additional range is added to cover those columns in the next row.
     *
     * The resulting core range is represented as a `CoreRangeSet`, which may consist of one or more `CoreRange`
     * objects depending on the configuration.
     */
    CoreRangeSet core_range;
    if (Wt_index >= total_number_of_cores) {
        core_range = CoreRangeSet(
            CoreRange({0, 0}, {compute_with_storage_grid_size.x - 1, compute_with_storage_grid_size.y - 1}));
    } else {
        const uint32_t core_grid_calculated_rows_number = Wt_index / compute_with_storage_grid_size.x;
        const uint32_t core_grid_calculated_columns_number = Wt_index % compute_with_storage_grid_size.x;

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
    const std::vector<uint32_t> reader_compile_time_args = {
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
    const std::string gather_reader_kernel_path =
        "ttnn/cpp/ttnn/operations/data_movement/gather/device/kernels/dataflow/gather_reader_single_row_multi_core.cpp";
    tt::tt_metal::KernelHandle gather_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        gather_reader_kernel_path,
        core_range,
        tt::tt_metal::ReaderDataMovementConfig{reader_compile_time_args});
    SetRuntimeArgs(
        program,
        gather_reader_kernel_id,
        core_range,
        {input_index_tensor_buffer->address(),
         all_core_utilization_loop_count ? all_core_utilization_loop_count : 1,
         tile_width,
         tile_height});

    const std::vector<uint32_t> writer_compile_time_args = {
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
    const std::string gather_writer_kernel_path =
        "ttnn/cpp/ttnn/operations/data_movement/gather/device/kernels/dataflow/gather_writer_single_row_multi_core.cpp";
    tt::tt_metal::KernelHandle gather_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        gather_writer_kernel_path,
        core_range,
        tt::tt_metal::WriterDataMovementConfig{writer_compile_time_args});
    SetRuntimeArgs(
        program,
        gather_writer_kernel_id,
        core_range,
        {input_tensor_buffer->address(),
         output_tensor_buffer->address(),
         all_core_utilization_loop_count ? all_core_utilization_loop_count : 1});

    // Adjust runtime arguments if cores are used more than once
    if (all_core_utilization_loop_residuum != 0 && all_core_utilization_loop_count != 0) {
        uint32_t residuum_count = 0;
        for (uint32_t core_y = 0; core_y < compute_with_storage_grid_size.y; core_y++) {
            for (uint32_t core_x = 0; core_x < compute_with_storage_grid_size.x; core_x++) {
                const uint32_t new_loop_count = all_core_utilization_loop_count + 1;
                const CoreCoord core = {core_x, core_y};

                SetRuntimeArgs(
                    program,
                    gather_reader_kernel_id,
                    core,
                    {input_index_tensor_buffer->address(), new_loop_count, tile_width, tile_height});

                SetRuntimeArgs(
                    program,
                    gather_writer_kernel_id,
                    core,
                    {input_tensor_buffer->address(), output_tensor_buffer->address(), new_loop_count});

                residuum_count++;
                if (residuum_count >= all_core_utilization_loop_residuum) {
                    core_y = compute_with_storage_grid_size.y;  // Break outer loop
                    break;
                }
            }  // core_x loop
        }  // core_y loop
    }  // if all_core_utilization_loop_residuum != 0 && all_core_utilization_loop_count != 0

    return {std::move(program), {gather_reader_kernel_id, gather_writer_kernel_id, compute_with_storage_grid_size}};
}

void GatherProgramFactorySingleRowMultiCore::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    auto input_tensor_buffer = tensor_args.input_tensor.buffer();
    auto input_index_tensor_buffer = tensor_args.input_tensor.buffer();
    auto output_tensor_buffer = output_tensor.buffer();

    const auto input_index_shape = tensor_args.input_index_tensor.padded_shape();
    const auto tile_width = tensor_args.input_tensor.tensor_spec().tile().get_width();
    const uint32_t Wt_index = tensor_args.input_index_tensor.padded_shape()[3] / tile_width;
    const uint32_t total_number_of_cores =
        cached_program.shared_variables.storage_grid_size.x * cached_program.shared_variables.storage_grid_size.y;

    // Calculate the number of cores utilized based on the input tensor shape
    const uint32_t all_core_utilization_loop_count = Wt_index / total_number_of_cores;
    const uint32_t all_core_utilization_loop_residuum = Wt_index % total_number_of_cores;

    uint32_t residuum_count = 0;
    for (uint32_t core_y = 0; core_y < cached_program.shared_variables.storage_grid_size.y; core_y++) {
        for (uint32_t core_x = 0; core_x < cached_program.shared_variables.storage_grid_size.x; core_x++) {
            const CoreCoord core = {core_x, core_y};
            auto& gather_reader_runtime_args =
                GetRuntimeArgs(cached_program.program, cached_program.shared_variables.gather_reader_kernel_id, core);
            gather_reader_runtime_args[0] = input_index_tensor_buffer->address();

            auto& gather_writer_runtime_args =
                GetRuntimeArgs(cached_program.program, cached_program.shared_variables.gather_writer_kernel_id, core);
            gather_writer_runtime_args[0] = input_tensor_buffer->address();
            gather_writer_runtime_args[1] = output_tensor_buffer->address();

            if (all_core_utilization_loop_count < 1 && all_core_utilization_loop_residuum != 0) {
                residuum_count++;
                if (residuum_count >= all_core_utilization_loop_residuum) {
                    core_y = cached_program.shared_variables.storage_grid_size.y;  // Break outer loop
                    break;
                }
            }
        }  // core_x loop
    }  // core_y loop
}
}  // namespace ttnn::operations::data_movement::gather::program
