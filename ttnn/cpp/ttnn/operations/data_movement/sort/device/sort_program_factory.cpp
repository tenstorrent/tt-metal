// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "sort_program_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include <cmath>
#include <cstdint>

namespace ttnn::prim {

// Single row - single core
SortProgramFactorySingleRowSingleCore::cached_program_t SortProgramFactorySingleRowSingleCore::create(
    const SortParams& attributes, const SortInputs& tensor_args, std::vector<Tensor>& output_tensors) {
    // Program config
    tt::tt_metal::Program program{};

    const tt::DataFormat input_tensor_cb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(tensor_args.input_tensor.dtype());
    const tt::DataFormat value_tensor_cb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(output_tensors.at(0).dtype());
    const tt::DataFormat index_tensor_cb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(output_tensors.at(1).dtype());

    const uint32_t input_tensor_tile_size = tile_size(input_tensor_cb_data_format);
    const uint32_t value_tensor_tile_size = tile_size(value_tensor_cb_data_format);
    const uint32_t index_tensor_tile_size = tile_size(index_tensor_cb_data_format);

    auto* input_buffer = tensor_args.input_tensor.buffer();
    auto* value_buffer = output_tensors.at(0).buffer();
    auto* index_buffer = output_tensors.at(1).buffer();

    const auto input_shape = tensor_args.input_tensor.padded_shape();
    const uint32_t Ht = (input_shape[0] * input_shape[1] * input_shape[2]) / tt::constants::TILE_HEIGHT;
    const uint32_t Wt = input_shape[3] / tt::constants::TILE_WIDTH;

    // Double buffering config
    constexpr uint32_t num_cb_unit = 2;                // Number of circular buffer units for double buffering
    constexpr uint32_t cb_in_units = 2 * num_cb_unit;  // Total number of circular buffer units

    // Calculate the number of cores available for computation
    auto* device = tensor_args.input_tensor.device();
    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t total_number_of_cores = compute_with_storage_grid_size.y * compute_with_storage_grid_size.x;

    // Calculate the number of cores utilized based on the input tensor shape
    const uint32_t all_core_utilization_loop_count = Ht / total_number_of_cores;
    const uint32_t all_core_utilization_loop_residuum = Ht % total_number_of_cores;

    // uint32 index tensor support
    const bool is_32_bit_data = index_tensor_cb_data_format == tt::DataFormat::UInt32;

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
    tt::tt_metal::CreateCircularBuffer(program, core_range, input_tensor_cb_config);

    constexpr uint32_t index_tensor_cb_index = tt::CBIndex::c_1;
    const tt::tt_metal::CircularBufferConfig index_tensor_cb_config =
        tt::tt_metal::CircularBufferConfig(
            cb_in_units * index_tensor_tile_size, {{index_tensor_cb_index, index_tensor_cb_data_format}})
            .set_page_size(index_tensor_cb_index, index_tensor_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, core_range, index_tensor_cb_config);

    constexpr uint32_t input_tensor_transposed_cb_index = tt::CBIndex::c_2;
    const tt::tt_metal::CircularBufferConfig input_tensor_transposed_cb_config =
        tt::tt_metal::CircularBufferConfig(
            Wt * input_tensor_tile_size, {{input_tensor_transposed_cb_index, input_tensor_cb_data_format}})
            .set_page_size(input_tensor_transposed_cb_index, input_tensor_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, core_range, input_tensor_transposed_cb_config);

    constexpr uint32_t index_tensor_transposed_cb_index = tt::CBIndex::c_3;
    const tt::tt_metal::CircularBufferConfig index_tensor_transposed_cb_config =
        tt::tt_metal::CircularBufferConfig(
            Wt * index_tensor_tile_size, {{index_tensor_transposed_cb_index, index_tensor_cb_data_format}})
            .set_page_size(index_tensor_transposed_cb_index, index_tensor_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, core_range, index_tensor_transposed_cb_config);

    constexpr uint32_t value_tensor_cb_index = tt::CBIndex::c_4;
    const tt::tt_metal::CircularBufferConfig value_tensor_cb_config =
        tt::tt_metal::CircularBufferConfig(
            num_cb_unit * value_tensor_tile_size, {{value_tensor_cb_index, value_tensor_cb_data_format}})
            .set_page_size(value_tensor_cb_index, index_tensor_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, core_range, value_tensor_cb_config);

    constexpr uint32_t index_tensor_output_cb_index = tt::CBIndex::c_5;
    const tt::tt_metal::CircularBufferConfig index_tensor_output_cb_config =
        tt::tt_metal::CircularBufferConfig(
            num_cb_unit * index_tensor_tile_size, {{index_tensor_output_cb_index, index_tensor_cb_data_format}})
            .set_page_size(index_tensor_output_cb_index, index_tensor_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, core_range, index_tensor_output_cb_config);

    constexpr uint32_t synchronization_cb_index = tt::CBIndex::c_6;
    constexpr uint32_t synchronization_cb_size = tt::constants::TILE_HW * sizeof(uint8_t);
    const tt::tt_metal::CircularBufferConfig synchronization_cb_config =
        tt::tt_metal::CircularBufferConfig(synchronization_cb_size, {{synchronization_cb_index, tt::DataFormat::UInt8}})
            .set_page_size(synchronization_cb_index, synchronization_cb_size);
    tt::tt_metal::CreateCircularBuffer(program, core_range, synchronization_cb_config);

    // Kernels
    std::vector<uint32_t> reader_compile_time_args = {
        input_tensor_cb_index,
        index_tensor_output_cb_index,
        Wt,
        Ht,
        total_number_of_cores,
        compute_with_storage_grid_size.x,
        compute_with_storage_grid_size.y};
    TensorAccessorArgs(*input_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*index_buffer).append_to(reader_compile_time_args);
    const std::string reader_kernel_path =
        "ttnn/cpp/ttnn/operations/data_movement/sort/device/kernels/dataflow/reader_single_row_single_core.cpp";
    tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program, reader_kernel_path, core_range, tt::tt_metal::ReaderDataMovementConfig{reader_compile_time_args});
    SetRuntimeArgs(
        program,
        reader_kernel_id,
        core_range,
        {input_buffer->address(),
         index_buffer->address(),
         all_core_utilization_loop_count ? all_core_utilization_loop_count : 1});

    std::vector<uint32_t> writer_compile_time_args = {
        value_tensor_cb_index,
        index_tensor_cb_index,
        Wt,
        Ht,
        total_number_of_cores,
        compute_with_storage_grid_size.x,
        compute_with_storage_grid_size.y,
        static_cast<uint32_t>(is_32_bit_data)};
    TensorAccessorArgs(*value_buffer).append_to(writer_compile_time_args);
    const std::string writer_kernel_path =
        "ttnn/cpp/ttnn/operations/data_movement/sort/device/kernels/dataflow/writer_single_row_single_core.cpp";
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
        synchronization_cb_index};
    const std::string compute_kernel_path =
        "ttnn/cpp/ttnn/operations/data_movement/sort/device/kernels/compute/sort_single_row_single_core.cpp";
    tt::tt_metal::KernelHandle compute_kernel_id = tt::tt_metal::CreateKernel(
        program,
        compute_kernel_path,
        core_range,
        tt::tt_metal::ComputeConfig{.fp32_dest_acc_en = is_32_bit_data, .compile_args = compute_compile_time_args});
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

void SortProgramFactorySingleRowSingleCore::override_runtime_arguments(
    cached_program_t& cached_program,
    const SortParams& /*attributes*/,
    const SortInputs& tensor_args,
    std::vector<Tensor>& output_tensors) {
    auto* input_tensor_buffer = tensor_args.input_tensor.buffer();
    auto* value_tensor_buffer = output_tensors.at(0).buffer();
    auto* index_tensor_buffer = output_tensors.at(1).buffer();

    const auto input_shape = tensor_args.input_tensor.padded_shape();
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
            reader_runtime_args[1] = index_tensor_buffer->address();

            auto& writer_runtime_args =
                GetRuntimeArgs(cached_program.program, cached_program.shared_variables.writer_kernel_id, core);
            writer_runtime_args[0] = value_tensor_buffer->address();

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

// SortProgramFactoryCrossCoreDataExchange - single row, multi core with processing multiple tiles on one core with
// cross core data exchange
SortProgramFactoryCrossCoreDataExchange::cached_program_t SortProgramFactoryCrossCoreDataExchange::create(
    const SortParams& attributes, const SortInputs& tensor_args, std::vector<Tensor>& output_tensors) {
    // Program config
    tt::tt_metal::Program program{};

    const tt::DataFormat input_tensor_cb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(tensor_args.input_tensor.dtype());
    const tt::DataFormat value_tensor_cb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(output_tensors.at(0).dtype());
    const tt::DataFormat index_tensor_cb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(output_tensors.at(1).dtype());
    const tt::DataFormat packer_unpacker_sync_cb_data_format = tt::DataFormat::Float16_b;

    const uint32_t input_tensor_tile_size = tile_size(input_tensor_cb_data_format);
    const uint32_t value_tensor_tile_size = tile_size(value_tensor_cb_data_format);
    const uint32_t index_tensor_tile_size = tile_size(index_tensor_cb_data_format);
    const uint32_t packer_unpacker_sync_tile_size = tile_size(packer_unpacker_sync_cb_data_format);

    auto* input_buffer = tensor_args.input_tensor.buffer();
    auto* value_buffer = output_tensors.at(0).buffer();
    auto* index_buffer = output_tensors.at(1).buffer();

    const auto tile_width = tensor_args.input_tensor.tensor_spec().tile().get_width();
    const auto tile_height = tensor_args.input_tensor.tensor_spec().tile().get_height();

    const auto input_shape = tensor_args.input_tensor.padded_shape();
    const uint32_t Ht = (input_shape[0] * input_shape[1] * input_shape[2]) / tile_height;
    const uint32_t Wt = input_shape[3] / tile_width;

    // Calculate the number of cores available for computation
    auto* const device = tensor_args.input_tensor.device();
    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t total_number_of_cores_physical = compute_with_storage_grid_size.y * compute_with_storage_grid_size.x;
    const uint32_t total_number_of_cores_virtual = rounddown_pow2(total_number_of_cores_physical);
    uint32_t number_of_tiles_per_core = get_number_of_tiles_per_core(
        total_number_of_cores_virtual,
        Wt,
        tensor_args.input_tensor.dtype(),
        output_tensors.at(1).dtype(),
        CrossCoreDataExchangeSortSlicingStrategy::USE_AS_MANY_CORES);
    number_of_tiles_per_core = std::min(number_of_tiles_per_core, Wt);

    // Calculate the number of cores utilized based on the input tensor shape
    const uint32_t all_core_utilization_count = (Wt + number_of_tiles_per_core - 1) / number_of_tiles_per_core;

    TT_FATAL(
        all_core_utilization_count <= total_number_of_cores_virtual,
        "All core utilization count exceeds total number of cores. Utilized cores: {}, Total cores: {}",
        all_core_utilization_count,
        total_number_of_cores_virtual);

    // uint32 index tensor support
    const bool is_32_bit_data = index_tensor_cb_data_format == tt::DataFormat::UInt32;

    /**
     * Calculates the core range based on the number of work units (all_core_utilization_count) and the total number of
     * available cores in the device's compute grid. The core range determines which cores will be utilized for
     * computation.
     *
     * The calculation works as follows:
     * 1. If all available cores are needed (all_core_utilization_count == total_number_of_cores), the core range covers
     * the entire grid.
     * 2. Otherwise, the number of rows (core_grid_calculated_rows_number) and columns
     * (core_grid_calculated_columns_number) required to cover all_core_utilization_count are calculated based on the
     * grid dimensions.
     *    - If both rows and columns are zero, only a single core is used.
     *    - If only rows are zero, the core range is set to cover the required number of columns in the first row.
     *    - Otherwise, the core range is set to cover the required rows, and if there are remaining columns,
     *      an additional range is added to cover those columns in the next row.
     *
     * The resulting core range is represented as a `CoreRangeSet`, which may consist of one or more `CoreRange`
     * objects depending on the configuration.
     */
    CoreRangeSet core_range;
    if (all_core_utilization_count == total_number_of_cores_physical) {
        // All cores used
        core_range = CoreRangeSet(
            CoreRange({0, 0}, {compute_with_storage_grid_size.x - 1, compute_with_storage_grid_size.y - 1}));
    } else if (all_core_utilization_count == total_number_of_cores_virtual) {
        const uint32_t core_grid_calculated_rows_number =
            (all_core_utilization_count / compute_with_storage_grid_size.x) - 1;
        const uint32_t core_grid_calculated_columns_number =
            all_core_utilization_count % compute_with_storage_grid_size.x;
        // All virtual cores used
        core_range =
            CoreRangeSet(CoreRange({0, 0}, {compute_with_storage_grid_size.x - 1, core_grid_calculated_rows_number}));
        if (core_grid_calculated_columns_number != 0) {
            const CoreRange additional_range(
                {0, core_grid_calculated_rows_number + 1},
                {core_grid_calculated_columns_number - 1, core_grid_calculated_rows_number + 1});
            core_range = core_range.merge(CoreRangeSet(additional_range));
        }
    } else {
        const uint32_t core_grid_calculated_rows_number = all_core_utilization_count / compute_with_storage_grid_size.x;
        const uint32_t core_grid_calculated_columns_number =
            all_core_utilization_count % compute_with_storage_grid_size.x;

        if (core_grid_calculated_rows_number == 0 && core_grid_calculated_columns_number == 0) {
            // Only one core used
            core_range = CoreRangeSet(CoreCoord({0, 0}));
        } else if (core_grid_calculated_rows_number == 0) {
            // Only cores from first row used
            core_range = CoreRangeSet(CoreRange({0, 0}, {core_grid_calculated_columns_number - 1, 0}));
        } else {
            // Rows and columns used
            core_range = CoreRangeSet(
                CoreRange({0, 0}, {compute_with_storage_grid_size.x - 1, core_grid_calculated_rows_number - 1}));
            if (core_grid_calculated_columns_number != 0) {
                const CoreRange additional_range(
                    {0, core_grid_calculated_rows_number},
                    {core_grid_calculated_columns_number - 1, core_grid_calculated_rows_number});
                core_range = core_range.merge(CoreRangeSet(additional_range));
            }
        }
    }

    // Lookup tensor data with physical core coordinates
    std::vector<uint32_t> physical_core_lookup_table_data;
    for (const auto& core_range : core_range.ranges()) {
        for (const auto& core_coord : core_range) {
            const auto physical_core = device->worker_core_from_logical_core(core_coord);
            physical_core_lookup_table_data.emplace_back(physical_core.x);
            physical_core_lookup_table_data.emplace_back(physical_core.y);
        }
    }
    const TensorSpec physical_core_lookup_table_spec(
        ttnn::Shape{1, physical_core_lookup_table_data.size()},
        TensorLayout{DataType::UINT32, PageConfig{Layout::ROW_MAJOR}, MemoryConfig()});
    Tensor physical_core_lookup_table_tensor =
        Tensor::from_vector(std::move(physical_core_lookup_table_data), physical_core_lookup_table_spec);
    physical_core_lookup_table_tensor = physical_core_lookup_table_tensor.to_device(device);
    auto* const physical_core_lookup_table_tensor_buffer = physical_core_lookup_table_tensor.buffer();
    const tt::DataFormat physical_core_lookup_table_cb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(physical_core_lookup_table_tensor.dtype());
    const uint32_t physical_core_lookup_table_tile_size = tile_size(physical_core_lookup_table_cb_data_format);

    // Circular buffers
    constexpr uint32_t cb_scale_factor = 2;

    constexpr uint32_t input_tensor_cb_index = tt::CBIndex::c_0;
    const tt::tt_metal::CircularBufferConfig input_tensor_cb_config =
        tt::tt_metal::CircularBufferConfig(
            cb_scale_factor * input_tensor_tile_size, {{input_tensor_cb_index, input_tensor_cb_data_format}})
            .set_page_size(input_tensor_cb_index, input_tensor_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, core_range, input_tensor_cb_config);

    constexpr uint32_t index_tensor_cb_index = tt::CBIndex::c_1;
    const tt::tt_metal::CircularBufferConfig index_tensor_cb_config =
        tt::tt_metal::CircularBufferConfig(
            cb_scale_factor * index_tensor_tile_size, {{index_tensor_cb_index, index_tensor_cb_data_format}})
            .set_page_size(index_tensor_cb_index, index_tensor_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, core_range, index_tensor_cb_config);

    constexpr uint32_t input_tensor_transposed_cb_index = tt::CBIndex::c_2;
    const tt::tt_metal::CircularBufferConfig input_tensor_transposed_cb_config =
        tt::tt_metal::CircularBufferConfig(
            number_of_tiles_per_core * input_tensor_tile_size,
            {{input_tensor_transposed_cb_index, input_tensor_cb_data_format}})
            .set_page_size(input_tensor_transposed_cb_index, input_tensor_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, core_range, input_tensor_transposed_cb_config);

    constexpr uint32_t index_tensor_transposed_cb_index = tt::CBIndex::c_3;
    const tt::tt_metal::CircularBufferConfig index_tensor_transposed_cb_config =
        tt::tt_metal::CircularBufferConfig(
            number_of_tiles_per_core * index_tensor_tile_size,
            {{index_tensor_transposed_cb_index, index_tensor_cb_data_format}})
            .set_page_size(index_tensor_transposed_cb_index, index_tensor_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, core_range, index_tensor_transposed_cb_config);

    constexpr uint32_t value_tensor_cb_index = tt::CBIndex::c_4;
    const tt::tt_metal::CircularBufferConfig value_tensor_cb_config =
        tt::tt_metal::CircularBufferConfig(
            cb_scale_factor * value_tensor_tile_size, {{value_tensor_cb_index, value_tensor_cb_data_format}})
            .set_page_size(value_tensor_cb_index, index_tensor_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, core_range, value_tensor_cb_config);

    constexpr uint32_t index_tensor_output_cb_index = tt::CBIndex::c_5;
    const tt::tt_metal::CircularBufferConfig index_tensor_output_cb_config =
        tt::tt_metal::CircularBufferConfig(
            cb_scale_factor * index_tensor_tile_size, {{index_tensor_output_cb_index, index_tensor_cb_data_format}})
            .set_page_size(index_tensor_output_cb_index, index_tensor_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, core_range, index_tensor_output_cb_config);

    constexpr uint32_t value_tensor_intermediate_cb_index = tt::CBIndex::c_6;
    const tt::tt_metal::CircularBufferConfig value_tensor_intermediate_cb_config =
        tt::tt_metal::CircularBufferConfig(
            cb_scale_factor * value_tensor_tile_size,
            {{value_tensor_intermediate_cb_index, value_tensor_cb_data_format}})
            .set_page_size(value_tensor_intermediate_cb_index, index_tensor_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, core_range, value_tensor_intermediate_cb_config);

    constexpr uint32_t index_tensor_intermediate_cb_index = tt::CBIndex::c_7;
    const tt::tt_metal::CircularBufferConfig index_tensor_intermediate_cb_config =
        tt::tt_metal::CircularBufferConfig(
            cb_scale_factor * index_tensor_tile_size,
            {{index_tensor_intermediate_cb_index, index_tensor_cb_data_format}})
            .set_page_size(index_tensor_intermediate_cb_index, index_tensor_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, core_range, index_tensor_intermediate_cb_config);

    constexpr uint32_t value_tensor_peer_cb_index = tt::CBIndex::c_8;
    const tt::tt_metal::CircularBufferConfig value_tensor_peer_cb_config =
        tt::tt_metal::CircularBufferConfig(
            cb_scale_factor * value_tensor_tile_size, {{value_tensor_peer_cb_index, value_tensor_cb_data_format}})
            .set_page_size(value_tensor_peer_cb_index, index_tensor_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, core_range, value_tensor_peer_cb_config);

    constexpr uint32_t index_tensor_peer_cb_index = tt::CBIndex::c_9;
    const tt::tt_metal::CircularBufferConfig index_tensor_peer_cb_config =
        tt::tt_metal::CircularBufferConfig(
            cb_scale_factor * index_tensor_tile_size, {{index_tensor_peer_cb_index, index_tensor_cb_data_format}})
            .set_page_size(index_tensor_peer_cb_index, index_tensor_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, core_range, index_tensor_peer_cb_config);

    constexpr uint32_t physical_core_lookup_table_cb_index = tt::CBIndex::c_10;
    const tt::tt_metal::CircularBufferConfig physical_core_lookup_table_cb_config =
        tt::tt_metal::CircularBufferConfig(
            physical_core_lookup_table_tile_size,
            {{physical_core_lookup_table_cb_index, physical_core_lookup_table_cb_data_format}})
            .set_page_size(physical_core_lookup_table_cb_index, physical_core_lookup_table_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, core_range, physical_core_lookup_table_cb_config);

    constexpr uint32_t packer_unpacker_sync_cb_index = tt::CBIndex::c_11;
    const tt::tt_metal::CircularBufferConfig packer_unpacker_sync_cb_config =
        tt::tt_metal::CircularBufferConfig(
            packer_unpacker_sync_tile_size, {{packer_unpacker_sync_cb_index, packer_unpacker_sync_cb_data_format}})
            .set_page_size(packer_unpacker_sync_cb_index, packer_unpacker_sync_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, core_range, packer_unpacker_sync_cb_config);

    // Semaphores
    const uint32_t semaphore_exchange_readers = CreateSemaphore(program, core_range, 0);
    CreateSemaphore(program, core_range, 0);
    const uint32_t semaphore_barrier = CreateSemaphore(program, core_range, 0);

    // Kernels
    std::vector<uint32_t> reader_compile_time_args = {
        compute_with_storage_grid_size.x,
        compute_with_storage_grid_size.y,
        input_tensor_cb_index,
        index_tensor_output_cb_index,
        value_tensor_intermediate_cb_index,
        index_tensor_intermediate_cb_index,
        value_tensor_peer_cb_index,
        index_tensor_peer_cb_index,
        physical_core_lookup_table_cb_index,
        Ht,
        Wt,
        number_of_tiles_per_core,
        all_core_utilization_count,
        !attributes.descending,
        semaphore_exchange_readers,
        semaphore_barrier,
    };
    TensorAccessorArgs(*input_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*index_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*physical_core_lookup_table_tensor_buffer).append_to(reader_compile_time_args);
    const std::string reader_kernel_path =
        "ttnn/cpp/ttnn/operations/data_movement/sort/device/kernels/dataflow/reader_cross_core_data_exchange.cpp";
    tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program, reader_kernel_path, core_range, tt::tt_metal::ReaderDataMovementConfig{reader_compile_time_args});
    SetRuntimeArgs(
        program,
        reader_kernel_id,
        core_range,
        {input_buffer->address(), index_buffer->address(), physical_core_lookup_table_tensor_buffer->address()});

    std::vector<uint32_t> writer_compile_time_args = {
        compute_with_storage_grid_size.x,
        compute_with_storage_grid_size.y,
        index_tensor_cb_index,
        value_tensor_cb_index,
        value_tensor_peer_cb_index,
        physical_core_lookup_table_cb_index,
        Wt,
        Ht,
        number_of_tiles_per_core,
        total_number_of_cores_virtual,
        semaphore_exchange_readers,
        static_cast<uint32_t>(is_32_bit_data)};
    TensorAccessorArgs(*value_buffer).append_to(writer_compile_time_args);
    const std::string writer_kernel_path =
        "ttnn/cpp/ttnn/operations/data_movement/sort/device/kernels/dataflow/writer_cross_core_data_exchange.cpp";
    tt::tt_metal::KernelHandle writer_kernel_id = tt::tt_metal::CreateKernel(
        program, writer_kernel_path, core_range, tt::tt_metal::WriterDataMovementConfig{writer_compile_time_args});
    SetRuntimeArgs(program, writer_kernel_id, core_range, {value_buffer->address()});

    const std::vector<uint32_t> compute_compile_time_args = {
        compute_with_storage_grid_size.x,
        compute_with_storage_grid_size.y,
        Ht,
        Wt,
        number_of_tiles_per_core,
        all_core_utilization_count,
        !attributes.descending,
        input_tensor_cb_index,
        index_tensor_cb_index,
        input_tensor_transposed_cb_index,
        index_tensor_transposed_cb_index,
        value_tensor_cb_index,
        index_tensor_output_cb_index,
        value_tensor_intermediate_cb_index,
        index_tensor_intermediate_cb_index,
        value_tensor_peer_cb_index,
        index_tensor_peer_cb_index,
        packer_unpacker_sync_cb_index,
    };
    const std::string compute_kernel_path =
        "ttnn/cpp/ttnn/operations/data_movement/sort/device/kernels/compute/sort_cross_core_data_exchange.cpp";
    tt::tt_metal::KernelHandle compute_kernel_id = tt::tt_metal::CreateKernel(
        program,
        compute_kernel_path,
        core_range,
        tt::tt_metal::ComputeConfig{.fp32_dest_acc_en = is_32_bit_data, .compile_args = compute_compile_time_args});

    return {std::move(program), {reader_kernel_id, compute_kernel_id, writer_kernel_id, core_range}};
}

void SortProgramFactoryCrossCoreDataExchange::override_runtime_arguments(
    cached_program_t& cached_program,
    const SortParams& /*attributes*/,
    const SortInputs& tensor_args,
    std::vector<Tensor>& output_tensors) {
    auto* const input_tensor_buffer = tensor_args.input_tensor.buffer();
    auto* const value_tensor_buffer = output_tensors.at(0).buffer();
    auto* const index_tensor_buffer = output_tensors.at(1).buffer();
    auto* const device = tensor_args.input_tensor.device();

    // Lookup tensor data with physical core coordinates
    std::vector<uint32_t> physical_core_lookup_table_data;
    for (const auto& core_range : cached_program.shared_variables.core_range_set.ranges()) {
        for (const auto& core_coord : core_range) {
            const auto physical_core = device->worker_core_from_logical_core(core_coord);
            physical_core_lookup_table_data.emplace_back(physical_core.x);
            physical_core_lookup_table_data.emplace_back(physical_core.y);
        }
    }
    const TensorSpec physical_core_lookup_table_spec(
        ttnn::Shape{1, physical_core_lookup_table_data.size()},
        TensorLayout{DataType::UINT32, PageConfig{Layout::ROW_MAJOR}, MemoryConfig()});
    Tensor physical_core_lookup_table_tensor =
        Tensor::from_vector(std::move(physical_core_lookup_table_data), physical_core_lookup_table_spec);
    physical_core_lookup_table_tensor = physical_core_lookup_table_tensor.to_device(device);

    // Update runtime args
    for (const auto& core_range : cached_program.shared_variables.core_range_set.ranges()) {
        for (const auto& core_coord : core_range) {
            auto& reader_runtime_args =
                GetRuntimeArgs(cached_program.program, cached_program.shared_variables.reader_kernel_id, core_coord);
            reader_runtime_args[0] = input_tensor_buffer->address();
            reader_runtime_args[1] = index_tensor_buffer->address();
            reader_runtime_args[2] = physical_core_lookup_table_tensor.buffer()->address();

            auto& writer_runtime_args =
                GetRuntimeArgs(cached_program.program, cached_program.shared_variables.writer_kernel_id, core_coord);
            writer_runtime_args[0] = value_tensor_buffer->address();
        }  // core_coord loop
    }  // core_range loop
}

uint32_t SortProgramFactoryCrossCoreDataExchange::get_number_of_tiles_per_core(
    uint32_t total_number_of_cores,
    uint32_t Wt,
    const DataType& input_dtype,
    const DataType& index_dtype,
    CrossCoreDataExchangeSortSlicingStrategy slicing_strategy) {
    switch (slicing_strategy) {
        case CrossCoreDataExchangeSortSlicingStrategy::USE_AS_MANY_CORES: {
            // Minimum of 2 tiles per core is required because the LLK (Low-Level Kernel) needs at least two tiles per
            // core to perform sorting operations. Maximum is capped at 128 tiles (power of 2) based on hardware memory
            // constraints, ensuring that tiles can fit into a single core's available memory.
            constexpr uint32_t MIN_TILES_PER_CORE = 2;
            constexpr uint32_t MAX_TILES_PER_CORE = 128;
            const auto max_val = std::max(Wt / total_number_of_cores, MIN_TILES_PER_CORE);
            return std::min(MAX_TILES_PER_CORE, max_val);
        }
        case CrossCoreDataExchangeSortSlicingStrategy::FILL_CORES_FIRST:
        default: {
            if (input_dtype == DataType::FLOAT32 || input_dtype == DataType::UINT32 || input_dtype == DataType::INT32 ||
                index_dtype == DataType::INT32 || index_dtype == DataType::UINT32) {
                return 64;
            }
            break;
        }
    }

    return 128;
}

uint32_t SortProgramFactoryCrossCoreDataExchange::rounddown_pow2(uint32_t n) {
    if (n == 0) {
        return 0;
    }
    return 1 << (31 - std::countl_zero(n));
}

// Single row - multi core
SortProgramFactorySingleRowMultiCore::cached_program_t SortProgramFactorySingleRowMultiCore::create(
    const SortParams& attributes, const SortInputs& tensor_args, std::vector<Tensor>& output_tensors) {
    // Program config
    tt::tt_metal::Program program{};

    const tt::DataFormat input_tensor_cb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(tensor_args.input_tensor.dtype());
    const tt::DataFormat value_tensor_cb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(output_tensors.at(0).dtype());
    const tt::DataFormat index_tensor_cb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(output_tensors.at(1).dtype());

    const uint32_t input_tensor_tile_size = tile_size(input_tensor_cb_data_format);
    const uint32_t value_tensor_tile_size = tile_size(value_tensor_cb_data_format);
    const uint32_t index_tensor_tile_size = tile_size(index_tensor_cb_data_format);

    auto* const input_buffer = tensor_args.input_tensor.buffer();
    auto* const value_buffer = output_tensors.at(0).buffer();
    auto* const index_buffer = output_tensors.at(1).buffer();

    const auto tile_width = tensor_args.input_tensor.tensor_spec().tile().get_width();
    const auto tile_height = tensor_args.input_tensor.tensor_spec().tile().get_height();

    const auto input_shape = tensor_args.input_tensor.padded_shape();
    const uint32_t Ht = (input_shape[0] * input_shape[1] * input_shape[2]) / tile_height;
    const uint32_t Wt = input_shape[3] / tile_width;

    // Calculate the number of cores available for computation
    auto* device = tensor_args.input_tensor.device();
    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t total_number_of_cores = compute_with_storage_grid_size.y * compute_with_storage_grid_size.x;

    // Calculate the number of cores utilized based on the input tensor shape
    // We process pairs of tiles - need Wt / 2 work units
    const uint32_t total_work_units = Wt / 2;
    const uint32_t number_of_available_cores = total_number_of_cores - 1;  // One core for coordinator

    const uint32_t all_core_utilization_loop_count = total_work_units / number_of_available_cores;

    // uint32 index tensor support
    const bool is_32_bit_data = index_tensor_cb_data_format == tt::DataFormat::UInt32;

    // Log 2 of Wt for compute kernel
    const uint32_t log2Wt = std::log2(Wt);

    /**
     * Calculates the core range based on the input tensor shape (Wt) and the total number of cores available
     * in the device's compute grid (minus one reserved for coordinator). The core range determines which
     * cores will be utilized for computation.
     *
     * The calculation works as follows:
     * 1. The coordinator core is set to the last core in the compute grid.
     *
     * 2. If the width (Wt) of the input tensor is greater than or equal to the total number of available cores,
     *    all cores in the compute grid are utilized. The core range is set to cover the entire grid.
     *
     * 3. If Wt is smaller than the total number of cores:
     *    - The number of rows (`core_grid_calculated_rows_number`) and columns
     * (`core_grid_calculated_columns_number`) required to cover Wt are calculated based on the grid dimensions.
     *    - If both rows and columns are zero, only a single core is used.
     *    - If only rows are zero, the core range is set to cover the required number of columns in the first
     * row.
     *    - Otherwise, the core range is set to cover the required rows, and if there are remaining columns,
     *      an additional range is added to cover those columns in the next row.
     *
     * The resulting core range is represented as a `CoreRangeSet`, which may consist of one or more `CoreRange`
     * objects depending on the configuration.
     */
    CoreCoord coordinator_core = {compute_with_storage_grid_size.x - 1, compute_with_storage_grid_size.y - 1};
    CoreRangeSet core_range;
    if (all_core_utilization_loop_count > 0) {
        core_range = CoreRangeSet(
            CoreRange({0, 0}, {compute_with_storage_grid_size.x - 1, compute_with_storage_grid_size.y - 2}));
        core_range = core_range.merge<CoreRangeSet>(CoreRangeSet(CoreRange(
            {0, compute_with_storage_grid_size.y - 1},
            {compute_with_storage_grid_size.x - 2, compute_with_storage_grid_size.y - 1})));
    } else {
        const uint32_t core_grid_calculated_rows_number = total_work_units / compute_with_storage_grid_size.x;
        const uint32_t core_grid_calculated_columns_number = total_work_units % compute_with_storage_grid_size.x;

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
    CoreRangeSet all_core_set({CoreRange(coordinator_core)});
    all_core_set = all_core_set.merge<CoreRangeSet>(core_range);

    // Circular buffers
    constexpr uint32_t buffer_scale_factor = 2;

    constexpr uint32_t input_tensor_cb_index = tt::CBIndex::c_0;
    const tt::tt_metal::CircularBufferConfig input_tensor_cb_config =
        tt::tt_metal::CircularBufferConfig(
            buffer_scale_factor * input_tensor_tile_size, {{input_tensor_cb_index, input_tensor_cb_data_format}})
            .set_page_size(input_tensor_cb_index, input_tensor_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_core_set, input_tensor_cb_config);

    constexpr uint32_t index_tensor_cb_index = tt::CBIndex::c_1;
    const tt::tt_metal::CircularBufferConfig index_tensor_cb_config =
        tt::tt_metal::CircularBufferConfig(
            buffer_scale_factor * index_tensor_tile_size, {{index_tensor_cb_index, index_tensor_cb_data_format}})
            .set_page_size(index_tensor_cb_index, index_tensor_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_core_set, index_tensor_cb_config);

    constexpr uint32_t input_tensor_transposed_cb_index = tt::CBIndex::c_2;
    const tt::tt_metal::CircularBufferConfig input_tensor_transposed_cb_config =
        tt::tt_metal::CircularBufferConfig(
            buffer_scale_factor * input_tensor_tile_size,
            {{input_tensor_transposed_cb_index, input_tensor_cb_data_format}})
            .set_page_size(input_tensor_transposed_cb_index, input_tensor_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_core_set, input_tensor_transposed_cb_config);

    constexpr uint32_t index_tensor_transposed_cb_index = tt::CBIndex::c_3;
    const tt::tt_metal::CircularBufferConfig index_tensor_transposed_cb_config =
        tt::tt_metal::CircularBufferConfig(
            buffer_scale_factor * index_tensor_tile_size,
            {{index_tensor_transposed_cb_index, index_tensor_cb_data_format}})
            .set_page_size(index_tensor_transposed_cb_index, index_tensor_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_core_set, index_tensor_transposed_cb_config);

    constexpr uint32_t input_tensor_output_cb_index = tt::CBIndex::c_4;
    const tt::tt_metal::CircularBufferConfig input_tensor_output_cb_config =
        tt::tt_metal::CircularBufferConfig(
            buffer_scale_factor * value_tensor_tile_size, {{input_tensor_output_cb_index, value_tensor_cb_data_format}})
            .set_page_size(input_tensor_output_cb_index, value_tensor_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_core_set, input_tensor_output_cb_config);

    constexpr uint32_t index_tensor_output_cb_index = tt::CBIndex::c_5;
    const tt::tt_metal::CircularBufferConfig index_tensor_output_cb_config =
        tt::tt_metal::CircularBufferConfig(
            buffer_scale_factor * index_tensor_tile_size, {{index_tensor_output_cb_index, index_tensor_cb_data_format}})
            .set_page_size(index_tensor_output_cb_index, index_tensor_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_core_set, index_tensor_output_cb_config);

    // Semaphores
    const uint32_t coordinator_to_cores_semaphore_id = CreateSemaphore(program, all_core_set, 0);
    const uint32_t cores_to_coordinator_semaphore_id = CreateSemaphore(program, all_core_set, 0);
    const auto coordinator_core_physical_coord = device->worker_core_from_logical_core(coordinator_core);

    const auto start_core_logical = core_range.ranges()[0].start_coord;
    const auto start_core_physical_coord = device->worker_core_from_logical_core(start_core_logical);
    const auto end_core_physical_coord = device->worker_core_from_logical_core(coordinator_core);

    // Kernels
    std::vector<uint32_t> coordinator_compile_time_args = {
        total_work_units,
        Wt,
        Ht,
        total_number_of_cores,
        number_of_available_cores,
        input_tensor_cb_index,
        index_tensor_cb_index,
        static_cast<uint32_t>(is_32_bit_data)};
    TensorAccessorArgs(*input_buffer).append_to(coordinator_compile_time_args);
    TensorAccessorArgs(*value_buffer).append_to(coordinator_compile_time_args);
    TensorAccessorArgs(*index_buffer).append_to(coordinator_compile_time_args);
    const std::string coordinator_kernel_path =
        "ttnn/cpp/ttnn/operations/data_movement/sort/device/kernels/dataflow/coordinator_single_row_multi_core.cpp";
    tt::tt_metal::KernelHandle coordinator_kernel_id = tt::tt_metal::CreateKernel(
        program,
        coordinator_kernel_path,
        coordinator_core,
        tt::tt_metal::ReaderDataMovementConfig{coordinator_compile_time_args});
    SetRuntimeArgs(
        program,
        coordinator_kernel_id,
        coordinator_core,
        {start_core_physical_coord.x,
         start_core_physical_coord.y,
         end_core_physical_coord.x,
         end_core_physical_coord.y,
         coordinator_to_cores_semaphore_id,
         cores_to_coordinator_semaphore_id,
         core_range.num_cores(),
         input_buffer->address(),
         value_buffer->address(),
         index_buffer->address()});

    std::vector<uint32_t> reader_compile_time_args = {
        input_tensor_cb_index,
        index_tensor_cb_index,
        Wt,
        Ht,
        total_number_of_cores,
        compute_with_storage_grid_size.x,
        compute_with_storage_grid_size.y,
        number_of_available_cores};
    TensorAccessorArgs(*value_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*index_buffer).append_to(reader_compile_time_args);
    const std::string reader_kernel_path =
        "ttnn/cpp/ttnn/operations/data_movement/sort/device/kernels/dataflow/reader_single_row_multi_core.cpp";
    tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program, reader_kernel_path, core_range, tt::tt_metal::ReaderDataMovementConfig{reader_compile_time_args});
    SetRuntimeArgs(
        program,
        reader_kernel_id,
        core_range,
        {value_buffer->address(),
         index_buffer->address(),
         coordinator_core_physical_coord.x,
         coordinator_core_physical_coord.y,
         coordinator_to_cores_semaphore_id,
         cores_to_coordinator_semaphore_id});

    std::vector<uint32_t> writer_compile_time_args = {
        input_tensor_output_cb_index,
        index_tensor_output_cb_index,
        Wt,
        Ht,
        total_number_of_cores,
        compute_with_storage_grid_size.x,
        compute_with_storage_grid_size.y,
        number_of_available_cores};
    TensorAccessorArgs(*value_buffer).append_to(writer_compile_time_args);
    TensorAccessorArgs(*index_buffer).append_to(writer_compile_time_args);
    const std::string writer_kernel_path =
        "ttnn/cpp/ttnn/operations/data_movement/sort/device/kernels/dataflow/writer_single_row_multi_core.cpp";
    tt::tt_metal::KernelHandle writer_kernel_id = tt::tt_metal::CreateKernel(
        program, writer_kernel_path, core_range, tt::tt_metal::WriterDataMovementConfig{writer_compile_time_args});
    SetRuntimeArgs(
        program,
        writer_kernel_id,
        core_range,
        {value_buffer->address(),
         index_buffer->address(),
         coordinator_core_physical_coord.x,
         coordinator_core_physical_coord.y,
         coordinator_to_cores_semaphore_id,
         cores_to_coordinator_semaphore_id});

    const std::vector<uint32_t> compute_compile_time_args = {
        input_tensor_cb_index,
        index_tensor_cb_index,
        input_tensor_transposed_cb_index,
        index_tensor_transposed_cb_index,
        input_tensor_output_cb_index,
        index_tensor_output_cb_index,
        Wt,
        Ht,
        number_of_available_cores,
        compute_with_storage_grid_size.x,
        compute_with_storage_grid_size.y,
        static_cast<uint32_t>(attributes.descending),
        static_cast<uint32_t>(attributes.stable),
        log2Wt};
    const std::string compute_kernel_path =
        "ttnn/cpp/ttnn/operations/data_movement/sort/device/kernels/compute/sort_single_row_multi_core.cpp";
    tt::tt_metal::KernelHandle compute_kernel_id = tt::tt_metal::CreateKernel(
        program,
        compute_kernel_path,
        core_range,
        tt::tt_metal::ComputeConfig{.fp32_dest_acc_en = is_32_bit_data, .compile_args = compute_compile_time_args});

    return {
        std::move(program),
        {coordinator_kernel_id, reader_kernel_id, compute_kernel_id, writer_kernel_id, coordinator_core, core_range}};
}

void SortProgramFactorySingleRowMultiCore::override_runtime_arguments(
    cached_program_t& cached_program,
    const SortParams& /*attributes*/,
    const SortInputs& tensor_args,
    std::vector<Tensor>& output_tensors) {
    auto* const input_tensor_buffer = tensor_args.input_tensor.buffer();
    auto* const value_tensor_buffer = output_tensors.at(0).buffer();
    auto* const index_tensor_buffer = output_tensors.at(1).buffer();

    auto& coordinator_core_runtime_args = GetRuntimeArgs(
        cached_program.program,
        cached_program.shared_variables.coordinator_kernel_id,
        cached_program.shared_variables.coordinator_core);
    coordinator_core_runtime_args[7] = input_tensor_buffer->address();
    coordinator_core_runtime_args[8] = value_tensor_buffer->address();
    coordinator_core_runtime_args[9] = index_tensor_buffer->address();

    for (const auto& core_range : cached_program.shared_variables.worker_core_range.ranges()) {
        for (const auto& core : core_range) {
            auto& reader_runtime_args =
                GetRuntimeArgs(cached_program.program, cached_program.shared_variables.reader_kernel_id, core);
            reader_runtime_args[0] = value_tensor_buffer->address();
            reader_runtime_args[1] = index_tensor_buffer->address();

            auto& writer_runtime_args =
                GetRuntimeArgs(cached_program.program, cached_program.shared_variables.writer_kernel_id, core);
            writer_runtime_args[0] = value_tensor_buffer->address();
            writer_runtime_args[1] = index_tensor_buffer->address();
        }  // core loop
    }  // core_range loop
}
}  // namespace ttnn::prim
