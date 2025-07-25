// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include "ttnn/operation.hpp"

#include <iostream>
#include <cmath>

using namespace tt::tt_metal;
using namespace std;
namespace ttnn::operations::reduction::detail {

operation::ProgramWithCallbacks topk_single_core_interleaved(
    const Tensor& input_tensor,
    const uint32_t k,
    const int8_t dim,
    const bool largest,
    const bool sorted,
    const bool uint16_output,
    const CoreRangeSet& sub_core_grids,
    Tensor& value_tensor,
    Tensor& index_tensor) {
    using namespace tt::constants;
    tt::tt_metal::Program program{};
    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    tt::DataFormat output_val_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(value_tensor.dtype());
    tt::DataFormat output_ind_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(index_tensor.dtype());

    auto core = corerange_to_cores(sub_core_grids, 1, true).at(0);

    uint32_t input_tile_size = tile_size(input_cb_data_format);
    uint32_t value_tile_size = tile_size(output_val_cb_data_format);
    uint32_t index_tile_size = tile_size(output_ind_cb_data_format);

    auto input_buffer = input_tensor.buffer();
    auto values_buffer = value_tensor.buffer();
    auto index_buffer = index_tensor.buffer();

    bool input_is_dram = input_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    bool values_is_dram = values_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    bool index_is_dram = index_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;

    uint32_t num_input_tiles = input_tensor.physical_volume() / TILE_HW;
    uint32_t num_value_tiles = value_tensor.physical_volume() / TILE_HW;

    auto input_shape = input_tensor.padded_shape();
    uint32_t Ht = (input_shape[0] * input_shape[1] * input_shape[2]) / TILE_HEIGHT;
    uint32_t Wt = input_shape[3] / TILE_WIDTH;

    uint32_t Ktiles = tt::div_up(k, tt::constants::TILE_WIDTH);

    // for streaming in input
    uint32_t num_cb_unit = 2;
    uint32_t cb_in_units = 2 * num_cb_unit;

    uint32_t input_cb_tile_count = cb_in_units;
    uint32_t transposed_cb_tile_count = 4;
    uint32_t result_prep_cb_tile_count = 2 * Ktiles;  // intermediate output
    uint32_t output_cb_tile_count = Ktiles;           // final output

    // Two tiles are loaded in for topk_local_sort at a time, and we double buffer to avoid stalls, so allocate four
    // tiles of space
    // TODO: In theory if we have enough memory we could allocate 2*Wt tiles to reduce stalls
    uint32_t input_cb_index = tt::CBIndex::c_0;
    tt::tt_metal::CircularBufferConfig input_cb_config =
        tt::tt_metal::CircularBufferConfig(
            input_cb_tile_count * value_tile_size, {{input_cb_index, input_cb_data_format}})
            .set_page_size(input_cb_index, input_tile_size);
    auto cb_input_tensor = tt::tt_metal::CreateCircularBuffer(program, core, input_cb_config);

    // Two tiles are loaded in for topk_local_sort at a time, and we double buffer to avoid stalls, so allocate four
    // tiles of space. This CB carries the indices that are created in the reader kernel
    uint32_t index_cb_index = tt::CBIndex::c_1;
    tt::tt_metal::CircularBufferConfig index_input_intermed0_config =
        tt::tt_metal::CircularBufferConfig(
            input_cb_tile_count * index_tile_size, {{index_cb_index, output_ind_cb_data_format}})
            .set_page_size(index_cb_index, index_tile_size);
    auto cb_index_tensor = tt::tt_metal::CreateCircularBuffer(program, core, index_input_intermed0_config);

    // Single buffered circular buffer that holds the transposed input tiles
    uint32_t transposed_val_cb_index = tt::CBIndex::c_2;
    tt::tt_metal::CircularBufferConfig transposed_val_cb_config =
        tt::tt_metal::CircularBufferConfig(
            transposed_cb_tile_count * value_tile_size, {{transposed_val_cb_index, input_cb_data_format}})
            .set_page_size(transposed_val_cb_index, input_tile_size);
    auto cb_input_transposed_tiles = tt::tt_metal::CreateCircularBuffer(program, core, transposed_val_cb_config);

    // Single buffered circular buffer that holds the transposed index tiles
    uint32_t transposed_ind_cb_index = tt::CBIndex::c_3;
    tt::tt_metal::CircularBufferConfig transposed_ind_cb_config =
        tt::tt_metal::CircularBufferConfig(
            transposed_cb_tile_count * index_tile_size, {{transposed_ind_cb_index, output_ind_cb_data_format}})
            .set_page_size(transposed_ind_cb_index, index_tile_size);
    auto cb_index_transposed_tiles = tt::tt_metal::CreateCircularBuffer(program, core, transposed_ind_cb_config);

    // Single buffered circular buffer that holds the result_prep input tiles
    uint32_t result_prep_val_cb_index = tt::CBIndex::c_4;
    tt::tt_metal::CircularBufferConfig result_prep_val_cb_config =
        tt::tt_metal::CircularBufferConfig(
            result_prep_cb_tile_count * value_tile_size, {{result_prep_val_cb_index, input_cb_data_format}})
            .set_page_size(result_prep_val_cb_index, input_tile_size);
    auto cb_input_result_prep_tiles = tt::tt_metal::CreateCircularBuffer(program, core, result_prep_val_cb_config);

    // Single buffered circular buffer that holds the result_prep index tiles
    uint32_t result_prep_ind_cb_index = tt::CBIndex::c_5;
    tt::tt_metal::CircularBufferConfig result_prep_ind_cb_config =
        tt::tt_metal::CircularBufferConfig(
            result_prep_cb_tile_count * index_tile_size, {{result_prep_ind_cb_index, output_ind_cb_data_format}})
            .set_page_size(result_prep_ind_cb_index, index_tile_size);
    auto cb_index_result_prep_tiles = tt::tt_metal::CreateCircularBuffer(program, core, result_prep_ind_cb_config);

    // Output topk values
    uint32_t output_val_cb_index = tt::CBIndex::c_6;
    tt::tt_metal::CircularBufferConfig output_val_cb_config =
        tt::tt_metal::CircularBufferConfig(
            output_cb_tile_count * value_tile_size, {{output_val_cb_index, output_val_cb_data_format}})
            .set_page_size(output_val_cb_index, value_tile_size);
    auto cb_values_tensor = tt::tt_metal::CreateCircularBuffer(program, core, output_val_cb_config);

    // Output topk indices
    uint32_t output_ind_cb_index = tt::CBIndex::c_7;
    tt::tt_metal::CircularBufferConfig output_ind_cb_config =
        tt::tt_metal::CircularBufferConfig(
            output_cb_tile_count * index_tile_size, {{output_ind_cb_index, output_ind_cb_data_format}})
            .set_page_size(output_ind_cb_index, index_tile_size);
    auto cb_output_ind_tensor = tt::tt_metal::CreateCircularBuffer(program, core, output_ind_cb_config);

    std::vector<uint32_t> reader_compile_time_args = {
        input_cb_index, index_cb_index, (uint32_t)input_is_dram, Ht, Wt, (uint32_t)uint16_output};
    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/dataflow/reader_create_index_tensor.cpp",
        core,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    SetRuntimeArgs(
        program,
        unary_reader_kernel_id,
        core,
        {
            input_buffer->address(),
        });

    std::vector<uint32_t> writer_compile_time_args = {
        output_val_cb_index,
        output_ind_cb_index,
        (std::uint32_t)values_is_dram,
        (std::uint32_t)index_is_dram,
        Ht,
        Ktiles};
    tt::tt_metal::KernelHandle binary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/dataflow/writer_binary_interleaved.cpp",
        core,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    SetRuntimeArgs(
        program,
        binary_writer_kernel_id,
        core,
        {
            values_buffer->address(),
            index_buffer->address(),

        });

    std::vector<uint32_t> compute_args = {
        input_cb_index,
        index_cb_index,
        transposed_val_cb_index,
        transposed_ind_cb_index,
        result_prep_val_cb_index,
        result_prep_ind_cb_index,
        output_val_cb_index,
        output_ind_cb_index,
        Ht,
        Wt,
        Ktiles,
        (std::uint32_t)largest,
    };
    tt::tt_metal::KernelHandle topk_compute_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/compute/topk.cpp",
        core,
        tt::tt_metal::ComputeConfig{.fp32_dest_acc_en = !uint16_output, .compile_args = compute_args});

    auto override_runtime_args_callback = [unary_reader_kernel_id, binary_writer_kernel_id, core](
                                              const void* operation,
                                              const Program& program,
                                              const std::vector<Tensor>& input_tensors,
                                              const std::vector<std::optional<const Tensor>>&,
                                              const std::vector<Tensor>& output_tensors) {
        auto input_buffer = input_tensors.at(0).buffer();

        auto values_buffer = output_tensors.at(0).buffer();
        auto index_buffer = output_tensors.at(1).buffer();

        {
            auto& reader_runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, core);
            reader_runtime_args[0] = input_buffer->address();

            auto& writer_runtime_args = GetRuntimeArgs(program, binary_writer_kernel_id, core);
            writer_runtime_args[0] = values_buffer->address();
            writer_runtime_args[1] = index_buffer->address();
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

static inline uint32_t largest_power_of_two(std::uint32_t x) { return x == 0 ? 0 : (1 << (31 - __builtin_clz(x))); }

/**
 * Split the work along the width such that the width is divisible by min_dim and the number of cores used is less than
 * or equal to max_cores. Each core must have a minimum of two tiles - min_dim = 64 as that's the minimum size for the
 * llk. Return the number of cores utilized for the split, the size of the split along the width, the width on the
 * remainder core if any, and the remaining elements that the gather core has to process If less than the max number of
 * cores are used, then we can try splitting on height as well. Eg) if only 2 cores are used for the split and then 1
 * for the gather, we only need 3 cores per row. Then take cores_per_row = 3 and try to split the height such that the
 * number of cores used is less than or equal to max_cores.
 */
static inline std::tuple<uint16_t, uint16_t, uint16_t, uint16_t> cores_utilized(
    uint16_t width,
    uint16_t min_dim,
    uint16_t max_dim,
    CoreRange core_range,
    uint32_t k,
    const uint32_t l1_size,
    const uint32_t value_tile_size,
    const uint32_t index_tile_size) {
    const auto max_cores = core_range.end_coord.y - core_range.start_coord.y - 1;
    uint16_t start_split_size = width / largest_power_of_two(max_cores);
    for (uint16_t split_size = start_split_size; split_size <= max_dim; split_size *= 2) {
        uint16_t rem = width % split_size;
        uint16_t num_cores = width / split_size + (rem > 0);
        uint32_t memory_cost_gather =
            2 * num_cores * (value_tile_size + index_tile_size);  // gathering one index and one value tile from each
                                                                  // local core, allocating two CBs for each
        uint32_t memory_cost_local =
            (split_size / tt::constants::TILE_WIDTH) *
            (value_tile_size + index_tile_size);  // we divide the width into split_size chunks and each chunk, as well
                                                  // as a matching set of indices, is processed by a core
        if (num_cores <= max_cores && (memory_cost_gather + memory_cost_local * num_cores) < (l1_size * num_cores) &&
            num_cores > 1 && split_size >= min_dim) {
            return {
                num_cores + 1,
                split_size,
                rem,
                num_cores * std::max(static_cast<uint32_t>(k), static_cast<uint32_t>(tt::constants::TILE_WIDTH))};
        }
    }
    return {max_cores + 1, width, 0, width * k};
}

/**
 * Split work along width dimension and compute topk values and indices for each split in parallel, on different cores.
 * Then gather the results of each split onto a single core, where the final topk values and indices are computed.
 *
 */
operation::ProgramWithCallbacks topk_multicore_interleaved(
    const Tensor& input_tensor,
    const std::optional<Tensor>& input_indices_tensor,
    const uint32_t k,
    const int8_t dim,
    const bool largest,
    const bool sorted,
    const CoreRangeSet& sub_core_grids,
    Tensor& value_tensor,
    Tensor& index_tensor) {
    using namespace tt::constants;
    tt::tt_metal::Program program{};

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    tt::DataFormat value_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(value_tensor.dtype());
    tt::DataFormat index_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(index_tensor.dtype());

    auto first_core_range = sub_core_grids.ranges().at(0);
    auto first_core_range_set = CoreRangeSet(first_core_range);

    uint32_t input_tile_size = tile_size(input_cb_data_format);
    uint32_t value_tile_size = tile_size(value_cb_data_format);
    uint32_t index_tile_size = tile_size(index_cb_data_format);

    auto input_buffer = input_tensor.buffer();
    auto values_buffer = value_tensor.buffer();
    auto index_buffer = index_tensor.buffer();
    auto input_indices_buffer = input_indices_tensor.has_value() ? input_indices_tensor->buffer() : nullptr;

    bool input_is_dram = input_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    bool values_is_dram = values_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    bool index_is_dram = index_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    bool input_indices_is_dram = input_indices_tensor.has_value()
                                     ? input_indices_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM
                                     : false;

    uint32_t num_input_tiles = input_tensor.physical_volume() / TILE_HW;
    uint32_t num_value_tiles = value_tensor.physical_volume() / TILE_HW;
    auto device = input_tensor.device();

    auto input_shape = input_tensor.padded_shape();
    uint32_t Ht = (input_shape[0] * input_shape[1] * input_shape[2]) / TILE_HEIGHT;
    const auto& [num_cores, local_topk_input_size, rem, final_topk_input_size] = cores_utilized(
        input_shape[dim],
        64,
        input_shape[dim] / 2,
        first_core_range,
        k,
        device->l1_size_per_core(),
        value_tile_size,
        index_tile_size);

    auto all_cores_range_set = select_from_corerange(first_core_range_set, 0, num_cores - 1u, false);

    auto local_cores_range_set = select_from_corerange(first_core_range_set, 0, num_cores - 2u, false);
    auto local_cores = corerange_to_cores(local_cores_range_set, num_cores - 1u, false);

    auto final_cores_range_set = select_from_corerange(first_core_range_set, num_cores - 1u, num_cores - 1u, false);
    auto final_core = corerange_to_cores(final_cores_range_set, 1u, false).at(0);

    uint32_t Wt_local = local_topk_input_size / TILE_WIDTH;
    uint32_t Wt_final = final_topk_input_size / TILE_WIDTH;
    uint32_t Kt = k % TILE_WIDTH == 0 ? k / TILE_WIDTH : k / TILE_WIDTH + 1;

    // for streaming in input
    uint32_t num_cb_unit = 2;
    uint32_t cb_in_units = 2 * num_cb_unit;

    // Two tiles are loaded in for topk_local_sort at a time, and we double buffer to avoid stalls, so allocate four
    // tiles of space
    // TODO: In theory if we have enough memory we could allocate 2*Wt tiles to reduce stalls
    uint32_t input_cb_index = tt::CBIndex::c_0;
    tt::tt_metal::CircularBufferConfig input_cb_config =
        tt::tt_metal::CircularBufferConfig(cb_in_units * value_tile_size, {{input_cb_index, input_cb_data_format}})
            .set_page_size(input_cb_index, input_tile_size);
    auto cb_input_tensor = tt::tt_metal::CreateCircularBuffer(program, all_cores_range_set, input_cb_config);

    // Two tiles are loaded in for topk_local_sort at a time, and we double buffer to avoid stalls, so allocate four
    // tiles of space This CB carries the indices that are created in the reader kernel
    uint32_t index_cb_index = tt::CBIndex::c_1;
    tt::tt_metal::CircularBufferConfig index_input_intermed0_config =
        tt::tt_metal::CircularBufferConfig(cb_in_units * index_tile_size, {{index_cb_index, index_cb_data_format}})
            .set_page_size(index_cb_index, index_tile_size);
    auto cb_index_tensor =
        tt::tt_metal::CreateCircularBuffer(program, all_cores_range_set, index_input_intermed0_config);

    // Single buffered circular buffer that holds the transposed input tiles
    uint32_t input_transposed_cb_index = tt::CBIndex::c_24;
    tt::tt_metal::CircularBufferConfig input_transposed_cb_config =
        tt::tt_metal::CircularBufferConfig(
            Wt_local * value_tile_size, {{input_transposed_cb_index, input_cb_data_format}})
            .set_page_size(input_transposed_cb_index, input_tile_size);
    auto cb_input_transposed_tiles =
        tt::tt_metal::CreateCircularBuffer(program, all_cores_range_set, input_transposed_cb_config);

    // Single buffered circular buffer that holds the transposed index tiles
    uint32_t index_transposed_cb_index = tt::CBIndex::c_25;
    tt::tt_metal::CircularBufferConfig index_transposed_cb_config =
        tt::tt_metal::CircularBufferConfig(
            Wt_local * index_tile_size, {{index_transposed_cb_index, index_cb_data_format}})
            .set_page_size(index_transposed_cb_index, index_tile_size);
    auto cb_index_transposed_tiles =
        tt::tt_metal::CreateCircularBuffer(program, all_cores_range_set, index_transposed_cb_config);

    uint32_t gathered_values_cb_index = tt::CBIndex::c_26;
    tt::tt_metal::CircularBufferConfig gathered_values_cb_config =
        tt::tt_metal::CircularBufferConfig(
            Wt_final * value_tile_size, {{gathered_values_cb_index, value_cb_data_format}})
            .set_page_size(gathered_values_cb_index, value_tile_size);
    auto cb_gathered_topk_values_tensor =
        tt::tt_metal::CreateCircularBuffer(program, all_cores_range_set, gathered_values_cb_config);

    uint32_t gathered_indices_cb_index = tt::CBIndex::c_27;
    tt::tt_metal::CircularBufferConfig gathered_indices_cb_config =
        tt::tt_metal::CircularBufferConfig(
            Wt_final * index_tile_size, {{gathered_indices_cb_index, index_cb_data_format}})
            .set_page_size(gathered_indices_cb_index, index_tile_size);
    auto cb_gathered_topk_indices_tensor =
        tt::tt_metal::CreateCircularBuffer(program, all_cores_range_set, gathered_indices_cb_config);

    uint32_t final_values_cb_index = tt::CBIndex::c_28;
    tt::tt_metal::CircularBufferConfig final_values_cb_config =
        tt::tt_metal::CircularBufferConfig(Wt_final * value_tile_size, {{final_values_cb_index, value_cb_data_format}})
            .set_page_size(final_values_cb_index, value_tile_size);
    auto cb_final_topk_values_tensor =
        tt::tt_metal::CreateCircularBuffer(program, all_cores_range_set, final_values_cb_config);

    uint32_t final_indices_cb_index = tt::CBIndex::c_29;
    tt::tt_metal::CircularBufferConfig final_indices_cb_config =
        tt::tt_metal::CircularBufferConfig(Wt_final * index_tile_size, {{final_indices_cb_index, index_cb_data_format}})
            .set_page_size(final_indices_cb_index, index_tile_size);
    auto cb_final_topk_index_tensor =
        tt::tt_metal::CreateCircularBuffer(program, all_cores_range_set, final_indices_cb_config);

    // Output topk values
    uint32_t values_cb_index = tt::CBIndex::c_16;
    tt::tt_metal::CircularBufferConfig values_cb_config =
        tt::tt_metal::CircularBufferConfig(num_cb_unit * value_tile_size, {{values_cb_index, value_cb_data_format}})
            .set_page_size(values_cb_index, value_tile_size);
    auto cb_values_tensor = tt::tt_metal::CreateCircularBuffer(program, all_cores_range_set, values_cb_config);

    // Output topk indices
    uint32_t output_ind_cb_index = tt::CBIndex::c_17;
    tt::tt_metal::CircularBufferConfig output_ind_cb_config =
        tt::tt_metal::CircularBufferConfig(num_cb_unit * index_tile_size, {{output_ind_cb_index, index_cb_data_format}})
            .set_page_size(output_ind_cb_index, index_tile_size);
    auto cb_output_ind_tensor = tt::tt_metal::CreateCircularBuffer(program, all_cores_range_set, output_ind_cb_config);

    // Create semaphores
    auto sender_semaphore_id = tt::tt_metal::CreateSemaphore(program, all_cores_range_set, INVALID);
    auto receiver_semaphore_id = tt::tt_metal::CreateSemaphore(program, all_cores_range_set, INVALID);
    std::vector<uint32_t> reader_local_compile_time_args = {
        input_cb_index,
        index_cb_index,
        (uint32_t)input_is_dram,
        (uint32_t)input_indices_is_dram,
        Ht,
        Wt_local,
        input_shape[-1] / TILE_WIDTH,  // Wt
    };
    std::string reader_kernel_path =
        "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/dataflow/reader_create_index_local_topk.cpp";
    if (input_indices_tensor.has_value()) {
        reader_kernel_path =
            "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/dataflow/reader_read_index_local_topk.cpp";
    }
    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        reader_kernel_path,
        local_cores_range_set,
        tt::tt_metal::ReaderDataMovementConfig(reader_local_compile_time_args));

    CoreCoord local_cores_physical_start = device->worker_core_from_logical_core(local_cores.at(0));
    CoreCoord local_cores_physical_end = device->worker_core_from_logical_core(local_cores.at(num_cores - 2u));
    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)receiver_semaphore_id,
        (std::uint32_t)sender_semaphore_id,
        (std::uint32_t)local_cores_physical_start.x,
        (std::uint32_t)local_cores_physical_start.y,
        (std::uint32_t)local_cores_physical_end.x,
        (std::uint32_t)local_cores_physical_end.y,
        (std::uint32_t)Ht,
        (std::uint32_t)Wt_final,
        (std::uint32_t)num_cores - 1,
    };

    tt::tt_metal::KernelHandle unary_reader_final_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/dataflow/reader_final_topk.cpp",
        final_cores_range_set,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    CoreCoord final_cores_physical = device->worker_core_from_logical_core(final_core);
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t)receiver_semaphore_id,
        (std::uint32_t)sender_semaphore_id,
        (std::uint32_t)final_cores_physical.x,
        (std::uint32_t)final_cores_physical.y,
        Ht,
        k,
        Kt,
    };
    tt::tt_metal::KernelHandle binary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/dataflow/writer_local_topk.cpp",
        local_cores_range_set,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    std::vector<uint32_t> writer_compile_time_args_final = {
        values_cb_index, output_ind_cb_index, (std::uint32_t)values_is_dram, (std::uint32_t)index_is_dram, Ht, Kt};
    tt::tt_metal::KernelHandle binary_writer_final_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/dataflow/writer_final_topk.cpp",
        final_cores_range_set,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args_final));

    std::vector<uint32_t> compute_args = {
        input_cb_index,
        index_cb_index,
        input_transposed_cb_index,
        index_transposed_cb_index,
        values_cb_index,
        output_ind_cb_index,
        Ht,
        Wt_local,
        k,
        Kt,
        (std::uint32_t)std::log2(k),
        (std::uint32_t)std::log2(Wt_local),
        (std::uint32_t)largest,
        (std::uint32_t)sorted,
    };
    tt::tt_metal::KernelHandle topk_compute_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/compute/topk_local.cpp",
        local_cores_range_set,
        tt::tt_metal::ComputeConfig{.compile_args = compute_args});

    std::vector<uint32_t> compute_args_final = {
        gathered_values_cb_index,
        gathered_indices_cb_index,
        final_values_cb_index,
        final_indices_cb_index,
        values_cb_index,
        output_ind_cb_index,
        Ht,
        Wt_final,
        k,
        Kt,
        (std::uint32_t)std::log2(k),
        (std::uint32_t)std::log2(Wt_final),
        (std::uint32_t)largest,
        (std::uint32_t)sorted,
    };

    tt::tt_metal::KernelHandle topk_final_compute_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/compute/topk_final.cpp",
        final_cores_range_set,
        tt::tt_metal::ComputeConfig{.compile_args = compute_args_final});

    int core_h = 0;
    int core_w = 0;
    bool ascending = !largest;
    for (auto core : local_cores) {
        if (input_indices_tensor.has_value()) {
            SetRuntimeArgs(
                program,
                unary_reader_kernel_id,
                core,
                {
                    input_buffer->address(),
                    input_indices_buffer->address(),
                    0,  // no height parallelism for now
                    core_w * Wt_local,
                });
        } else {
            SetRuntimeArgs(
                program,
                unary_reader_kernel_id,
                core,
                {
                    input_buffer->address(),
                    0,  // no height parallelism for now
                    core_w * Wt_local,
                });
        }

        SetRuntimeArgs(
            program,
            binary_writer_kernel_id,
            core,
            {
                core_w,
            });

        SetRuntimeArgs(
            program,
            topk_compute_kernel_id,
            core,
            {
                ascending,
            });
        core_w++;
        ascending = !ascending;
    }
    SetRuntimeArgs(
        program,
        binary_writer_final_kernel_id,
        final_core,
        {
            values_buffer->address(),
            index_buffer->address(),
        });

    auto override_runtime_args_callback =
        [unary_reader_kernel_id, binary_writer_final_kernel_id, local_cores, final_core](
            const void* operation,
            const Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            auto input_buffer = input_tensors.at(0).buffer();
            auto values_buffer = output_tensors.at(0).buffer();
            auto index_buffer = output_tensors.at(1).buffer();
            auto input_indices_buffer =
                optional_input_tensors.at(0).has_value() ? optional_input_tensors.at(0).value().buffer() : nullptr;

            for (auto core : local_cores) {
                auto& reader_runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, core);
                reader_runtime_args[0] = input_buffer->address();
                if (optional_input_tensors.at(0).has_value()) {
                    reader_runtime_args[1] = input_indices_buffer->address();
                }
            }

            auto& writer_runtime_args = GetRuntimeArgs(program, binary_writer_final_kernel_id, final_core);
            writer_runtime_args[0] = values_buffer->address();
            writer_runtime_args[1] = index_buffer->address();
        };

    return {std::move(program), override_runtime_args_callback};
}
}  // namespace ttnn::operations::reduction::detail
