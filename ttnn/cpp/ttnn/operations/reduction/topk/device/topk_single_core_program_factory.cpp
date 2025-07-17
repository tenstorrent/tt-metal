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

}  // namespace ttnn::operations::reduction::detail
