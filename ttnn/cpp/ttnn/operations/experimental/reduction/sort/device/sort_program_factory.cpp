// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sort_program_factory.hpp"

#include <cstdint>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>

namespace ttnn::operations::experimental::reduction::sort::program {

SortProgramFactory::cached_program_t SortProgramFactory::create(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args, tensor_return_value_t& output_tensors) {
    // Program config
    tt::tt_metal::Program program{};
    const CoreCoord core = {0, 0};

    // Tensor config info
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

    // Circular buffers
    constexpr uint32_t input_tensor_cb_index = tt::CBIndex::c_0;
    const tt::tt_metal::CircularBufferConfig input_tensor_cb_config =
        tt::tt_metal::CircularBufferConfig(
            cb_in_units * input_tensor_tile_size, {{input_tensor_cb_index, input_tensor_cb_data_format}})
            .set_page_size(input_tensor_cb_index, input_tensor_tile_size);
    auto cb_input_tensor = tt::tt_metal::CreateCircularBuffer(program, core, input_tensor_cb_config);

    constexpr uint32_t index_tensor_cb_index = tt::CBIndex::c_1;
    const tt::tt_metal::CircularBufferConfig index_tensor_cb_config =
        tt::tt_metal::CircularBufferConfig(
            cb_in_units * index_tensor_tile_size, {{index_tensor_cb_index, index_tensor_cb_data_format}})
            .set_page_size(index_tensor_cb_index, index_tensor_tile_size);
    auto cb_index_tensor = tt::tt_metal::CreateCircularBuffer(program, core, index_tensor_cb_config);

    constexpr uint32_t input_tensor_transposed_cb_index = tt::CBIndex::c_24;
    const tt::tt_metal::CircularBufferConfig input_tensor_transposed_cb_config =
        tt::tt_metal::CircularBufferConfig(
            Wt * input_tensor_tile_size, {{input_tensor_transposed_cb_index, input_tensor_cb_data_format}})
            .set_page_size(input_tensor_transposed_cb_index, input_tensor_tile_size);
    auto cb_input_tensor_transposed =
        tt::tt_metal::CreateCircularBuffer(program, core, input_tensor_transposed_cb_config);

    constexpr uint32_t index_tensor_transposed_cb_index = tt::CBIndex::c_25;
    const tt::tt_metal::CircularBufferConfig index_tensor_transposed_cb_config =
        tt::tt_metal::CircularBufferConfig(
            Wt * index_tensor_tile_size, {{index_tensor_transposed_cb_index, index_tensor_cb_data_format}})
            .set_page_size(index_tensor_transposed_cb_index, index_tensor_tile_size);
    auto cb_index_tensor_transposed =
        tt::tt_metal::CreateCircularBuffer(program, core, index_tensor_transposed_cb_config);

    constexpr uint32_t value_tensor_cb_index = tt::CBIndex::c_16;
    const tt::tt_metal::CircularBufferConfig value_tensor_cb_config =
        tt::tt_metal::CircularBufferConfig(
            num_cb_unit * value_tensor_tile_size, {{value_tensor_cb_index, value_tensor_cb_data_format}})
            .set_page_size(value_tensor_cb_index, index_tensor_tile_size);
    auto cb_value_tensor = tt::tt_metal::CreateCircularBuffer(program, core, value_tensor_cb_config);

    constexpr uint32_t index_tensor_output_cb_index = tt::CBIndex::c_17;
    const tt::tt_metal::CircularBufferConfig index_tensor_output_cb_config =
        tt::tt_metal::CircularBufferConfig(
            num_cb_unit * index_tensor_tile_size, {{index_tensor_output_cb_index, index_tensor_cb_data_format}})
            .set_page_size(index_tensor_output_cb_index, index_tensor_tile_size);
    auto cb_index_tensor_output = tt::tt_metal::CreateCircularBuffer(program, core, index_tensor_output_cb_config);

    // Kernels
    const std::vector<uint32_t> reader_compile_time_args = {
        input_tensor_cb_index,
        index_tensor_output_cb_index,
        static_cast<uint32_t>(input_tensor_is_dram),
        static_cast<uint32_t>(index_tensor_is_dram),
        Ht,
        Wt};
    const std::string reader_kernel_path =
        "ttnn/cpp/ttnn/operations/experimental/reduction/sort/device/kernels/dataflow/reader.cpp";
    tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program, reader_kernel_path, core, tt::tt_metal::ReaderDataMovementConfig{reader_compile_time_args});

    SetRuntimeArgs(program, reader_kernel_id, core, {input_buffer->address(), index_buffer->address()});

    const std::vector<uint32_t> writer_compile_time_args = {
        value_tensor_cb_index, index_tensor_cb_index, static_cast<uint32_t>(value_tensor_is_dram), Ht, Wt};
    const std::string writer_kernel_path =
        "ttnn/cpp/ttnn/operations/experimental/reduction/sort/device/kernels/dataflow/writer.cpp";
    tt::tt_metal::KernelHandle writer_kernel_id = tt::tt_metal::CreateKernel(
        program, writer_kernel_path, core, tt::tt_metal::WriterDataMovementConfig{writer_compile_time_args});

    SetRuntimeArgs(program, writer_kernel_id, core, {value_buffer->address()});

    const std::vector<uint32_t> compute_compile_time_args = {
        input_tensor_cb_index,
        index_tensor_cb_index,
        input_tensor_transposed_cb_index,
        index_tensor_transposed_cb_index,
        value_tensor_cb_index,
        index_tensor_output_cb_index,
        Ht,
        Wt,
        static_cast<uint32_t>(attributes.descending),
        static_cast<uint32_t>(attributes.stable)};
    const std::string compute_kernel_path =
        "ttnn/cpp/ttnn/operations/experimental/reduction/sort/device/kernels/compute/sort.cpp";
    tt::tt_metal::KernelHandle compute_kernel_id = tt::tt_metal::CreateKernel(
        program, compute_kernel_path, core, tt::tt_metal::ComputeConfig{.compile_args = compute_compile_time_args});

    return {std::move(program), {reader_kernel_id, compute_kernel_id, writer_kernel_id}};
}

void SortProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensors) {
    auto input_tensor_buffer = tensor_args.input_tensor.buffer();
    auto value_tensor_buffer = output_tensors.at(0).buffer();
    auto index_tensor_buffer = output_tensors.at(1).buffer();

    CoreCoord core = {0, 0};

    {
        auto& reader_runtime_args =
            GetRuntimeArgs(cached_program.program, cached_program.shared_variables.reader_kernel_id, core);
        reader_runtime_args[0] = input_tensor_buffer->address();
        reader_runtime_args[1] = index_tensor_buffer->address();

        auto& writer_runtime_args =
            GetRuntimeArgs(cached_program.program, cached_program.shared_variables.writer_kernel_id, core);
        writer_runtime_args[0] = value_tensor_buffer->address();
    }
}

}  // namespace ttnn::operations::experimental::reduction::sort::program
