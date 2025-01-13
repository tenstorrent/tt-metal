// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_log.h"

namespace ttnn::operations::reduction::detail {

operation::ProgramWithCallbacks sampling_multicore_interleaved(
    const Tensor& input_values_tensor,
    const Tensor& input_indices_tensor,
    const std::vector<uint16_t>& k,
    const std::vector<uint16_t>& p,
    Tensor& output_tensor) {
    using namespace tt::constants;
    tt::tt_metal::Program program{};

    tt::DataFormat input_values_cb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(input_values_tensor.get_dtype());
    tt::DataFormat input_indices_cb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(input_indices_tensor.get_dtype());
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.get_dtype());
    tt::DataFormat index_cb_data_format = tt::DataFormat::UInt16;

    uint32_t input_values_tile_size = tile_size(input_values_cb_data_format);
    uint32_t input_indices_tile_size = tile_size(input_indices_cb_data_format);
    uint32_t index_tile_size = tile_size(index_cb_data_format);

    auto input_values_buffer = input_values_tensor.buffer();
    auto input_indices_buffer = input_indices_tensor.buffer();
    auto output_buffer = output_tensor.buffer();

    bool input_values_is_dram = input_values_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    bool input_indices_is_dram = input_indices_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    bool output_is_dram = output_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;

    uint32_t num_input_values_tiles = input_values_tensor.volume() / TILE_HW;
    uint32_t num_input_indices_tiles = input_indices_tensor.volume() / TILE_HW;
    auto device = input_tensor.device();

    auto input_shape = input_values_tensor.get_legacy_shape();
    uint32_t Ht = (input_shape[0] * input_shape[1] * input_shape[2]) / TILE_HEIGHT;
    uint32_t Wt = input_shape[3] / TILE_WIDTH;
    const auto& num_cores = Ht * TILE_HEIGHT;
    CoreRange cores({0, 0}, {0, num_cores - 1u});

    // for streaming in input
    uint32_t num_cb_unit = 2;
    uint32_t cb_in_units = 2 * num_cb_unit;

    // Two tiles are loaded in for sampling_local_sort at a time, and we double buffer to avoid stalls, so allocate
    // four tiles of space
    // TODO: In theory if we have enough memory we could allocate 2*Wt tiles to reduce stalls
    uint32_t input_values_cb_index = tt::CBIndex::c_0;
    tt::tt_metal::CircularBufferConfig input_values_cb_config =
        tt::tt_metal::CircularBufferConfig(
            cb_in_units * input_values_tile_size, {{input_values_cb_index, input_values_cb_data_format}})
            .set_page_size(input_values_cb_index, input_values_tile_size);
    auto cb_input_values_tensor = tt::tt_metal::CreateCircularBuffer(program, core, input_values_cb_config);

    uint32_t input_indices_cb_index = tt::CBIndex::c_1;
    tt::tt_metal::CircularBufferConfig input_indices_cb_config =
        tt::tt_metal::CircularBufferConfig(
            cb_in_units * input_indices_tile_size, {{input_indices_cb_index, input_indices_cb_data_format}})
            .set_page_size(input_indices_cb_index, input_indices_tile_size);
    auto cb_input_indices_tensor = tt::tt_metal::CreateCircularBuffer(program, core, input_indices_cb_config);

    uint32_t index_cb_index = tt::CBIndex::c_24;
    tt::tt_metal::CircularBufferConfig index_input_intermed0_config =
        tt::tt_metal::CircularBufferConfig(cb_in_units * index_tile_size, {{index_cb_index, index_cb_data_format}})
            .set_page_size(index_cb_index, index_tile_size);
    auto cb_index_tensor = tt::tt_metal::CreateCircularBuffer(program, core, index_input_intermed0_config);

    // // Two tiles are loaded in for sampling_local_sort at a time, and we double buffer to avoid stalls, so allocate
    // four
    // // tiles of space This CB carries the indices that are created in the reader kernel
    // uint32_t index_cb_index = tt::CBIndex::c_1;
    // tt::tt_metal::CircularBufferConfig index_input_intermed0_config =
    //     tt::tt_metal::CircularBufferConfig(cb_in_units * index_tile_size, {{index_cb_index, index_cb_data_format}})
    //         .set_page_size(index_cb_index, index_tile_size);
    // auto cb_index_tensor = tt::tt_metal::CreateCircularBuffer(program, core, index_input_intermed0_config);

    // // Single buffered circular buffer that holds the transposed input tiles
    // uint32_t input_transposed_cb_index = tt::CBIndex::c_24;
    // tt::tt_metal::CircularBufferConfig input_transposed_cb_config =
    //     tt::tt_metal::CircularBufferConfig(
    //         Wt_local * value_tile_size, {{input_transposed_cb_index, input_cb_data_format}})
    //         .set_page_size(input_transposed_cb_index, input_tile_size);
    // auto cb_input_transposed_tiles = tt::tt_metal::CreateCircularBuffer(program, core, input_transposed_cb_config);

    // // Single buffered circular buffer that holds the transposed index tiles
    // uint32_t index_transposed_cb_index = tt::CBIndex::c_25;
    // tt::tt_metal::CircularBufferConfig index_transposed_cb_config =
    //     tt::tt_metal::CircularBufferConfig(
    //         Wt_local * index_tile_size, {{index_transposed_cb_index, index_cb_data_format}})
    //         .set_page_size(index_transposed_cb_index, index_tile_size);
    // auto cb_index_transposed_tiles = tt::tt_metal::CreateCircularBuffer(program, core, index_transposed_cb_config);

    // // Output sampling values
    // uint32_t values_cb_index = tt::CBIndex::c_16;
    // tt::tt_metal::CircularBufferConfig values_cb_config =
    //     tt::tt_metal::CircularBufferConfig(num_cb_unit * value_tile_size, {{values_cb_index, value_cb_data_format}})
    //         .set_page_size(values_cb_index, value_tile_size);
    // auto cb_values_tensor = tt::tt_metal::CreateCircularBuffer(program, core, values_cb_config);

    // // Output cumsum values
    // uint32_t cumsum_values_cb_index = tt::CBIndex::c_17;
    // tt::tt_metal::CircularBufferConfig cumsum_values_cb_config =
    //     tt::tt_metal::CircularBufferConfig(
    //         num_cb_unit * value_tile_size, {{cumsum_values_cb_index, value_cb_data_format}})
    //         .set_page_size(cumsum_values_cb_index, value_tile_size);
    // auto cb_cumsum_values_tensor = tt::tt_metal::CreateCircularBuffer(program, core, cumsum_values_cb_config);

    // // Output sampling indices
    uint32_t num_out0_units = 32;
    uint32_t output_unit_size = output_tensor.element_size();
    uint32_t aligned_out0_unit_size = 32 * output_unit_size;
    uint32_t output_cb_index = tt::CBIndex::c_17;
    tt::tt_metal::CircularBufferConfig output_cb_config =
        tt::tt_metal::CircularBufferConfig(aligned_out0_unit_size, {{output_ind_cb_index, index_cb_data_format}})
            .set_page_size(output_ind_cb_index, aligned_out0_unit_size);
    auto cb_output_tensor = tt::tt_metal::CreateCircularBuffer(program, cores, output_cb_config);

    std::vector<uint32_t> reader_compile_time_args = {
        input_values_cb_index,
        input_indices_cb_index,
        index_cb_index,
        (uint32_t)input_values_is_dram,
        (uint32_t)input_indices_is_dram,
        Ht,
        Wt};
    tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/sampling/device/kernels/dataflow/reader_values_indices_tensor.cpp",
        cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    SetRuntimeArgs(
        program,
        reader_kernel_id,
        cores,
        {
            input_values_buffer->address(),
            input_indices_buffer->address(),
        });

    for (uint32_t i = 0; i < cores.size(); ++i) {
        const CoreCoord& core = cores.at(i);
        std::vector<uint32_t> writer_compile_time_args = {
            output_cb_index, (std::uint32_t)output_is_dram, k[i], i, aligned_out0_unit_size};
        tt::tt_metal::KernelHandle writer_kernel_id = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/reduction/sampling/device/kernels/dataflow/writer_interleaved.cpp",
            core,
            tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

        tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, {output_buffer->address()});
    }

    // std::vector<uint32_t> compute_args = {
    //     input_cb_index,
    //     index_cb_index,
    //     input_transposed_cb_index,
    //     index_transposed_cb_index,
    //     values_cb_index,
    //     output_ind_cb_index,
    //     Ht,
    //     Wt,
    //     k,
    //     (std::uint32_t)std::log2(k),
    //     (std::uint32_t)std::log2(Wt),
    // };
    // tt::tt_metal::KernelHandle topk_compute_kernel_id = tt::tt_metal::CreateKernel(
    //     program,
    //     "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/compute/topk.cpp",
    //     core,
    //     tt::tt_metal::ComputeConfig{.compile_args = compute_args});

    auto override_runtime_args_callback = [reader_kernel_id, writer_kernel_id](
                                              const Program& program,
                                              const std::vector<Buffer*>& input_buffers,
                                              const std::vector<Buffer*>& output_buffers) {
        auto input_values_buffer = input_buffers.at(0);
        auto input_indices_buffer = input_buffers.at(1);
        auto output_buffer = output_buffers.at(0);

        for (const auto& core : cores) {
            auto& reader_runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
            reader_runtime_args[0] = input_values_buffer->address();
            reader_runtime_args[1] = input_indices_buffer->address();

            auto& writer_runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
            writer_runtime_args[0] = output_buffer->address();
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

}  // namespace ttnn::operations::reduction::detail
