// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "common/constants.hpp"
#include "common/math.hpp"
#include "expand_device_operation.hpp"
#include "host_api.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::expand {

ExpandOperation::ExpandTileFactory::cached_program_t ExpandOperation::ExpandTileFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto input = tensor_args.input;

    auto output_mem_config = operation_attributes.memory_config;
    auto compute_kernel_config = operation_attributes.compute_kernel_config;

    // Device Setup
    auto* device = input.device();
    Program program = CreateProgram();

    // Initialize data
    auto input_shape_tmp = input.get_shape();
    std::vector<int64_t> input_shape;

    // Strip empty leading dimensions (for what we are doing next, this spell P-A-I-N)
    for (int i = 0; i < input_shape_tmp.size(); i++) {
        if (input_shape_tmp[i] > 1) {
            // Push the rest of the shape
            for (int j = i; j < input_shape_tmp.size(); j++) {
                input_shape.push_back(input_shape_tmp[j]);
            }
            break;
        }
    }

    // If it's LITERALLY {1}, handle it
    if (input_shape.size() == 0) {
        input_shape.push_back(1);
    }

    auto output_shape = output.get_shape();
    uint32_t data_size = input.element_size();
    tt::DataFormat data_format = datatype_to_dataformat_converter(input.get_dtype());

    // These are needed for the 2d case where the page size actually changes
    uint32_t input_tsr_rank = input_shape.size();
    uint32_t output_tsr_rank = output_shape.size();
    uint32_t n_rows = input_tsr_rank == 1 ? 1 : input_shape[input_tsr_rank - 2];

    uint32_t unexpanded_row_size = input_shape[input_tsr_rank - 1] * data_size;
    uint32_t expanded_row_size = output_shape[output_tsr_rank - 1] * data_size;
    uint32_t horz_expand_count = expanded_row_size / unexpanded_row_size;

    uint32_t nd_expand_count = output.volume() / input.volume() / horz_expand_count;

    uint32_t n_input_tiles = tt::div_up(input.volume(), tt::constants::TILE_HW);
    uint32_t n_output_tiles = tt::div_up(output.volume(), tt::constants::TILE_HW);
    auto tile_size = tt::constants::TILE_HW * data_size;

#ifdef DEBUG
    tt::log_debug("Data size = %d\n", data_size);

    tt::log_debug("Input Page size = %lu\n", input.buffer()->page_size());
    tt::log_debug("Output Page size = %lu\n", output.buffer()->page_size());

    std::stringstream debug_stream;

    debug_stream << "Input Shape = ";
    for (auto i = 0; i < input_shape.size(); i++) {
        debug_stream << input_shape[i] << " ";
    }
    debug_stream << std::endl;

    debug_stream << "Output Shape = ";
    for (auto i = 0; i < output_shape.size(); i++) {
        debug_stream << output_shape[i] << " ";
    }
    debug_stream << std::endl;

    tt::log_debug("%s", debug_stream.str().c_str());

    tt::log_debug("Horz Expand Ratio = %d\n", horz_expand_count);
    tt::log_debug("Vert Expand Ratio = %d\n", nd_expand_count);
#endif

    auto core = CoreCoord{0, 0};

    const auto src_is_dram = static_cast<const uint32_t>(input.buffer()->is_dram());
    const auto dst_is_dram = static_cast<const uint32_t>(output.buffer()->is_dram());
    const uint32_t sram_buffer_length = 32;

    // Scratch buffer
    uint32_t reader_scratch_cb_id = tt::CB::c_intermed0;
    auto scratch_config = CircularBufferConfig(tile_size * 2, {{reader_scratch_cb_id, data_format}})
                              .set_page_size(reader_scratch_cb_id, tile_size);
    auto scratch_handle = CreateCircularBuffer(program, core, scratch_config);

    // IO SRAM
    uint32_t io_sram_cb_id = tt::CB::c_out0;
    auto io_sram_config = CircularBufferConfig(unexpanded_row_size * sram_buffer_length, {{io_sram_cb_id, data_format}})
                              .set_page_size(io_sram_cb_id, unexpanded_row_size * data_size);
    auto io_sram_handle = CreateCircularBuffer(program, core, io_sram_config);

    std::vector<uint32_t> reader_compile_time_args = {
        src_is_dram,
        reader_scratch_cb_id,
        io_sram_cb_id,
        data_size,
        tt::constants::TILE_WIDTH,
        tt::constants::TILE_HEIGHT,
    };

    std::vector<uint32_t> writer_compile_time_args = {
        dst_is_dram,
        io_sram_cb_id,
        data_size,
        tt::constants::TILE_WIDTH,
        tt::constants::TILE_HEIGHT,
    };

    KernelHandle reader_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/expand/device/kernels/reader_tile_expand.cpp",
        core,
        ReaderDataMovementConfig(reader_compile_time_args));

    KernelHandle writer_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/expand/device/kernels/writer_tile_expand.cpp",
        core,
        WriterDataMovementConfig(writer_compile_time_args));

    tt::tt_metal::SetRuntimeArgs(
        program,
        reader_id,
        core,
        {
            input.buffer()->address(),
            n_rows,
            unexpanded_row_size,
            horz_expand_count,
        });

    tt::tt_metal::SetRuntimeArgs(
        program,
        writer_id,
        core,
        {
            output.buffer()->address(),
            n_rows,
            expanded_row_size,
            nd_expand_count,
        });

    return {std::move(program), {reader_id, writer_id, core}};
}

void ExpandOperation::ExpandTileFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto& program = cached_program.program;
    auto& reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    auto& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;
    auto core = cached_program.shared_variables.core;

    auto input = tensor_args.input;

    {
        auto runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
        runtime_args[0] = input.buffer()->address();
    }
    {
        auto runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
        runtime_args[0] = output.buffer()->address();
    }
}
}  // namespace ttnn::operations::expand
