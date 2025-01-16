// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <string>
#include <vector>

#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/work_split.hpp>
#include "expand_device_operation.hpp"
#include <tt-metalium/host_api.hpp>
#include "hostdevcommon/kernel_structs.h"
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/circular_buffer_types.hpp>
#include <tt-metalium/kernel_types.hpp>
#include "ttnn/tensor/types.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::expand {
ExpandOperation::ExpandRowMajorFactory::cached_program_t ExpandOperation::ExpandRowMajorFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto input = tensor_args.input;

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

    uint32_t nd_expand_count = output.get_logical_volume() / input.get_logical_volume() / horz_expand_count;

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

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_copies_per_core_group_1, num_copies_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, nd_expand_count);

#ifdef DEBUG
    tt::log_debug("Num Cores = %d\n", num_cores);
    tt::log_debug("Num Rows Per Core Group 1 = %d\n", num_copies_per_core_group_1);
    tt::log_debug("Num Rows Per Core Group 2 = %d\n", num_copies_per_core_group_2);
#endif

    const auto src_is_dram = static_cast<const uint32_t>(input.buffer()->is_dram());
    const auto dst_is_dram = static_cast<const uint32_t>(output.buffer()->is_dram());

    const auto sram_buffer_length = 32;

    // Scratch SRAM buffer
    uint32_t scratch_buf_id = tt::CBIndex::c_24;
    auto scratch_config =
        CircularBufferConfig(unexpanded_row_size * sram_buffer_length, {{scratch_buf_id, data_format}})
            .set_page_size(scratch_buf_id, unexpanded_row_size);
    auto scratch_handle = CreateCircularBuffer(program, all_cores, scratch_config);

    // IO SRAM Buffer
    uint32_t io_buf_id = tt::CBIndex::c_16;
    auto io_config = CircularBufferConfig(expanded_row_size * sram_buffer_length, {{io_buf_id, data_format}})
                         .set_page_size(io_buf_id, expanded_row_size);
    auto io_handle = CreateCircularBuffer(program, all_cores, io_config);

    std::vector<uint32_t> reader_compile_runtime_args = {
        src_is_dram,
        scratch_buf_id,
        io_buf_id,
        data_size,
    };
    std::vector<uint32_t> writer_compile_runtime_args = {
        dst_is_dram,
        io_buf_id,
    };

    KernelHandle reader_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/expand/device/kernels/reader_rm_expand.cpp",
        all_cores,
        ReaderDataMovementConfig(reader_compile_runtime_args));

    KernelHandle writer_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/expand/device/kernels/writer_rm_expand.cpp",
        all_cores,
        WriterDataMovementConfig(writer_compile_runtime_args));

    uint32_t rows_offset = 0;
    uint32_t group1_cores = core_group_1.num_cores();
    auto cores = grid_to_cores(num_cores, num_cores_x, num_cores_y);

    uint32_t num_copies_this_core;
    for (auto core : cores) {
        if (core_group_1.contains(core)) {
            num_copies_this_core = num_copies_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_copies_this_core = num_copies_per_core_group_2;
        }

        SetRuntimeArgs(
            program,
            reader_id,
            core,
            {
                input.buffer()->address(),
                n_rows,
                input_shape[input_tsr_rank - 1],
                horz_expand_count,
                input.buffer()->page_size(),
            });

        SetRuntimeArgs(
            program,
            writer_id,
            core,
            {
                output.buffer()->address(),
                n_rows,
                output.buffer()->page_size(),
                num_copies_this_core,
                rows_offset,
            });

        // Buffer page size is exactly one row in ROW_MAJOR mode
        rows_offset += num_copies_this_core * n_rows;
    }
    return {std::move(program), {reader_id, writer_id, cores}};
}

void ExpandOperation::ExpandRowMajorFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto& program = cached_program.program;
    auto& reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    auto& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;
    auto& cores = cached_program.shared_variables.cores;

    auto input = tensor_args.input;

    for (const auto& core : cores) {
        {
            // reader
            auto runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
            runtime_args[0] = input.buffer()->address();
        }
        {
            // writer
            auto runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
            runtime_args[0] = output.buffer()->address();
        }
    }
}
}  // namespace ttnn::operations::expand
