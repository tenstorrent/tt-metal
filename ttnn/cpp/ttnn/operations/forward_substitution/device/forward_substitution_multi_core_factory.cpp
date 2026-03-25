// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "forward_substitution_device_operation.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::forward_substitution {

ForwardSubstitutionOperation::MultiCore::cached_program_t ForwardSubstitutionOperation::MultiCore::create(
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const tt::tt_metal::Tensor& input = tensor_args.input;
    // ROW_MAJOR: padded_shape == logical_shape (enforced in validate)
    const auto input_shape = input.logical_shape();
    const auto rank = input_shape.rank();

    // C = matrix dimension (last dim)
    const uint32_t C = input_shape[-1];
    const uint32_t matrix_page_size = C * sizeof(float);      // one row in bytes
    const uint32_t matrix_total_size = C * matrix_page_size;  // full C×C matrix in bytes

    // batch = product of all dims except last two
    uint32_t batch = 1;
    for (uint32_t i = 0; i < rank - 2; i++) {
        batch *= input_shape[i];
    }

    ////////////////////////////////////////////////////////////////////////////
    //                            Program Setup
    ////////////////////////////////////////////////////////////////////////////
    Program program{};

    auto compute_with_storage_grid_size = input.device()->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, batches_per_core_group_1, batches_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, batch);

    auto data_format = tt::DataFormat::Float32;

    // CB c_0: input rows from reader (double-buffered, one row at a time)
    uint32_t cb_in_depth = 2;
    auto cb_in_index = tt::CBIndex::c_0;
    CreateCircularBuffer(
        program,
        all_cores,
        tt::tt_metal::CircularBufferConfig(cb_in_depth * matrix_page_size, {{cb_in_index, data_format}})
            .set_page_size(cb_in_index, matrix_page_size));

    // CB c_1: work buffer (holds one full C×C matrix for in-place forward sub)
    auto cb_work_index = tt::CBIndex::c_1;
    CreateCircularBuffer(
        program,
        all_cores,
        tt::tt_metal::CircularBufferConfig(matrix_total_size, {{cb_work_index, data_format}})
            .set_page_size(cb_work_index, matrix_total_size));

    // CB c_2: temp row buffer (holds one row during forward sub to avoid read-after-write hazard)
    auto cb_temp_index = tt::CBIndex::c_2;
    CreateCircularBuffer(
        program,
        all_cores,
        tt::tt_metal::CircularBufferConfig(matrix_page_size, {{cb_temp_index, data_format}})
            .set_page_size(cb_temp_index, matrix_page_size));

    // Create reader kernel
    std::vector<uint32_t> reader_compile_time_args = {
        matrix_page_size,  // page size (one row)
        C,                 // matrix dimension
    };
    tt::tt_metal::TensorAccessorArgs(input.buffer()).append_to(reader_compile_time_args);

    auto reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/forward_substitution/device/kernels/reader_forward_substitution.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    // Create writer kernel
    std::vector<uint32_t> writer_compile_time_args = {
        matrix_page_size,  // page size (one row)
        C,                 // matrix dimension
    };
    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(writer_compile_time_args);

    auto writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/forward_substitution/device/kernels/writer_forward_substitution.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    // Set runtime args per core
    uint32_t start_batch = 0;
    auto cores = corerange_to_cores(all_cores);
    for (const auto& core : cores) {
        uint32_t batches_per_core{};
        if (core_group_1.contains(core)) {
            batches_per_core = batches_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            batches_per_core = batches_per_core_group_2;
        } else {
            TT_FATAL(false, "Core not in specified core ranges");
        }

        SetRuntimeArgs(
            program,
            reader_kernel_id,
            core,
            {
                input.buffer()->address(),
                start_batch,
                start_batch + batches_per_core,
            });

        SetRuntimeArgs(
            program,
            writer_kernel_id,
            core,
            {
                output.buffer()->address(),
                start_batch,
                start_batch + batches_per_core,
            });

        start_batch += batches_per_core;
    }

    return {std::move(program), {reader_kernel_id, writer_kernel_id, cores}};
}

void ForwardSubstitutionOperation::MultiCore::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto& program = cached_program.program;
    auto& reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    auto& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;
    auto& cores = cached_program.shared_variables.cores;

    auto src_buffer = tensor_args.input.buffer()->address();
    auto output_buffer = output.buffer()->address();

    for (const auto& core : cores) {
        {
            auto& runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
            runtime_args[0] = src_buffer;
        }
        {
            auto& runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
            runtime_args[0] = output_buffer;
        }
    }
}

}  // namespace ttnn::operations::forward_substitution
