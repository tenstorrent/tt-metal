// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#include "transpose_hc_rm_program_factory.hpp"

#include <tt_stl/assert.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-logger/tt-logger.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

namespace {

void set_runtime_args_hc_rm(
    Program& program,
    KernelHandle reader_kernel_id,
    KernelHandle writer_kernel_id,
    const Tensor& input_tensor,
    Tensor& output_tensor,
    uint32_t num_cores_total,
    uint32_t num_cores_y,
    const CoreRangeSet& core_group_1,
    uint32_t num_sticks_per_core_group_1,
    const CoreRangeSet& core_group_2,
    uint32_t num_sticks_per_core_group_2,
    bool is_create) {
    auto* input_buffer = input_tensor.buffer();
    auto* output_buffer = output_tensor.buffer();
    auto input_shape = input_tensor.padded_shape();

    uint32_t W = input_shape[3], H = input_shape[2], C = input_shape[1];
    uint32_t W_bytes = W * input_tensor.element_size();

    uint32_t max_read_size = 2048;
    uint32_t curr_c = 0, curr_h = 0, curr_n = 0;

    auto& cached_reader_args = GetRuntimeArgs(program, reader_kernel_id);
    auto& cached_writer_args = GetRuntimeArgs(program, writer_kernel_id);

    for (uint32_t i = 0, curr_sticks_read = 0, curr_sticks_write = 0; i < num_cores_total; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_sticks_per_core;

        if (core_group_1.contains(core)) {
            num_sticks_per_core = num_sticks_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_sticks_per_core = num_sticks_per_core_group_2;
        } else {
            num_sticks_per_core = 0;
        }

        uint32_t num_sticks_per_core_read = 0, num_read_per_barrier = 0;
        if (num_sticks_per_core != 0) {
            num_sticks_per_core_read = merge_num_sticks_to_read(num_sticks_per_core, W_bytes, max_read_size);
            num_read_per_barrier = num_sticks_per_core / num_sticks_per_core_read;
        }

        if (is_create) {
            SetRuntimeArgs(
                program,
                reader_kernel_id,
                core,
                {input_buffer->address(),
                 num_sticks_per_core_read,
                 num_read_per_barrier,
                 curr_sticks_read,
                 curr_c,
                 curr_h,
                 curr_n});

            SetRuntimeArgs(
                program,
                writer_kernel_id,
                core,
                {output_buffer->address(), num_sticks_per_core_read, num_read_per_barrier, curr_sticks_write});
        } else {
            auto& reader_args = cached_reader_args.at(core.x).at(core.y);
            auto& writer_args = cached_writer_args.at(core.x).at(core.y);

            reader_args[0] = input_buffer->address();
            reader_args[1] = num_sticks_per_core_read;
            reader_args[2] = num_read_per_barrier;
            reader_args[3] = curr_sticks_read;
            reader_args[4] = curr_c;
            reader_args[5] = curr_h;
            reader_args[6] = curr_n;

            writer_args[0] = output_buffer->address();
            writer_args[1] = num_sticks_per_core_read;
            writer_args[2] = num_read_per_barrier;
            writer_args[3] = curr_sticks_write;
        }

        curr_sticks_write += num_sticks_per_core;

        for (uint32_t j = 0; j < num_sticks_per_core; ++j) {
            curr_c++;
            curr_sticks_read += H;
            if (curr_c == C) {
                curr_h++;
                curr_c = 0;
                if (curr_h == H) {
                    curr_n++;
                    curr_c = 0;
                    curr_h = 0;
                    curr_sticks_read = curr_sticks_read - H + 1;
                } else {
                    curr_sticks_read = curr_sticks_read - C * H + 1;
                }
            }
        }
    }
}

}  // namespace

TransposeHCRMProgramFactory::cached_program_t TransposeHCRMProgramFactory::create(
    const TransposeParams& /*operation_attributes*/, const TransposeInputs& tensor_args, Tensor& output_tensor) {
    const auto& input_tensor = tensor_args.input;

    TT_ASSERT(input_tensor.storage_type() == StorageType::DEVICE, "Operand to transpose_hc needs to be on device!");
    TT_ASSERT(input_tensor.buffer() != nullptr, "Operand to transpose_hc needs to be allocated in a buffer on device!");

    const auto& a_shape = input_tensor.logical_shape();
    uint32_t W = a_shape[3], H = a_shape[2], C = a_shape[1], N = a_shape[0];
    uint32_t NCH = N * C * H;

    Program program = CreateProgram();

    tt::DataFormat cb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());

    log_debug(tt::LogOp, "transpose_hc_rm");
    log_debug(tt::LogOp, "cb_data_format: {}", cb_data_format);

    IDevice* device = input_tensor.device();
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_cores_total = num_cores_x * num_cores_y;
    CoreRange total_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    auto [num_cores, all_cores, core_group_1, core_group_2, num_sticks_per_core_group_1, num_sticks_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, NCH);

    Buffer* dst_buffer = output_tensor.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t src0_cb_index = 0;

    auto num_sticks = num_sticks_per_core_group_1 > num_sticks_per_core_group_2 ? num_sticks_per_core_group_1
                                                                                : num_sticks_per_core_group_2;
    auto stick_size = W * input_tensor.element_size();
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(num_sticks * stick_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, stick_size);
    CreateCircularBuffer(program, total_cores, cb_src0_config);

    Buffer* src0_buffer = input_tensor.buffer();
    std::vector<uint32_t> reader_compile_time_args;
    reader_compile_time_args.push_back(N);
    reader_compile_time_args.push_back(H);
    reader_compile_time_args.push_back(C);
    reader_compile_time_args.push_back(W * input_tensor.element_size());
    reader_compile_time_args.push_back(W * input_tensor.element_size());
    TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {src0_cb_index};
    writer_compile_time_args.push_back(W * input_tensor.element_size());
    writer_compile_time_args.push_back(W * input_tensor.element_size());
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
        "reader_unary_transpose_hc_interleaved_partitioned_rm.cpp",
        total_cores,
        ReaderDataMovementConfig(reader_compile_time_args));

    KernelHandle writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
        "writer_unary_transpose_hc_interleaved_start_id_rm.cpp",
        total_cores,
        WriterDataMovementConfig(writer_compile_time_args));

    set_runtime_args_hc_rm(
        program,
        reader_kernel_id,
        writer_kernel_id,
        input_tensor,
        output_tensor,
        num_cores_total,
        num_cores_y,
        core_group_1,
        num_sticks_per_core_group_1,
        core_group_2,
        num_sticks_per_core_group_2,
        true);

    return {
        std::move(program),
        {.reader_kernel_id = reader_kernel_id,
         .writer_kernel_id = writer_kernel_id,
         .core_group_1 = core_group_1,
         .core_group_2 = core_group_2,
         .num_cores_total = num_cores_total,
         .num_cores_y = num_cores_y,
         .num_sticks_per_core_group_1 = num_sticks_per_core_group_1,
         .num_sticks_per_core_group_2 = num_sticks_per_core_group_2}};
}

void TransposeHCRMProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const TransposeParams& /*operation_attributes*/,
    const TransposeInputs& tensor_args,
    Tensor& output_tensor) {
    auto& program = cached_program.program;
    auto& shared_variables = cached_program.shared_variables;

    set_runtime_args_hc_rm(
        program,
        shared_variables.reader_kernel_id,
        shared_variables.writer_kernel_id,
        tensor_args.input,
        output_tensor,
        shared_variables.num_cores_total,
        shared_variables.num_cores_y,
        shared_variables.core_group_1,
        shared_variables.num_sticks_per_core_group_1,
        shared_variables.core_group_2,
        shared_variables.num_sticks_per_core_group_2,
        false);
}

}  // namespace ttnn::prim
