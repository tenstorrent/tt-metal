// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/reshape_view/device/reshape_row_major_program_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#define MASK_64 0xFFFFFFFFFFFFFFC0
#define MASK_16 0xFFFFFFFFFFFFFFF0

namespace ttnn::operations::data_movement::reshape {

ReshapeRMProgramFactory::cached_program_t ReshapeRMProgramFactory::create(
    const ReshapeParams& operation_attributes, const ReshapeInputs& tensor_args, Tensor& tensor_return_value) {
    const auto& input = tensor_args.input;
    const auto& output = tensor_return_value;
    const auto& sub_core_grid = operation_attributes.sub_core_grid;

    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();
    // get datum size
    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    const uint32_t data_size = input.element_size();
    tt::tt_metal::IDevice* device = input.device();
    // Multi device pre-computation
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    CoreRange default_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});
    CoreRangeSet total_cores = sub_core_grid.has_value() ? sub_core_grid.value() : CoreRangeSet(default_cores);
    uint32_t num_cores_total = total_cores.num_cores();

    auto input_log_shape = input.logical_shape();
    auto output_log_shape = output.logical_shape();

    log_debug(tt::LogOp, "reshape_view: row major program factory");
    log_debug(tt::LogOp, "input shape: {}", input_log_shape);
    log_debug(tt::LogOp, "output shape: {}", output_log_shape);
    log_debug(tt::LogOp, "data size: {}", data_size);

    uint32_t source_page_size_bytes = input_log_shape[-1] * data_size;
    uint32_t dest_page_size_bytes = output_log_shape[-1] * data_size;
    uint32_t source_read_size_bytes = ((source_page_size_bytes - 1) & MASK_64) + 128;
    uint32_t read_start_page = 0;
    uint32_t write_start_page = 0;
    tt::tt_metal::Buffer* src_buffer = input.buffer();
    tt::tt_metal::Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");
    // Find how many input pages each core is responsible for so that we always start at the beginning of a read and
    // write page Since the logical volumes match, we are guaranteed that the very last page is aligned
    uint32_t responsibility = ((input_log_shape[-2] - 1) / num_cores_total) + 1;
    while ((responsibility * source_page_size_bytes) % dest_page_size_bytes != 0) {
        responsibility++;
    }
    const uint32_t cb_size0 = source_read_size_bytes;
    const uint32_t cb_size1 = ((dest_page_size_bytes - 1) & MASK_64) + 80;

    bool can_use_dual_kernel =
        (source_page_size_bytes % dest_page_size_bytes == 0 || dest_page_size_bytes % source_page_size_bytes == 0);

    uint32_t src0_cb_index = 0;
    uint32_t src1_cb_index = 1;
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(cb_size0 * 2, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, cb_size0);
    tt::tt_metal::CreateCircularBuffer(program, total_cores, cb_src0_config);
    tt::tt_metal::CircularBufferConfig cb_src1_config =
        tt::tt_metal::CircularBufferConfig(cb_size1, {{src1_cb_index, cb_data_format}})
            .set_page_size(src1_cb_index, cb_size1);
    tt::tt_metal::CreateCircularBuffer(program, total_cores, cb_src1_config);
    std::vector<uint32_t> compile_time_args = {
        (std::uint32_t)(source_page_size_bytes % 64 == 0) ? 1 : 0,
        (std::uint32_t)(source_page_size_bytes % 16 == 0) ? 1 : 0,
        src0_cb_index,
        src1_cb_index,
        source_page_size_bytes,
        dest_page_size_bytes};
    tt::tt_metal::TensorAccessorArgs(*src_buffer).append_to(compile_time_args);
    tt::tt_metal::TensorAccessorArgs(*dst_buffer).append_to(compile_time_args);

    tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/reshape_view/device/device/rm_reshape_interleaved.cpp",
        total_cores,
        tt::tt_metal::ReaderDataMovementConfig(compile_time_args));
    uint32_t src2_cb_index = 2;
    uint32_t src3_cb_index = 3;
    tt::tt_metal::KernelHandle reader_kernel_id2 = 0;
    if (can_use_dual_kernel) {
        tt::tt_metal::CircularBufferConfig cb_src2_config =
            tt::tt_metal::CircularBufferConfig(cb_size0 * 2, {{src2_cb_index, cb_data_format}})
                .set_page_size(src2_cb_index, cb_size0);
        tt::tt_metal::CreateCircularBuffer(program, total_cores, cb_src2_config);
        tt::tt_metal::CircularBufferConfig cb_src3_config =
            tt::tt_metal::CircularBufferConfig(cb_size1, {{src3_cb_index, cb_data_format}})
                .set_page_size(src3_cb_index, cb_size1);
        tt::tt_metal::CreateCircularBuffer(program, total_cores, cb_src3_config);
        compile_time_args[2] = src2_cb_index;
        compile_time_args[3] = src3_cb_index;
        reader_kernel_id2 = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/data_movement/reshape_view/device/device/rm_reshape_interleaved.cpp",
            total_cores,
            tt::tt_metal::WriterDataMovementConfig(compile_time_args));
    }
    uint32_t done = 0;
    for (auto core : corerange_to_cores(total_cores, std::nullopt)) {
        if (done == 1) {
            const std::vector<uint32_t> reader_runtime_args = {
                src_buffer->address(), dst_buffer->address(), source_read_size_bytes, 0, 0, 0, 0, 1

            };
            tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);
            if (can_use_dual_kernel) {
                tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id2, core, reader_runtime_args);
            }
        } else {
            // Create the circular buffers

            // set the runtime args
            // set the compile time args
            const uint32_t start_of_read = read_start_page;
            uint32_t end_of_read = read_start_page + responsibility;
            end_of_read = end_of_read < input_log_shape[-2] ? end_of_read : input_log_shape[-2];
            uint32_t pages_for_this_core = end_of_read - start_of_read;
            uint32_t write_jump = (pages_for_this_core * source_page_size_bytes) / dest_page_size_bytes;

            if (can_use_dual_kernel) {
                // Split work in half - determine split point and second write position
                uint32_t mid_read, second_write_pos;
                if (source_page_size_bytes >= dest_page_size_bytes) {
                    // Split by input pages
                    uint32_t half_pages = pages_for_this_core / 2;
                    mid_read = start_of_read + half_pages;
                    second_write_pos = write_start_page + (half_pages * source_page_size_bytes / dest_page_size_bytes);
                } else {
                    // Split by output pages
                    uint32_t total_bytes_for_core = pages_for_this_core * source_page_size_bytes;
                    uint32_t total_output_pages_for_core = total_bytes_for_core / dest_page_size_bytes;
                    uint32_t half_output_pages = total_output_pages_for_core / 2;
                    mid_read = start_of_read + (half_output_pages * dest_page_size_bytes / source_page_size_bytes);
                    second_write_pos = write_start_page + half_output_pages;
                }

                std::vector<uint32_t> runtime_args = {
                    src_buffer->address(),
                    dst_buffer->address(),
                    source_read_size_bytes,
                    start_of_read,
                    mid_read,
                    write_start_page,
                    0,
                    0};

                tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, runtime_args);

                runtime_args[3] = mid_read;
                runtime_args[4] = end_of_read;
                runtime_args[5] = second_write_pos;

                tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id2, core, runtime_args);
            } else {
                // Original single kernel approach
                const std::vector<uint32_t> reader_runtime_args = {
                    src_buffer->address(),
                    dst_buffer->address(),
                    source_read_size_bytes,
                    start_of_read,
                    end_of_read,
                    write_start_page,
                    0,  // write_start_offset removed (always 0)
                    done};
                tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);
            }
            write_start_page += write_jump;
            read_start_page = end_of_read;
            done = (end_of_read == input_log_shape[-2]) ? 1 : 0;
        }
    }

    return {std::move(program), {reader_kernel_id, reader_kernel_id2, can_use_dual_kernel, num_cores_x, num_cores_y}};
}

void ReshapeRMProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const ReshapeParams& operation_attributes,
    const ReshapeInputs& tensor_args,
    Tensor& tensor_return_value) {
    auto& shared_variables = cached_program.shared_variables;
    const auto& reader_kernel_id = shared_variables.reader_kernel_id;
    const auto& reader_kernel_id2 = shared_variables.reader_kernel_id2;
    const auto& can_use_dual_kernel = shared_variables.can_use_dual_kernel;
    const auto& num_cores_x = shared_variables.num_cores_x;
    const auto& num_cores_y = shared_variables.num_cores_y;

    tt::tt_metal::Buffer* src_buffer = tensor_args.input.buffer();
    tt::tt_metal::Buffer* dst_buffer = tensor_return_value.buffer();

    auto& program = cached_program.program;

    CoreRange default_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});
    CoreRangeSet total_cores = operation_attributes.sub_core_grid.has_value()
                                   ? operation_attributes.sub_core_grid.value()
                                   : CoreRangeSet(default_cores);

    for (auto core : corerange_to_cores(total_cores, std::nullopt)) {
        // Update buffer addresses for primary kernel
        {
            auto& runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
            runtime_args[0] = src_buffer->address();  // src_buffer address
            runtime_args[1] = dst_buffer->address();  // dst_buffer address
        }

        // Update buffer addresses for dual kernel if enabled
        if (can_use_dual_kernel) {
            auto& runtime_args = GetRuntimeArgs(program, reader_kernel_id2, core);
            runtime_args[0] = src_buffer->address();  // src_buffer address
            runtime_args[1] = dst_buffer->address();  // dst_buffer address
        }
    }
}

}  // namespace ttnn::operations::data_movement::reshape
