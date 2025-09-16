// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <math.h>

#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/data_movement/reshape_view/reshape_common.hpp"

#include <optional>
#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/core.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/types.hpp"
#include "ttnn/decorators.hpp"

#include "reshape_program_factory.hpp"

#define MASK_64 0xFFFFFFFFFFFFFFC0
#define OFFSET_64 0x000000000000003F
#define MASK_16 0xFFFFFFFFFFFFFFF0
#define OFFSET_16 0x000000000000000F

namespace ttnn::operations::data_movement::reshape {

tt::tt_metal::operation::ProgramWithCallbacks rm_reshape_preparer_single_risk(
    const Tensor& input, const Tensor& output) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();
    // get datum size
    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    const uint32_t data_size = input.element_size();
    tt::tt_metal::IDevice* device = input.device();
    // Multi device pre-computation
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_cores_total = num_cores_x * num_cores_y;
    CoreRange total_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});
    auto input_log_shape = input.logical_shape();
    auto output_log_shape = output.logical_shape();
    log_debug(tt::LogOp, "row major reshape");
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
    const uint32_t write_jump = (responsibility * source_page_size_bytes) / dest_page_size_bytes;
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
    for (int core_x = 0; core_x < num_cores_x; core_x++) {
        for (int core_y = 0; core_y < num_cores_y; core_y++) {
            CoreCoord core = {core_x, core_y};
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

                if (can_use_dual_kernel) {
                    // Split work in half - determine split point and second write position
                    uint32_t mid_read, second_write_pos;
                    if (source_page_size_bytes >= dest_page_size_bytes) {
                        // Split by input pages
                        uint32_t half_responsibility = responsibility / 2;
                        mid_read = start_of_read + half_responsibility;
                        second_write_pos =
                            write_start_page + (half_responsibility * source_page_size_bytes / dest_page_size_bytes);
                    } else {
                        // Split by output pages
                        uint32_t half_output_pages =
                            ((responsibility * source_page_size_bytes) / dest_page_size_bytes) / 2;
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
    }
    auto override_runtime_args_callback =
        [reader_kernel_id, reader_kernel_id2, can_use_dual_kernel, num_cores_x, num_cores_y](
            const void* operation,
            const tt::tt_metal::Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>&,
            const std::vector<Tensor>& output_tensors) {
            tt::tt_metal::Buffer* src_buffer = input_tensors.at(0).buffer();
            tt::tt_metal::Buffer* dst_buffer = output_tensors.at(0).buffer();

            for (int core_x = 0; core_x < num_cores_x; core_x++) {
                for (int core_y = 0; core_y < num_cores_y; core_y++) {
                    CoreCoord core = {core_x, core_y};

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
        };
    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_args_callback};
}

tt::tt_metal::operation::ProgramWithCallbacks rm_reshape_preparer(const Tensor& input, const Tensor& output) {
    return rm_reshape_preparer_single_risk(input, output);
}

};  // namespace ttnn::operations::data_movement::reshape
