// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include <cmath>
#include <optional>
#include <variant>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "ttnn/core.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

#include "ttnn/operations/data_movement/repeat/device/repeat_program_factory_last_dim.hpp"

namespace ttnn::operations::data_movement::repeat::program {

RepeatProgramFactoryLastDim::cached_program_t RepeatProgramFactoryLastDim::create(
    const RepeatParams& operation_attributes, const RepeatInputs& tensor_args, Tensor& tensor_return_value) {
    // We are repeating the last dim on a 2D shape
    const auto& input = tensor_args.input;
    const auto& output = tensor_return_value;
    const uint32_t num_repeats = operation_attributes.m_num_repeats;
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
    ttnn::Shape input_log_shape = ttnn::Shape(input.logical_shape().view());
    ttnn::Shape output_log_shape = ttnn::Shape(output.logical_shape().view());
    uint32_t source_page_size_bytes = input_log_shape[-1] * data_size;
    uint32_t dest_page_size_bytes = source_page_size_bytes * num_repeats;
    TT_FATAL(
        dest_page_size_bytes == output_log_shape[-1] * data_size,
        "Data size of output does not match requirement for repeat last dim");
    uint32_t read_start_page = 0;
    tt::tt_metal::Buffer* src_buffer = input.buffer();
    tt::tt_metal::Buffer* dst_buffer = output.buffer();
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");
    // Find how many input pages each core is responsible for so that we always start at the beginning of a read and
    // write page Since the logical volumes match, we are guaranteed that the very last page is aligned
    uint32_t number_of_pages = input_log_shape[-2];
    uint32_t responsibility = ((number_of_pages - 1) / num_cores_total) + 1;
    uint32_t cb_size_bytes = READ_ALIGNMENT * 2 + (source_page_size_bytes & 0xF) == 0 ? source_page_size_bytes
                             : (source_page_size_bytes & 0x7) == 0                    ? source_page_size_bytes * 2
                             : (source_page_size_bytes & 0x3) == 0                    ? source_page_size_bytes * 4
                             : (source_page_size_bytes & 0x1) == 0                    ? source_page_size_bytes * 8
                                                                                      : source_page_size_bytes * 16;
    uint32_t src0_cb_index = 0;
    uint32_t src1_cb_index = 1;
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(cb_size_bytes, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, cb_size_bytes);
    tt::tt_metal::CreateCircularBuffer(program, total_cores, cb_src0_config);
    tt::tt_metal::CircularBufferConfig cb_src1_config =
        tt::tt_metal::CircularBufferConfig(cb_size_bytes, {{src1_cb_index, cb_data_format}})
            .set_page_size(src1_cb_index, cb_size_bytes);
    tt::tt_metal::CreateCircularBuffer(program, total_cores, cb_src1_config);
    std::vector<uint32_t> compile_time_args = {
        (std::uint32_t)source_page_size_bytes, (std::uint32_t)num_repeats, src0_cb_index, src1_cb_index};
    tt::tt_metal::TensorAccessorArgs(*src_buffer).append_to(compile_time_args);
    tt::tt_metal::TensorAccessorArgs(*dst_buffer).append_to(compile_time_args);

    tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/repeat/device/kernels/repeat_last_dim_rm.cpp",
        total_cores,
        tt::tt_metal::ReaderDataMovementConfig(compile_time_args));
    uint32_t done = 0;
    for (int core_x = 0; core_x < num_cores_x; core_x++) {
        for (int core_y = 0; core_y < num_cores_y; core_y++) {
            CoreCoord core = {core_x, core_y};
            if (done == 1) {
                const std::vector<uint32_t> reader_runtime_args = {
                    src_buffer->address(), dst_buffer->address(), 0, 0, 1};
                tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);
            } else {
                const uint32_t start_of_read = read_start_page;
                uint32_t end_of_read = read_start_page + responsibility;
                end_of_read = end_of_read < number_of_pages ? end_of_read : number_of_pages;

                const std::vector<uint32_t> reader_runtime_args = {
                    src_buffer->address(), dst_buffer->address(), start_of_read, end_of_read, 0

                };
                read_start_page = end_of_read;
                done = (end_of_read == input_log_shape[-2]) ? 1 : 0;
                tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);
            }
        }
    }
    return RepeatProgramFactoryLastDim::cached_program_t{std::move(program), {reader_kernel_id, total_cores}};
}

void RepeatProgramFactoryLastDim::override_runtime_arguments(
    cached_program_t& cached_program,
    const RepeatParams& /*operation_attributes*/,
    const RepeatInputs& tensor_args,
    Tensor& tensor_return_value) {
    auto& program = cached_program.program;
    auto& shared_vars = cached_program.shared_variables;

    auto& reader_kernel_id = shared_vars.reader_kernel_id;
    auto& total_cores = shared_vars.total_cores;

    const auto& input = tensor_args.input;
    const auto& output = tensor_return_value;

    auto& runtime_args_by_core = GetRuntimeArgs(program, reader_kernel_id);
    for (const auto& core : total_cores) {
        auto& runtime_args = runtime_args_by_core[core.x][core.y];
        runtime_args.at(0) = input.buffer()->address();
        runtime_args.at(1) = output.buffer()->address();
    }
}

}  // namespace ttnn::operations::data_movement::repeat::program
