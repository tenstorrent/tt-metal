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

#include "ttnn/operations/data_movement/repeat/device/repeat_program_factory_higher_dim.hpp"

namespace ttnn::prim {

RepeatProgramFactoryHigherDim::cached_program_t RepeatProgramFactoryHigherDim::create(
    const RepeatParams& operation_attributes, const RepeatInputs& tensor_args, Tensor& tensor_return_value) {
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
    uint32_t page_size_bytes = input_log_shape[3] * data_size;
    TT_FATAL(
        page_size_bytes == output_log_shape[3] * data_size,
        "Data size of output does not match requirement for repeat last dim");
    uint32_t read_start_page = 0;
    tt::tt_metal::Buffer* src_buffer = input.buffer();
    tt::tt_metal::Buffer* dst_buffer = output.buffer();
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");
    // Find how many input pages each core is responsible for so that we always start at the beginning of a read and
    // write page Since the logical volumes match, we are guaranteed that the very last page is aligned
    uint32_t number_of_higher_pages = input_log_shape[0];
    uint32_t number_of_lower_pages = input_log_shape[2];
    uint32_t number_of_rep_dim_pages = input_log_shape[1];
    uint32_t cb_size_bytes = (READ_ALIGNMENT * 2) + page_size_bytes;
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
        (std::uint32_t)page_size_bytes, src0_cb_index, src1_cb_index, number_of_lower_pages, number_of_rep_dim_pages};
    tt::tt_metal::TensorAccessorArgs(*src_buffer).append_to(compile_time_args);
    tt::tt_metal::TensorAccessorArgs(*dst_buffer).append_to(compile_time_args);

    tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/repeat/device/kernels/repeat_higher_dim_rm.cpp",
        total_cores,
        tt::tt_metal::ReaderDataMovementConfig(compile_time_args));
    uint32_t done = 0;
    // Determine runtime argumens
    bool divide_on_higher = number_of_higher_pages > number_of_lower_pages;

    uint32_t responsibility_chunk =
        (divide_on_higher ? number_of_higher_pages : number_of_lower_pages) / num_cores_total;
    uint32_t responsibility_mod = (divide_on_higher ? number_of_higher_pages : number_of_lower_pages) % num_cores_total;
    uint32_t core_count = 0;
    for (int core_x = 0; core_x < num_cores_x; core_x++) {
        for (int core_y = 0; core_y < num_cores_y; core_y++) {
            uint32_t responsibility =
                core_count++ < responsibility_mod ? responsibility_chunk + 1 : responsibility_chunk;
            CoreCoord core = {core_x, core_y};
            if (done == 1) {
                const std::vector<uint32_t> reader_runtime_args = {0, 0, 0, 0, 0, 0, 0, 1};
                tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);
            } else if (divide_on_higher) {
                // set the runtime args
                // set the compile time args
                const uint32_t start_of_read = read_start_page;
                uint32_t end_of_read = read_start_page + responsibility;
                end_of_read = end_of_read < number_of_higher_pages ? end_of_read : number_of_higher_pages;

                const std::vector<uint32_t> reader_runtime_args = {
                    src_buffer->address(),
                    dst_buffer->address(),
                    start_of_read,
                    end_of_read,
                    0,
                    number_of_lower_pages,
                    num_repeats,
                    0};
                read_start_page = end_of_read;
                done = (end_of_read == number_of_higher_pages) ? 1 : 0;
                tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);
            } else {
                // set the runtime args
                // set the compile time args
                const uint32_t start_of_read = read_start_page;
                uint32_t end_of_read = read_start_page + responsibility;
                end_of_read = end_of_read < number_of_lower_pages ? end_of_read : number_of_lower_pages;

                const std::vector<uint32_t> reader_runtime_args = {
                    src_buffer->address(),
                    dst_buffer->address(),
                    0,
                    number_of_higher_pages,
                    start_of_read,
                    end_of_read,
                    num_repeats,
                    0};
                read_start_page = end_of_read;
                done = (end_of_read == number_of_lower_pages) ? 1 : 0;
                tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);
            }
        }
    }
    return RepeatProgramFactoryHigherDim::cached_program_t{std::move(program), {reader_kernel_id, total_cores}};
}

void RepeatProgramFactoryHigherDim::override_runtime_arguments(
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

}  // namespace ttnn::prim
