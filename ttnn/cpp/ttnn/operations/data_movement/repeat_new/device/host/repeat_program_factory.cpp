// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <math.h>

#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

#include <optional>
#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/core.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/types.hpp"
#include "ttnn/decorators.hpp"

namespace ttnn::operations::data_movement::repeat {

tt::tt_metal::operation::ProgramWithCallbacks rm_repeater(
    // We are repeating the second dim on a 4D shape
    const Tensor& input,
    uint32_t num_repeats,
    const Tensor& output) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();
    // get datum size
    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.get_dtype());
    const uint32_t data_size = input.element_size();
    tt::tt_metal::Device* device = input.device();
    // Multi device pre-computation
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_cores_total = num_cores_x * num_cores_y;
    CoreRange total_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});
    ttnn::Shape input_log_shape = ttnn::Shape(input.get_logical_shape().view());
    ttnn::Shape output_log_shape = ttnn::Shape(output.get_logical_shape().view());
    tt::log_debug("row major reshape");
    tt::log_debug("input shape: {}", input_log_shape);
    tt::log_debug("output shape: {}", output_log_shape);
    tt::log_debug("data size: {}", data_size);
    uint32_t page_size_bytes = input_log_shape[3] * data_size;
    TT_ASSERT(
        page_size_bytes == output_log_shape[3] * data_size,
        "Data size of output does not match requirement for repeat last dim");
    uint32_t read_start_page = 0;
    tt::tt_metal::Buffer* src_buffer = input.buffer();
    tt::tt_metal::Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");
    // Find how many input pages each core is responsible for so that we always start at the begining of a read and
    // write page Since the logical volumes match, we are guaranteed that the very last page is aligned
    uint32_t number_of_higher_pages = input_log_shape[0];
    uint32_t number_of_lower_pages = input_log_shape[2];
    uint32_t number_of_rep_dim_pages = input_log_shape[1];
    uint32_t src0_is_dram = src_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    uint32_t cb_size_bytes = 128 + page_size_bytes;
    uint32_t src0_cb_index = 0;
    uint32_t src1_cb_index = 1;
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(cb_size_bytes, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, cb_size_bytes);
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, total_cores, cb_src0_config);
    tt::tt_metal::CircularBufferConfig cb_src1_config =
        tt::tt_metal::CircularBufferConfig(cb_size_bytes, {{src1_cb_index, cb_data_format}})
            .set_page_size(src1_cb_index, cb_size_bytes);

    auto cb_src1 = tt::tt_metal::CreateCircularBuffer(program, total_cores, cb_src1_config);
    bool page_is_pow_2 = tt::tt_metal::is_power_of_two_at_least_32(page_size_bytes);
    uint32_t page_pow_2 = page_is_pow_2 ? (std::uint32_t)std::log2(page_size_bytes) : 0;
    std::vector<uint32_t> compile_time_args = {
        (std::uint32_t)src0_is_dram,
        (std::uint32_t)page_size_bytes,
        src0_cb_index,
        src1_cb_index,
        page_is_pow_2,
        page_pow_2,
        number_of_lower_pages,
        number_of_higher_pages,
        number_of_rep_dim_pages};

    tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/repeat_new/device/device/repeat_higher_dim_rm.cpp",
        total_cores,
        tt::tt_metal::ReaderDataMovementConfig(compile_time_args));
    uint32_t done = 0;
    // Determine runtime argumens
    bool divide_on_higher = number_of_higher_pages > number_of_lower_pages;
    uint32_t responsibility = divide_on_higher ? ((number_of_higher_pages - 1) / number_of_higher_pages) + 1
                                               : ((number_of_lower_pages - 1) / number_of_lower_pages) + 1;
    for (int core_x = 0; core_x < num_cores_x; core_x++) {
        for (int core_y = 0; core_y < num_cores_y; core_y++) {
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
                done = (end_of_read == number_of_higher_pages) ? 1 : 0;
                tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);
            }
        }
    }
    return {.program = std::move(program)};
}

};  // namespace ttnn::operations::data_movement::repeat
