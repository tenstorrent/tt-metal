// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <math.h>
#include <string>

#include "tt-metalium/work_split.hpp"
#include "upsample_op.hpp"
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/math.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/math.hpp>

#include <tt_stl/reflection.hpp>

namespace ttnn::operations::upsample {
tt::tt_metal::operation::ProgramWithCallbacks upsample_multi_core_interleaved(
    const Tensor& input, Tensor& output, const uint32_t scale_factor_h, const uint32_t scale_factor_w) {
    tt::tt_metal::Program program{};

    const tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    const uint32_t input_unit_size = input.padded_shape()[-1] * input.element_size();
    const tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    const uint32_t output_unit_size = output.padded_shape()[-1] * output.element_size();

    const uint32_t output_num_units = output.physical_volume() / output.padded_shape()[-1];  // N*H*W for outout
    const uint32_t input_num_units = input.physical_volume() / input.padded_shape()[-1];     // N*H*W for input

    const auto& output_shape = output.padded_shape();
    // This should allocate a DRAM buffer on the device
    tt::tt_metal::IDevice* const device = output.device();

    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_cores_x = compute_with_storage_grid_size.x;
    const uint32_t num_cores_y = compute_with_storage_grid_size.y;

    const auto
        [num_cores, all_cores, core_group_1, core_group_2, num_sticks_per_core_group_1, num_sticks_per_core_group_2] =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, input_num_units);

    // circular buffer for input
    uint32_t next_cb_index = tt::CBIndex::c_0;
    const uint32_t src0_cb_index = next_cb_index++;
    const uint32_t num_input_units = 2;
    const uint32_t aligned_input_unit_size = tt::round_up(input_unit_size, tt::tt_metal::hal::get_dram_alignment());

    tt::tt_metal::create_cb(
        src0_cb_index, program, all_cores, aligned_input_unit_size, num_input_units, input_cb_data_format);

    // circulat buffer same for input and output. No compute kernels.
    const uint32_t output_cb_index = src0_cb_index;  // same as input cb

    const auto src_buffer = input.buffer();
    const auto dst_buffer = output.buffer();
    const bool src_is_dram = src_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    const bool dst_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;

    /*
    The data layout is mapped in DRAM as follows:
        number of channel = stick size
        NHW is number of sticks
    */

    std::vector<uint32_t> reader_compile_time_args, writer_compile_time_args;
    const bool src_stick_size_is_power_of_two = tt::tt_metal::is_power_of_two_at_least_32(input_unit_size);
    const uint32_t src_log2_stick_size = src_stick_size_is_power_of_two ? (std::uint32_t)log2(input_unit_size) : 0;
    reader_compile_time_args = {
        (std::uint32_t)src0_cb_index,
        (std::uint32_t)src_is_dram,
        (std::uint32_t)input_unit_size,
        (std::uint32_t)src_stick_size_is_power_of_two,
        (std::uint32_t)src_log2_stick_size};

    const uint32_t block_height = 1;                // since input is row major, blocks are just one row tall
    const uint32_t num_units_per_output_stick = 1;  // 1 page in out_cb is needed to get a whole output stick

    const bool dst_stick_size_is_power_of_two = tt::tt_metal::is_power_of_two_at_least_32(output_unit_size);
    const uint32_t dst_log2_stick_size = dst_stick_size_is_power_of_two ? (std::uint32_t)log2(output_unit_size) : 0;
    writer_compile_time_args = {
        (std::uint32_t)output_cb_index,
        (std::uint32_t)dst_is_dram,
        (std::uint32_t)input_unit_size,
        (std::uint32_t)dst_stick_size_is_power_of_two,
        (std::uint32_t)dst_log2_stick_size,
        (std::uint32_t)scale_factor_h,
        (std::uint32_t)scale_factor_w,
        (std::uint32_t)output_shape[1],
        (std::uint32_t)output_shape[2],
        (std::uint32_t)block_height,
        num_units_per_output_stick};

    const std::map<std::string, std::string> kernel_defines;
    const tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/pool/upsample/device/kernels/dataflow/"
        "reader_upsample_unary_stick_layout_interleaved_start_id.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args, kernel_defines));

    const tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/pool/upsample/device/kernels/dataflow/writer_upsample_interleaved.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args, kernel_defines));

    std::vector<uint32_t> reader_rt_arguments{
        src_buffer->address(),
        0,  // set in loop, num of sticks on core
        0   // set in loop, start_id of stick in core
    };

    std::vector<uint32_t> writer_rt_arguments{
        dst_buffer->address(),
        0,  // set in loop, num of sticks on core
        0   // set in loop, stard_id of stick on core
    };

    for (uint32_t i = 0, num_sticks_written = 0; i < num_cores; i++) {
        const CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_sticks_per_core = 0;
        if (core_group_1.contains(core)) {
            num_sticks_per_core = num_sticks_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_sticks_per_core = num_sticks_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }

        reader_rt_arguments[1] = num_sticks_per_core;
        reader_rt_arguments[2] = num_sticks_written;

        writer_rt_arguments[1] = num_sticks_per_core;
        writer_rt_arguments[2] = num_sticks_written;

        tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_rt_arguments);

        tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_rt_arguments);

        num_sticks_written += num_sticks_per_core;
    }

    auto override_runtime_args_callback = [unary_reader_kernel_id, unary_writer_kernel_id, num_cores, num_cores_y](
                                              const void* operation,
                                              tt::tt_metal::Program& program,
                                              const std::vector<Tensor>& input_tensors,
                                              const std::vector<std::optional<const Tensor>>&,
                                              const std::vector<Tensor>& output_tensors) {
        const auto src_buffer = input_tensors.at(0).buffer();
        const auto dst_buffer = output_tensors.at(0).buffer();

        for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++) {
            const CoreCoord core = {i / num_cores_y, i % num_cores_y};
            {
                auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, unary_reader_kernel_id, core);
                runtime_args[0] = src_buffer->address();
            }
            {
                auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, unary_writer_kernel_id, core);
                runtime_args[0] = dst_buffer->address();
            }
        }
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_args_callback};
}

}  // namespace ttnn::operations::upsample
