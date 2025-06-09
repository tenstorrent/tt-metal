// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <math.h>

#include "upsample_op.hpp"
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/math.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/math.hpp>

#include <tt_stl/reflection.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::upsample {
using namespace tt;
operation::ProgramWithCallbacks upsample_single_core(
    const Tensor& input, Tensor& output, const uint32_t scale_factor_h, const uint32_t scale_factor_w) {
    Program program{};
    CoreRange core({0, 0}, {0, 0});

    tt::DataFormat input_cb_data_format = tt_metal::datatype_to_dataformat_converter(input.dtype());
    uint32_t input_unit_size = input.padded_shape()[-1] * input.element_size();
    tt::DataFormat output_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t output_unit_size = output.padded_shape()[-1] * output.element_size();

    uint32_t output_num_units = output.physical_volume() / output.padded_shape()[-1];  // N*H*W for outout
    uint32_t input_num_units = input.physical_volume() / input.padded_shape()[-1];     // N*H*W for input

    auto output_shape = output.padded_shape();
    // This should allocate a DRAM buffer on the device
    tt_metal::IDevice* device = output.device();

    // circulat buffer for input
    uint32_t next_cb_index = CBIndex::c_0;
    uint32_t src0_cb_index = next_cb_index++;
    uint32_t num_input_units = 2;
    uint32_t aligned_input_unit_size = tt::round_up(input_unit_size, hal::get_dram_alignment());

    tt::tt_metal::create_cb(
        src0_cb_index, program, core, aligned_input_unit_size, num_input_units, input_cb_data_format);

    // circulat buffer same for input and output. No compute kernels.
    uint32_t output_cb_index = src0_cb_index;  // same as input cb

    auto src_buffer = input.buffer();
    auto dst_buffer = output.buffer();
    bool src_is_dram = src_buffer->buffer_type() == tt_metal::BufferType::DRAM;
    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM;

    /*
    The data layout is mapped in DRAM as follows:
        number of channel = stick size
        NHW is number of sticks
    */

    std::vector<uint32_t> reader_compile_time_args, writer_compile_time_args;
    bool src_stick_size_is_power_of_two = is_power_of_two_at_least_32(input_unit_size);
    uint32_t src_log2_stick_size = src_stick_size_is_power_of_two ? (std::uint32_t)log2(input_unit_size) : 0;
    reader_compile_time_args = {
        (std::uint32_t)src0_cb_index,
        (std::uint32_t)src_is_dram,
        (std::uint32_t)src_stick_size_is_power_of_two,
        (std::uint32_t)src_log2_stick_size};

    bool dst_stick_size_is_power_of_two = is_power_of_two_at_least_32(output_unit_size);
    uint32_t dst_log2_stick_size = dst_stick_size_is_power_of_two ? (std::uint32_t)log2(output_unit_size) : 0;
    writer_compile_time_args = {
        (std::uint32_t)output_cb_index,
        (std::uint32_t)dst_is_dram,
        (std::uint32_t)dst_stick_size_is_power_of_two,
        (std::uint32_t)dst_log2_stick_size};

    std::map<string, string> kernel_defines;
    tt_metal::KernelHandle unary_reader_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/pool/upsample/device/kernels/dataflow/"
        "reader_upsample_unary_stick_layout_interleaved_start_id.cpp",
        core,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args, kernel_defines));

    tt_metal::KernelHandle unary_writer_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/pool/upsample/device/kernels/dataflow/"
        "writer_upsample_unary_stick_layout_interleaved_start_id.cpp",
        core,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args, kernel_defines));

    SetRuntimeArgs(program, unary_reader_kernel_id, core, {src_buffer->address(), input_unit_size, input_num_units});

    SetRuntimeArgs(
        program,
        unary_writer_kernel_id,
        core,
        {dst_buffer->address(),
         input_unit_size,
         input_num_units,
         (uint32_t)scale_factor_h,
         (uint32_t)scale_factor_w,
         (uint32_t)output_shape[1],
         (uint32_t)output_shape[2]});

    auto override_runtime_args_callback = [unary_reader_kernel_id, unary_writer_kernel_id](
                                              const void* operation,
                                              Program& program,
                                              const std::vector<Tensor>& input_tensors,
                                              const std::vector<std::optional<const Tensor>>&,
                                              const std::vector<Tensor>& output_tensors) {
        auto src_buffer = input_tensors.at(0).buffer();

        auto dst_buffer = output_tensors.at(0).buffer();

        CoreCoord core = {0, 0};

        {
            auto& runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, core);
            runtime_args[0] = src_buffer->address();
        }

        {
            auto& runtime_args = GetRuntimeArgs(program, unary_writer_kernel_id, core);
            runtime_args[0] = dst_buffer->address();
        }
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_args_callback};
}

}  // namespace ttnn::operations::upsample
