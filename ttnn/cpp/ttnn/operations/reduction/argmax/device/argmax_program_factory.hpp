// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>

#include "ttnn/deprecated/tt_dnn/op_library/math.hpp"
#include "ttnn/operations/core/work_split/work_split.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

namespace ttnn::operations::reduction::detail {

using namespace tt::constants;

operation::ProgramWithCallbacks argmax_multi_core(
    const Tensor &input, const Tensor &output, const std::optional<uint32_t> dim) {
    tt::tt_metal::Program program{};

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.get_dtype());
    uint32_t input_unit_size = input.get_legacy_shape()[-1] * input.element_size();
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    uint32_t output_unit_size = output.get_legacy_shape()[-1] * output.element_size();

    tt::tt_metal::Device *device = output.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_units = 1;  // single-core
    auto [num_cores, all_cores, core_group_1, core_group_2, num_units_per_core_group_1, num_units_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, num_units);

    const auto &input_shape = input.get_legacy_shape();
    const uint32_t B = input_shape[0];
    const uint32_t C = input_shape[1];
    const uint32_t H = input_shape[2];
    const uint32_t W = input_shape[3];

    uint32_t src0_cb_index = tt::CB::c_in0;
    uint32_t num_input_units = W;
    uint32_t aligned_input_unit_size = round_up_to_mul32(input_unit_size * num_input_units);
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(aligned_input_unit_size, {{src0_cb_index, input_cb_data_format}})
            .set_page_size(src0_cb_index, aligned_input_unit_size);
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    uint32_t intermed0_cb_index = tt::CB::c_intermed0;
    uint32_t num_intermed0_units = B * C * H * W;
    // TODO: output stick size should be output dim tensor innermost dim
    tt::tt_metal::CircularBufferConfig intermed0_cb_config =
        tt::tt_metal::CircularBufferConfig(
            num_intermed0_units * output.element_size(), {{intermed0_cb_index, output_cb_data_format}})
            .set_page_size(intermed0_cb_index, output.element_size());  /// page size shouldn't matter here
    auto cb_intermed0 = tt::tt_metal::CreateCircularBuffer(program, all_cores, intermed0_cb_config);

    /* NO WRITER FOR NOW
    uint32_t output_cb_index = 16; // same as input cb
    uint32_t num_output_units = 2;
    uint32_t aligned_output_unit_size = round_up_to_mul32(output_unit_size);
    tt::tt_metal::CircularBufferConfig output_cb_config = tt::tt_metal::CircularBufferConfig(num_output_units *
    aligned_output_unit_size, {{output_cb_index, output_cb_data_format}}) .set_page_size(output_cb_index,
    aligned_output_unit_size); auto cb_output = tt::tt_metal::CreateCircularBuffer(program, all_cores,
    output_cb_config);
    */

    auto src_buffer = input.buffer();
    auto dst_buffer = output.buffer();
    bool src_is_dram = src_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    bool dst_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;

    std::vector<uint32_t> reader_compile_time_args = {
        src0_cb_index,
        intermed0_cb_index,
        src_is_dram,
        dst_is_dram,
        input_unit_size,
        output_unit_size,
        B,
        C,
        H,
        W,
        dim.value_or(0),
        (uint32_t)(not dim.has_value()),
    };

    std::map<string, string> kernel_defines;
    tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/argmax/device/kernels/reader_argmax_interleaved.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args, kernel_defines));

    uint32_t g1_numcores = core_group_1.num_cores();
    uint32_t g2_numcores = core_group_2.num_cores();
    auto cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, false);

    for (uint32_t i = 0; i < cores.size(); ++i) {
        const CoreCoord &core = cores.at(i);

        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, {src_buffer->address(), dst_buffer->address()});
    }

    auto override_runtime_args_callback = [reader_kernel_id, cores](
                                              const Program &program,
                                              const std::vector<Buffer *> &input_buffers,
                                              const std::vector<Buffer *> &output_buffers) {
        auto src_buffer = input_buffers.at(0);

        auto dst_buffer = output_buffers.at(0);

        for (const auto &core : cores) {
            {
                auto &runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
                runtime_args[0] = src_buffer->address();
                runtime_args[1] = dst_buffer->address();
            }
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

}  // namespace ttnn::operations::reduction::detail
