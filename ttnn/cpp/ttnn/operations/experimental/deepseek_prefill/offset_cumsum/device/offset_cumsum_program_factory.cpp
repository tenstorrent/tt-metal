// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <algorithm>

#include "offset_cumsum_program_factory.hpp"
#include "offset_cumsum_device_operation_types.hpp"

#include "ttnn/operations/math.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include "ttnn/operation.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::experimental::prim {

OffsetCumsumProgramFactory::cached_program_t OffsetCumsumProgramFactory::create(
    const OffsetCumsumParams& /*operation_attributes*/, const Tensor& input, Tensor& tensor_return_value) {
    tt::tt_metal::Program program{};

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(tt::tt_metal::DataType::UINT32);

    CoreCoord core = {0, 0};
    CoreRangeSet core_set = CoreRangeSet(std::vector{CoreRange(core, core)});

    const auto& logical_shape = input.logical_shape();
    uint32_t W = logical_shape[-1];
    uint32_t H = logical_shape[-2];

    auto* src_buffer = input.buffer();
    auto* dst_buffer = tensor_return_value.buffer();
    bool src_is_dram = src_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    bool dst_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;

    uint32_t input_page_size = src_buffer->aligned_page_size();
    uint32_t output_page_size = dst_buffer->aligned_page_size();

    uint32_t cb_in0_index = tt::CBIndex::c_0;
    tt::tt_metal::CircularBufferConfig cb_in0_config =
        tt::tt_metal::CircularBufferConfig(input_page_size, {{cb_in0_index, cb_data_format}})
            .set_page_size(cb_in0_index, input_page_size);
    tt::tt_metal::CreateCircularBuffer(program, core_set, cb_in0_config);

    uint32_t cb_out0_index = tt::CBIndex::c_1;
    tt::tt_metal::CircularBufferConfig cb_out0_config =
        tt::tt_metal::CircularBufferConfig(output_page_size, {{cb_out0_index, cb_data_format}})
            .set_page_size(cb_out0_index, output_page_size);
    tt::tt_metal::CreateCircularBuffer(program, core_set, cb_out0_config);

    std::vector<uint32_t> compile_time_args = {
        cb_in0_index,
        cb_out0_index,
        (uint32_t)src_is_dram,
        (uint32_t)dst_is_dram,
        input_page_size,
        output_page_size,
        W,
        H,
    };
    tt::tt_metal::TensorAccessorArgs(src_buffer).append_to(compile_time_args);
    tt::tt_metal::TensorAccessorArgs(dst_buffer).append_to(compile_time_args);

    std::map<std::string, std::string> kernel_defines;
    tt::tt_metal::KernelHandle kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/offset_cumsum/device/kernels/"
        "reader_offset_cumsum_interleaved.cpp",
        core_set,
        tt::tt_metal::ReaderDataMovementConfig(compile_time_args, kernel_defines));

    tt::tt_metal::SetRuntimeArgs(program, kernel_id, core, {src_buffer->address(), dst_buffer->address()});

    return cached_program_t{
        std::move(program),
        {/* kernel_id = */ kernel_id,
         /* core      = */ core}};
}

void OffsetCumsumProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program, const OffsetCumsumParams&, const Tensor& input, Tensor& tensor_return_value) {
    auto& program = cached_program.program;
    const auto& core = cached_program.shared_variables.core;
    const auto& kernel_id = cached_program.shared_variables.kernel_id;

    auto& runtime_args = GetRuntimeArgs(program, kernel_id, core);
    runtime_args[0] = input.buffer()->address();
    runtime_args[1] = tensor_return_value.buffer()->address();
}

}  // namespace ttnn::experimental::prim
