// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <algorithm>

#include "plusone_program_factory.hpp"
#include "plusone_device_operation_types.hpp"

#include "ttnn/operations/math.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include "ttnn/operation.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::experimental::plusone::program {

PlusOneProgramFactory::cached_program_t PlusOneProgramFactory::create(
    const PlusoneParams& operation_attributes,
    const PlusoneInputs& tensor_args,
    tensor_return_value_t& /*tensor_return_value*/) {
    tt::tt_metal::Program program{};
    const auto& input = tensor_args.input;
    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    uint32_t input_unit_size = input.element_size();

    CoreRangeSet all_cores = CoreRangeSet(std::vector{CoreRange({0, 0}, {0, 0})});
    uint32_t num_cores = 1;  // single-core

    if (operation_attributes.sub_core_grids.has_value()) {
        all_cores = operation_attributes.sub_core_grids.value();
        num_cores = all_cores.num_cores();
    }

    const auto& input_shape = input.padded_shape();
    uint32_t W = input_shape[-1];
    uint32_t H = 1;
    if (!input.is_sharded() && input_shape.size() > 1) {
        for (uint32_t i = 0; i < input_shape.size() - 1; ++i) {
            H *= input_shape[i];
        }
    }

    uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t num_input_units = W;
    uint32_t aligned_input_page_size = round_up_to_mul32(num_input_units * input_unit_size);
    auto* src_buffer = input.buffer();
    bool src_is_dram = src_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;

    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(aligned_input_page_size, {{src0_cb_index, input_cb_data_format}})
            .set_page_size(src0_cb_index, aligned_input_page_size);
    if (input.is_sharded()) {
        cb_src0_config.set_globally_allocated_address(*src_buffer);
    }
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    std::vector<uint32_t> reader_compile_time_args = {
        src0_cb_index, src_is_dram, aligned_input_page_size, W, H, operation_attributes.skip_negative_entries};
    tt::tt_metal::TensorAccessorArgs(src_buffer).append_to(reader_compile_time_args);
    std::map<std::string, std::string> kernel_defines;
    tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/plusone/device/kernels/reader_plusone_interleaved.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args, kernel_defines));

    auto cores = corerange_to_cores(all_cores, num_cores, true);

    for (const auto& core : cores) {
        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, {src_buffer->address()});
    }

    return cached_program_t{
        std::move(program),
        {/* reader_kernel_id = */ reader_kernel_id,
         /* cores            = */ cores}};
}

void PlusOneProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program, const PlusoneParams&, const PlusoneInputs& tensor_args, tensor_return_value_t&) {
    auto* src_buffer = tensor_args.input.buffer();

    auto& program = cached_program.program;
    const auto& cores = cached_program.shared_variables.cores;
    const auto& reader_kernel_id = cached_program.shared_variables.reader_kernel_id;

    for (const auto& core : cores) {
        auto& runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
        runtime_args[0] = src_buffer->address();
    }
}

}  // namespace ttnn::operations::experimental::plusone::program
