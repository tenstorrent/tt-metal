// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "fill_rm_program_factory.hpp"
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/data_movement/common/common.hpp"

namespace ttnn::operations::data_movement::fill_rm::program {

FillRMProgramFactory::cached_program_t FillRMProgramFactory::create(
    const FillRmParams& operation_attributes, const FillRmInputs& tensor_args, Tensor& tensor_return_value) {
    using namespace tt::tt_metal;

    const Tensor& input = tensor_args.input;
    Tensor& output = tensor_return_value;
    const uint32_t N = operation_attributes.N;
    const uint32_t C = operation_attributes.C;
    const uint32_t H = operation_attributes.H;
    const uint32_t W = operation_attributes.W;
    const uint32_t hFill = operation_attributes.hFill;
    const uint32_t wFill = operation_attributes.wFill;
    const float val_hi = operation_attributes.val_hi;
    const float val_lo = operation_attributes.val_lo;

    Program program = CreateProgram();
    const CoreRange core({0, 0}, {0, 0});

    const tt::DataFormat cb_data_format = datatype_to_dataformat_converter(input.dtype());
    const uint32_t single_tile_size = tt::tile_size(cb_data_format);

    Buffer* dst_buffer = output.buffer();
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    const uint32_t num_cb_tiles = 16;
    TT_FATAL(
        W < 1024 * num_cb_tiles,
        "Width (W) must be less than {} for kernel simplification. Got W={}, num_cb_tiles={}",
        1024 * num_cb_tiles,
        W,
        num_cb_tiles);

    const CircularBufferConfig cb_src0_config =
        CircularBufferConfig(num_cb_tiles * single_tile_size, {{0, cb_data_format}}).set_page_size(0, single_tile_size);
    CreateCircularBuffer(program, core, cb_src0_config);

    const CircularBufferConfig cb_src1_config =
        CircularBufferConfig(num_cb_tiles * single_tile_size, {{1, cb_data_format}}).set_page_size(1, single_tile_size);
    CreateCircularBuffer(program, core, cb_src1_config);

    std::vector<uint32_t> reader_compile_time_args;
    TensorAccessorArgs(*dst_buffer).append_to(reader_compile_time_args);

    const KernelHandle reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/fill_rm/device/kernels/dataflow/fill_rm_interleaved.cpp",
        core,
        ReaderDataMovementConfig(reader_compile_time_args));

    SetRuntimeArgs(
        program,
        reader_kernel_id,
        core,
        {dst_buffer->address(),
         uint32_t(N * C),
         uint32_t(H),
         uint32_t(W),
         uint32_t(hFill),
         uint32_t(wFill),
         uint32_t(std::bit_cast<uint16_t>(bfloat16(val_hi))),
         uint32_t(std::bit_cast<uint16_t>(bfloat16(val_lo)))});

    return cached_program_t{std::move(program), shared_variables_t{.kernel_id = reader_kernel_id}};
}

void FillRMProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const FillRmParams& /*operation_attributes*/,
    const FillRmInputs& /*tensor_args*/,
    Tensor& tensor_return_value) {
    using namespace tt::tt_metal;

    Buffer* dst_buffer = tensor_return_value.buffer();

    Program& program = cached_program.program;
    const KernelHandle kernel_id = cached_program.shared_variables.kernel_id;

    const CoreCoord core = {0, 0};

    auto& runtime_args = GetRuntimeArgs(program, kernel_id, core);
    runtime_args[0] = dst_buffer->address();
}

}  // namespace ttnn::operations::data_movement::fill_rm::program
