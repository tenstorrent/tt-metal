// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#include "prod_all_program_factory.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::reduction::prod_all::program {

ProdAllProgramFactory::cached_program_t ProdAllProgramFactory::create(
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    Tensor& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;
    using namespace tt::constants;

    const auto& input = tensor_args.input;
    auto& output = tensor_return_value;

    Program program{};

    CoreRange core({0, 0}, {0, 0});

    DataFormat cb_data_format = datatype_to_dataformat_converter(input.dtype());
    uint32_t single_tile_size = tile_size(cb_data_format);

    uint32_t num_tiles = input.physical_volume() / TILE_HW;

    uint32_t num_input_tiles = 2;
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{CBIndex::c_0, cb_data_format}})
            .set_page_size(CBIndex::c_0, single_tile_size);
    CreateCircularBuffer(program, core, cb_src0_config);

    CircularBufferConfig cb_inter_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{CBIndex::c_2, cb_data_format}})
            .set_page_size(CBIndex::c_2, single_tile_size);
    CreateCircularBuffer(program, core, cb_inter_config);

    uint32_t output_cb_index = CBIndex::c_3;
    uint32_t num_output_tiles = 2;
    CircularBufferConfig cb_output_config =
        CircularBufferConfig(num_output_tiles * single_tile_size, {{output_cb_index, cb_data_format}})
            .set_page_size(output_cb_index, single_tile_size);
    CreateCircularBuffer(program, core, cb_output_config);

    auto* src_buffer = input.buffer();
    auto* dst_buffer = output.buffer();

    std::vector<uint32_t> reader_compile_time_args;
    TensorAccessorArgs(*src_buffer).append_to(reader_compile_time_args);
    std::vector<uint32_t> writer_compile_time_args = {static_cast<uint32_t>(output_cb_index)};
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    KernelHandle unary_reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp",
        core,
        ReaderDataMovementConfig{reader_compile_time_args});

    KernelHandle unary_writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        core,
        WriterDataMovementConfig{writer_compile_time_args});

    std::vector<uint32_t> compute_kernel_args = {
        num_tiles,  // per_core_block_cnt
        1           // per_core_block_size
    };

    bool fp32_dest_acc_en = false;
    bool math_approx_mode = true;
    CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/prod/device/kernels/compute/prod_all.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_kernel_args});

    SetRuntimeArgs(program, unary_reader_kernel_id, core, {src_buffer->address(), num_tiles, 0});

    SetRuntimeArgs(program, unary_writer_kernel_id, core, {dst_buffer->address(), /*num_tiles=*/1, 0});

    return {std::move(program), {unary_reader_kernel_id, unary_writer_kernel_id}};
}

void ProdAllProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    Tensor& tensor_return_value) {
    using namespace tt::tt_metal;

    auto& program = cached_program.program;
    auto& shared_variables = cached_program.shared_variables;

    auto* src_buffer = tensor_args.input.buffer();
    auto* dst_buffer = tensor_return_value.buffer();

    CoreCoord core = {0, 0};

    {
        auto& runtime_args = GetRuntimeArgs(program, shared_variables.unary_reader_kernel_id, core);
        runtime_args[0] = src_buffer->address();
    }

    {
        auto& runtime_args = GetRuntimeArgs(program, shared_variables.unary_writer_kernel_id, core);
        runtime_args[0] = dst_buffer->address();
    }
}

}  // namespace ttnn::operations::reduction::prod_all::program
