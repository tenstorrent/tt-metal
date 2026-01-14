// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <string>

#include "repeat_and_interleave_eltwise_mul_program_factory.hpp"

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::experimental::ssm::repeat_mul::program {

using namespace tt::constants;
using namespace tt::tt_metal;

namespace {
constexpr uint32_t HIDDEN_SIZE = 5120;
constexpr uint32_t ONE_TILE = 1;
}  // namespace

RepeatAndInterleaveEltwiseMulProgramFactory::cached_program_t RepeatAndInterleaveEltwiseMulProgramFactory::create(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args, Tensor& tensor_return_value) {
    const auto& a = tensor_args.a;
    const auto& b = tensor_args.b;
    auto& output = tensor_return_value;

    const auto& ashape = a.padded_shape();
    const auto& bshape = b.padded_shape();

    tt::tt_metal::Buffer* src0_buffer = a.buffer();
    tt::tt_metal::Buffer* src1_buffer = b.buffer();

    tt::tt_metal::Buffer* out_buffer = output.buffer();
    TT_ASSERT(out_buffer != nullptr, "Output buffer should be allocated on device!");

    tt::DataFormat in0_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    tt::DataFormat in1_data_format = tt::tt_metal::datatype_to_dataformat_converter(b.dtype());
    tt::DataFormat interm_data_format = tt::DataFormat::Float16_b;
    tt::DataFormat output_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());

    uint32_t in0_single_tile_size = tt::tile_size(in0_data_format);
    uint32_t in1_single_tile_size = tt::tile_size(in1_data_format);
    uint32_t interm_single_tile_size = tt::tile_size(interm_data_format);
    uint32_t output_single_tile_size = tt::tile_size(output_data_format);

    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    // Parallelize on bshape[-1]
    auto num_output_blocks_total = bshape[-1] / TILE_WIDTH;
    const bool row_major = false;
    auto device_compute_with_storage_grid_size = a.device()->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_blocks_per_core_group_1, num_blocks_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(device_compute_with_storage_grid_size, num_output_blocks_total, row_major);

    uint32_t g1_numcores = core_group_1.num_cores();
    uint32_t g2_numcores = core_group_2.num_cores();
    std::vector<CoreCoord> cores = grid_to_cores(
        num_cores, device_compute_with_storage_grid_size.x, device_compute_with_storage_grid_size.y, row_major);

    // Create circular buffers
    uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t cb0_tiles = ONE_TILE * 2;  // double buffer
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(cb0_tiles * in0_single_tile_size, {{src0_cb_index, in0_data_format}})
            .set_page_size(src0_cb_index, in0_single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    uint32_t src1_cb_index = tt::CBIndex::c_1;
    uint32_t cb1_tiles = ONE_TILE * 2;  // double buffer
    tt::tt_metal::CircularBufferConfig cb_src1_config =
        tt::tt_metal::CircularBufferConfig(cb1_tiles * in1_single_tile_size, {{src1_cb_index, in1_data_format}})
            .set_page_size(src1_cb_index, in1_single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src1_config);

    uint32_t output_cb_index = 16;
    uint32_t output_cb_tiles = ONE_TILE * 2;  // double buffer
    tt::tt_metal::CircularBufferConfig cb_output_config =
        tt::tt_metal::CircularBufferConfig(
            output_cb_tiles * output_single_tile_size, {{output_cb_index, output_data_format}})
            .set_page_size(output_cb_index, output_single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);

    uint32_t interm_num_tiles = ONE_TILE * 2;  // double buffer
    uint32_t interm_cb_size = interm_num_tiles * interm_single_tile_size;
    uint32_t cb_intermed0_index = tt::CBIndex::c_24;  // cb_in0_transposed
    tt::tt_metal::CircularBufferConfig cb_intermed0_config =
        tt::tt_metal::CircularBufferConfig(interm_cb_size, {{cb_intermed0_index, interm_data_format}})
            .set_page_size(cb_intermed0_index, interm_single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_intermed0_config);

    uint32_t cb_intermed1_index = tt::CBIndex::c_25;  // cb_in1_transposed
    tt::tt_metal::CircularBufferConfig cb_intermed1_config =
        tt::tt_metal::CircularBufferConfig(interm_cb_size, {{cb_intermed1_index, interm_data_format}})
            .set_page_size(cb_intermed1_index, interm_single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_intermed1_config);

    uint32_t cb_intermed2_index = tt::CBIndex::c_26;  // cb_in1_bcast_row
    tt::tt_metal::CircularBufferConfig cb_intermed2_config =
        tt::tt_metal::CircularBufferConfig(interm_cb_size, {{cb_intermed2_index, interm_data_format}})
            .set_page_size(cb_intermed2_index, interm_single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_intermed2_config);

    uint32_t cb_intermed3_index = tt::CBIndex::c_27;  // cb_out_transposed
    tt::tt_metal::CircularBufferConfig cb_intermed3_config =
        tt::tt_metal::CircularBufferConfig(interm_cb_size, {{cb_intermed3_index, interm_data_format}})
            .set_page_size(cb_intermed3_index, interm_single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_intermed3_config);

    // Compile time args
    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)src0_cb_index,
        (std::uint32_t)src1_cb_index,
        (std::uint32_t)cb_intermed1_index,
        (std::uint32_t)cb_intermed2_index,
    };
    tt::tt_metal::TensorAccessorArgs(src0_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(src1_buffer).append_to(reader_compile_time_args);
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t)output_cb_index,
    };
    tt::tt_metal::TensorAccessorArgs(out_buffer).append_to(writer_compile_time_args);
    std::vector<uint32_t> compute_args = {
        (std::uint32_t)src0_cb_index,
        (std::uint32_t)src1_cb_index,
        (std::uint32_t)output_cb_index,
        (std::uint32_t)cb_intermed0_index,
        (std::uint32_t)cb_intermed1_index,
        (std::uint32_t)cb_intermed2_index,
        (std::uint32_t)cb_intermed3_index,
    };

    std::map<std::string, std::string> ssm_eltwise_defines;
    if (ashape[-1] == TILE_WIDTH) {
        ssm_eltwise_defines["REPEAT_IN0"] = "1";
    }
    if (bshape[-1] == HIDDEN_SIZE) {
        ssm_eltwise_defines["REPEAT_INTERLEAVE_IN1"] = "1";
    }

    // Load kernels
    auto reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ssm/repeat_and_interleave_eltwise_mul/device/kernels/"
        "reader_ssm_eltwise_mul.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args, ssm_eltwise_defines));

    auto writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ssm/repeat_and_interleave_eltwise_mul/device/kernels/"
        "writer_ssm_eltwise_mul.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    auto compute_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ssm/repeat_and_interleave_eltwise_mul/device/kernels/"
        "ssm_eltwise_mul.cpp",
        all_cores,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = operation_attributes.math_fidelity,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_args,
            .defines = ssm_eltwise_defines});

    // Store shared variables
    shared_variables_t shared_variables;
    shared_variables.reader_kernel_id = reader_kernel_id;
    shared_variables.writer_kernel_id = writer_kernel_id;
    shared_variables.compute_kernel_id = compute_kernel_id;
    shared_variables.compute_with_storage_grid_size = device_compute_with_storage_grid_size;
    shared_variables.all_cores = all_cores;
    shared_variables.cores = cores;
    shared_variables.num_cores = num_cores;
    shared_variables.g1_numcores = g1_numcores;
    shared_variables.g2_numcores = g2_numcores;
    shared_variables.num_blocks_per_core_group_1 = num_blocks_per_core_group_1;
    shared_variables.num_blocks_per_core_group_2 = num_blocks_per_core_group_2;
    shared_variables.ashape = ashape;
    shared_variables.bshape = bshape;
    shared_variables.hidden_size = HIDDEN_SIZE;

    cached_program_t cached_program{std::move(program), std::move(shared_variables)};

    // Set initial runtime args
    override_runtime_arguments(cached_program, operation_attributes, tensor_args, tensor_return_value);

    return cached_program;
}

void RepeatAndInterleaveEltwiseMulProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t&,
    const tensor_args_t& tensor_args,
    Tensor& tensor_return_value) {
    const auto& a = tensor_args.a;
    const auto& b = tensor_args.b;
    const auto& output = tensor_return_value;

    tt::tt_metal::Buffer* src0_buffer = a.buffer();
    tt::tt_metal::Buffer* src1_buffer = b.buffer();
    tt::tt_metal::Buffer* dst_buffer = output.buffer();

    auto& program = cached_program.program;
    const auto& cores = cached_program.shared_variables.cores;
    const auto& num_cores = cached_program.shared_variables.num_cores;
    const auto& g1_numcores = cached_program.shared_variables.g1_numcores;
    const auto& num_blocks_per_core_group_1 = cached_program.shared_variables.num_blocks_per_core_group_1;
    const auto& num_blocks_per_core_group_2 = cached_program.shared_variables.num_blocks_per_core_group_2;
    const auto& bshape = cached_program.shared_variables.bshape;
    const auto& ashape = cached_program.shared_variables.ashape;
    const auto& hidden_size = cached_program.shared_variables.hidden_size;
    const auto& reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    const auto& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;
    const auto& compute_kernel_id = cached_program.shared_variables.compute_kernel_id;

    // Default reader runtime args
    std::vector<uint32_t> reader_runtime_args = {
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    };

    // Default writer runtime args
    std::vector<uint32_t> writer_runtime_args = {
        0,
        0,
        0,
        0,
        0,
    };

    // Default compute runtime args
    std::vector<uint32_t> compute_runtime_args = {
        0,
        0,
    };

    std::vector<std::vector<uint32_t>> all_reader_runtime_args = {cores.size(), reader_runtime_args};
    std::vector<std::vector<uint32_t>> all_writer_runtime_args = {cores.size(), writer_runtime_args};
    std::vector<std::vector<uint32_t>> all_compute_runtime_args = {cores.size(), compute_runtime_args};

    // Set runtime args
    uint32_t num_blocks_per_core;
    for (uint32_t i = 0, num_blocks_written = 0; i < num_cores; i++) {
        if (i < g1_numcores) {
            num_blocks_per_core = num_blocks_per_core_group_1;
        } else {
            num_blocks_per_core = num_blocks_per_core_group_2;
        }

        // Update core dependent runtime args
        all_reader_runtime_args[i][0] = src0_buffer->address();
        all_reader_runtime_args[i][1] = src1_buffer->address();
        all_reader_runtime_args[i][2] = num_blocks_per_core;
        all_reader_runtime_args[i][3] = num_blocks_written;
        all_reader_runtime_args[i][4] = bshape[2] / TILE_HEIGHT;
        all_reader_runtime_args[i][5] = bshape[-1] / TILE_WIDTH;
        all_reader_runtime_args[i][6] = ashape[-1] / TILE_WIDTH;

        all_writer_runtime_args[i][0] = dst_buffer->address();

        // update writer's num_tiles based on input_b already repeat_interleaved or not
        if (bshape[-1] == hidden_size) {
            all_writer_runtime_args[i][1] = num_blocks_per_core * TILE_WIDTH;
            all_writer_runtime_args[i][2] = num_blocks_written * TILE_WIDTH;
        } else {
            all_writer_runtime_args[i][1] = num_blocks_per_core;
            all_writer_runtime_args[i][2] = num_blocks_written;
        }

        all_writer_runtime_args[i][3] = bshape[2] / TILE_HEIGHT;
        all_writer_runtime_args[i][4] = hidden_size;

        all_compute_runtime_args[i][0] = num_blocks_per_core;
        all_compute_runtime_args[i][1] = bshape[2] / TILE_HEIGHT;

        num_blocks_written += num_blocks_per_core;
    }

    SetRuntimeArgs(program, reader_kernel_id, cores, all_reader_runtime_args);
    SetRuntimeArgs(program, writer_kernel_id, cores, all_writer_runtime_args);
    SetRuntimeArgs(program, compute_kernel_id, cores, all_compute_runtime_args);
}

}  // namespace ttnn::operations::experimental::ssm::repeat_mul::program
