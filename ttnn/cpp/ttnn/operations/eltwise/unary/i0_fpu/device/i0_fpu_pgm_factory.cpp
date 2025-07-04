// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>

#include "i0_fpu_pgm_factory.hpp"

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/host_api.hpp>

namespace ttnn::operations::unary::program {

static const std::string compute_root = "ttnn/cpp/ttnn/operations/eltwise/unary/i0_fpu/device/kernels/compute/";

using namespace tt::constants;

I0FpuProgramFactory::cached_program_t I0FpuProgramFactory::create(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input = tensor_args.input;

    tt::tt_metal::Program program{};

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    uint32_t single_tile_size = tt::tt_metal::detail::TileSize(cb_data_format);
    tt::DataFormat cb_data_format_output = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t single_tile_size_output = tt::tt_metal::detail::TileSize(cb_data_format_output);

    uint32_t num_tiles = input.physical_volume() / tt::constants::TILE_HW;

    tt::tt_metal::IDevice* device = input.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_tiles);

    // input buffer
    uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t num_input_tiles = 2;
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, single_tile_size);
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    // input_squared buffer
    uint32_t src1_cb_index = tt::CBIndex::c_1;
    tt::tt_metal::CircularBufferConfig cb_src1_config =
        tt::tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src1_cb_index, cb_data_format}})
            .set_page_size(src1_cb_index, single_tile_size);
    auto cb_src1 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src1_config);

    // coeff0 buffer
    uint32_t coeff0_cb_index = tt::CBIndex::c_3;
    tt::tt_metal::CircularBufferConfig cb_coeff0_config =
        tt::tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{coeff0_cb_index, cb_data_format}})
            .set_page_size(coeff0_cb_index, single_tile_size);
    auto cb_coeff0 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_coeff0_config);

    // coeff1 buffer
    uint32_t coeff1_cb_index = tt::CBIndex::c_4;
    tt::tt_metal::CircularBufferConfig cb_coeff1_config =
        tt::tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{coeff1_cb_index, cb_data_format}})
            .set_page_size(coeff1_cb_index, single_tile_size);
    auto cb_coeff1 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_coeff1_config);

    // coeff2 buffer
    uint32_t coeff2_cb_index = tt::CBIndex::c_5;
    tt::tt_metal::CircularBufferConfig cb_coeff2_config =
        tt::tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{coeff2_cb_index, cb_data_format}})
            .set_page_size(coeff2_cb_index, single_tile_size);
    auto cb_coeff2 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_coeff2_config);

    // coeff3 buffer
    uint32_t coeff3_cb_index = tt::CBIndex::c_6;
    tt::tt_metal::CircularBufferConfig cb_coeff3_config =
        tt::tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{coeff3_cb_index, cb_data_format}})
            .set_page_size(coeff3_cb_index, single_tile_size);
    auto cb_coeff3 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_coeff3_config);

    // coeff4 buffer
    uint32_t coeff4_cb_index = tt::CBIndex::c_7;
    tt::tt_metal::CircularBufferConfig cb_coeff4_config =
        tt::tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{coeff4_cb_index, cb_data_format}})
            .set_page_size(coeff4_cb_index, single_tile_size);
    auto cb_coeff4 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_coeff4_config);

    // coeff5 buffer
    uint32_t coeff5_cb_index = tt::CBIndex::c_8;
    tt::tt_metal::CircularBufferConfig cb_coeff5_config =
        tt::tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{coeff5_cb_index, cb_data_format}})
            .set_page_size(coeff5_cb_index, single_tile_size);
    auto cb_coeff5 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_coeff5_config);

    // coeff6 buffer
    uint32_t coeff6_cb_index = tt::CBIndex::c_9;
    tt::tt_metal::CircularBufferConfig cb_coeff6_config =
        tt::tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{coeff6_cb_index, cb_data_format}})
            .set_page_size(coeff6_cb_index, single_tile_size);
    auto cb_coeff6 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_coeff6_config);

    // coeff7 buffer
    uint32_t coeff7_cb_index = tt::CBIndex::c_10;
    tt::tt_metal::CircularBufferConfig cb_coeff7_config =
        tt::tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{coeff7_cb_index, cb_data_format}})
            .set_page_size(coeff7_cb_index, single_tile_size);
    auto cb_coeff7 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_coeff7_config);

    // coeff8 buffer
    uint32_t coeff8_cb_index = tt::CBIndex::c_11;
    tt::tt_metal::CircularBufferConfig cb_coeff8_config =
        tt::tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{coeff8_cb_index, cb_data_format}})
            .set_page_size(coeff8_cb_index, single_tile_size);
    auto cb_coeff8 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_coeff8_config);

    // coeff9 buffer
    uint32_t coeff9_cb_index = tt::CBIndex::c_12;
    tt::tt_metal::CircularBufferConfig cb_coeff9_config =
        tt::tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{coeff9_cb_index, cb_data_format}})
            .set_page_size(coeff9_cb_index, single_tile_size);
    auto cb_coeff9 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_coeff9_config);

    // coeff10 buffer
    uint32_t coeff10_cb_index = tt::CBIndex::c_13;
    tt::tt_metal::CircularBufferConfig cb_coeff10_config =
        tt::tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{coeff10_cb_index, cb_data_format}})
            .set_page_size(coeff10_cb_index, single_tile_size);
    auto cb_coeff10 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_coeff10_config);

    // one_cb buffer
    uint32_t one_cb_index = tt::CBIndex::c_14;
    tt::tt_metal::CircularBufferConfig cb_one_config =
        tt::tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{one_cb_index, cb_data_format}})
            .set_page_size(one_cb_index, single_tile_size);
    auto cb_one = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_one_config);

    // output buffer
    uint32_t output_cb_index = tt::CBIndex::c_2;
    uint32_t num_output_tiles = 2;
    tt::tt_metal::CircularBufferConfig cb_output_config =
        tt::tt_metal::CircularBufferConfig(
            num_output_tiles * single_tile_size_output, {{output_cb_index, cb_data_format_output}})
            .set_page_size(output_cb_index, single_tile_size_output);
    auto cb_output = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);

    auto src_buffer = input.buffer();
    auto dst_buffer = output.buffer();

    const auto coeff10 = pack_two_bfloat16_into_uint32({1.50E-22f, 1.50E-22f});
    const auto coeff9 = pack_two_bfloat16_into_uint32({7.24E-20f, 7.24E-20f});
    const auto coeff8 = pack_two_bfloat16_into_uint32({2.90E-17f, 2.90E-17f});
    const auto coeff7 = pack_two_bfloat16_into_uint32({9.39E-15f, 9.39E-15f});
    const auto coeff6 = pack_two_bfloat16_into_uint32({2.40E-12f, 2.40E-12f});
    const auto coeff5 = pack_two_bfloat16_into_uint32({4.71E-10f, 4.71E-10f});
    const auto coeff4 = pack_two_bfloat16_into_uint32({6.78E-08f, 6.78E-08f});
    const auto coeff3 = pack_two_bfloat16_into_uint32({0.000006781684028f, 0.000006781684028f});
    const auto coeff2 = pack_two_bfloat16_into_uint32({0.0004340277778f, 0.0004340277778f});
    const auto coeff1 = pack_two_bfloat16_into_uint32({0.015625f, 0.015625f});
    const auto coeff0 = pack_two_bfloat16_into_uint32({0.25f, 0.25f});
    const auto one_scalar = pack_two_bfloat16_into_uint32({1.0f, 1.0f});

    // const auto coeff10 = std::bit_cast<uint32_t>(1.50E-22f);
    // const auto coeff9 = std::bit_cast<uint32_t>(7.24E-20f);
    // const auto coeff8 = std::bit_cast<uint32_t>(2.90E-17f);
    // const auto coeff7 = std::bit_cast<uint32_t>(9.39E-15f);
    // const auto coeff6 = std::bit_cast<uint32_t>(2.40E-12f);
    // const auto coeff5 = std::bit_cast<uint32_t>(4.71E-10f);
    // const auto coeff4 = std::bit_cast<uint32_t>(6.78E-08f);
    // const auto coeff3 = std::bit_cast<uint32_t>(0.000006781684028f);
    // const auto coeff2 = std::bit_cast<uint32_t>(0.0004340277778f);
    // const auto coeff1 = std::bit_cast<uint32_t>(0.015625f);
    // const auto coeff0 = std::bit_cast<uint32_t>(0.25f);

    // POLYVAL5(0.00144462f, -0.01055479f, -0.01203685f, 0.24300185f, 0.50437757f, val);

    bool src_is_dram = src_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    std::vector<uint32_t> reader_compile_time_args = {
        (uint32_t)src_is_dram,
        coeff0,
        coeff1,
        coeff2,
        coeff3,
        coeff4,
        coeff5,
        coeff6,
        coeff7,
        coeff8,
        coeff9,
        coeff10,
        one_scalar};
    bool dst_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)output_cb_index, (std::uint32_t)dst_is_dram};

    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/i0_fpu/device/kernels/dataflow/reader_interleaved_i0_fpu.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    std::vector<uint32_t> compute_kernel_args_group_1 = {
        num_tiles_per_core_group_1,  // per_core_block_cnt
        1                            // per_core_block_size
    };

    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (args.preserve_fp32_precision) {
        unpack_to_dest_mode[src0_cb_index] = UnpackToDestMode::UnpackToDestFp32;
    }

    bool math_approx_mode = false;
    std::map<string, string> unary_defines;
    auto path = compute_root + "i0_fpu.cpp";

    auto eltwise_unary_kernel_group_1_id = tt::tt_metal::CreateKernel(
        program,
        path,
        core_group_1,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = args.fp32_dest_acc_en,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .bfp8_pack_precise = args.bfp8_pack_precise,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_kernel_args_group_1,
            .defines = unary_defines});

    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_kernel_args_group_2 = {
            num_tiles_per_core_group_2,  // per_core_block_cnt
            1                            // per_core_block_size
        };

        auto eltwise_unary_kernel_group_2_id = tt::tt_metal::CreateKernel(
            program,
            path,
            core_group_2,
            tt::tt_metal::ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = args.fp32_dest_acc_en,
                .unpack_to_dest_mode = unpack_to_dest_mode,
                .bfp8_pack_precise = args.bfp8_pack_precise,
                .math_approx_mode = math_approx_mode,
                .compile_args = compute_kernel_args_group_2,
                .defines = unary_defines});
    }

    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_tiles_per_core = 0;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }

        tt::tt_metal::SetRuntimeArgs(
            program, unary_reader_kernel_id, core, {src_buffer->address(), num_tiles_per_core, num_tiles_written});

        tt::tt_metal::SetRuntimeArgs(
            program, unary_writer_kernel_id, core, {dst_buffer->address(), num_tiles_per_core, num_tiles_written});
        num_tiles_written += num_tiles_per_core;
    }

    return cached_program_t{
        std::move(program), {unary_reader_kernel_id, unary_writer_kernel_id, num_cores, num_cores_y}};
}

void I0FpuProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto& unary_reader_kernel_id = cached_program.shared_variables.unary_reader_kernel_id;
    auto& unary_writer_kernel_id = cached_program.shared_variables.unary_writer_kernel_id;
    const uint32_t num_cores = cached_program.shared_variables.num_cores;
    const uint32_t num_cores_y = cached_program.shared_variables.num_cores_y;

    auto& program = cached_program.program;

    const auto& input = tensor_args.input;
    auto src_buffer = input.buffer();
    auto dst_buffer = output.buffer();

    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        {
            auto& runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, core);
            runtime_args[0] = src_buffer->address();
        }

        {
            auto& runtime_args = GetRuntimeArgs(program, unary_writer_kernel_id, core);
            runtime_args[0] = dst_buffer->address();
        }
    }
}

}  // namespace ttnn::operations::unary::program
