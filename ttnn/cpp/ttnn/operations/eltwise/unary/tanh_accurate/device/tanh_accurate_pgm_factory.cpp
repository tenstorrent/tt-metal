// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>

#include "tanh_accurate_pgm_factory.hpp"

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/host_api.hpp>

namespace ttnn::operations::unary::program {

static const std::string compute_root = "ttnn/cpp/ttnn/operations/eltwise/unary/tanh_accurate/device/kernels/compute/";

using namespace tt::constants;

TanhAccurateProgramFactory::cached_program_t TanhAccurateProgramFactory::create(
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

    // intermediate buffers

    // LUT tanh(x)
    uint32_t im1_cb_index = tt::CBIndex::c_1;
    tt::tt_metal::CircularBufferConfig cb_im1_config =
        tt::tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{im1_cb_index, cb_data_format}})
            .set_page_size(im1_cb_index, single_tile_size);
    auto cb_im1 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_im1_config);

    // exp(2x)
    uint32_t im2_cb_index = tt::CBIndex::c_3;
    tt::tt_metal::CircularBufferConfig cb_im2_config =
        tt::tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{im2_cb_index, cb_data_format}})
            .set_page_size(im2_cb_index, single_tile_size);
    auto cb_im2 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_im2_config);

    // exp(2x) - 1
    uint32_t im3_cb_index = tt::CBIndex::c_4;
    tt::tt_metal::CircularBufferConfig cb_im3_config =
        tt::tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{im3_cb_index, cb_data_format}})
            .set_page_size(im3_cb_index, single_tile_size);
    auto cb_im3 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_im3_config);

    // recip(exp(2x) + 1)
    uint32_t im4_cb_index = tt::CBIndex::c_5;
    tt::tt_metal::CircularBufferConfig cb_im4_config =
        tt::tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{im4_cb_index, cb_data_format}})
            .set_page_size(im4_cb_index, single_tile_size);
    auto cb_im4 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_im4_config);

    // exp(2x) - 1 * recip(exp(2x) - 1)
    uint32_t im5_cb_index = tt::CBIndex::c_6;
    tt::tt_metal::CircularBufferConfig cb_im5_config =
        tt::tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{im5_cb_index, cb_data_format}})
            .set_page_size(im5_cb_index, single_tile_size);
    auto cb_im5 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_im5_config);

    // output for x > 3.5
    uint32_t im6_cb_index = tt::CBIndex::c_7;
    tt::tt_metal::CircularBufferConfig cb_im6_config =
        tt::tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{im6_cb_index, cb_data_format}})
            .set_page_size(im6_cb_index, single_tile_size);
    auto cb_im6 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_im6_config);

    // output for x <= 3.5
    uint32_t im7_cb_index = tt::CBIndex::c_8;
    tt::tt_metal::CircularBufferConfig cb_im7_config =
        tt::tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{im7_cb_index, cb_data_format}})
            .set_page_size(im7_cb_index, single_tile_size);
    auto cb_im7 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_im7_config);

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

    bool src_is_dram = src_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    std::vector<uint32_t> reader_compile_time_args = {(uint32_t)src_is_dram};
    bool dst_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)output_cb_index, (std::uint32_t)dst_is_dram};

    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp",
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
    auto path = "ttnn/cpp/ttnn/operations/eltwise/unary/tanh_accurate/device/kernels/compute/tanh_accurate.cpp";

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

void TanhAccurateProgramFactory::override_runtime_arguments(
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
