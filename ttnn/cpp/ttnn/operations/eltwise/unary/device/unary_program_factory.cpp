// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>

#include "unary_program_factory.hpp"

#include "tt_metal/common/work_split.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"

namespace ttnn::operations::unary::program {

using namespace tt::constants;

UnaryProgramFactory::cached_program_t UnaryProgramFactory::create(
    const operation_attributes_t& args,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {

    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input = tensor_args.input;
    const auto& ops_chain = args.op_chain;

    tt::tt_metal::Program program{};

    bool tilized = output.get_layout() == Layout::TILE;
    tt::log_debug(tt::LogOp, "is tilized {} ", tilized);

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.get_dtype());
    uint32_t single_tile_size = tilized ? tt::tt_metal::detail::TileSize(cb_data_format) : input.get_legacy_shape()[-1] * input.element_size();
    tt::DataFormat cb_data_format_output = tt::tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    uint32_t single_tile_size_output = tilized ? tt::tt_metal::detail::TileSize(cb_data_format_output) : output.get_legacy_shape()[-1] * output.element_size();
    tt::log_debug(tt::LogOp, "is single_tile_size input {} ", single_tile_size);
    tt::log_debug(tt::LogOp, "is single_tile_size_output {} ", single_tile_size_output);

    uint32_t num_units = tilized ? output.volume() / TILE_HW : output.volume() / output.get_legacy_shape()[-1];
    tt::log_debug(tt::LogOp, "is tilized {}, num_units {} ", tilized, num_units);
    // uint32_t num_tiles = input.volume() / tt::constants::TILE_HW;

    tt::tt_metal::Device *device = input.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_units);

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;
    uint32_t aligned_input_unit_size = round_up_to_mul32(single_tile_size);
    tt::log_debug(tt::LogOp, "is aligned_input_unit_size {} ", aligned_input_unit_size);
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(num_input_tiles * aligned_input_unit_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, aligned_input_unit_size);
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    uint32_t output_cb_index = 16;  // output operands start at index 16
    uint32_t num_output_tiles = 2;
    uint32_t aligned_output_unit_size = round_up_to_mul32(single_tile_size_output);
    tt::log_debug(tt::LogOp, "is aligned_output_unit_size {} ", aligned_output_unit_size);
    tt::tt_metal::CircularBufferConfig cb_output_config =
        tt::tt_metal::CircularBufferConfig(
            num_output_tiles * aligned_output_unit_size, {{output_cb_index, cb_data_format_output}})
            .set_page_size(output_cb_index, aligned_output_unit_size);
    auto cb_output = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);

    auto src_buffer = input.buffer();

    auto dst_buffer = output.buffer();

    bool src_is_dram = src_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    bool dst_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;

    std::vector<uint32_t> reader_compile_time_args, writer_compile_time_args;

    if (tilized) {
        reader_compile_time_args = {(uint32_t)src_is_dram};
        writer_compile_time_args = {
            (std::uint32_t) output_cb_index,
            (std::uint32_t) dst_is_dram
        };
    } else {
        bool src_stick_size_is_power_of_two = is_power_of_two_at_least_32(single_tile_size);
        uint32_t src_log2_stick_size = src_stick_size_is_power_of_two ? (std::uint32_t)log2(single_tile_size) : 0;
        reader_compile_time_args = {
            (std::uint32_t) src0_cb_index,
            (std::uint32_t) src_is_dram,
            (std::uint32_t) src_stick_size_is_power_of_two,
            (std::uint32_t) src_log2_stick_size
        };
        bool dst_stick_size_is_power_of_two = is_power_of_two_at_least_32(single_tile_size_output);
        uint32_t dst_log2_stick_size = dst_stick_size_is_power_of_two ? (std::uint32_t)log2(single_tile_size_output) : 0;
        writer_compile_time_args = {
            (std::uint32_t) output_cb_index,
            (std::uint32_t) dst_is_dram,
            (std::uint32_t) dst_stick_size_is_power_of_two,
            (std::uint32_t) dst_log2_stick_size
        };
    }

    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        tilized ? "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp" : "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/reader_unary_stick_layout_interleaved_start_id.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        tilized ? "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp" : "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp",
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

    bool math_approx_mode = std::all_of(
        args.op_chain.begin(), args.op_chain.end(), [](const auto &u) { return utils::get_op_approx_mode(u.op_type); });
    std::map<string, string> unary_defines = utils::get_block_defines(args.op_chain);
    auto eltwise_unary_kernel_group_1_id = tt::tt_metal::CreateKernel(
        program,
        "tt_metal/kernels/compute/eltwise_sfpu.cpp",
        core_group_1,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = args.fp32_dest_acc_en,
            .unpack_to_dest_mode = unpack_to_dest_mode,
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
            "tt_metal/kernels/compute/eltwise_sfpu.cpp",
            core_group_2,
            tt::tt_metal::ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = args.fp32_dest_acc_en,
                .unpack_to_dest_mode = unpack_to_dest_mode,
                .math_approx_mode = math_approx_mode,
                .compile_args = compute_kernel_args_group_2,
                .defines = unary_defines});
    }

    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_tiles_per_core = 0;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }
        tt::log_debug(tt::LogOp, "is num_tiles_written {} ", num_tiles_written);
        tt::log_debug(tt::LogOp, "is num_tiles_per_core {} ", num_tiles_per_core);

        if (tilized) {
            tt::tt_metal::SetRuntimeArgs(
                program,
                unary_reader_kernel_id,
                core,
                {
                    src_buffer->address(),
                    num_tiles_per_core,
                    num_tiles_written
                }
            );

            tt::tt_metal::SetRuntimeArgs(
                program,
                unary_writer_kernel_id,
                core,
                {
                    dst_buffer->address(),
                    num_tiles_per_core,
                    num_tiles_written
                }
            );
        } else {
            tt::tt_metal::SetRuntimeArgs(
                program,
                unary_reader_kernel_id,
                core,
                {
                    src_buffer->address(),
                    single_tile_size,
                    num_tiles_per_core,
                    num_tiles_written
                }
            );

            tt::tt_metal::SetRuntimeArgs(
                program,
                unary_writer_kernel_id,
                core,
                {
                    dst_buffer->address(),
                    single_tile_size_output,
                    num_tiles_per_core,
                    num_tiles_written
                }
            );
        }

        num_tiles_written += num_tiles_per_core;
    }

    return cached_program_t{std::move(program), {unary_reader_kernel_id, unary_writer_kernel_id, num_cores, num_cores_y}};
}

void UnaryProgramFactory::override_runtime_arguments(
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
            auto &runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, core);
            runtime_args[0] = src_buffer->address();
        }

        {
            auto &runtime_args = GetRuntimeArgs(program, unary_writer_kernel_id, core);
            runtime_args[0] = dst_buffer->address();
        }
    }
}

}  // namespace ttnn::operations::unary::program
