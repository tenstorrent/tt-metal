// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rotate_half_program_factory.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"

namespace ttnn::operations::experimental::transformer::detail {

using namespace tt;
using namespace tt::constants;

operation::ProgramWithCallbacks rotate_half_single_core(const Tensor &input, Tensor &output) {
    auto program = tt::tt_metal::CreateProgram();

    CoreRange core({0, 0}, {0, 0});

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(input.get_dtype());
    uint32_t single_tile_size = tt_metal::detail::TileSize(cb_data_format);

    tt::DataFormat scalar_cb_data_format = tt::DataFormat::Float16_b;
    uint32_t scalar_single_tile_size = tt_metal::detail::TileSize(scalar_cb_data_format);

    uint32_t num_tiles = input.volume() / TILE_HW;
    uint32_t num_rows = input.volume()  / input.get_legacy_shape()[-1] / TILE_HEIGHT;
    uint32_t half_row_size = input.get_legacy_shape()[-1] / TILE_WIDTH / 2;

    tt_metal::Device *device = input.device();

    // Used for half of tensor that is multiplied
    uint32_t src_mul_cb_index = 0;
    uint32_t num_input_tiles = 2;
    tt_metal::CircularBufferConfig cb_src_mul_config = tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src_mul_cb_index, cb_data_format}})
		.set_page_size(src_mul_cb_index, single_tile_size);
    auto cb_src_mul = tt_metal::CreateCircularBuffer(program, core, cb_src_mul_config);

    // Used for bcast scalar
    uint32_t src_scalar_cb_index = 1;
    uint32_t num_scalar_tiles = 1;
    tt_metal::CircularBufferConfig cb_src1_config = tt_metal::CircularBufferConfig(num_scalar_tiles * scalar_single_tile_size, {{src_scalar_cb_index, cb_data_format}})
		.set_page_size(src_scalar_cb_index, scalar_single_tile_size);
    auto cb_src1 = tt_metal::CreateCircularBuffer(program, core, cb_src1_config);

    // Used for half of tensor that is not multiplied
    uint32_t src_no_mul_cb_index = 2;
    tt_metal::CircularBufferConfig cb_src_no_mul_config = tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src_no_mul_cb_index, cb_data_format}})
		.set_page_size(src_no_mul_cb_index, single_tile_size);
    auto cb_src_no_mul = tt_metal::CreateCircularBuffer(program, core, cb_src_no_mul_config);

    uint32_t output_mul_cb_index = 16; // output operands start at index 16
    uint32_t num_output_tiles = 2;
    tt_metal::CircularBufferConfig cb_output_config = tt_metal::CircularBufferConfig(num_output_tiles * single_tile_size, {{output_mul_cb_index, cb_data_format}})
		.set_page_size(output_mul_cb_index, single_tile_size);
    auto cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);
    uint32_t output_no_mul_cb_index = src_no_mul_cb_index;

    const uint16_t bfloat16_scalar = bfloat16(-1.0f).to_uint16();

    auto src_buffer = input.buffer();
    auto dst_buffer = output.buffer();

    bool src_is_dram = src_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {
        (uint32_t)src_no_mul_cb_index,
        (uint32_t)src_mul_cb_index,
        (uint32_t)src_scalar_cb_index,
        (uint32_t)src_is_dram,
        (uint32_t)bfloat16_scalar
    };
    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t) output_no_mul_cb_index,
        (std::uint32_t) output_mul_cb_index,
        (std::uint32_t) dst_is_dram
    };

    tt_metal::KernelHandle unary_reader_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/rotate_half/device/kernels/dataflow/reader_rotate_half_interleaved_start_id.cpp",
        core,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    tt_metal::KernelHandle unary_writer_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/rotate_half/device/kernels/dataflow/writer_rotate_half_interleaved_start_id.cpp",
        core,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    std::map<string, string> bcast_compute_defines = {
        {"BCAST_OP", "mul_tiles_bcast"},
        {"BCAST_LLKOP", "ELWMUL"},
        {"BCAST_DIM", "BroadcastType::SCALAR"},
		{"BCAST_SCALAR", "1"}
	};

	auto bcast_kernel_group_1_id = tt_metal::CreateKernel(
		program,
		"ttnn/cpp/ttnn/operations/data_movement/bcast/device/kernels/compute/bcast_hw.cpp",
		core,
		tt_metal::ComputeConfig{.compile_args = {}, .defines = bcast_compute_defines}
	);

    SetRuntimeArgs(
        program,
        unary_reader_kernel_id,
        core,
        {
            src_buffer->address(),
            num_rows,
            half_row_size,
            0
        }
    );

    SetRuntimeArgs(
        program,
        bcast_kernel_group_1_id,
        core,
        {
            1, // B
            1, // Ht
            num_tiles / 2  // Wt
        }
    );

    SetRuntimeArgs(
        program,
        unary_writer_kernel_id,
        core,
        {
            dst_buffer->address(),
            num_rows,
            half_row_size,
            0
        }
    );

    auto override_runtime_args_callback = [unary_reader_kernel_id, unary_writer_kernel_id](
        const ProgramHandle program,
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {

        auto src_buffer = input_buffers.at(0);

        auto dst_buffer = output_buffers.at(0);

        CoreCoord core = {0, 0};

        {
            auto &runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, core);
            runtime_args[0] = src_buffer->address();
        }

        {
            auto &runtime_args = GetRuntimeArgs(program, unary_writer_kernel_id, core);
            runtime_args[0] = dst_buffer->address();
        }
    };

    return {program, override_runtime_args_callback};
}

} // namespace ttnn::operations::experimental::transformer::detail
