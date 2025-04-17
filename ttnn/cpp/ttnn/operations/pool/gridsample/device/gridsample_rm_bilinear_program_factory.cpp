// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <math.h>

#include "tt-metalium/circular_buffer.hpp"
#include "tt-metalium/circular_buffer_types.hpp"
#include "gridsample_op.hpp"
#include "ttnn/operations/math.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/math.hpp>
// #include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/operations/reduction/generic/device/reduce_op.hpp"  // for reduce_op_utils

#include <tt_stl/reflection.hpp>
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"
#include "ttnn/operations/sliding_window/halo/halo.hpp"

#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include <tt-metalium/command_queue.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/device.hpp>

using namespace tt::constants;

namespace ttnn::operations::gridsample {
using namespace tt;
using namespace tt::tt_metal;

operation::ProgramWithCallbacks gridsample_rm_single_core(
    const Tensor& input,
    Tensor& output,
    const Tensor& reshaped_input,
    const std::vector<float>& normalized_grid,
    const std::string& mode,
    bool align_corners) {
    Program program{};
    CoreRange core({0, 0}, {0, 0});

    auto output_shape = output.get_logical_shape();

    auto input_shape = input.get_logical_shape();

    tt::DataFormat input_cb_data_format = tt_metal::datatype_to_dataformat_converter(input.get_dtype());
    uint32_t input_unit_size =
        input.volume() * input.element_size();  // N * C * H * W -> As we require entire input tensor
    tt::DataFormat output_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    uint32_t output_unit_size = output.element_size();  // 1 element size as we store 1 element at a time.

    int batch_size = input.get_logical_shape()[0];
    int channels = input.get_logical_shape()[1];

    int total_count = output_shape[2] * output_shape[3] * 2;  // for i and i + 1
    // we will iterate the Normalized Grid input_num_units times to retrieve the output

    // This should allocate a DRAM buffer on the device
    tt_metal::IDevice* device = output.device();

    // circular buffer for input
    uint32_t src0_cb_index = CBIndex::c_0;

    uint32_t num_input_units = 2;
    uint32_t aligned_input_unit_size = tt::round_up(input_unit_size, hal::get_dram_alignment());

    tt_metal::CircularBufferConfig cb_src0_config =
        tt_metal::CircularBufferConfig(
            num_input_units * aligned_input_unit_size, {{src0_cb_index, input_cb_data_format}})
            .set_page_size(src0_cb_index, aligned_input_unit_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    tt_metal::CircularBufferConfig cb_src1_config =
        tt_metal::CircularBufferConfig(num_input_units * aligned_input_unit_size, {{1, input_cb_data_format}})
            .set_page_size(1, aligned_input_unit_size);
    auto cb_src1 = tt_metal::CreateCircularBuffer(program, core, cb_src1_config);

    tt_metal::CircularBufferConfig cb_src2_config =
        tt_metal::CircularBufferConfig(num_input_units * aligned_input_unit_size, {{2, input_cb_data_format}})
            .set_page_size(2, aligned_input_unit_size);
    auto cb_src2 = tt_metal::CreateCircularBuffer(program, core, cb_src2_config);

    uint32_t output_cb_index = src0_cb_index;  // same as input cb

    auto src_buffer = reshaped_input.buffer();
    auto dst_buffer = output.buffer();
    bool src_is_dram = src_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;

    std::vector<uint32_t> reader_compile_time_args, writer_compile_time_args;

    int Gridoffset = output_shape[2] * output_shape[3] * 2;
    int row = input_shape[2];
    int col = input_shape[3];

    constexpr uint32_t single_tile_size = 2 * (32 * 32);
    constexpr uint32_t num_tiles = 50;
    constexpr uint32_t dram_buffer_size = single_tile_size * num_tiles;

    tt::tt_metal::InterleavedBufferConfig l1_config{
        .device = device,
        .size = dram_buffer_size,
        .page_size = dram_buffer_size,
        .buffer_type = tt::tt_metal::BufferType::L1};

    ttnn::Shape gridshape{1, 1, 1, batch_size * output_shape[2] * output_shape[3] * 2};

    TensorSpec output_spec =
        TensorSpec(gridshape, TensorLayout(DataType::FLOAT32, PageConfig(input.get_layout()), input.memory_config()));

    ttnn::Tensor grid = ttnn::Tensor::from_vector<float>(normalized_grid, output_spec, device);

    auto grid_buffer = grid.buffer();

    uint32_t grid_size = grid.volume() * grid.element_size();

    reader_compile_time_args = {
        (std::uint32_t)src0_cb_index,
        (std::uint32_t)src_is_dram,
        (std::uint32_t)channels,
        (std::uint32_t)batch_size,
        (std::uint32_t)total_count,
        (std::uint32_t)Gridoffset,
        (std::uint32_t)row,
        (std::uint32_t)col};

    std::map<string, string> kernel_defines;

    // Perform Reading and Writing the data in the reader kernel
    tt_metal::KernelHandle unary_reader_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/pool/gridsample/device/kernels/"
        "reader_gridsample_rm_bilinear_layout_interleaved_start_id.cpp",
        core,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args, kernel_defines));

    SetRuntimeArgs(
        program,
        unary_reader_kernel_id,
        core,
        {src_buffer->address(),
         dst_buffer->address(),
         grid_buffer->address(),
         input_unit_size,
         output_unit_size,
         grid_size});

    auto override_runtime_args_callback = [unary_reader_kernel_id, grid_buffer](
                                              const void* operation,
                                              Program& program,
                                              const std::vector<Tensor>& input_tensors,
                                              const std::vector<std::optional<const Tensor>>&,
                                              const std::vector<Tensor>& output_tensors) {
        auto src_buffer = input_tensors.at(0).buffer();

        auto dst_buffer = output_tensors.at(0).buffer();

        CoreCoord core = {0, 0};

        {
            auto& runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, core);
            runtime_args[0] = src_buffer->address();
            runtime_args[1] = dst_buffer->address();
            runtime_args[2] = grid_buffer->address();
        }
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_args_callback};
}
}  // namespace ttnn::operations::gridsample
