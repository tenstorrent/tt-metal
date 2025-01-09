// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/fill_pad/device/fill_pad_op.hpp"
#include "tt_metal/common/test_tiles.hpp"
#include "ttnn/operations/core/core.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/util.hpp"

using namespace tt::tt_metal;

using uint32_t = uint32_t;

namespace ttnn::operations::data_movement {

operation::ProgramWithCallbacks fill_pad_single_core(const Tensor& input_tensor, float fill_value) {
    tt::tt_metal::Device* device = input_tensor.device();
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();
    CoreRange core({0, 0}, {0, 0});

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());
    uint32_t single_tile_size = tt::tt_metal::detail::TileSize(cb_data_format);
    uint32_t page_size = single_tile_size;
    std::cout << "single_tile_size: " << single_tile_size << std::endl;

    tt::tt_metal::Buffer* tens_buffer = input_tensor.buffer();
    TT_ASSERT(tens_buffer != nullptr, "Input buffer should be allocated on device!");

    uint32_t input_element_size_bytes = input_tensor.element_size();
    uint32_t height = input_tensor.get_logical_shape()[-2];
    uint32_t width = input_tensor.get_logical_shape()[-1];

    constexpr uint32_t src0_cb_index = tt::CBIndex::c_0;
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(input_element_size_bytes * page_size * 2, {{src0_cb_index, cb_data_format}})
            .set_page_size(0, input_element_size_bytes * page_size);
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    bool src_is_dram = tens_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;

    bool output_stick_size_is_power_of_two = is_power_of_two_at_least_32(page_size);
    uint32_t output_log2_stick_size = output_stick_size_is_power_of_two ? (std::uint32_t)std::log2(page_size) : 0;
    // create kernel
    // reader compile time args
    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)src0_cb_index,
        (std::uint32_t)src_is_dram,
        (std::uint32_t)output_stick_size_is_power_of_two,
        (std::uint32_t)output_log2_stick_size};
    (std::uint32_t)fill_value
};

tt::tt_metal::KernelHandle binary_reader_kernel_id = tt::tt_metal::CreateKernel(
    program,
    "ttnn/cpp/ttnn/operations/data_movement/fill_pad/device/kernels/dataflow/fill_pad_writer.cpp",
    core,
    // tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));
    tt_metal::WriterDataMovementConfig(writer_compile_time_args));  // gonna be writer only

std::vector<uint32_t> writer_runtime_args = {
    (std::uint32_t)tens_buffer()->address(),
    (std::uint32_t)page_size,
    (std::uint32_t)height,
    (std::uint32_t)width,
    (std::uint32_t)(((height + TILE_HEIGHT - 1) / TILE_HEIGHT) * TILE_HEIGHT),
    (std::uint32_t)(((width + TILE_HEIGHT - 1) / TILE_HEIGHT) * TILE_HEIGHT),
};
tt::tt_metal::SetRuntimeArgs(program, binary_reader_kernel_id, core, {tens_buffer->address(), height, width});

auto override_runtime_args_callback =
    [kernel_id = binary_reader_kernel_id](
        const Program& program, const std::vector<Buffer*>& input_buffers, const std::vector<Buffer*>& output_buffers) {
        auto tens_buffer = input_buffers.at(0);

        CoreCoord core = {0, 0};

        {
            auto& runtime_args = GetRuntimeArgs(program, kernel_id, core);
            runtime_args[0] = tens_buffer->address();
        }
    };

return {std::move(program), override_runtime_args_callback};
}

void FillPad::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    TT_FATAL(input_tensor_a.get_dtype() == DataType::BFLOAT16, "Error");
    TT_FATAL(
        input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED,
        "FillPad does not currently support sharding");
    TT_FATAL(
        this->output_mem_config.memory_layout == TensorMemoryLayout::INTERLEAVED,
        "FillPad does not currently support sharding");
}

operation::ProgramWithCallbacks FillPad::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    return fill_pad_single_core(input_tensor, this->fill_value);
}

}  // namespace ttnn::operations::data_movement
