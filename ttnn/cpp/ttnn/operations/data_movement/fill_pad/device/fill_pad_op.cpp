// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/fill_rm/device/fill_rm_op.hpp"
#include "tt_metal/common/test_tiles.hpp"

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
    std::cout << "single_tile_size: " << single_tile_size << std::endl;

    tt::tt_metal::Buffer* tens_buffer = input_tensor.buffer();
    TT_ASSERT(tens_buffer != nullptr, "Input buffer should be allocated on device!");

    uint32_t input_element_size_bytes = input_tensor.element_size();
    uint32_t height = a.get_logical_shape()[-2];
    uint32_t width = a.get_logical_shape()[-1];

    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(input_element_size_bytes * single_tile_size, {{0, cb_data_format}})
            .set_page_size(0, single_tile_size);
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    // tt::tt_metal::CircularBufferConfig cb_src1_config =
    //     tt::tt_metal::CircularBufferConfig(num_cb_tiles * single_tile_size, {{1, cb_data_format}})
    //         .set_page_size(1, single_tile_size);
    // auto cb_src1 = tt::tt_metal::CreateCircularBuffer(program, core, cb_src1_config);

    bool src_is_dram = tens_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;

    // create kernel
    // reader compile time args
    std::vector<uint32_t> reader_compile_time_args = {(std::uint32_t)src_is_dram, (std::uint32_t)fill_value};
    ,
};

tt::tt_metal::KernelHandle binary_reader_kernel_id = tt::tt_metal::CreateKernel(
    program,
    "ttnn/cpp/ttnn/operations/data_movement/fill_pad/device/kernels/dataflow/fill_pad_reader.cpp",
    core,
    tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

tt::tt_metal::SetRuntimeArgs(program, binary_reader_kernel_id, core, {tens_buffer->address(), height, width});

auto override_runtime_args_callback = [kernel_id = binary_reader_kernel_id](
                                          const Program& program, const std::vector<Buffer*>& input_buffers) {
    auto tens_buffer = input_buffers.at(0);

    CoreCoord core = {0, 0};

    {
        auto& runtime_args = GetRuntimeArgs(program, kernel_id, core);
        runtime_args[0] = tens_buffer->address();
    }
};

return {std::move(program), override_runtime_args_callback};
}

void FillRM::validate(const std::vector<Tensor>& input_tensors) const {
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
