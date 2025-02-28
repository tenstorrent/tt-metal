// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "llama_reduce_scatter_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <vector>
#include <tt-metalium/hal_exp.hpp>

namespace tnn::operations::experimental::ccl {

LlamaReduceScatterDeviceOperation::LlamaReduceScatterAdd::cached_program_t
LlamaReduceScatterDeviceOperation::LlamaReduceScatterAdd::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    // X = output width
    // Y = output height
    // input shape = (..., H, W)
    // output shape = (..., Y, X)

    /**
     * The algorithm is as follows:
     * 1. Read in blocks of data along the X and W dimensions (XW blocks, W is contiguous)
     *  a. TILE_HEIGHT rows along X with TILE_WIDTH elements across W
     * 2. Tilize, transpose, and untilize the data into a WX block
     * 3. Write out all the data in WX block to its correct position in the permuted output tensor buffer
     *  a. We write out on face/subtile line at a time
     *  a. X is the output width dimension, but it's tiled so we can only write out face/subtile line at a time
     * 4. Repeat until all XW blocks are processed
     * 5. If X is not a multiple of TILE_WIDTH, we pad the last face/subtile line with the pad value
     * 6. If Y is not a multiple of TILE_HEIGHT, we pad the last set of tiles on the Y dimension with the pad value
     *
     */

    using namespace tt;
    using namespace tt::tt_metal;
    const std::optional<float> pad_value = operation_attributes.pad_value;

    const auto& input_tensor = tensor_args.input_tensor;
    const auto& input_shape = input_tensor.get_logical_shape();
    const auto& dims = operation_attributes.dims;
    uint32_t rank = dims.size();
    auto& output_tensor = tensor_return_value;
    auto& output_shape = output_tensor.get_logical_shape();
    auto& padded_output_shape = output_tensor.get_padded_shape();
    const auto& tile_shape = input_tensor.get_tensor_spec().tile().get_tile_shape();
    const auto& face_shape = input_tensor.get_tensor_spec().tile().get_face_shape();

    auto src_buffer = input_tensor.buffer();
    auto dst_buffer = output_tensor.buffer();

    tt::tt_metal::Program program{};
    uint32_t element_size = input_tensor.element_size();

    tt::tt_metal::IDevice* device = input_tensor.device();

    uint32_t src0_cb_index = tt::CBIndex::c_0;

    uint32_t output_cb_index = src0_cb_index;

    uint32_t num_input_pages_to_read = 2;
    uint32_t num_tiles = 10;

    // auto compute_with_storage_grid_size = input_tensor.device()->compute_with_storage_grid_size();
    CoreCoord compute_with_storage_grid_size = {1u, 1u};
    auto [num_cores, all_cores, core_group_1, core_group_2, num_blocks_per_core_group_1, num_blocks_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_tiles);

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());
    uint32_t input_page_size =
        input_tensor.data_types() != tt::DataType::BFLOAT8_B ? tile_shape[0] * tile_shape[1] : 1088;

    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(num_input_pages_to_read * input_page_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, input_page_size);
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    bool src_is_dram = src_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    bool dst_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;

    std::vector<uint32_t> reader_compile_time_args = {(uint32_t)src_is_dram, input_page_size, src0_cb_index};

    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/permute/device/kernels/dataflow/"
        "reader_permute_interleaved_tiled_generic.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    std::vector<uint32_t> compute_kernel_args = {};

    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)dst_is_dram, input_page_size, src0_cb_index};

    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/llama_reduce_scatter/device/kernels/dataflow/"
        "writer_llama_reduce_scatter.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    std::vector<uint32_t> reader_runtime_args = {src_buffer->address(), 0, 0};
    std::vector<uint32_t> compute_runtime_args = {0, 0};
    std::vector<uint32_t> writer_runtime_args = {dst_buffer->address(), 0, 0};

    auto cores = corerange_to_cores(all_cores, std::nullopt);
    uint32_t start_block = 0;
    uint32_t num_blocks_per_core = 0;
    for (const auto& core : cores) {
        if (core_group_1.contains(core)) {
            num_blocks_per_core = num_blocks_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_blocks_per_core = num_blocks_per_core_group_2;
        } else {
            // no-op
            num_blocks_per_core = 0;
        }

        uint32_t end_block = start_block + num_blocks_per_core;
        reader_runtime_args[1] = start_block;
        reader_runtime_args[2] = end_block;

        writer_runtime_args[1] = start_block;
        writer_runtime_args[2] = end_block;

        tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_runtime_args);
        tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_runtime_args);

        start_block = end_block;
    }

    return {
        std::move(program),
        {.unary_reader_kernel_id = unary_reader_kernel_id,
         .unary_writer_kernel_id = unary_writer_kernel_id,
         .core_range = all_cores}};
}

void LlamaReduceScatterDeviceOperation::LlamaReduceScatterAdd::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& program = cached_program.program;
    auto& unary_reader_kernel_id = cached_program.shared_variables.unary_reader_kernel_id;
    auto& unary_writer_kernel_id = cached_program.shared_variables.unary_writer_kernel_id;

    const auto& input_tensor = tensor_args.input_tensor;
    auto& output_tensor = tensor_return_value;

    auto src_buffer = input_tensor.buffer();
    auto dst_buffer = output_tensor.buffer();
    auto& all_cores = cached_program.shared_variables.core_range;

    auto cores = corerange_to_cores(all_cores, std::nullopt);
    for (const auto& core : cores) {
        auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, unary_reader_kernel_id, core);
        runtime_args[0] = src_buffer->address();
        auto& runtime_args_writer = tt::tt_metal::GetRuntimeArgs(program, unary_writer_kernel_id, core);
        runtime_args_writer[0] = dst_buffer->address();
    }
}

}  // namespace tnn::operations::experimental::ccl
