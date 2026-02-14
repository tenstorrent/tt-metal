// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/math.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "ttnn/operations/embedding_backward/device/embedding_backward_program_factory.hpp"

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

EmbeddingBackwardProgramFactory::cached_program_t EmbeddingBackwardProgramFactory::create(
    const EmbeddingBackwardParams& operation_attributes,
    const EmbeddingBackwardInputs& tensor_args,
    Tensor& tensor_return_value) {
    ////////////////////////////////////////////////////////////////////////////
    //                 Buffer Setup
    ////////////////////////////////////////////////////////////////////////////

    tt_metal::Buffer* index_tensor_buffer = tensor_args.index_tensor.buffer();
    tt_metal::Buffer* grad_tensor_buffer = tensor_args.grad_tensor.buffer();
    tt_metal::Buffer* out_buffer = tensor_return_value.buffer();

    auto* device = tensor_args.grad_tensor.device();

    const auto& index_tensor = tensor_args.index_tensor;
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////

    Program program{};

    uint32_t index_element_size_bytes = index_tensor.element_size();
    constexpr uint32_t INPUT_SIZE = 32;

    tt::DataFormat grad_cb_data_format = datatype_to_dataformat_converter(tensor_args.grad_tensor.dtype());
    uint32_t grad_single_tile_size = tt::tile_size(grad_cb_data_format);

    tt::DataFormat index_cb_data_format = datatype_to_dataformat_converter(tensor_args.index_tensor.dtype());
    uint32_t index_single_page_size =
        INPUT_SIZE * index_element_size_bytes;  // Only need 32 at most at a time, which is less than full page size
    uint32_t index_page_size = index_tensor.padded_shape()[-1] * index_element_size_bytes;

    tt::DataFormat mask_cb_data_format = tt::DataFormat::UInt8;
    uint32_t mask_single_page_size = INPUT_SIZE * 1;  // UInt8 is 1 byte per element

    tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(tensor_return_value.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    uint32_t embedding_dim = tensor_args.grad_tensor.padded_shape()[-1];
    uint32_t embedding_tiles = embedding_dim / TILE_WIDTH;

    uint32_t batch_size = tensor_args.index_tensor.padded_shape()[0];
    uint32_t seq_len_tiles = tensor_args.index_tensor.padded_shape()[-1] / TILE_WIDTH;
    uint32_t input_height_tiles = batch_size * seq_len_tiles;

    uint32_t num_embeddings_tiles = operation_attributes.num_embeddings / TILE_HEIGHT;

    // We split work based on the number of tiles in the embedding dimension
    auto grid_size = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(grid_size, embedding_tiles);
    uint32_t max_tiles_per_core = std::max(num_tiles_per_core_group_1, num_tiles_per_core_group_2);

    log_debug(LogType::LogOp, "Embedding hidden size tiles: {}", embedding_tiles);
    log_debug(LogType::LogOp, "Num parallel cores: {}", num_cores);
    log_debug(LogType::LogOp, "Max hidden size tiles per core: {}", max_tiles_per_core);

    ////////////////////////////////////////////////////////////////////////////
    //                 Circular buffers
    ////////////////////////////////////////////////////////////////////////////

    // To read from grad tensor
    create_cb(CBIndex::c_0, program, all_cores, grad_single_tile_size, max_tiles_per_core, grad_cb_data_format);

    // To store index values for a single tile
    create_cb(CBIndex::c_1, program, all_cores, index_single_page_size, 1, index_cb_data_format);

    // To read from output tensor
    create_cb(CBIndex::c_2, program, all_cores, output_single_tile_size, max_tiles_per_core, output_cb_data_format);

    // To store mask values for a single tile
    create_cb(CBIndex::c_24, program, all_cores, mask_single_page_size, 1, mask_cb_data_format);

    // L1 scratch space to pass chunk_count from reader to UNPACK
    create_cb(
        CBIndex::c_25, program, all_cores, 16, 1, grad_cb_data_format);  // grad_cb_data_format doesn't matter here

    // For tiles to be written to the output
    create_cb(CBIndex::c_16, program, all_cores, output_single_tile_size, max_tiles_per_core, output_cb_data_format);

    ////////////////////////////////////////////////////////////////////////////
    //                 Kernels
    ////////////////////////////////////////////////////////////////////////////

    // reader

    std::vector<uint32_t> reader_compile_time_args = {
        (uint32_t)max_tiles_per_core,
        (uint32_t)batch_size,
        (uint32_t)seq_len_tiles,
        (uint32_t)num_embeddings_tiles,
        (uint32_t)index_page_size,
        (uint32_t)(tensor_args.index_tensor.dtype() == DataType::BFLOAT16),
        (uint32_t)(tensor_return_value.dtype() == DataType::BFLOAT16)};
    TensorAccessorArgs(*grad_tensor_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*index_tensor_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*out_buffer).append_to(reader_compile_time_args);

    auto reader_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/embedding_backward/device/kernels/dataflow/reader_embedding_backward.cpp",
        all_cores,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    std::vector<uint32_t> reader_runtime_args = {
        grad_tensor_buffer->address(),
        index_tensor_buffer->address(),
        out_buffer->address(),
        embedding_tiles,  // how many pages to skip to get to the next row
        0,                // offset to the first tile in a row
        0,                // how many tiles to process in a row
    };

    auto compute_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/embedding_backward/device/kernels/compute/embedding_backward.cpp",
        all_cores,
        tt_metal::ComputeConfig{.compile_args = {max_tiles_per_core, input_height_tiles}});

    ////////////////////////////////////////////////////////////////////////////
    //                 Run-time arguments
    ////////////////////////////////////////////////////////////////////////////

    auto cores = corerange_to_cores(all_cores);
    uint32_t offset = 0;
    for (auto core : cores) {
        reader_runtime_args[4] = offset;
        if (core_group_1.contains(core)) {
            reader_runtime_args[5] = num_tiles_per_core_group_1;
        } else {
            reader_runtime_args[5] = num_tiles_per_core_group_2;
        }
        SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);
        SetRuntimeArgs(program, compute_kernel_id, core, {reader_runtime_args[5]});

        offset += reader_runtime_args[5];
    }
    return cached_program_t{
        std::move(program), {.reader_kernel_id = reader_kernel_id, .cores = cores, .device = device}};
}

void EmbeddingBackwardProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const EmbeddingBackwardParams&,
    const EmbeddingBackwardInputs& tensor_args,
    Tensor& tensor_return_value) {
    auto* index_dram_buffer = tensor_args.index_tensor.buffer();
    auto* grad_dram_buffer = tensor_args.grad_tensor.buffer();
    auto* output_dram_buffer = tensor_return_value.buffer();

    auto& program = cached_program.program;
    const auto& shared_variables = cached_program.shared_variables;

    auto& runtime_args_by_core = GetRuntimeArgs(program, shared_variables.reader_kernel_id);
    for (const auto& core : shared_variables.cores) {
        auto& runtime_args = runtime_args_by_core[core.x][core.y];
        runtime_args[0] = grad_dram_buffer->address();
        runtime_args[1] = index_dram_buffer->address();
        runtime_args[2] = output_dram_buffer->address();
    }
}

}  // namespace ttnn::prim
