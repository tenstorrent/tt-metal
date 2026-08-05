// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/embedding_backward/device/embedding_backward_device_operation.hpp"

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

ProgramDescriptor EmbeddingBackwardDeviceOperation::create_descriptor(
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

    ProgramDescriptor desc;

    // To read from grad tensor
    desc.cbs.push_back(CBDescriptor{
        .total_size = max_tiles_per_core * grad_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_0),
            .data_format = grad_cb_data_format,
            .page_size = grad_single_tile_size,
        }}},
    });

    // To store index values for a single tile
    desc.cbs.push_back(CBDescriptor{
        .total_size = index_single_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_1),
            .data_format = index_cb_data_format,
            .page_size = index_single_page_size,
        }}},
    });

    // To read from output tensor
    desc.cbs.push_back(CBDescriptor{
        .total_size = max_tiles_per_core * output_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_2),
            .data_format = output_cb_data_format,
            .page_size = output_single_tile_size,
        }}},
    });

    // To store mask values for a single tile
    desc.cbs.push_back(CBDescriptor{
        .total_size = mask_single_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_24),
            .data_format = mask_cb_data_format,
            .page_size = mask_single_page_size,
        }}},
    });

    // L1 scratch space to pass chunk_count from reader to UNPACK
    desc.cbs.push_back(CBDescriptor{
        .total_size = 16,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_25),
            .data_format = grad_cb_data_format,
            .page_size = 16,
        }}},
    });  // grad_cb_data_format doesn't matter here

    // For tiles to be written to the output
    desc.cbs.push_back(CBDescriptor{
        .total_size = max_tiles_per_core * output_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_16),
            .data_format = output_cb_data_format,
            .page_size = output_single_tile_size,
        }}},
    });

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

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/embedding_backward/device/kernels/dataflow/reader_embedding_backward.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor compute_desc;
    compute_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/embedding_backward/device/kernels/compute/embedding_backward.cpp";
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = all_cores;
    compute_desc.compile_time_args = {max_tiles_per_core, input_height_tiles};
    compute_desc.config = ComputeConfigDescriptor{};

    ////////////////////////////////////////////////////////////////////////////
    //                 Run-time arguments
    ////////////////////////////////////////////////////////////////////////////

    auto cores = corerange_to_cores(all_cores);
    uint32_t offset = 0;
    for (auto core : cores) {
        uint32_t tiles_this_core = 0;
        if (core_group_1.contains(core)) {
            tiles_this_core = num_tiles_per_core_group_1;
        } else {
            tiles_this_core = num_tiles_per_core_group_2;
        }
        // Buffer* entries register for fast cache-hit address patching (see PR #42992).
        reader_desc.emplace_runtime_args(
            core, {grad_tensor_buffer, index_tensor_buffer, out_buffer, embedding_tiles, offset, tiles_this_core});
        compute_desc.emplace_runtime_args(core, {tiles_this_core});

        offset += tiles_this_core;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

}  // namespace ttnn::prim
