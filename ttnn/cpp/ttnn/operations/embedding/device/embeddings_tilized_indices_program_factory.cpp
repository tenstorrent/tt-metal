// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "embeddings_tilized_indices_program_factory.hpp"
#include "embedding_program_factory_common.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_align.hpp>
#include <algorithm>

namespace ttnn::prim {

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;

tt::tt_metal::ProgramDescriptor EmbeddingsTilizedIndicesProgramFactory::create_descriptor(
    const EmbeddingParams& operation_attributes, const EmbeddingInputs& tensor_args, Tensor& tensor_return_value) {
    const auto& a = tensor_args.input_tensor_arg;
    const auto& weights = tensor_args.weight_arg;
    auto& output = tensor_return_value;
    const auto& embeddings_type = operation_attributes.embeddings_type;
    const auto& pad_token = operation_attributes.pad_token;

    ProgramDescriptor desc;

    ////////////////////////////////////////////////////////////////////////////
    //                 Buffer Setup
    ////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////
    //                      Grayskull Device Setup
    ////////////////////////////////////////////////////////////////////////////
    // This should allocate a DRAM buffer on the device
    auto* device = a.device();

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////

    uint32_t input_element_size_bytes = a.element_size();
    uint32_t weights_element_size_bytes = weights.element_size();
    uint32_t output_element_size_bytes = output.element_size();

    // row major, page size is last dim
    uint32_t input_page_size = a.logical_shape()[-1] * input_element_size_bytes;
    uint32_t weight_page_size = weights.padded_shape()[-1] * weights_element_size_bytes;
    uint32_t output_page_size = output.padded_shape()[-1] * output_element_size_bytes;

    // weights shape is [1, 1, num_embeddings, num_dim]

    uint32_t batch_size = a.logical_shape()[0];  // num rows
    uint32_t num_cols = a.logical_shape()[-1];
    uint32_t volume = num_cols * batch_size;
    auto alignment = a.buffer()->alignment();

    // setup problem and grid size

    uint32_t problem_size = volume;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    CoreSplitResult work = split_work_to_cores_aligned(compute_with_storage_grid_size, problem_size, FACE_HEIGHT);

    uint32_t num_cores = work.required_cores;
    CoreRangeSet all_cores = work.all_cores;
    CoreRangeSet core_group_1 = work.core_group_1;
    CoreRangeSet core_group_2 = work.core_group_2;
    uint32_t num_blocks_per_core_group_1 = work.units_per_core_group_1;
    uint32_t num_blocks_per_core_group_2 = work.units_per_core_group_2;

    uint32_t g1_numcores = core_group_1.num_cores();

    // Create Buffers
    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());

    tt::DataFormat weights_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(weights.dtype());

    constexpr uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t rounded_weight_page_size = tt::align(weight_page_size, alignment);
    desc.cbs.push_back(CBDescriptor{
        .total_size = 2 * rounded_weight_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = src0_cb_index,
            .data_format = weights_cb_data_format,
            .page_size = rounded_weight_page_size,
        }}},
    });

    constexpr uint32_t src1_cb_index = tt::CBIndex::c_1;
    // The reader reads a full aligned index page (one tile) into this scratch CB, so it must be at
    // least one aligned page; FACE_HEIGHT sticks under-allocates it and the NOC read overruns.
    uint32_t index_page_size = round_up_to_mul32(input_element_size_bytes);
    uint32_t index_cb_page_size =
        std::max<uint32_t>(FACE_HEIGHT * index_page_size, static_cast<uint32_t>(a.buffer()->aligned_page_size()));
    desc.cbs.push_back(CBDescriptor{
        .total_size = index_cb_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = src1_cb_index,
            .data_format = input_cb_data_format,
            .page_size = index_cb_page_size,
        }}},
    });

    constexpr uint32_t src2_cb_index = tt::CBIndex::c_2;
    if (embeddings_type == EmbeddingsType::PADDED) {
        uint32_t cache_page_size = round_up_to_mul32(weight_page_size);
        desc.cbs.push_back(CBDescriptor{
            .total_size = cache_page_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = src2_cb_index,
                .data_format = weights_cb_data_format,
                .page_size = cache_page_size,
            }}},
        });
    } else if (embeddings_type == EmbeddingsType::BINARY) {
        uint32_t cache_page_size = round_up_to_mul32(weight_page_size);
        desc.cbs.push_back(CBDescriptor{
            .total_size = 2 * cache_page_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = src2_cb_index,
                .data_format = weights_cb_data_format,
                .page_size = cache_page_size,
            }}},
        });
    }

    uint32_t output_cb_index = src0_cb_index;

    // Create Kernels
    // reader
    std::vector<uint32_t> embedding_compile_time_args = {
        (std::uint32_t)src0_cb_index,
        (std::uint32_t)src1_cb_index,
        (std::uint32_t)src2_cb_index,
        (std::uint32_t)input_page_size,
        (std::uint32_t)weight_page_size,
        (std::uint32_t)a.logical_shape()[-1],  // width/length of a row
        (std::uint32_t)FACE_HEIGHT};
    tt::tt_metal::TensorAccessorArgs(*a.buffer()).append_to(embedding_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(*weights.buffer()).append_to(embedding_compile_time_args);

    EmbeddingsIndexType embeddings_index_type;
    if (a.dtype() == DataType::BFLOAT16) {
        embeddings_index_type = EmbeddingsIndexType::BFP16;
    } else {
        embeddings_index_type = EmbeddingsIndexType::UINT32;
    }

    KernelDescriptor::Defines embedding_defines = {
        {enchantum::to_string(embeddings_type).data(), "1"}, {enchantum::to_string(embeddings_index_type).data(), "1"}};

    if (a.logical_shape()[-1] <= FACE_HEIGHT) {
        embedding_defines.emplace_back("ONLY_ONE_FACE_COLUMN", "1");
    }

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = "ttnn/cpp/ttnn/operations/embedding/device/kernels/dataflow/embedding_ind_tilized.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(embedding_compile_time_args);
    reader_desc.defines = std::move(embedding_defines);
    reader_desc.config = ReaderConfigDescriptor{};

    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)output_cb_index, (std::uint32_t)output_page_size};
    tt::tt_metal::TensorAccessorArgs(*output.buffer()).append_to(writer_compile_time_args);

    // Tilized writer
    KernelDescriptor writer_desc;
    writer_desc.kernel_source = "ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.config = WriterConfigDescriptor{};

    uint32_t col_offset = 0;
    uint32_t weight_offset = 0;

    auto cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, false);
    auto* a_buffer = a.buffer();
    auto* weights_buffer = weights.buffer();
    auto* output_buffer = output.buffer();

    uint32_t row = 0;
    uint32_t tiles_per_tile_row = (num_cols + TILE_HEIGHT - 1) / TILE_HEIGHT;

    reader_desc.runtime_args.reserve(cores.size());
    writer_desc.runtime_args.reserve(cores.size());

    for (uint32_t i = 0; i < cores.size(); ++i) {
        const CoreCoord& core = cores[i];
        col_offset = weight_offset % num_cols;
        row = weight_offset / num_cols;

        uint32_t local_num_blocks = i < g1_numcores ? num_blocks_per_core_group_1 : num_blocks_per_core_group_2;
        uint32_t r_f_offset = (((row % TILE_HEIGHT) / FACE_HEIGHT) * 2 * FACE_HW) + ((row % FACE_HEIGHT) * FACE_HEIGHT);
        // Offset by one face size if we are in the right half of the tile + where we are in the row
        uint32_t c_f_offset = ((col_offset % TILE_HEIGHT) / FACE_HEIGHT) * FACE_HW;
        uint32_t face_offset = r_f_offset + c_f_offset;
        uint32_t curr_tile = ((row / TILE_HEIGHT) * tiles_per_tile_row) + (col_offset / TILE_HEIGHT);

        // Reader
        {
            KernelDescriptor::RTArgList reader_args;
            reader_args.push_back(a_buffer);
            reader_args.push_back(weights_buffer);
            reader_args.push_back(curr_tile);
            reader_args.push_back(face_offset);
            reader_args.push_back(local_num_blocks);
            reader_args.push_back(col_offset);
            reader_args.push_back(static_cast<uint32_t>(col_offset % FACE_HEIGHT));  // starting col in the face row
            if (embeddings_type == EmbeddingsType::PADDED) {
                reader_args.push_back(pad_token.value());
            }
            reader_desc.emplace_runtime_args(core, reader_args);
        }

        // Writer
        writer_desc.emplace_runtime_args(
            core, {output_buffer, static_cast<uint32_t>(output_page_size), local_num_blocks, weight_offset});

        weight_offset += local_num_blocks;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));

    return desc;
}

}  // namespace ttnn::prim
