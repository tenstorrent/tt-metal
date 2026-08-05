// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "embeddings_rm_program_factory.hpp"
#include "embedding_program_factory_common.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_align.hpp>

namespace ttnn::prim {

using namespace tt;
using namespace tt::tt_metal;

tt::tt_metal::ProgramDescriptor EmbeddingsRMProgramFactory::create_descriptor(
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

    tt::tt_metal::Buffer* out_buffer = output.buffer();

    ////////////////////////////////////////////////////////////////////////////
    //                      Grayskull Device Setup
    ////////////////////////////////////////////////////////////////////////////
    // This should allocate a DRAM buffer on the device
    IDevice* device = a.device();

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////

    bool output_sharded = is_sharded(output.buffer()->buffer_layout());

    uint32_t input_element_size_bytes = a.element_size();
    uint32_t weights_element_size_bytes = weights.element_size();
    uint32_t output_element_size_bytes = output.element_size();

    // row major, page size is last dim
    uint32_t input_page_size = a.padded_shape()[-1] * input_element_size_bytes;
    uint32_t weight_page_size = weights.padded_shape()[-1] * weights_element_size_bytes;
    uint32_t output_page_size = output.padded_shape()[-1] * output_element_size_bytes;

    // weights shape is [1, 1, num_embeddings, num_dim]

    uint32_t batch_size = a.padded_shape()[0];
    uint32_t num_output_rows_per_batch = a.padded_shape()[-1];
    uint32_t num_output_rows = num_output_rows_per_batch * batch_size;
    auto alignment = a.buffer()->alignment();
    uint32_t block_height = (alignment / input_element_size_bytes);
    uint32_t num_blocks = num_output_rows;
    uint32_t num_blocks_per_batch = num_output_rows_per_batch;

    // setup problem and grid size

    uint32_t problem_size = num_blocks;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();

    uint32_t num_blocks_per_core_group_1 = 0, num_blocks_per_core_group_2 = 0;
    CoreRangeSet all_cores, core_group_1, core_group_2;
    bool row_major = false;
    if (output_sharded) {
        const auto& shard_spec = output.shard_spec().value();
        all_cores = shard_spec.grid;
        core_group_1 = all_cores;
        num_blocks_per_core_group_1 = shard_spec.shape[0];
        num_blocks_per_core_group_2 = 0;
        row_major = shard_spec.orientation == ShardOrientation::ROW_MAJOR;
    } else {
        std::tie(
            std::ignore,
            all_cores,
            core_group_1,
            core_group_2,
            num_blocks_per_core_group_1,
            num_blocks_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, problem_size);
    }
    uint32_t g1_numcores = core_group_1.num_cores();

    // Create Buffers
    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());

    tt::DataFormat weights_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(weights.dtype());

    constexpr uint32_t out_cb_index = tt::CBIndex::c_0;
    uint32_t rounded_weight_page_size = tt::align(weight_page_size, alignment);

    constexpr uint32_t max_l1_budget_bytes = 1024 * 1024;  // 1MB budget for embedding CB
    uint32_t chunk_size;
    uint32_t num_chunks;
    uint32_t last_chunk_size;
    bool use_chunked = !output_sharded && rounded_weight_page_size > max_l1_budget_bytes;
    if (use_chunked) {
        chunk_size = (max_l1_budget_bytes / alignment) * alignment;
        chunk_size = std::max(chunk_size, alignment);
        num_chunks = (rounded_weight_page_size + chunk_size - 1) / chunk_size;
        last_chunk_size = rounded_weight_page_size - (num_chunks - 1) * chunk_size;
    } else {
        chunk_size = rounded_weight_page_size;
        num_chunks = 1;
        last_chunk_size = rounded_weight_page_size;
    }

    uint32_t out_cb_size;
    if (output_sharded) {
        out_cb_size = output.buffer()->aligned_size_per_bank();
    } else {
        uint32_t buffering_size = (num_blocks_per_core_group_1 > 1 || num_blocks_per_core_group_2 > 1) ? 2 : 1;
        out_cb_size = buffering_size * chunk_size;
    }
    CBDescriptor out_cb_desc{
        .total_size = out_cb_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = out_cb_index,
            .data_format = weights_cb_data_format,
            .page_size = chunk_size,
        }}},
    };
    if (output_sharded) {
        out_cb_desc.buffer = out_buffer;
    }
    desc.cbs.push_back(std::move(out_cb_desc));

    constexpr uint32_t src1_cb_index = tt::CBIndex::c_1;
    uint32_t index_page_size = round_up_to_mul32(input_element_size_bytes);
    desc.cbs.push_back(CBDescriptor{
        .total_size = block_height * index_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = src1_cb_index,
            .data_format = input_cb_data_format,
            .page_size = block_height * index_page_size,
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

    // Create Kernels
    // reader
    std::vector<uint32_t> embedding_compile_time_args = {
        (std::uint32_t)out_cb_index,
        (std::uint32_t)src1_cb_index,
        (std::uint32_t)src2_cb_index,
        (std::uint32_t)input_page_size,
        (std::uint32_t)weight_page_size,
        (std::uint32_t)block_height,
        (std::uint32_t)block_height * input_element_size_bytes,
        (std::uint32_t)chunk_size,
        (std::uint32_t)num_chunks,
        (std::uint32_t)last_chunk_size};
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

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = "ttnn/cpp/ttnn/operations/embedding/device/kernels/dataflow/embeddings.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(embedding_compile_time_args);
    reader_desc.defines = embedding_defines;
    reader_desc.config = ReaderConfigDescriptor{};

    // Writer
    std::optional<KernelDescriptor> writer_desc;
    if (!output_sharded) {
        if (use_chunked) {
            std::vector<uint32_t> writer_compile_time_args = {
                (std::uint32_t)out_cb_index,
                (std::uint32_t)output_page_size,
                (std::uint32_t)chunk_size,
                (std::uint32_t)num_chunks,
                (std::uint32_t)last_chunk_size};
            tt::tt_metal::TensorAccessorArgs(*output.buffer()).append_to(writer_compile_time_args);

            KernelDescriptor w;
            w.kernel_source =
                "ttnn/cpp/ttnn/operations/embedding/device/kernels/dataflow/embeddings_rm_writer_chunked.cpp";
            w.source_type = KernelDescriptor::SourceType::FILE_PATH;
            w.core_ranges = all_cores;
            w.compile_time_args = std::move(writer_compile_time_args);
            w.config = WriterConfigDescriptor{};
            writer_desc = std::move(w);
        } else {
            std::vector<uint32_t> writer_compile_time_args = {
                (std::uint32_t)out_cb_index, (std::uint32_t)output_page_size};
            tt::tt_metal::TensorAccessorArgs(*output.buffer()).append_to(writer_compile_time_args);

            KernelDescriptor w;
            w.kernel_source = "ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp";
            w.source_type = KernelDescriptor::SourceType::FILE_PATH;
            w.core_ranges = all_cores;
            w.compile_time_args = std::move(writer_compile_time_args);
            w.config = WriterConfigDescriptor{};
            writer_desc = std::move(w);
        }
    }

    uint32_t input_offset = 0;

    auto cores = corerange_to_cores(all_cores, std::nullopt, row_major);
    auto* a_buffer = a.buffer();
    auto* weights_buffer = weights.buffer();
    auto* output_buffer = output.buffer();

    reader_desc.runtime_args.reserve(cores.size());
    if (writer_desc.has_value()) {
        writer_desc->runtime_args.reserve(cores.size());
    }

    for (uint32_t i = 0; i < cores.size(); ++i) {
        const CoreCoord& core = cores[i];

        uint32_t local_num_blocks = i < g1_numcores ? num_blocks_per_core_group_1 : num_blocks_per_core_group_2;

        // Reader
        {
            KernelDescriptor::RTArgList reader_args;
            reader_args.push_back(a_buffer);
            reader_args.push_back(weights_buffer);
            reader_args.push_back(input_offset / num_blocks_per_batch);
            reader_args.push_back(
                tt::round_down(input_offset % num_blocks_per_batch, block_height) * input_element_size_bytes);
            reader_args.push_back(local_num_blocks);
            reader_args.push_back(input_offset % num_blocks_per_batch % block_height);
            if (embeddings_type == EmbeddingsType::PADDED) {
                reader_args.push_back(pad_token.value());
            }
            reader_desc.emplace_runtime_args(core, reader_args);
        }

        // Writer
        if (!output_sharded) {
            if (use_chunked) {
                writer_desc->emplace_runtime_args(core, {output_buffer, local_num_blocks, input_offset});
            } else {
                writer_desc->emplace_runtime_args(
                    core, {output_buffer, static_cast<uint32_t>(output_page_size), local_num_blocks, input_offset});
            }
        }

        input_offset += local_num_blocks;
    }

    desc.kernels.push_back(std::move(reader_desc));
    if (writer_desc.has_value()) {
        desc.kernels.push_back(std::move(*writer_desc));
    }

    return desc;
}

}  // namespace ttnn::prim
