// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "embeddings_fused_program_factory.hpp"
#include "embedding_program_factory_common.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::prim {

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;

tt::tt_metal::ProgramDescriptor EmbeddingsFusedProgramFactory::create_descriptor(
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

    // row major, page size is last dim
    uint32_t input_page_size = a.padded_shape()[-1] * input_element_size_bytes;
    uint32_t weight_page_size = weights.padded_shape()[-1] * weights_element_size_bytes;

    // weights shape is [1, 1, num_embeddings, num_dim]

    uint32_t batch_size = a.padded_shape()[0];
    uint32_t num_output_rows_per_batch = a.padded_shape()[-1];
    uint32_t num_output_rows = num_output_rows_per_batch * batch_size;
    // Note: num_blocks is just blocks along height
    uint32_t num_blocks = num_output_rows / TILE_HEIGHT;
    uint32_t num_blocks_per_batch = num_output_rows_per_batch / TILE_HEIGHT;
    uint32_t num_blocks_per_core_group_1 = 0, num_blocks_per_core_group_2 = 0, num_tiles_per_block = 0;
    CoreRangeSet all_cores, core_group_1, core_group_2;
    bool row_major = false;
    if (output_sharded) {
        const auto& shard_spec = output.shard_spec().value();
        all_cores = shard_spec.grid;
        core_group_1 = all_cores;
        num_blocks_per_core_group_1 = shard_spec.shape[0] / TILE_HEIGHT;
        num_blocks_per_core_group_2 = 0;
        num_tiles_per_block = shard_spec.shape[1] / TILE_WIDTH;
        row_major = shard_spec.orientation == ShardOrientation::ROW_MAJOR;
    } else {
        auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
        std::tie(
            std::ignore,
            all_cores,
            core_group_1,
            core_group_2,
            num_blocks_per_core_group_1,
            num_blocks_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_blocks);
        num_tiles_per_block = weights.padded_shape()[-1] / TILE_WIDTH;
    }
    uint32_t g1_numcores = core_group_1.num_cores();

    // Create Buffers
    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());

    EmbeddingsIndexType embeddings_index_type;
    if (a.dtype() == DataType::BFLOAT16) {
        embeddings_index_type = EmbeddingsIndexType::BFP16;
    } else {
        embeddings_index_type = EmbeddingsIndexType::UINT32;
    }

    tt::DataFormat weights_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(weights.dtype());
    uint32_t weights_single_tile_size = tt::tile_size(weights_cb_data_format);
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    // Hardcoded limit to reduce L1 usage. Should be updated to be tuned based on overall L1 usage
    constexpr uint32_t max_double_buffer_tiles = 64;

    constexpr uint32_t max_l1_budget_bytes = 1024 * 1024;  // 1MB budget for embedding CB
    uint32_t max_tiles_per_chunk = std::min(max_l1_budget_bytes / weights_single_tile_size, num_tiles_per_block);
    max_tiles_per_chunk = std::max(max_tiles_per_chunk, 1U);

    uint32_t required_memory_bytes = 2 * num_tiles_per_block * weights_single_tile_size;
    bool use_chunked_processing = required_memory_bytes > max_l1_budget_bytes;

    // For very large embeddings, use chunked processing
    uint32_t tiles_per_chunk;
    uint32_t num_chunks;
    uint32_t last_chunk_tiles;
    uint32_t buffering;

    if (use_chunked_processing) {
        // Keep tiles_per_chunk near the cap and let the last chunk be partial.
        // Reader/compute kernels handle the partial trailing chunk explicitly
        // via last_chunk_tiles.
        tiles_per_chunk = std::min(max_tiles_per_chunk, max_double_buffer_tiles);
        num_chunks = (num_tiles_per_block + tiles_per_chunk - 1) / tiles_per_chunk;
        last_chunk_tiles = num_tiles_per_block - (num_chunks - 1) * tiles_per_chunk;
        buffering = tiles_per_chunk > max_double_buffer_tiles ? 1 : 2;
    } else {
        // Use original non-chunked approach for smaller embeddings
        tiles_per_chunk = num_tiles_per_block;
        num_chunks = 1;
        last_chunk_tiles = num_tiles_per_block;
        buffering = num_tiles_per_block > max_double_buffer_tiles ? 1 : 2;
    }

    constexpr uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t cb0_size = buffering * tiles_per_chunk * weights_single_tile_size;
    desc.cbs.push_back(CBDescriptor{
        .total_size = cb0_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = src0_cb_index,
            .data_format = weights_cb_data_format,
            .page_size = weights_single_tile_size,
        }}},
    });

    constexpr uint32_t src1_cb_index = tt::CBIndex::c_1;
    desc.cbs.push_back(CBDescriptor{
        .total_size = TILE_HEIGHT * input_element_size_bytes,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = src1_cb_index,
            .data_format = input_cb_data_format,
            .page_size = TILE_HEIGHT * input_element_size_bytes,
        }}},
    });

    constexpr uint32_t output_cb_index = tt::CBIndex::c_2;
    uint32_t output_cb_size;
    if (output_sharded) {
        output_cb_size = output.buffer()->aligned_size_per_bank();
    } else {
        output_cb_size = buffering * tiles_per_chunk * output_single_tile_size;
    }
    CBDescriptor output_cb_desc{
        .total_size = output_cb_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = output_cb_index,
            .data_format = output_cb_data_format,
            .page_size = output_single_tile_size,
        }}},
    };
    if (output_sharded) {
        output_cb_desc.buffer = out_buffer;
    }
    desc.cbs.push_back(std::move(output_cb_desc));

    constexpr uint32_t src2_cb_index = tt::CBIndex::c_3;
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
    uint32_t weight_block_size;
    if (output_sharded) {
        weight_block_size = output.shard_spec().value().shape[1] * weights_element_size_bytes;
    } else {
        weight_block_size = weight_page_size;
    }

    // TODO: Can increase size for larger reads
    uint32_t input_block_size_bytes = TILE_HEIGHT * input_element_size_bytes;
    // Create Kernels
    // reader
    std::vector<uint32_t> embedding_compile_time_args = {
        (std::uint32_t)src0_cb_index,
        (std::uint32_t)src1_cb_index,
        (std::uint32_t)src2_cb_index,
        (std::uint32_t)input_page_size,
        (std::uint32_t)weight_page_size,
        (std::uint32_t)weight_block_size,
        (std::uint32_t)tiles_per_chunk,
        (std::uint32_t)input_block_size_bytes,
        (std::uint32_t)num_chunks,
        (std::uint32_t)last_chunk_tiles};
    tt::tt_metal::TensorAccessorArgs(*a.buffer()).append_to(embedding_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(*weights.buffer()).append_to(embedding_compile_time_args);

    KernelDescriptor::Defines embedding_defines = {
        {enchantum::to_string(embeddings_type).data(), "1"}, {enchantum::to_string(embeddings_index_type).data(), "1"}};

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = "ttnn/cpp/ttnn/operations/embedding/device/kernels/dataflow/embeddings_tilize.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(embedding_compile_time_args);
    reader_desc.defines = embedding_defines;
    reader_desc.config = ReaderConfigDescriptor{};

    // Compute kernels: split across the two core groups, each with its own
    // per_core_block_cnt compile-time arg. We must build them as separate
    // KernelDescriptors because their compile_time_args / core_ranges differ.
    std::optional<KernelDescriptor> compute_desc_1;
    std::optional<KernelDescriptor> compute_desc_2;
    const char* compute_kernel_path =
        use_chunked_processing ? "ttnn/cpp/ttnn/operations/embedding/device/kernels/compute/tilize_chunked.cpp"
                               : "ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/compute/tilize.cpp";

    if (num_blocks_per_core_group_1 > 0) {
        std::vector<uint32_t> compute_args_1 = {
            uint32_t(src0_cb_index),                // input embeddings_cb_index
            uint32_t(output_cb_index),              // output_cb_index
            uint32_t(num_blocks_per_core_group_1),  // per_core_block_cnt
            uint32_t(tiles_per_chunk),              // tiles_per_chunk
            uint32_t(num_chunks),                   // num_chunks per block
            uint32_t(last_chunk_tiles)              // last_chunk_tiles
        };
        KernelDescriptor compute_desc;
        compute_desc.kernel_source = compute_kernel_path;
        compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        compute_desc.core_ranges = core_group_1;
        compute_desc.compile_time_args = std::move(compute_args_1);
        compute_desc.config = ComputeConfigDescriptor{};
        compute_desc_1 = std::move(compute_desc);
    }

    if (num_blocks_per_core_group_2 > 0) {
        std::vector<uint32_t> compute_args_2 = {
            uint32_t(src0_cb_index),                // input embeddings_cb_index
            uint32_t(output_cb_index),              // output_cb_index
            uint32_t(num_blocks_per_core_group_2),  // per_core_block_cnt
            uint32_t(tiles_per_chunk),              // tiles_per_chunk
            uint32_t(num_chunks),                   // num_chunks per block
            uint32_t(last_chunk_tiles)              // last_chunk_tiles
        };
        KernelDescriptor compute_desc;
        compute_desc.kernel_source = compute_kernel_path;
        compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        compute_desc.core_ranges = core_group_2;
        compute_desc.compile_time_args = std::move(compute_args_2);
        compute_desc.config = ComputeConfigDescriptor{};
        compute_desc_2 = std::move(compute_desc);
    }

    // TODO: We can use the second risc to do more work in parallel
    std::optional<KernelDescriptor> writer_desc;
    if (!output_sharded) {
        std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)output_cb_index};
        tt::tt_metal::TensorAccessorArgs(*output.buffer()).append_to(writer_compile_time_args);

        // Tilized writer
        KernelDescriptor w;
        w.kernel_source =
            "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp";
        w.source_type = KernelDescriptor::SourceType::FILE_PATH;
        w.core_ranges = all_cores;
        w.compile_time_args = std::move(writer_compile_time_args);
        w.config = WriterConfigDescriptor{};
        writer_desc = std::move(w);
    }

    auto cores = corerange_to_cores(all_cores, std::nullopt, row_major);

    auto* a_buffer = a.buffer();
    auto* weights_buffer = weights.buffer();
    auto* output_buffer = output.buffer();

    reader_desc.runtime_args.reserve(cores.size());
    if (writer_desc.has_value()) {
        writer_desc->runtime_args.reserve(cores.size());
    }

    uint32_t input_offset = 0;
    uint32_t weight_offset = 0;
    uint32_t tile_offset = 0;
    for (uint32_t i = 0; i < cores.size(); ++i) {
        const CoreCoord& core = cores[i];

        uint32_t local_num_blocks = i < g1_numcores ? num_blocks_per_core_group_1 : num_blocks_per_core_group_2;

        // Reader
        {
            KernelDescriptor::RTArgList reader_args;
            reader_args.push_back(a_buffer);
            reader_args.push_back(weights_buffer);
            reader_args.push_back(input_offset / num_blocks_per_batch);
            reader_args.push_back(input_offset % num_blocks_per_batch * input_block_size_bytes);
            reader_args.push_back(weight_offset);
            reader_args.push_back(local_num_blocks);
            if (embeddings_type == EmbeddingsType::PADDED) {
                reader_args.push_back(pad_token.value());
            }
            reader_desc.emplace_runtime_args(core, reader_args);
        }

        // Writer
        if (!output_sharded) {
            writer_desc->emplace_runtime_args(
                core, {output_buffer, static_cast<uint32_t>(num_tiles_per_block * local_num_blocks), tile_offset});
            tile_offset += local_num_blocks * num_tiles_per_block;
            input_offset += local_num_blocks;
        } else {
            weight_offset += weight_block_size;
            if (weight_offset == weight_page_size) {
                weight_offset = 0;
                input_offset += local_num_blocks;
            }
        }
    }

    desc.kernels.push_back(std::move(reader_desc));
    if (writer_desc.has_value()) {
        desc.kernels.push_back(std::move(*writer_desc));
    }
    if (compute_desc_1.has_value()) {
        desc.kernels.push_back(std::move(*compute_desc_1));
    }
    if (compute_desc_2.has_value()) {
        desc.kernels.push_back(std::move(*compute_desc_2));
    }

    return desc;
}

}  // namespace ttnn::prim
