// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "post_combine_reduce_program_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::post_combine_reduce {

namespace {

uint32_t get_num_pages(const ttnn::Tensor& tensor) { return (uint32_t)tensor.buffer()->num_pages(); }
uint32_t get_page_size(const ttnn::Tensor& tensor) { return (uint32_t)tensor.buffer()->page_size(); }
uint32_t get_aligned_page_size(const ttnn::Tensor& tensor) { return (uint32_t)tensor.buffer()->aligned_page_size(); }

}  // namespace

tt::tt_metal::ProgramDescriptor PostCombineReduceProgramFactory::create_descriptor(
    const PostCombineReduceParams& operation_attributes,
    const PostCombineReduceInputs& tensor_args,
    ttnn::Tensor& tensor_return_value) {
    tt::tt_metal::ProgramDescriptor desc;

    const auto& combine_output = tensor_args.combine_output;
    const auto& weights = tensor_args.weights;
    const auto& indices_opt = tensor_args.indices;
    const auto& dispatch_table_opt = tensor_args.expert_dispatch_table;
    // both-or-neither enforced in validate(); here we just pick the skip mode
    const bool use_dispatch_table_skip = indices_opt.has_value();
    auto* device = combine_output.device();

    const auto& combine_shape = combine_output.padded_shape();

    const uint32_t expert_dim = operation_attributes.expert_dim;

    const uint32_t emb_dim = combine_shape[-1];
    const uint32_t num_experts = combine_shape[expert_dim];

    uint32_t num_tokens = 1;
    for (uint32_t i = 0; i < expert_dim; ++i) {
        num_tokens *= combine_shape[i];
    }

    constexpr uint32_t TILE_SIZE = 1024;  // 32 x 32 bfloat16 tile (element count)
    constexpr uint32_t TILE_WIDTH = 32;
    constexpr uint32_t BF16_BYTES = 2;

    // Number of tile-sized CB pages needed to hold one emb_dim row.
    // ceil(emb_dim / 1024) supports non-1024-aligned dims (e.g. GPT-OSS 2880).
    const uint32_t emb_dim_cb_tiles = (emb_dim + TILE_SIZE - 1) / TILE_SIZE;
    // Number of real 32x32 output tiles per 32-token block.
    const uint32_t emb_dim_out_tiles = emb_dim / TILE_WIDTH;
    // Raw byte count for NoC reads in the reader (handles non-aligned emb_dim).
    const uint32_t emb_dim_bytes = emb_dim * BF16_BYTES;

    TT_FATAL(
        emb_dim % TILE_WIDTH == 0,
        "Embedding dimension {} must be divisible by tile width ({}); remainder is {}",
        emb_dim,
        TILE_WIDTH,
        emb_dim % TILE_WIDTH);
    TT_FATAL(
        emb_dim_cb_tiles <= 8,
        "Embedding dimension tiles {} must fit in 8 DST registers for batching",
        emb_dim_cb_tiles);

    TT_FATAL(num_experts <= 32, "post_combine_reduce: at most 32 slots per token, got {}", num_experts);

    constexpr uint32_t TOKENS_PER_CHUNK = 32;
    TT_FATAL(num_tokens > 0, "post_combine_reduce: num_tokens must be > 0, got {}", num_tokens);
    TT_FATAL(
        num_tokens % TOKENS_PER_CHUNK == 0,
        "Number of tokens {} must be divisible by {} for hardware tilization",
        num_tokens,
        TOKENS_PER_CHUNK);

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_cores_total = num_cores_x * num_cores_y;

    const uint32_t total_chunks = num_tokens / TOKENS_PER_CHUNK;
    const uint32_t num_cores = std::min(total_chunks, num_cores_total);
    const uint32_t base_chunks_per_core = total_chunks / num_cores;
    const uint32_t extra_chunks = total_chunks % num_cores;

    constexpr bool row_major = true;

    auto core_range_set = tt::tt_metal::num_cores_to_corerangeset(num_cores, compute_with_storage_grid_size, row_major);

    auto cores = tt::tt_metal::grid_to_cores(num_cores, num_cores_x, num_cores_y, row_major);

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(combine_output.dtype());
    tt::DataFormat weight_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(weights.dtype());
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(tensor_return_value.dtype());

    uint32_t tile_size = tt::tile_size(input_cb_data_format);

    const auto push_cb = [&](tt::CBIndex index, uint32_t num_pages, uint32_t page_size, tt::DataFormat format) {
        desc.cbs.push_back(tt::tt_metal::CBDescriptor{
            .total_size = num_pages * page_size,
            .core_ranges = core_range_set,
            .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(index),
                .data_format = format,
                .page_size = page_size,
            }}},
        });
    };

    // The reader streams only active (token, slot) pairs: one combine row (c_0) and one weight scalar in
    // element 0 of a tile (c_1) each, double-buffered so the next read overlaps the multiply.
    push_cb(tt::CBIndex::c_0, 2 * emb_dim_cb_tiles, tile_size, input_cb_data_format);
    push_cb(tt::CBIndex::c_1, 2, tile_size, weight_cb_data_format);

    // c_4: per-chunk active-slot count of each token (reader -> compute).
    constexpr uint32_t token_counts_page_size = TOKENS_PER_CHUNK * sizeof(uint32_t);
    push_cb(tt::CBIndex::c_4, 2, token_counts_page_size, tt::DataFormat::UInt32);

    // c_5: reader-private scratch for one chunk's routing weights, one aligned weight page per (token, slot).
    const uint32_t weight_aligned_page_size = get_aligned_page_size(weights);
    TT_FATAL(
        get_num_pages(weights) == num_tokens * num_experts,
        "post_combine_reduce: expected one weight page per (token, slot), got {} pages for {} x {}",
        get_num_pages(weights),
        num_tokens,
        num_experts);
    push_cb(tt::CBIndex::c_5, TOKENS_PER_CHUNK * num_experts, weight_aligned_page_size, weight_cb_data_format);

    // c_2 / c_3: reader-private dispatch table and one chunk's expert indices (dispatch-table mode only).
    uint32_t dispatch_table_num_pages = 0;
    uint32_t dispatch_table_aligned_page_size = 0;
    uint32_t dispatch_table_entries = 0;
    uint32_t indices_aligned_page_size = 0;
    if (use_dispatch_table_skip) {
        const auto& indices = *indices_opt;
        const auto& expert_dispatch_table = *dispatch_table_opt;

        dispatch_table_num_pages = get_num_pages(expert_dispatch_table);
        dispatch_table_aligned_page_size = get_aligned_page_size(expert_dispatch_table);
        // The reader indexes the table as one contiguous int32 array.
        TT_FATAL(
            dispatch_table_num_pages == 1 || get_page_size(expert_dispatch_table) == dispatch_table_aligned_page_size,
            "post_combine_reduce: a multi-page expert_dispatch_table must have aligned pages");
        dispatch_table_entries = expert_dispatch_table.logical_volume();
        push_cb(
            tt::CBIndex::c_2,
            dispatch_table_num_pages,
            dispatch_table_aligned_page_size,
            tt::tt_metal::datatype_to_dataformat_converter(expert_dispatch_table.dtype()));

        indices_aligned_page_size = get_aligned_page_size(indices);
        push_cb(
            tt::CBIndex::c_3,
            TOKENS_PER_CHUNK,
            indices_aligned_page_size,
            tt::tt_metal::datatype_to_dataformat_converter(indices.dtype()));
    }

    // c_16: tilized output, one chunk at a time.
    push_cb(tt::CBIndex::c_16, TOKENS_PER_CHUNK * emb_dim_cb_tiles, tile_size, output_cb_data_format);
    // c_17: row-major accumulator (tilize input), one chunk at a time.
    push_cb(tt::CBIndex::c_17, TOKENS_PER_CHUNK * emb_dim_cb_tiles, tile_size, output_cb_data_format);

    auto* combine_buffer = combine_output.buffer();
    auto* weight_buffer = weights.buffer();
    auto* output_buffer = tensor_return_value.buffer();
    auto* indices_buffer = use_dispatch_table_skip ? indices_opt->buffer() : nullptr;
    auto* dispatch_table_buffer = use_dispatch_table_skip ? dispatch_table_opt->buffer() : nullptr;

    // Reader compile-time args: fixed layout in both modes. In weight mode the dispatch-table metadata is zero
    // and the dispatch_table / indices accessor slots carry the weight tensor's args as placeholders.
    std::vector<uint32_t> reader_compile_time_args = {
        num_experts,
        emb_dim_cb_tiles,
        emb_dim_bytes,
        weight_aligned_page_size,
        dispatch_table_num_pages,
        dispatch_table_aligned_page_size,
        dispatch_table_entries,
        indices_aligned_page_size,
        static_cast<uint32_t>(use_dispatch_table_skip ? 1 : 0),
    };
    tt::tt_metal::TensorAccessorArgs(combine_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(weight_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(use_dispatch_table_skip ? dispatch_table_buffer : weight_buffer)
        .append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(use_dispatch_table_skip ? indices_buffer : weight_buffer)
        .append_to(reader_compile_time_args);

    std::vector<uint32_t> compute_compile_time_args = {emb_dim_cb_tiles};

    std::vector<uint32_t> writer_compile_time_args = {emb_dim_cb_tiles, emb_dim_out_tiles};
    tt::tt_metal::TensorAccessorArgs(output_buffer).append_to(writer_compile_time_args);

    // Build kernel descriptors and push them onto desc.kernels.  Stable indices
    // (0=reader, 1=compute, 2=writer) below let emplace_runtime_args identify
    // each kernel.
    tt::tt_metal::KernelDescriptor reader_kernel_desc;
    reader_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/post_combine_reduce/device/kernels/"
        "deepseek_moe_post_combine_reduce_reader.cpp";
    reader_kernel_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    reader_kernel_desc.core_ranges = core_range_set;
    reader_kernel_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_kernel_desc.config = tt::tt_metal::ReaderConfigDescriptor{};

    tt::tt_metal::KernelDescriptor compute_kernel_desc;
    compute_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/post_combine_reduce/device/kernels/"
        "deepseek_moe_post_combine_reduce_compute.cpp";
    compute_kernel_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    compute_kernel_desc.core_ranges = core_range_set;
    compute_kernel_desc.compile_time_args = std::move(compute_compile_time_args);
    compute_kernel_desc.config = tt::tt_metal::ComputeConfigDescriptor{
        .math_fidelity = MathFidelity::HiFi4,
        .fp32_dest_acc_en = false,
        .dst_full_sync_en = false,
    };

    tt::tt_metal::KernelDescriptor writer_kernel_desc;
    writer_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/post_combine_reduce/device/kernels/"
        "deepseek_moe_post_combine_reduce_writer.cpp";
    writer_kernel_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    writer_kernel_desc.core_ranges = core_range_set;
    writer_kernel_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_kernel_desc.config = tt::tt_metal::WriterConfigDescriptor{};

    // Distribute chunks of 32 tokens across cores. The first `extra_chunks` cores
    // get (base_chunks_per_core + 1) chunks; the remaining get base_chunks_per_core.
    // Buffer addresses go in as Buffer* so the framework records bindings for the cache-hit fast path.
    uint32_t token_start = 0;
    for (uint32_t i = 0; i < num_cores; ++i) {
        const CoreCoord& core = cores[i];
        const uint32_t chunks_this_core = base_chunks_per_core + (i < extra_chunks ? 1 : 0);

        // Reader: combine, weights, (dispatch-table mode: dispatch_table, indices), token_start, chunks.
        tt::tt_metal::KernelDescriptor::RTArgList reader_rt_args;
        reader_rt_args.push_back(combine_buffer);
        reader_rt_args.push_back(weight_buffer);
        if (use_dispatch_table_skip) {
            reader_rt_args.push_back(dispatch_table_buffer);
            reader_rt_args.push_back(indices_buffer);
        }
        reader_rt_args.push_back(token_start);
        reader_rt_args.push_back(chunks_this_core);
        reader_kernel_desc.emplace_runtime_args(core, reader_rt_args);

        tt::tt_metal::KernelDescriptor::RTArgList compute_rt_args;
        compute_rt_args.push_back(chunks_this_core);
        compute_kernel_desc.emplace_runtime_args(core, compute_rt_args);

        tt::tt_metal::KernelDescriptor::RTArgList writer_rt_args;
        writer_rt_args.push_back(output_buffer);
        writer_rt_args.push_back(token_start);
        writer_rt_args.push_back(chunks_this_core);
        writer_kernel_desc.emplace_runtime_args(core, writer_rt_args);

        token_start += chunks_this_core * TOKENS_PER_CHUNK;
    }

    desc.kernels.push_back(std::move(reader_kernel_desc));
    desc.kernels.push_back(std::move(compute_kernel_desc));
    desc.kernels.push_back(std::move(writer_kernel_desc));

    return desc;
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::post_combine_reduce
