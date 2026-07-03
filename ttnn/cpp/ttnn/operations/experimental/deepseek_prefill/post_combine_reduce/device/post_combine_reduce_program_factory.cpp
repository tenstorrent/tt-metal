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

    auto cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, row_major);

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(combine_output.dtype());
    tt::DataFormat weight_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(weights.dtype());
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(tensor_return_value.dtype());

    uint32_t tile_size = tt::tile_size(input_cb_data_format);

    // c_0: Stream one expert at a time through c_0 to minimize L1 footprint.
    uint32_t combine_cb_size = emb_dim_cb_tiles * tile_size;
    desc.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = combine_cb_size,
        .core_ranges = core_range_set,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_0),
            .data_format = input_cb_data_format,
            .page_size = tile_size,
        }}},
    });

    // c_1: Stream one weight at a time (matching expert-by-expert input streaming).
    uint32_t weight_cb_size = tile_size;
    desc.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = weight_cb_size,
        .core_ranges = core_range_set,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_1),
            .data_format = weight_cb_data_format,
            .page_size = tile_size,
        }}},
    });

    // c_2 / c_3 CBs (dispatch table, indices) are only allocated when the
    // DeepSeek skip path is in use; the GPT-OSS path does not touch them.
    uint32_t dispatch_table_num_pages = 0;
    uint32_t dispatch_table_page_size_val = 0;
    uint32_t dispatch_table_aligned_page_size = 0;
    uint32_t indices_page_size_val = 0;
    uint32_t indices_aligned_page_size = 0;
    uint32_t indices_pages_per_core = 0;

    if (use_dispatch_table_skip) {
        const auto& indices = *indices_opt;
        const auto& expert_dispatch_table = *dispatch_table_opt;

        tt::DataFormat indices_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(indices.dtype());
        tt::DataFormat dispatch_table_cb_data_format =
            tt::tt_metal::datatype_to_dataformat_converter(expert_dispatch_table.dtype());

        // c_2: Dispatch table scratch — loaded once by writer, read by compute.
        dispatch_table_num_pages = get_num_pages(expert_dispatch_table);
        dispatch_table_page_size_val = get_page_size(expert_dispatch_table);
        dispatch_table_aligned_page_size = get_aligned_page_size(expert_dispatch_table);
        uint32_t dispatch_table_cb_size = dispatch_table_num_pages * dispatch_table_aligned_page_size;
        desc.cbs.push_back(tt::tt_metal::CBDescriptor{
            .total_size = dispatch_table_cb_size,
            .core_ranges = core_range_set,
            .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_2),
                .data_format = dispatch_table_cb_data_format,
                .page_size = dispatch_table_aligned_page_size,
            }}},
        });

        // c_3: Indices scratch — loaded one chunk at a time (reused per chunk).
        indices_page_size_val = get_page_size(indices);
        indices_aligned_page_size = get_aligned_page_size(indices);
        indices_pages_per_core = TOKENS_PER_CHUNK;
        uint32_t indices_cb_size = indices_pages_per_core * indices_aligned_page_size;
        desc.cbs.push_back(tt::tt_metal::CBDescriptor{
            .total_size = indices_cb_size,
            .core_ranges = core_range_set,
            .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_3),
                .data_format = indices_cb_data_format,
                .page_size = indices_aligned_page_size,
            }}},
        });
    }

    // c_16: Output — one chunk at a time (compute produces TOKENS_PER_CHUNK tiles per iteration)
    uint32_t output_cb_size = TOKENS_PER_CHUNK * emb_dim_cb_tiles * tile_size;
    desc.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = output_cb_size,
        .core_ranges = core_range_set,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_16),
            .data_format = output_cb_data_format,
            .page_size = tile_size,
        }}},
    });

    // c_17: Row-major scratch for tilize — one chunk at a time
    uint32_t rowmajor_cb_size = TOKENS_PER_CHUNK * emb_dim_cb_tiles * tile_size;
    desc.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = rowmajor_cb_size,
        .core_ranges = core_range_set,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_17),
            .data_format = output_cb_data_format,
            .page_size = tile_size,
        }}},
    });

    auto* combine_buffer = combine_output.buffer();
    auto* weight_buffer = weights.buffer();
    auto* output_buffer = tensor_return_value.buffer();
    auto* indices_buffer = use_dispatch_table_skip ? indices_opt->buffer() : nullptr;
    auto* dispatch_table_buffer = use_dispatch_table_skip ? dispatch_table_opt->buffer() : nullptr;

    // Reader compile-time args: num_experts, emb_dim_cb_tiles, emb_dim_bytes, combine accessor.
    // Reader does not need to know about expert-skip; that logic lives in compute + writer.
    std::vector<uint32_t> reader_compile_time_args = {
        num_experts,
        emb_dim_cb_tiles,
        emb_dim_bytes,
    };
    tt::tt_metal::TensorAccessorArgs(combine_buffer).append_to(reader_compile_time_args);

    // Compute compile-time args. The skip-mode toggle is appended last so the
    // DeepSeek-only arg (dispatch_table page count) retains a stable position
    // across both modes (it is zero in the GPT-OSS path).
    std::vector<uint32_t> compute_compile_time_args = {
        num_experts,
        emb_dim_cb_tiles,
        dispatch_table_num_pages,
        static_cast<uint32_t>(use_dispatch_table_skip ? 1 : 0),
    };

    // Writer compile-time args use a fixed layout across both paths. In the
    // GPT-OSS path the dispatch_table / indices metadata slots carry zeros and
    // the dispatch_table_accessor_args / indices_accessor_args slots reuse the
    // weight tensor's TensorAccessorArgs as an always-valid placeholder; the
    // kernel guards every use of them with `if constexpr (use_dispatch_table_skip)`.
    std::vector<uint32_t> writer_compile_time_args = {
        num_experts,
        emb_dim_cb_tiles,
        emb_dim_out_tiles,
        dispatch_table_num_pages,
        dispatch_table_page_size_val,
        dispatch_table_aligned_page_size,
        indices_page_size_val,
        indices_aligned_page_size,
    };
    tt::tt_metal::TensorAccessorArgs(weight_buffer).append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(output_buffer).append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(use_dispatch_table_skip ? dispatch_table_buffer : weight_buffer)
        .append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(use_dispatch_table_skip ? indices_buffer : weight_buffer)
        .append_to(writer_compile_time_args);
    writer_compile_time_args.push_back(static_cast<uint32_t>(use_dispatch_table_skip ? 1 : 0));

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
    uint32_t token_start = 0;
    for (uint32_t i = 0; i < num_cores; ++i) {
        const CoreCoord& core = cores[i];
        const uint32_t chunks_this_core = base_chunks_per_core + (i < extra_chunks ? 1 : 0);

        // Reader RT args: [combine_buffer*, token_start, chunks_this_core].
        // Push the buffer pointer first so the framework records a BufferBinding
        // for the cache-hit fast path.
        tt::tt_metal::KernelDescriptor::RTArgList reader_rt_args;
        reader_rt_args.push_back(combine_buffer);
        reader_rt_args.push_back(token_start);
        reader_rt_args.push_back(chunks_this_core);
        reader_kernel_desc.emplace_runtime_args(core, reader_rt_args);

        // Compute RT args (no Buffer*).
        tt::tt_metal::KernelDescriptor::RTArgList compute_rt_args;
        compute_rt_args.push_back(token_start);
        compute_rt_args.push_back(chunks_this_core);
        compute_kernel_desc.emplace_runtime_args(core, compute_rt_args);

        // Writer runtime args: weight_addr, output_addr,
        //   (deepseek only: dispatch_table_addr, indices_addr),
        //   token_start, chunks_this_core.  All buffer addresses are pushed as
        //   Buffer* so the framework records bindings for the fast path.
        tt::tt_metal::KernelDescriptor::RTArgList writer_rt_args;
        writer_rt_args.push_back(weight_buffer);
        writer_rt_args.push_back(output_buffer);
        if (use_dispatch_table_skip) {
            writer_rt_args.push_back(dispatch_table_buffer);
            writer_rt_args.push_back(indices_buffer);
        }
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
