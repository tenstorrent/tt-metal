// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "post_combine_reduce_program_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/math.hpp>

#include "ttnn/operations/experimental/deepseek_prefill/per_token_cast_to_fp8/per_token_cast_to_fp8.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::post_combine_reduce {

namespace {

uint32_t get_num_pages(const ttnn::Tensor& tensor) { return (uint32_t)tensor.buffer()->num_pages(); }
uint32_t get_page_size(const ttnn::Tensor& tensor) { return (uint32_t)tensor.buffer()->page_size(); }
uint32_t get_aligned_page_size(const ttnn::Tensor& tensor) { return (uint32_t)tensor.buffer()->aligned_page_size(); }

// 1xB scaled-fp8 combine rows [H e4m3 | H/B fp32 scales] (fused DQ2, sparse SDPA style): per 32-token block the
// reader stacks each token's compacted local slots as row-major e4m3 rows and stages weight * scale per row and scale
// block, and compute tilizes the rows (tile row = token) and applies it with column-broadcast multiplies on the FPU.
tt::tt_metal::ProgramDescriptor create_fp8_block_descriptor(
    const PostCombineReduceParams& operation_attributes,
    const PostCombineReduceInputs& tensor_args,
    ttnn::Tensor& tensor_return_value) {
    using tt::tt_metal::CBDescriptor;
    using tt::tt_metal::CBFormatDescriptor;
    tt::tt_metal::ProgramDescriptor desc;

    const auto& combine_output = tensor_args.combine_output;
    const auto& weights = tensor_args.weights;
    const auto& indices = *tensor_args.indices;
    const auto& expert_dispatch_table = *tensor_args.expert_dispatch_table;
    auto* device = combine_output.device();
    const auto& combine_shape = combine_output.padded_shape();
    const uint32_t expert_dim = operation_attributes.expert_dim;

    constexpr uint32_t TILE_WIDTH = 32;
    constexpr uint32_t TOKENS_PER_CHUNK = 32;
    constexpr uint32_t DRAM_ALIGN = 64;  // Blackhole DRAM reads keep source and destination congruent mod 64 B
    const uint32_t row_bytes = combine_shape[-1];
    // Rows are [H e4m3 | H / sbw fp32 scales], maybe padded to whole 64 B lines (combine_fabric2d's are); one scale
    // covers tiles_per_scale = sbw / 32 tiles of a row.
    const uint32_t sbw = operation_attributes.fp8_scale_block_w;
    const uint32_t tiles_per_scale = sbw / TILE_WIDTH;
    const uint32_t emb_dim = per_token_cast_to_fp8::packed_row_h(row_bytes, sbw);
    TT_FATAL(emb_dim != 0, "bad 1x{} fp8 row width {}", sbw, row_bytes);
    const uint32_t emb_wt = emb_dim / TILE_WIDTH;
    const uint32_t num_experts = combine_shape[expert_dim];
    uint32_t num_tokens = 1;
    for (uint32_t i = 0; i < expert_dim; ++i) {
        num_tokens *= combine_shape[i];
    }
    TT_FATAL(num_tokens % TOKENS_PER_CHUNK == 0, "Number of tokens {} must be divisible by 32", num_tokens);
    TT_FATAL(num_experts <= 255, "top-k {} too large", num_experts);
    const uint32_t total_chunks = num_tokens / TOKENS_PER_CHUNK;

    const auto grid = device->compute_with_storage_grid_size();
    const uint32_t num_cores_total = grid.x * grid.y;
    // Column parts: part_wt tiles per core, a multiple of the 4-tile DEST batch and of a scale block. A block costs a
    // core about the same at any width, its reads being latency-bound, so the busiest core's block count sets the
    // op's time: pick the width that gives it the fewest blocks, then the narrowest such. c_0 holds a whole block's
    // top-k stages, which bounds the width.
    constexpr uint32_t RM_BYTES_MAX = 512 * 1024;
    const uint32_t fp8_tile_bytes = tt::tile_size(tt::DataFormat::Fp8_e4m3);
    uint32_t part_wt = 0;
    uint32_t best_blocks = 0;
    for (uint32_t d = 4; d <= emb_wt && num_experts * d * fp8_tile_bytes <= RM_BYTES_MAX; d += 4) {
        if (emb_wt % d != 0 || d % tiles_per_scale != 0 || emb_wt / d > num_cores_total) {
            continue;
        }
        const uint32_t cores = std::min(total_chunks, num_cores_total / (emb_wt / d));
        const uint32_t blocks = (total_chunks + cores - 1) / cores;
        if (part_wt == 0 || blocks < best_blocks) {
            part_wt = d;
            best_blocks = blocks;
        }
    }
    TT_FATAL(part_wt != 0, "fp8 post_combine_reduce: {} tiles per row have no part width that fits", emb_wt);
    const uint32_t num_parts = emb_wt / part_wt;
    const uint32_t cores_per_part = std::max(1u, std::min(total_chunks, num_cores_total / num_parts));
    TT_FATAL(
        cores_per_part * num_parts <= num_cores_total,
        "1xB fp8 post_combine_reduce: {} parts exceed the grid",
        num_parts);
    const uint32_t num_cores = cores_per_part * num_parts;
    const uint32_t base_chunks_per_core = total_chunks / cores_per_part;
    const uint32_t extra_chunks = total_chunks % cores_per_part;
    const uint32_t part_row_bytes = part_wt * TILE_WIDTH;
    TT_FATAL(part_row_bytes % DRAM_ALIGN == 0, "part row {} B must be 64-B aligned", part_row_bytes);
    const uint32_t scale_slot_bytes = tt::round_up(DRAM_ALIGN - 4 + part_wt * 4, DRAM_ALIGN);

    auto core_range_set = tt::tt_metal::num_cores_to_corerangeset(num_cores, grid, true);
    auto cores = tt::tt_metal::grid_to_cores(num_cores, grid.x, grid.y, true);

    const auto bf16 = tt::DataFormat::Float16_b;
    const uint32_t bf16_tile = tt::tile_size(bf16);
    const uint32_t fp8_tile = tt::tile_size(tt::DataFormat::Fp8_e4m3);
    auto add_cb = [&](tt::CBIndex index, tt::DataFormat df, uint32_t page_size, uint32_t num_pages) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = page_size * num_pages,
            .core_ranges = core_range_set,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(index), .data_format = df, .page_size = page_size}}},
        });
    };
    // Sizes stay 64-B multiples so every CB base keeps the DRAM-read congruence.
    add_cb(tt::CBIndex::c_0, tt::DataFormat::Fp8_e4m3, fp8_tile, num_experts * part_wt);  // a block's stage rows
    add_cb(tt::CBIndex::c_1, bf16, bf16_tile, 2);  // unused here, but the shared writer reads its tile size
    const uint32_t dispatch_table_num_pages = get_num_pages(expert_dispatch_table);
    const uint32_t dispatch_table_aligned_page_size = get_aligned_page_size(expert_dispatch_table);
    add_cb(
        tt::CBIndex::c_2,
        tt::tt_metal::datatype_to_dataformat_converter(expert_dispatch_table.dtype()),
        tt::round_up(dispatch_table_num_pages * dispatch_table_aligned_page_size, DRAM_ALIGN),
        1);
    const uint32_t indices_aligned_page_size = get_aligned_page_size(indices);
    add_cb(
        tt::CBIndex::c_3,
        tt::tt_metal::datatype_to_dataformat_converter(indices.dtype()),
        tt::round_up(TOKENS_PER_CHUNK * indices_aligned_page_size, DRAM_ALIGN),
        1);
    add_cb(tt::CBIndex::c_4, bf16, bf16_tile, 2 * (part_wt / tiles_per_scale));  // weight * scale per block (column 0)
    add_cb(tt::CBIndex::c_6, bf16, bf16_tile, part_wt);                          // tilized stage
    add_cb(tt::CBIndex::c_7, tt::DataFormat::Float32, DRAM_ALIGN, 2);            // stage-count mailbox
    const uint32_t weight_aligned_page_size = get_aligned_page_size(weights);
    add_cb(
        tt::CBIndex::c_8,
        tt::DataFormat::UInt32,
        tt::round_up(TOKENS_PER_CHUNK * num_experts * (weight_aligned_page_size + scale_slot_bytes), DRAM_ALIGN),
        1);
    const auto out_df = tt::tt_metal::datatype_to_dataformat_converter(tensor_return_value.dtype());
    add_cb(tt::CBIndex::c_16, out_df, tt::tile_size(out_df), 2 * part_wt);

    auto* combine_buffer = combine_output.buffer();
    auto* weight_buffer = weights.buffer();
    auto* output_buffer = tensor_return_value.buffer();
    auto* indices_buffer = indices.buffer();
    auto* dispatch_table_buffer = expert_dispatch_table.buffer();

    std::vector<uint32_t> reader_ct_args = {
        num_experts,
        part_wt,
        dispatch_table_num_pages,
        get_page_size(expert_dispatch_table),
        dispatch_table_aligned_page_size,
        get_page_size(indices),
        indices_aligned_page_size,
        scale_slot_bytes,
        num_experts * part_wt * fp8_tile,
    };
    tt::tt_metal::TensorAccessorArgs(combine_buffer).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(weight_buffer).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(dispatch_table_buffer).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(indices_buffer).append_to(reader_ct_args);

    std::vector<uint32_t> writer_ct_args = {
        num_experts,
        part_wt,
        emb_wt,
        dispatch_table_num_pages,
        get_page_size(expert_dispatch_table),
        dispatch_table_aligned_page_size,
        get_page_size(indices),
        indices_aligned_page_size,
    };
    tt::tt_metal::TensorAccessorArgs(weight_buffer).append_to(writer_ct_args);
    tt::tt_metal::TensorAccessorArgs(output_buffer).append_to(writer_ct_args);
    tt::tt_metal::TensorAccessorArgs(dispatch_table_buffer).append_to(writer_ct_args);
    tt::tt_metal::TensorAccessorArgs(indices_buffer).append_to(writer_ct_args);
    writer_ct_args.push_back(1);  // use_dispatch_table_skip (runtime-arg layout)
    writer_ct_args.push_back(1);  // fp8_block

    const std::string kdir =
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/post_combine_reduce/device/kernels/";
    tt::tt_metal::KernelDescriptor reader_kernel_desc;
    reader_kernel_desc.kernel_source = kdir + "deepseek_moe_post_combine_reduce_fp8_block_reader.cpp";
    reader_kernel_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    reader_kernel_desc.core_ranges = core_range_set;
    reader_kernel_desc.compile_time_args = std::move(reader_ct_args);
    reader_kernel_desc.defines = {{"TILES_PER_SCALE", std::to_string(tiles_per_scale)}};
    reader_kernel_desc.config = tt::tt_metal::ReaderConfigDescriptor{};

    tt::tt_metal::KernelDescriptor compute_kernel_desc;
    compute_kernel_desc.kernel_source = kdir + "deepseek_moe_post_combine_reduce_fp8_block_compute.cpp";
    compute_kernel_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    compute_kernel_desc.core_ranges = core_range_set;
    compute_kernel_desc.compile_time_args = {part_wt, num_experts};
    compute_kernel_desc.defines = {{"TILES_PER_SCALE", std::to_string(tiles_per_scale)}};
    // Fp8_e4m3 CBs on Blackhole need 32-bit DEST (as per_token_cast_back's bf16 datapath).
    tt::tt_metal::ComputeConfigDescriptor::UnpackToDestModes m(
        NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
    m[tt::CBIndex::c_0] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    compute_kernel_desc.config = tt::tt_metal::ComputeConfigDescriptor{
        .math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
        .fp32_dest_acc_en = true,
        .dst_full_sync_en = false,
        .unpack_to_dest_mode = std::move(m),
    };

    tt::tt_metal::KernelDescriptor writer_kernel_desc;
    writer_kernel_desc.kernel_source = kdir + "deepseek_moe_post_combine_reduce_fp8_block_writer.cpp";
    writer_kernel_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    writer_kernel_desc.core_ranges = core_range_set;
    writer_kernel_desc.compile_time_args = std::move(writer_ct_args);
    writer_kernel_desc.config = tt::tt_metal::WriterConfigDescriptor{};

    const uint32_t combine_aligned_page_size = combine_buffer->aligned_page_size();
    uint32_t token_start = 0;
    for (uint32_t i = 0; i < num_cores; ++i) {
        const CoreCoord& core = cores[i];
        const uint32_t part = i / cores_per_part;
        const uint32_t j = i % cores_per_part;
        if (j == 0) {
            token_start = 0;
        }
        const uint32_t chunks_this_core = base_chunks_per_core + (j < extra_chunks ? 1 : 0);
        const uint32_t part_offset = part * part_row_bytes;
        const uint32_t scale_first = emb_dim + part * part_wt / tiles_per_scale * 4;
        const uint32_t scale_window = scale_first & ~(DRAM_ALIGN - 1);
        const uint32_t scale_skip_words = (scale_first - scale_window) / 4;
        const uint32_t scale_read_bytes =
            tt::round_up(scale_first - scale_window + part_wt / tiles_per_scale * 4, DRAM_ALIGN);
        TT_FATAL(
            scale_window + scale_read_bytes <= combine_aligned_page_size && scale_read_bytes <= scale_slot_bytes,
            "1xB fp8 scale window [{}, +{}) exceeds the {}-B page",
            scale_window,
            scale_read_bytes,
            combine_aligned_page_size);

        tt::tt_metal::KernelDescriptor::RTArgList reader_rt_args;
        reader_rt_args.push_back(combine_buffer);
        reader_rt_args.push_back(weight_buffer);
        reader_rt_args.push_back(dispatch_table_buffer);
        reader_rt_args.push_back(indices_buffer);
        reader_rt_args.push_back(token_start);
        reader_rt_args.push_back(chunks_this_core);
        reader_rt_args.push_back(part_offset);
        reader_rt_args.push_back(scale_window);
        reader_rt_args.push_back(scale_skip_words);
        reader_rt_args.push_back(scale_read_bytes);
        reader_kernel_desc.emplace_runtime_args(core, reader_rt_args);

        tt::tt_metal::KernelDescriptor::RTArgList compute_rt_args;
        compute_rt_args.push_back(chunks_this_core);
        compute_kernel_desc.emplace_runtime_args(core, compute_rt_args);

        tt::tt_metal::KernelDescriptor::RTArgList writer_rt_args;
        writer_rt_args.push_back(weight_buffer);
        writer_rt_args.push_back(output_buffer);
        writer_rt_args.push_back(dispatch_table_buffer);
        writer_rt_args.push_back(indices_buffer);
        writer_rt_args.push_back(token_start);
        writer_rt_args.push_back(chunks_this_core);
        writer_rt_args.push_back(part * part_wt);  // part's first output tile column
        writer_rt_args.push_back(part_wt);         // part's output tiles per 32-token block
        writer_kernel_desc.emplace_runtime_args(core, writer_rt_args);

        token_start += chunks_this_core * TOKENS_PER_CHUNK;
    }

    desc.kernels.push_back(std::move(reader_kernel_desc));
    desc.kernels.push_back(std::move(compute_kernel_desc));
    desc.kernels.push_back(std::move(writer_kernel_desc));
    return desc;
}

}  // namespace

tt::tt_metal::ProgramDescriptor PostCombineReduceProgramFactory::create_descriptor(
    const PostCombineReduceParams& operation_attributes,
    const PostCombineReduceInputs& tensor_args,
    ttnn::Tensor& tensor_return_value) {
    if (operation_attributes.fp8_scale_block_w != 0) {
        return create_fp8_block_descriptor(operation_attributes, tensor_args, tensor_return_value);
    }
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
        .math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
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
