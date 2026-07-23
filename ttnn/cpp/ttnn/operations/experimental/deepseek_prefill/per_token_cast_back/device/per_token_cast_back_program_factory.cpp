// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "per_token_cast_back_program_factory.hpp"

#include <tuple>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/circular_buffer_constants.h>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/tt_align.hpp>

#include "ttnn/operations/experimental/deepseek_prefill/per_token_cast_to_fp8/per_token_cast_to_fp8.hpp"

// per_token_cast_back: Convert (e4m3 input, scale) -> (bfloat16 or float32 output)
// Implementation requires row-major inputs and outputs.
// Internally, it relies on intermediate CBs for tilization / untilization, as well as for conversion.
// To efficiently perform this tilization, we use a block-based approach with a [32, 128] circular buffer ( = 4 tiles).
//
// Two paths share this single program factory (and, via a TOKEN_COUNT_AWARE compile-time define, the same reader /
// writer / compute kernels):
//   * plain  (operation_attributes.token_count_aware == false): dequantize the whole [M, H] buffer; the rows are
//            split across the grid host-side (split_work_to_cores).
//   * token_count_aware (== true): from the per-expert token counts, each core works out on-device how
//            many rows of the buffer are valid and dequantizes only that leading part of the tensor.

namespace ttnn::experimental::prim::per_token_cast_back {

namespace fp8 = ttnn::operations::experimental::deepseek_prefill::per_token_cast_to_fp8;

namespace {

std::pair<uint32_t, uint32_t> fold_M_H(const ttnn::Shape& shape) {
    uint64_t M = 1;
    for (size_t i = 0; i + 1 < shape.size(); ++i) {
        M *= static_cast<uint64_t>(shape[i]);
    }
    return {static_cast<uint32_t>(M), static_cast<uint32_t>(shape[shape.size() - 1])};
}

uint32_t rows_for_core_from_split(
    const CoreCoord& core,
    const CoreRangeSet& core_range_set_1,
    const CoreRangeSet& core_range_set_2,
    uint32_t rows_per_core_g1,
    uint32_t rows_per_core_g2) {
    if (core_range_set_1.contains(core)) {
        return rows_per_core_g1;
    }
    if (core_range_set_2.contains(core)) {
        return rows_per_core_g2;
    }
    return 0;
}

}  // namespace

PerTokenCastBackProgramFactory::cached_program_t PerTokenCastBackProgramFactory::create(
    const PerTokenCastBackParams& operation_attributes,
    const PerTokenCastBackInputs& tensor_args,
    Tensor& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const bool token_count_aware = operation_attributes.token_count_aware;

    const auto& input_e4m3 = tensor_args.input_e4m3;
    const auto& input_scale = tensor_args.input_scale;
    auto& output = tensor_return_value;

    const auto& shape = input_e4m3.logical_shape();
    auto [M, H] = fold_M_H(shape);  // M = rows, H = width (last dim)

    // Tile / face dims come from the tensor's tile spec.
    const auto tile_shape = input_e4m3.tensor_spec().tile().get_tile_shape();
    const auto face_shape = input_e4m3.tensor_spec().tile().get_face_shape();
    const uint32_t tile_h = tile_shape[0];
    const uint32_t tile_w = tile_shape[1];
    const uint32_t face_h = face_shape[0];
    const uint32_t face_w = face_shape[1];

    // A block is exactly one 128-element block per row. The input_e4m3/output CBs keep one-tile
    // pages, and the reader fills the [tile_h x 128] block as one contiguous run.
    constexpr uint32_t block_w = fp8::BLOCK_W;  // BlockW: 128 elements

    const uint32_t block_wt = block_w / tile_w;  // BlockWt: tiles across the 128-wide block
    constexpr uint32_t block_ht = 1;             // BlockHt: one tile-height batch
    const uint32_t tiles_per_block = block_ht * block_wt;

    const uint32_t input_e4m3_block_bytes = block_w;  // one 128-element row, 1 byte/elem
    const uint32_t out_elem_bytes = output.element_size();
    const uint32_t out_block_bytes = block_w * out_elem_bytes;         // one 128-element row of output
    const uint32_t input_e4m3_tile_bytes = tile_h * tile_w;            // cb_input_e4m3 page = one e4m3 tile
    const uint32_t out_tile_bytes = tile_h * tile_w * out_elem_bytes;  // cb_out page = one output tile
    const uint32_t scale_aligned_page_bytes = input_scale.buffer()->aligned_page_size();

    // Token-count-aware-only metadata tensors (unset on the plain path).
    const Tensor* expert_region_offsets = token_count_aware ? &tensor_args.expert_region_offsets.value() : nullptr;
    const Tensor* expert_token_counts = token_count_aware ? &tensor_args.expert_token_counts.value() : nullptr;
    const Tensor* global_expert_idx_table = token_count_aware ? &tensor_args.global_expert_idx_table.value() : nullptr;

    // Length of the region/counts vectors; the token_count_aware kernels assert every table entry stays within it.
    const uint32_t num_routed_experts =
        token_count_aware ? static_cast<uint32_t>(expert_region_offsets->logical_shape()[-1]) : 0;

    // Offset (in elements) from the start of each metadata entry to that token's H/128 scales.
    // Plain scale tensor -> 0. Metadata entry -> skip the leading routing header:
    //   entry = [ x x x s1 s2 ... sn ]   (x = routing data, not ours; s = scales)
    //   scales_start_offset = scale_entry_length - scales_per_token   (n scales in the tail)
    const uint32_t scales_per_token = H / block_w;  // H/128
    const uint32_t scale_entry_length = static_cast<uint32_t>(input_scale.logical_shape()[-1]);
    const uint32_t scales_start_offset =
        (token_count_aware && operation_attributes.scales_from_metadata) ? (scale_entry_length - scales_per_token) : 0;

    // The whole datapath is fp32; the scale is always fp32 (plain fp32 tensor or the int32-backed metadata tail).
    const uint32_t TILE_BYTES = tile_h * tile_w * sizeof(float);
    const DataFormat fp8_df = DataFormat::Fp8_e4m3;
    const DataFormat fp32_df = DataFormat::Float32;
    const DataFormat output_df = datatype_to_dataformat_converter(operation_attributes.output_dtype);

    auto* src_e4m3_buffer = input_e4m3.buffer();
    auto* src_scale_buffer = input_scale.buffer();
    auto* dst_buffer = output.buffer();
    auto* region_buffer = token_count_aware ? expert_region_offsets->buffer() : nullptr;
    auto* counts_buffer = token_count_aware ? expert_token_counts->buffer() : nullptr;
    auto* table_buffer = token_count_aware ? global_expert_idx_table->buffer() : nullptr;

    Program program{};

    auto* device = input_e4m3.device();
    auto compute_grid = device->compute_with_storage_grid_size();

    // Split the work across the cores.
    uint32_t num_cores = 0;
    CoreRangeSet all_cores;
    CoreRangeSet core_range_set_1;
    CoreRangeSet core_range_set_2;
    uint32_t rows_per_core_g1 = 0;
    uint32_t rows_per_core_g2 = 0;
    if (token_count_aware) {
        // Light up the whole grid; the host commits no per-core work here. Each core decides at runtime,
        // while the kernels run, how much of the work it takes on (from the device-side token counts).
        num_cores = compute_grid.x * compute_grid.y;
        all_cores = CoreRangeSet{CoreRange{{0, 0}, {compute_grid.x - 1, compute_grid.y - 1}}};
    } else {
        // Split on rows (not tile-rows) so horizontal tensors (small M, large H) use the whole grid;
        // the op is DRAM/NoC-bound, so spreading rows across more cores spreads the data movement. Each
        // core's contiguous row range need not be tile-aligned (kernels address by absolute DRAM page).
        // std::tie unpacks the returned tuple into these already-declared variables (a structured
        // binding `auto [...]` can only declare fresh ones, not assign existing).
        std::tie(num_cores, all_cores, core_range_set_1, core_range_set_2, rows_per_core_g1, rows_per_core_g2) =
            split_work_to_cores(compute_grid, M);
    }

    constexpr uint32_t cb_input_e4m3_idx = CBIndex::c_0;
    constexpr uint32_t cb_in_rm_idx = CBIndex::c_1;
    constexpr uint32_t cb_in_tile_idx = CBIndex::c_2;
    constexpr uint32_t cb_loop_count_idx = CBIndex::c_3;  // token_count_aware: reader -> compute num_blocks mailbox
    constexpr uint32_t cb_scale_bcast_idx = CBIndex::c_4;
    constexpr uint32_t cb_out_tile_idx = CBIndex::c_5;
    constexpr uint32_t cb_scale_scratch_idx = CBIndex::c_6;
    constexpr uint32_t cb_region_scratch_r_idx = CBIndex::c_7;   // token_count_aware only
    constexpr uint32_t cb_counts_scratch_r_idx = CBIndex::c_8;   // token_count_aware only
    constexpr uint32_t cb_table_scratch_r_idx = CBIndex::c_9;    // token_count_aware only
    constexpr uint32_t cb_region_scratch_w_idx = CBIndex::c_10;  // token_count_aware only
    constexpr uint32_t cb_counts_scratch_w_idx = CBIndex::c_11;  // token_count_aware only
    constexpr uint32_t cb_table_scratch_w_idx = CBIndex::c_12;   // token_count_aware only
    constexpr uint32_t cb_out_idx = CBIndex::c_16;

    auto make_fp32_tile_cb = [&](uint32_t cb_idx, uint32_t num_tiles) {
        CircularBufferConfig cfg =
            CircularBufferConfig(num_tiles * TILE_BYTES, {{cb_idx, fp32_df}}).set_page_size(cb_idx, TILE_BYTES);
        CreateCircularBuffer(program, all_cores, cfg);
    };

    // cb_input_e4m3: input_e4m3, one tile per page; tiles_per_block pages = one block, double-buffered.
    // double-buffered. The reader fills the [tile_h x 128] block contiguously across these pages.
    CircularBufferConfig cb_input_e4m3_cfg =
        CircularBufferConfig(2 * tiles_per_block * input_e4m3_tile_bytes, {{cb_input_e4m3_idx, fp8_df}})
            .set_page_size(cb_input_e4m3_idx, input_e4m3_tile_bytes);
    CreateCircularBuffer(program, all_cores, cb_input_e4m3_cfg);

    make_fp32_tile_cb(cb_in_rm_idx, tiles_per_block);     // input_e4m3 -> fp32 RM
    make_fp32_tile_cb(cb_in_tile_idx, tiles_per_block);   // tilized fp32 input
    make_fp32_tile_cb(cb_scale_bcast_idx, 2 * block_ht);  // col0 = scale
    make_fp32_tile_cb(cb_out_tile_idx, tiles_per_block);  // divided tiles -> untilize

    // cb_out: row-major output (bf16/fp32), one tile per page; tiles_per_block pages = one block,
    // double-buffered.
    CircularBufferConfig cb_out_cfg =
        CircularBufferConfig(2 * tiles_per_block * out_tile_bytes, {{cb_out_idx, output_df}})
            .set_page_size(cb_out_idx, out_tile_bytes);
    CreateCircularBuffer(program, all_cores, cb_out_cfg);

    // cb_scale_scratch: reader-private staging for 32 tokens' full scale rows (page-aligned stride).
    const uint32_t scale_scratch_bytes = tile_h * scale_aligned_page_bytes;
    CircularBufferConfig cb_scale_scratch_cfg =
        CircularBufferConfig(scale_scratch_bytes, {{cb_scale_scratch_idx, fp32_df}})
            .set_page_size(cb_scale_scratch_idx, scale_scratch_bytes);
    CreateCircularBuffer(program, all_cores, cb_scale_scratch_cfg);

    uint32_t region_aligned_page_bytes = 0;
    uint32_t counts_aligned_page_bytes = 0;
    uint32_t table_aligned_page_bytes = 0;
    if (token_count_aware) {
        region_aligned_page_bytes = region_buffer->aligned_page_size();
        counts_aligned_page_bytes = counts_buffer->aligned_page_size();
        table_aligned_page_bytes = table_buffer->aligned_page_size();

        const DataFormat idx_df = datatype_to_dataformat_converter(expert_token_counts->dtype());
        auto make_index_scratch_cb = [&](uint32_t cb_idx, uint32_t page_bytes) {
            CircularBufferConfig cfg =
                CircularBufferConfig(page_bytes, {{cb_idx, idx_df}}).set_page_size(cb_idx, page_bytes);
            CreateCircularBuffer(program, all_cores, cfg);
        };

        // cb_loop_count: reader -> compute mailbox for the device-computed num_blocks (see read_tile_value).
        // one uint32 (num_blocks), rounded up to the L1 alignment
        const uint32_t loop_count_page_bytes = tt::align(sizeof(uint32_t), hal::get_l1_alignment());
        CircularBufferConfig cb_loop_count_cfg =
            CircularBufferConfig(loop_count_page_bytes, {{cb_loop_count_idx, fp32_df}})
                .set_page_size(cb_loop_count_idx, loop_count_page_bytes);
        CreateCircularBuffer(program, all_cores, cb_loop_count_cfg);

        // Per-kernel private scratch for the three metadata vectors (reader and writer each read them).
        make_index_scratch_cb(cb_region_scratch_r_idx, region_aligned_page_bytes);
        make_index_scratch_cb(cb_counts_scratch_r_idx, counts_aligned_page_bytes);
        make_index_scratch_cb(cb_table_scratch_r_idx, table_aligned_page_bytes);
        make_index_scratch_cb(cb_region_scratch_w_idx, region_aligned_page_bytes);
        make_index_scratch_cb(cb_counts_scratch_w_idx, counts_aligned_page_bytes);
        make_index_scratch_cb(cb_table_scratch_w_idx, table_aligned_page_bytes);
    }

    const std::map<std::string, std::string> token_count_aware_defines =
        token_count_aware ? std::map<std::string, std::string>{{"TOKEN_COUNT_AWARE", "1"}}
                          : std::map<std::string, std::string>{};

    // Reader (RISCV_1): input_e4m3 blocks + builds the column-0 scale broadcast operands. On the token_count_aware
    // path it first works out how many compute-blocks it must process, then publishes that count (num_blocks) to
    // compute.
    std::vector<uint32_t> reader_ct_args = {
        cb_input_e4m3_idx,
        cb_scale_bcast_idx,
        cb_scale_scratch_idx,
        input_e4m3_block_bytes,
        block_ht,
        scale_aligned_page_bytes,
        tile_h,
        tile_w,
        face_h,
        face_w};
    if (token_count_aware) {
        reader_ct_args.insert(
            reader_ct_args.end(),
            {cb_loop_count_idx,
             cb_region_scratch_r_idx,
             cb_counts_scratch_r_idx,
             cb_table_scratch_r_idx,
             num_cores,
             operation_attributes.experts_per_chip,
             scales_start_offset,
             num_routed_experts,
             M});
    }
    TensorAccessorArgs(src_e4m3_buffer).append_to(reader_ct_args);
    TensorAccessorArgs(src_scale_buffer).append_to(reader_ct_args);
    if (token_count_aware) {
        TensorAccessorArgs(region_buffer).append_to(reader_ct_args);
        TensorAccessorArgs(counts_buffer).append_to(reader_ct_args);
        TensorAccessorArgs(table_buffer).append_to(reader_ct_args);
    }
    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/per_token_cast_back/device/kernels/dataflow/"
        "reader_per_token_cast_back.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_ct_args,
            .defines = token_count_aware_defines});

    // Writer (RISCV_0): column-block-major writes of the row-major output. On the token_count_aware path it derives
    // the same slice as the reader (identical (core_id, num_cores) formula).
    std::vector<uint32_t> writer_ct_args = {cb_out_idx, out_block_bytes, tile_h, tiles_per_block};
    if (token_count_aware) {
        writer_ct_args.insert(
            writer_ct_args.end(),
            {cb_region_scratch_w_idx,
             cb_counts_scratch_w_idx,
             cb_table_scratch_w_idx,
             num_cores,
             operation_attributes.experts_per_chip,
             num_routed_experts,
             M});
    }
    TensorAccessorArgs(dst_buffer).append_to(writer_ct_args);
    if (token_count_aware) {
        TensorAccessorArgs(region_buffer).append_to(writer_ct_args);
        TensorAccessorArgs(counts_buffer).append_to(writer_ct_args);
        TensorAccessorArgs(table_buffer).append_to(writer_ct_args);
    }
    KernelHandle writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/per_token_cast_back/device/kernels/dataflow/"
        "writer_per_token_cast_back.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_ct_args,
            .defines = token_count_aware_defines});

    // Compute (TRISC): input_e4m3 -> fp32 RM -> tilize -> scale bcast multiply -> untilize to output.
    // Plain: num_blocks arrives as a runtime arg. Token-count-aware: num_blocks arrives via cb_loop_count.
    std::vector<uint32_t> compute_ct_args = {
        cb_input_e4m3_idx,
        cb_in_rm_idx,
        cb_in_tile_idx,
        cb_scale_bcast_idx,
        cb_out_tile_idx,
        cb_out_idx,
        tile_h,
        tile_w,
        cb_loop_count_idx};
    // fp32_dest_acc_en=True required (input_e4m3 CB on core); HiFi4 (the ComputeConfig default) keeps the
    // broadcast multiply precise.
    KernelHandle compute_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/per_token_cast_back/device/kernels/compute/"
        "compute_per_token_cast_back.cpp",
        all_cores,
        ComputeConfig{.fp32_dest_acc_en = true, .compile_args = compute_ct_args, .defines = token_count_aware_defines});

    auto all_cores_vec = corerange_to_cores(all_cores, num_cores, true);
    if (token_count_aware) {
        for (uint32_t i = 0; i < num_cores; ++i) {
            const auto& core = all_cores_vec[i];
            SetRuntimeArgs(
                program,
                reader_kernel_id,
                core,
                {src_e4m3_buffer->address(),
                 src_scale_buffer->address(),
                 region_buffer->address(),
                 counts_buffer->address(),
                 table_buffer->address(),
                 i,
                 H});
            SetRuntimeArgs(
                program,
                writer_kernel_id,
                core,
                {dst_buffer->address(),
                 region_buffer->address(),
                 counts_buffer->address(),
                 table_buffer->address(),
                 i,
                 H});
            // Compute reads num_blocks from the reader via cb_loop_count; no per-core runtime args needed.
            SetRuntimeArgs(program, compute_kernel_id, core, {});
        }
    } else {
        // Each core's rows form a flat stream of 128-element scale blocks read/written in tile_h-block batches.
        const uint32_t scale_blocks_per_row = H / fp8::BLOCK_W;  // H / 128
        uint32_t row_offset = 0;
        for (uint32_t i = 0; i < num_cores; ++i) {
            const auto& core = all_cores_vec[i];
            const uint32_t rows_for_core =
                rows_for_core_from_split(core, core_range_set_1, core_range_set_2, rows_per_core_g1, rows_per_core_g2);
            const uint32_t total_scale_blocks = rows_for_core * scale_blocks_per_row;
            const uint32_t num_blocks = tt::div_up(total_scale_blocks, tile_h);  // last block may be partial

            TT_FATAL(
                num_blocks == tt::div_up(rows_for_core * scale_blocks_per_row, tile_h),
                "per_token_cast_back: num_blocks invariant violated on a core");

            // Pass num_blocks to the (unchanged) compute as (num_tile_rows = num_blocks, num_col_blocks
            // = 1) so its for-tr/for-c loop runs exactly num_blocks times.
            SetRuntimeArgs(
                program,
                reader_kernel_id,
                core,
                {src_e4m3_buffer->address(), src_scale_buffer->address(), num_blocks, row_offset, rows_for_core, H});
            SetRuntimeArgs(
                program, writer_kernel_id, core, {dst_buffer->address(), num_blocks, row_offset, rows_for_core, H});
            SetRuntimeArgs(program, compute_kernel_id, core, {num_blocks});
            row_offset += rows_for_core;
        }
    }

    return cached_program_t{
        std::move(program), {reader_kernel_id, writer_kernel_id, compute_kernel_id, std::move(all_cores_vec)}};
}

void PerTokenCastBackProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const PerTokenCastBackParams& /*operation_attributes*/,
    const PerTokenCastBackInputs& tensor_args,
    Tensor& tensor_return_value) {
    auto& program = cached_program.program;
    auto& shared = cached_program.shared_variables;

    uint32_t src_e4m3_addr = tensor_args.input_e4m3.buffer()->address();
    uint32_t src_scale_addr = tensor_args.input_scale.buffer()->address();
    uint32_t dst_addr = tensor_return_value.buffer()->address();
    const bool token_count_aware = tensor_args.expert_region_offsets.has_value();

    if (token_count_aware) {
        const uint32_t region_addr = tensor_args.expert_region_offsets->buffer()->address();
        const uint32_t counts_addr = tensor_args.expert_token_counts->buffer()->address();
        const uint32_t table_addr = tensor_args.global_expert_idx_table->buffer()->address();
        for (const auto& core : shared.all_cores_vec) {
            auto& reader_args = tt::tt_metal::GetRuntimeArgs(program, shared.reader_kernel_id, core);
            reader_args[0] = src_e4m3_addr;
            reader_args[1] = src_scale_addr;
            reader_args[2] = region_addr;
            reader_args[3] = counts_addr;
            reader_args[4] = table_addr;

            auto& writer_args = tt::tt_metal::GetRuntimeArgs(program, shared.writer_kernel_id, core);
            writer_args[0] = dst_addr;
            writer_args[1] = region_addr;
            writer_args[2] = counts_addr;
            writer_args[3] = table_addr;
        }
    } else {
        for (const auto& core : shared.all_cores_vec) {
            auto& reader_args = tt::tt_metal::GetRuntimeArgs(program, shared.reader_kernel_id, core);
            reader_args[0] = src_e4m3_addr;
            reader_args[1] = src_scale_addr;
            auto& writer_args = tt::tt_metal::GetRuntimeArgs(program, shared.writer_kernel_id, core);
            writer_args[0] = dst_addr;
        }
    }
}

}  // namespace ttnn::experimental::prim::per_token_cast_back
