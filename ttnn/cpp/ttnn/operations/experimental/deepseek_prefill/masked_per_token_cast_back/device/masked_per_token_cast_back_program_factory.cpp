// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "masked_per_token_cast_back_program_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/circular_buffer_constants.h>

#include "ttnn/operations/experimental/deepseek_prefill/per_token_cast_to_fp8/per_token_cast_to_fp8.hpp"

// masked_per_token_cast_back: Convert (e4m3 input, scale) -> (bfloat16 or float32 output), but only for
// the valid, contiguously-packed prefix of a MoE dispatch buffer. Unlike per_token_cast_back, the row
// range is not split host-side: the whole Tensix grid is used and each core derives its own balanced
// tile-row slice on-device from the per-expert token counts / region offsets (same technique as the
// extract op). The reader publishes its computed num_blocks to the compute kernel via a small control CB.

namespace ttnn::experimental::prim::masked_per_token_cast_back {

namespace fp8 = ttnn::operations::experimental::deepseek_prefill::per_token_cast_to_fp8;

namespace {

std::pair<uint32_t, uint32_t> fold_M_H(const ttnn::Shape& shape) {
    uint64_t M = 1;
    for (size_t i = 0; i + 1 < shape.size(); ++i) {
        M *= static_cast<uint64_t>(shape[i]);
    }
    return {static_cast<uint32_t>(M), static_cast<uint32_t>(shape[shape.size() - 1])};
}

}  // namespace

MaskedPerTokenCastBackProgramFactory::cached_program_t MaskedPerTokenCastBackProgramFactory::create(
    const MaskedPerTokenCastBackParams& operation_attributes,
    const MaskedPerTokenCastBackInputs& tensor_args,
    Tensor& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input_e4m3 = tensor_args.input_e4m3;
    const auto& input_scale = tensor_args.input_scale;
    const auto& expert_region_offsets = tensor_args.expert_region_offsets;
    const auto& expert_token_counts = tensor_args.expert_token_counts;
    const auto& global_expert_idx_table = tensor_args.global_expert_idx_table;
    auto& output = tensor_return_value;

    const auto& shape = input_e4m3.logical_shape();
    auto [M, H] = fold_M_H(shape);  // M = rows (buffer capacity), H = width (last dim)

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

    // Column offset (in fp32/int32 elements) of the per-token scale tail within each scale-source row.
    // Plain scale tensor: 0 (row is exactly H/128 scales). Metadata tensor: skip the leading routing
    // columns, so offset = scale_last_dim - H/128 (== 5 for the standard 5-field metadata header).
    const uint32_t blocks_per_row_hint = H / block_w;
    const uint32_t scale_last_dim = static_cast<uint32_t>(input_scale.logical_shape()[-1]);
    const uint32_t scale_col_offset =
        operation_attributes.scales_from_metadata ? (scale_last_dim - blocks_per_row_hint) : 0;

    const uint32_t region_aligned_page_bytes = expert_region_offsets.buffer()->aligned_page_size();
    const uint32_t counts_aligned_page_bytes = expert_token_counts.buffer()->aligned_page_size();
    const uint32_t table_aligned_page_bytes = global_expert_idx_table.buffer()->aligned_page_size();

    auto* src_e4m3_buffer = input_e4m3.buffer();
    auto* src_scale_buffer = input_scale.buffer();
    auto* region_buffer = expert_region_offsets.buffer();
    auto* counts_buffer = expert_token_counts.buffer();
    auto* table_buffer = global_expert_idx_table.buffer();
    auto* dst_buffer = output.buffer();

    Program program{};

    auto* device = input_e4m3.device();
    auto compute_grid = device->compute_with_storage_grid_size();
    // Use the whole grid; each core derives its own balanced tile-row slice on-device (no host split).
    const uint32_t num_cores = compute_grid.x * compute_grid.y;
    const CoreRange all_cores_range{{0, 0}, {compute_grid.x - 1, compute_grid.y - 1}};
    const CoreRangeSet all_cores{all_cores_range};

    const DataFormat fp8_df = DataFormat::Fp8_e4m3;
    const DataFormat fp32_df = DataFormat::Float32;
    const DataFormat output_df = datatype_to_dataformat_converter(operation_attributes.output_dtype);
    const DataFormat idx_df = datatype_to_dataformat_converter(expert_token_counts.dtype());
    // Scale values are always fp32 (plain fp32 tensor or the int32-backed metadata tail). When bf16_scale
    // is set, narrow them to bf16 on-device (the packer, in the compute kernel) and run the whole datapath
    // in bf16 (HiFi2): decode E4M3 -> bf16, then tilize / bcast-multiply / untilize in bf16. The FPU bcast
    // path addresses SrcA and SrcB with a shared stride, so the narrowed bf16 scale matches the bf16 input.
    const uint32_t scale_elem_bytes = input_scale.element_size();
    const bool bf16_scale = operation_attributes.bf16_scale;
    const bool convert_scale = bf16_scale;  // fp32 scale -> bf16 on-device when the bf16 datapath is requested
    const bool compute_bf16 = bf16_scale;
    const DataFormat compute_df = compute_bf16 ? DataFormat::Float16_b : fp32_df;
    const uint32_t compute_tile_bytes = tile_h * tile_w * (compute_bf16 ? 2u : 4u);
    // The scale bcast operand is always the raw fp32 scale (the reader copies col-0 values); it is narrowed
    // to bf16 on-device only when convert_scale is set.
    const DataFormat scale_bcast_df = fp32_df;
    const uint32_t scale_bcast_tile_bytes = tile_h * tile_w * scale_elem_bytes;

    constexpr uint32_t cb_input_e4m3_idx = CBIndex::c_0;
    constexpr uint32_t cb_in_rm_idx = CBIndex::c_1;
    constexpr uint32_t cb_in_tile_idx = CBIndex::c_2;
    constexpr uint32_t cb_control_idx = CBIndex::c_3;
    constexpr uint32_t cb_scale_bcast_idx = CBIndex::c_4;
    constexpr uint32_t cb_out_tile_idx = CBIndex::c_5;
    constexpr uint32_t cb_scale_scratch_idx = CBIndex::c_6;
    constexpr uint32_t cb_region_scratch_r_idx = CBIndex::c_7;
    constexpr uint32_t cb_counts_scratch_r_idx = CBIndex::c_8;
    constexpr uint32_t cb_table_scratch_r_idx = CBIndex::c_9;
    constexpr uint32_t cb_region_scratch_w_idx = CBIndex::c_10;
    constexpr uint32_t cb_counts_scratch_w_idx = CBIndex::c_11;
    constexpr uint32_t cb_table_scratch_w_idx = CBIndex::c_12;
    constexpr uint32_t cb_scale_bcast_bf16_idx = CBIndex::c_13;  // packer fp32->bf16 scale (convert_scale)
    constexpr uint32_t cb_out_idx = CBIndex::c_16;

    auto make_compute_tile_cb = [&](uint32_t cb_idx, uint32_t num_tiles) {
        CircularBufferConfig cfg = CircularBufferConfig(num_tiles * compute_tile_bytes, {{cb_idx, compute_df}})
                                       .set_page_size(cb_idx, compute_tile_bytes);
        CreateCircularBuffer(program, all_cores, cfg);
    };

    auto make_index_scratch_cb = [&](uint32_t cb_idx, uint32_t page_bytes) {
        CircularBufferConfig cfg =
            CircularBufferConfig(page_bytes, {{cb_idx, idx_df}}).set_page_size(cb_idx, page_bytes);
        CreateCircularBuffer(program, all_cores, cfg);
    };

    // cb_input_e4m3: input_e4m3, one tile per page; tiles_per_block pages = one block,
    // double-buffered. The reader fills the [tile_h x 128] block contiguously across these pages.
    CircularBufferConfig cb_input_e4m3_cfg =
        CircularBufferConfig(2 * tiles_per_block * input_e4m3_tile_bytes, {{cb_input_e4m3_idx, fp8_df}})
            .set_page_size(cb_input_e4m3_idx, input_e4m3_tile_bytes);
    CreateCircularBuffer(program, all_cores, cb_input_e4m3_cfg);

    // All streaming CBs are double-buffered so the reader/compute/writer kernels (and the compute
    // UNPACK/MATH/PACK sub-threads) can run one block ahead of each other without stalling.
    make_compute_tile_cb(cb_in_rm_idx, 2 * tiles_per_block);     // input_e4m3 -> compute_df RM
    make_compute_tile_cb(cb_in_tile_idx, 2 * tiles_per_block);   // tilized input (compute_df)
    make_compute_tile_cb(cb_out_tile_idx, 2 * tiles_per_block);  // multiplied tiles -> untilize

    // cb_scale_bcast: col0 = raw fp32 scale. In the fp32 datapath the multiply consumes it directly; in the
    // bf16 datapath the packer narrows it into cb_scale_bcast_bf16 first.
    CircularBufferConfig cb_scale_bcast_cfg =
        CircularBufferConfig(2 * block_ht * scale_bcast_tile_bytes, {{cb_scale_bcast_idx, scale_bcast_df}})
            .set_page_size(cb_scale_bcast_idx, scale_bcast_tile_bytes);
    CreateCircularBuffer(program, all_cores, cb_scale_bcast_cfg);
    if (convert_scale) {
        make_compute_tile_cb(cb_scale_bcast_bf16_idx, 2 * block_ht);  // packer fp32->bf16 scale operand
    }

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

    // cb_control: reader -> compute mailbox for the device-computed num_blocks (see read_tile_value).
    constexpr uint32_t control_page_bytes = 16;  // one uint32 padded to L1 alignment
    CircularBufferConfig cb_control_cfg = CircularBufferConfig(control_page_bytes, {{cb_control_idx, fp32_df}})
                                              .set_page_size(cb_control_idx, control_page_bytes);
    CreateCircularBuffer(program, all_cores, cb_control_cfg);

    // Per-kernel private scratch for the three metadata vectors (reader and writer each read them).
    make_index_scratch_cb(cb_region_scratch_r_idx, region_aligned_page_bytes);
    make_index_scratch_cb(cb_counts_scratch_r_idx, counts_aligned_page_bytes);
    make_index_scratch_cb(cb_table_scratch_r_idx, table_aligned_page_bytes);
    make_index_scratch_cb(cb_region_scratch_w_idx, region_aligned_page_bytes);
    make_index_scratch_cb(cb_counts_scratch_w_idx, counts_aligned_page_bytes);
    make_index_scratch_cb(cb_table_scratch_w_idx, table_aligned_page_bytes);

    // Reader (RISCV_1): computes its balanced tile-row slice from the metadata vectors, streams
    // input_e4m3 blocks, builds column-0 scale broadcast operands, and publishes num_blocks to compute.
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
        face_w,
        cb_control_idx,
        cb_region_scratch_r_idx,
        cb_counts_scratch_r_idx,
        cb_table_scratch_r_idx,
        num_cores,
        operation_attributes.experts_per_chip,
        scale_col_offset,
        scale_elem_bytes};
    TensorAccessorArgs(src_e4m3_buffer).append_to(reader_ct_args);
    TensorAccessorArgs(src_scale_buffer).append_to(reader_ct_args);
    TensorAccessorArgs(region_buffer).append_to(reader_ct_args);
    TensorAccessorArgs(counts_buffer).append_to(reader_ct_args);
    TensorAccessorArgs(table_buffer).append_to(reader_ct_args);
    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/masked_per_token_cast_back/device/kernels/dataflow/"
        "reader_masked_per_token_cast_back.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = reader_ct_args});

    // Writer (RISCV_0): computes the same slice and does column-block-major writes of the row-major output.
    std::vector<uint32_t> writer_ct_args = {
        cb_out_idx,
        out_block_bytes,
        tile_h,
        tiles_per_block,
        cb_region_scratch_w_idx,
        cb_counts_scratch_w_idx,
        cb_table_scratch_w_idx,
        num_cores,
        operation_attributes.experts_per_chip};
    TensorAccessorArgs(dst_buffer).append_to(writer_ct_args);
    TensorAccessorArgs(region_buffer).append_to(writer_ct_args);
    TensorAccessorArgs(counts_buffer).append_to(writer_ct_args);
    TensorAccessorArgs(table_buffer).append_to(writer_ct_args);
    KernelHandle writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/masked_per_token_cast_back/device/kernels/dataflow/"
        "writer_masked_per_token_cast_back.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = writer_ct_args});

    // Compute (TRISC): input_e4m3 -> fp32 RM -> tilize -> scale bcast multiply -> untilize to output.
    // num_blocks arrives from the reader via cb_control (read_tile_value mailbox).
    std::vector<uint32_t> compute_ct_args = {
        cb_input_e4m3_idx,
        cb_in_rm_idx,
        cb_in_tile_idx,
        cb_scale_bcast_idx,
        cb_out_tile_idx,
        cb_out_idx,
        tile_h,
        tile_w,
        cb_control_idx,
        static_cast<uint32_t>(compute_bf16),
        static_cast<uint32_t>(convert_scale),
        cb_scale_bcast_bf16_idx};
    // fp32_dest_acc_en=True required (input_e4m3 CB on core). A bf16 scale carries only 8 mantissa bits
    // and the input is fp8 (3), so the broadcast multiply does not need HiFi4's full fp32 passes — use
    // HiFi2 for the bf16-scale path; fp32 scale keeps HiFi4.
    const MathFidelity math_fidelity = bf16_scale ? MathFidelity::HiFi2 : MathFidelity::HiFi4;
    // The fp8 input CB forces a 32-bit DEST (fp32_dest_acc_en). Unpack the e4m3 input straight to a
    // 32-bit DEST so the bf16 datapath decodes/accumulates correctly (mirrors typecast's fp8 path).
    std::vector<tt::tt_metal::UnpackToDestMode> unpack_to_dest_mode(
        NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
    unpack_to_dest_mode[cb_input_e4m3_idx] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    // The fp32 scale operand must unpack losslessly to DEST before the packer rounds it to bf16.
    if (convert_scale) {
        unpack_to_dest_mode[cb_scale_bcast_idx] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    }
    KernelHandle compute_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/masked_per_token_cast_back/device/kernels/compute/"
        "compute_masked_per_token_cast_back.cpp",
        all_cores,
        ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = true,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .compile_args = compute_ct_args});

    auto all_cores_vec = corerange_to_cores(all_cores, num_cores, true);
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
            {dst_buffer->address(), region_buffer->address(), counts_buffer->address(), table_buffer->address(), i, H});
        // Compute reads num_blocks from the reader via cb_control; no per-core runtime args needed.
        SetRuntimeArgs(program, compute_kernel_id, core, {});
    }

    return cached_program_t{
        std::move(program), {reader_kernel_id, writer_kernel_id, compute_kernel_id, std::move(all_cores_vec)}};
}

void MaskedPerTokenCastBackProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const MaskedPerTokenCastBackParams& /*operation_attributes*/,
    const MaskedPerTokenCastBackInputs& tensor_args,
    Tensor& tensor_return_value) {
    auto& program = cached_program.program;
    auto& shared = cached_program.shared_variables;

    uint32_t src_e4m3_addr = tensor_args.input_e4m3.buffer()->address();
    uint32_t src_scale_addr = tensor_args.input_scale.buffer()->address();
    uint32_t region_addr = tensor_args.expert_region_offsets.buffer()->address();
    uint32_t counts_addr = tensor_args.expert_token_counts.buffer()->address();
    uint32_t table_addr = tensor_args.global_expert_idx_table.buffer()->address();
    uint32_t dst_addr = tensor_return_value.buffer()->address();

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
}

}  // namespace ttnn::experimental::prim::masked_per_token_cast_back
