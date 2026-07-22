// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "per_token_cast_back_program_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/circular_buffer_constants.h>

#include "ttnn/operations/experimental/deepseek_prefill/per_token_cast_to_fp8/per_token_cast_to_fp8.hpp"

// per_token_cast_back: Convert (e4m3 input, scale) -> (bfloat16 or float32 output)
// Implementation requires row-major inputs and outputs.
// Internally, it relies on intermediate CBs for tilization / untilization, as well as for conversion.
// To efficiently perform this tilization, we use a block-based approach with a [32, 128] circular buffer ( = 4 tiles).

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

    const uint32_t TILE_BYTES = tile_h * tile_w * sizeof(float);
    const uint32_t block_wt = block_w / tile_w;  // BlockWt: tiles across the 128-wide block
    constexpr uint32_t block_ht = 1;             // BlockHt: one tile-height batch
    const uint32_t tiles_per_block = block_ht * block_wt;

    const uint32_t input_e4m3_block_bytes = block_w;  // one 128-element row, 1 byte/elem
    const uint32_t out_elem_bytes = output.element_size();
    const uint32_t out_block_bytes = block_w * out_elem_bytes;         // one 128-element row of output
    const uint32_t input_e4m3_tile_bytes = tile_h * tile_w;            // cb_input_e4m3 page = one e4m3 tile
    const uint32_t out_tile_bytes = tile_h * tile_w * out_elem_bytes;  // cb_out page = one output tile
    const uint32_t scale_aligned_page_bytes = input_scale.buffer()->aligned_page_size();

    auto* src_e4m3_buffer = input_e4m3.buffer();
    auto* src_scale_buffer = input_scale.buffer();
    auto* dst_buffer = output.buffer();

    Program program{};

    auto* device = input_e4m3.device();
    auto compute_grid = device->compute_with_storage_grid_size();
    // Split on rows (not tile-rows) so horizontal tensors (small M, large H) use the whole grid;
    // the op is DRAM/NoC-bound, so spreading rows across more cores spreads the data movement. Each
    // core's contiguous row range need not be tile-aligned (kernels address by absolute DRAM page).
    auto [num_cores, all_cores, core_range_set_1, core_range_set_2, rows_per_core_g1, rows_per_core_g2] =
        split_work_to_cores(compute_grid, M);

    const DataFormat fp8_df = DataFormat::Fp8_e4m3;
    const DataFormat fp32_df = DataFormat::Float32;
    const DataFormat output_df = datatype_to_dataformat_converter(operation_attributes.output_dtype);

    // narrow_scales_to_bf16: perform the multiply in bf16/HiFi2 (vs fp32/HiFi4)
    const bool narrow_scales_to_bf16 = operation_attributes.narrow_scales_to_bf16;
    const DataFormat compute_df = narrow_scales_to_bf16 ? DataFormat::Float16_b : fp32_df;
    const uint32_t compute_tile_bytes = tile_h * tile_w * (narrow_scales_to_bf16 ? 2u : 4u);

    constexpr uint32_t cb_input_e4m3_idx = CBIndex::c_0;
    constexpr uint32_t cb_in_rm_fp32_idx = CBIndex::c_1;  // fp32 rm buffer (fp32 datapath only)
    constexpr uint32_t cb_in_tile_idx = CBIndex::c_2;     // reused: tilized input
    constexpr uint32_t cb_scale_bcast_bf16_idx =
        CBIndex::c_3;  // bf16: packer packs fp32 scales to bf16 scales (bf16 datapath)
    constexpr uint32_t cb_scale_bcast_fp32_idx = CBIndex::c_4;  // fp32: raw fp32 scales read from DRAM
    constexpr uint32_t cb_scale_scratch_idx = CBIndex::c_6;
    constexpr uint32_t cb_out_idx = CBIndex::c_16;  // reused: output

    auto make_fp32_tile_cb = [&](uint32_t cb_idx, uint32_t num_tiles) {
        CircularBufferConfig cfg =
            CircularBufferConfig(num_tiles * TILE_BYTES, {{cb_idx, fp32_df}}).set_page_size(cb_idx, TILE_BYTES);
        CreateCircularBuffer(program, all_cores, cfg);
    };

    // Compute tiles carry the datapath format (bf16 or fp32) selected by narrow_scales_to_bf16.
    auto make_compute_tile_cb = [&](uint32_t cb_idx, uint32_t num_tiles) {
        CircularBufferConfig cfg = CircularBufferConfig(num_tiles * compute_tile_bytes, {{cb_idx, compute_df}})
                                       .set_page_size(cb_idx, compute_tile_bytes);
        CreateCircularBuffer(program, all_cores, cfg);
    };

    // cb_input_e4m3: input_e4m3, one tile per page; tiles_per_block pages = one block,
    // double-buffered. The reader fills the [tile_h x 128] block contiguously across these pages.
    CircularBufferConfig cb_input_e4m3_cfg =
        CircularBufferConfig(2 * tiles_per_block * input_e4m3_tile_bytes, {{cb_input_e4m3_idx, fp8_df}})
            .set_page_size(cb_input_e4m3_idx, input_e4m3_tile_bytes);
    CreateCircularBuffer(program, all_cores, cb_input_e4m3_cfg);

    // Double-buffered so there is not stalling between the reader/compute/writer.
    make_compute_tile_cb(cb_in_tile_idx, 2 * tiles_per_block);  // reused: tilized input
    make_fp32_tile_cb(cb_scale_bcast_fp32_idx, 2 * block_ht);   // col0 = scale
    if (narrow_scales_to_bf16) {
        // bf16 datapath: scale narrowed to bf16 by the packer.
        make_compute_tile_cb(cb_scale_bcast_bf16_idx, 2 * block_ht);
    } else {
        // fp32 datapath: fp8 e4m3 -> fp32 row-major before tilize.
        make_fp32_tile_cb(cb_in_rm_fp32_idx, 2 * tiles_per_block);
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

    // Reader (RISCV_1): input_e4m3 blocks + builds column-0 scale broadcast operands.
    std::vector<uint32_t> reader_ct_args = {
        cb_input_e4m3_idx,
        cb_scale_bcast_fp32_idx,
        cb_scale_scratch_idx,
        input_e4m3_block_bytes,
        block_ht,
        scale_aligned_page_bytes,
        tile_h,
        tile_w,
        face_h,
        face_w};
    TensorAccessorArgs(src_e4m3_buffer).append_to(reader_ct_args);
    TensorAccessorArgs(src_scale_buffer).append_to(reader_ct_args);
    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/per_token_cast_back/device/kernels/dataflow/"
        "reader_per_token_cast_back.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = reader_ct_args});

    // Writer (RISCV_0): column-block-major writes of the row-major output.
    std::vector<uint32_t> writer_ct_args = {cb_out_idx, out_block_bytes, tile_h, tiles_per_block};
    TensorAccessorArgs(dst_buffer).append_to(writer_ct_args);
    KernelHandle writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/per_token_cast_back/device/kernels/dataflow/"
        "writer_per_token_cast_back.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = writer_ct_args});

    // Compute (TRISC): input_e4m3 -> compute data format tiles -> scale bcast multiply -> untilize to output.
    std::vector<uint32_t> compute_ct_args = {
        cb_input_e4m3_idx,
        cb_in_rm_fp32_idx,
        cb_in_tile_idx,
        cb_scale_bcast_fp32_idx,
        cb_out_idx,
        tile_h,
        tile_w,
        static_cast<uint32_t>(narrow_scales_to_bf16),
        cb_scale_bcast_bf16_idx};

    // HiFi2: bf16 datapath; HiFi4: fp32 datapath.
    const MathFidelity math_fidelity = narrow_scales_to_bf16 ? MathFidelity::HiFi2 : MathFidelity::HiFi4;
    // Skip trip through srcA register. fp8 goes straight to the DEST.
    std::vector<tt::tt_metal::UnpackToDestMode> unpack_to_dest_mode(
        NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
    unpack_to_dest_mode[cb_input_e4m3_idx] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    // On the bf16 path the fp32 scale must unpack losslessly to DEST before the packer rounds it to bf16.
    if (narrow_scales_to_bf16) {
        unpack_to_dest_mode[cb_scale_bcast_fp32_idx] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    }
    KernelHandle compute_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/per_token_cast_back/device/kernels/compute/"
        "compute_per_token_cast_back.cpp",
        all_cores,
        ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = true,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .compile_args = compute_ct_args});

    // Each core's rows form a flat stream of 128-element scale blocks read/written in tile_h-block batches.
    const uint32_t scale_blocks_per_row = H / fp8::BLOCK_W;  // H / 128
    auto all_cores_vec = corerange_to_cores(all_cores, num_cores, true);
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

    for (const auto& core : shared.all_cores_vec) {
        auto& reader_args = tt::tt_metal::GetRuntimeArgs(program, shared.reader_kernel_id, core);
        reader_args[0] = src_e4m3_addr;
        reader_args[1] = src_scale_addr;
        auto& writer_args = tt::tt_metal::GetRuntimeArgs(program, shared.writer_kernel_id, core);
        writer_args[0] = dst_addr;
    }
}

}  // namespace ttnn::experimental::prim::per_token_cast_back
