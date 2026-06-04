// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "per_token_cast_to_fp8_program_factory.hpp"

#include <bit>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>

#include "ttnn/operations/experimental/deepseek_prefill/common/fp8_quant_common.hpp"

// per_token_cast_to_fp8: LLK implementation (promoted from the experiments/token-cast-to-fp8 spike).
// Per tile_h x 128 block, the compute kernel tilizes the input, computes a per-128-element
// amax = max(|x|), forms scale = clamp(amax, 1e-4) / 448 and 1/scale, divides, and untilizes
// to output_e4m3. The writer extracts column 0 of the scale tiles into the [.., H/128] scale
// output. Requires H % 128 == 0; work is split across cores over rows (each core gets a contiguous,
// not necessarily tile-aligned, row range).

namespace ttnn::experimental::prim::per_token_cast_to_fp8 {

namespace common = ttnn::operations::experimental::deepseek_prefill::fp8_quant_common;

PerTokenCastToFp8ProgramFactory::cached_program_t PerTokenCastToFp8ProgramFactory::create(
    const PerTokenCastToFp8Params& /*operation_attributes*/,
    const PerTokenCastToFp8Inputs& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input = tensor_args.input_tensor;
    const auto& [output_e4m3, output_scale] = tensor_return_value;

    const auto& input_shape = input.logical_shape();
    auto [M, H] = common::infer_M_H(input_shape);  // M = rows, H = width (last dim)

    // Tile / face dims come from the tensor's tile spec.
    const auto tile_shape = input.tensor_spec().tile().get_tile_shape();
    const auto face_shape = input.tensor_spec().tile().get_face_shape();
    const uint32_t tile_h = tile_shape[0];
    const uint32_t tile_w = tile_shape[1];
    const uint32_t face_h = face_shape[0];
    const uint32_t face_w = face_shape[1];

    // M and H are now arbitrary; the last tile-row / column-block may be partial (zero-padded by the
    // reader, written back only for real rows/columns by the writer). H stays a multiple of 128 so
    // 128-element scale blocks are always full and the scale tensor's last dim is H/128.
    TT_FATAL(
        H % common::SCALE_GROUP_SIZE == 0,
        "per_token_cast_to_fp8: H={} must be a multiple of SCALE_GROUP_SIZE={}",
        H,
        common::SCALE_GROUP_SIZE);

    // A block is exactly one 128-element scale block per row. The input/output_e4m3 CBs keep
    // one-tile pages, and the reader fills the [tile_h x 128] block as one contiguous run.
    constexpr uint32_t BLOCK_ELEMS = common::SCALE_GROUP_SIZE;  // 128

    const uint32_t TILE_BYTES_FP32 = tile_h * tile_w * 4;
    const uint32_t COL_BLOCK_TILES = BLOCK_ELEMS / tile_w;                         // 4 for 32-wide tiles
    constexpr uint32_t GROUPS_PER_BLOCK = BLOCK_ELEMS / common::SCALE_GROUP_SIZE;  // 1

    const uint32_t scale_blocks_per_row = H / common::SCALE_GROUP_SIZE;  // H / 128
    const uint32_t in_elem_bytes = input.element_size();
    const uint32_t input_block_bytes = BLOCK_ELEMS * in_elem_bytes;  // one 128-element row of a block
    const uint32_t output_e4m3_block_bytes = BLOCK_ELEMS;            // one 128-element row, 1 byte/elem
    const uint32_t in_tile_bytes = tile_h * tile_w * in_elem_bytes;  // cb_in page = one input tile
    const uint32_t output_e4m3_page_bytes = tile_h * tile_w;         // cb_output_e4m3 page = one tile
    const uint32_t scale_aligned_page_bytes = output_scale.buffer()->aligned_page_size();

    auto* src_buffer = input.buffer();
    auto* dst_e4m3_buffer = output_e4m3.buffer();
    auto* dst_scale_buffer = output_scale.buffer();

    Program program{};

    auto* device = input.device();
    auto compute_grid = device->compute_with_storage_grid_size();
    // Split on rows (not tile-rows) so horizontal tensors (small M, large H) use the whole grid;
    // the op is DRAM/NoC-bound, so spreading rows across more cores spreads the data movement. Each
    // core's contiguous row range need not be tile-aligned (kernels address by absolute DRAM page).
    auto [num_cores, all_cores, core_group_1, core_group_2, rows_per_core_g1, rows_per_core_g2] =
        split_work_to_cores(compute_grid, M);

    const DataFormat input_df = datatype_to_dataformat_converter(input.dtype());
    const DataFormat fp32_df = DataFormat::Float32;
    const DataFormat fp8_df = DataFormat::Fp8_e4m3;

    // CB indices mirror the generic_op spike (op.py).
    constexpr uint32_t cb_in_idx = CBIndex::c_0;
    constexpr uint32_t cb_tile_idx = CBIndex::c_1;
    constexpr uint32_t cb_scaler_idx = CBIndex::c_3;
    constexpr uint32_t cb_abs_idx = CBIndex::c_4;
    constexpr uint32_t cb_scale_tiles_idx = CBIndex::c_5;
    constexpr uint32_t cb_scale_scratch_idx = CBIndex::c_6;
    constexpr uint32_t cb_inv_scale_tiles_idx = CBIndex::c_7;
    constexpr uint32_t cb_out_tile_idx = CBIndex::c_8;
    constexpr uint32_t cb_output_e4m3_idx = CBIndex::c_16;

    auto make_fp32_tile_cb = [&](uint32_t cb_idx, uint32_t num_tiles) {
        CircularBufferConfig cfg = CircularBufferConfig(num_tiles * TILE_BYTES_FP32, {{cb_idx, fp32_df}})
                                       .set_page_size(cb_idx, TILE_BYTES_FP32);
        CreateCircularBuffer(program, all_cores, cfg);
    };

    // cb_in: input row-major, one tile per page; COL_BLOCK_TILES pages = one 128-wide block,
    // double-buffered. The reader fills the block ([tile_h x 128]) contiguously across these pages.
    CircularBufferConfig cb_in_cfg = CircularBufferConfig(2 * COL_BLOCK_TILES * in_tile_bytes, {{cb_in_idx, input_df}})
                                         .set_page_size(cb_in_idx, in_tile_bytes);
    CreateCircularBuffer(program, all_cores, cb_in_cfg);

    make_fp32_tile_cb(cb_tile_idx, COL_BLOCK_TILES);                  // tilized input
    make_fp32_tile_cb(cb_scaler_idx, 1);                              // reduce scaler (1.0), reader-filled
    make_fp32_tile_cb(cb_abs_idx, 2 * COL_BLOCK_TILES);               // per-group abs tiles
    make_fp32_tile_cb(cb_scale_tiles_idx, 2 * GROUPS_PER_BLOCK);      // col0 = scale
    make_fp32_tile_cb(cb_inv_scale_tiles_idx, 2 * GROUPS_PER_BLOCK);  // col0 = 1/scale
    make_fp32_tile_cb(cb_out_tile_idx, COL_BLOCK_TILES);              // divided tiles -> untilize

    // cb_output_e4m3: output_e4m3 row-major output, one tile per page; COL_BLOCK_TILES pages = one
    // block, double-buffered.
    CircularBufferConfig cb_output_e4m3_cfg =
        CircularBufferConfig(2 * COL_BLOCK_TILES * output_e4m3_page_bytes, {{cb_output_e4m3_idx, fp8_df}})
            .set_page_size(cb_output_e4m3_idx, output_e4m3_page_bytes);
    CreateCircularBuffer(program, all_cores, cb_output_e4m3_cfg);

    // cb_scale_scratch: writer-private staging for ONE token's full scale row (page-aligned),
    // accumulated across blocks and flushed when the token's last scale block is emitted.
    const uint32_t scale_scratch_bytes = scale_aligned_page_bytes;
    CircularBufferConfig cb_scale_scratch_cfg =
        CircularBufferConfig(scale_scratch_bytes, {{cb_scale_scratch_idx, fp32_df}})
            .set_page_size(cb_scale_scratch_idx, scale_scratch_bytes);
    CreateCircularBuffer(program, all_cores, cb_scale_scratch_cfg);

    // Reader (RISCV_1): fills the reduce scaler, then column-block-major input reads.
    std::vector<uint32_t> reader_ct_args = {
        cb_in_idx, input_block_bytes, cb_scaler_idx, tile_h, tile_w, face_h, face_w};
    TensorAccessorArgs(src_buffer).append_to(reader_ct_args);
    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/per_token_cast_to_fp8/device/kernels/dataflow/"
        "reader_per_token_cast_to_fp8.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = reader_ct_args});

    // Writer (RISCV_0): output_e4m3 block writes + scale extraction (col 0 of cb_scale_tiles) -> scale.
    std::vector<uint32_t> writer_ct_args = {
        cb_output_e4m3_idx,
        output_e4m3_block_bytes,
        cb_scale_tiles_idx,
        cb_scale_scratch_idx,
        scale_blocks_per_row,
        scale_aligned_page_bytes,
        tile_h,
        tile_w,
        face_h,
        face_w};
    TensorAccessorArgs(dst_e4m3_buffer).append_to(writer_ct_args);
    TensorAccessorArgs(dst_scale_buffer).append_to(writer_ct_args);
    KernelHandle writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/per_token_cast_to_fp8/device/kernels/dataflow/"
        "writer_per_token_cast_to_fp8.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = writer_ct_args});

    // Compute (TRISC): tilize -> per-group amax scale + 1/scale -> divide -> untilize to output_e4m3.
    const uint32_t clamp_min_bits = std::bit_cast<uint32_t>(common::SCALE_CLAMP_MIN);
    const uint32_t clamp_max_bits = std::bit_cast<uint32_t>(3.0e38f);
    const uint32_t inv_e4m3_max_bits = std::bit_cast<uint32_t>(1.0f / common::E4M3_MAX_NORMAL);
    std::vector<uint32_t> compute_ct_args = {
        cb_in_idx,
        cb_tile_idx,
        cb_scaler_idx,
        cb_abs_idx,
        cb_scale_tiles_idx,
        cb_inv_scale_tiles_idx,
        cb_out_tile_idx,
        cb_output_e4m3_idx,
        clamp_min_bits,
        clamp_max_bits,
        inv_e4m3_max_bits,
        tile_w};
    // fp32_dest_acc_en=True is required whenever an 8-bit-float CB (output_e4m3) is on the core (DEST in
    // 32-bit family-agnostic mode); it also gives fp32 precision for the reduce/divide stages.
    KernelHandle compute_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/per_token_cast_to_fp8/device/kernels/compute/"
        "compute_per_token_cast_to_fp8.cpp",
        all_cores,
        ComputeConfig{.fp32_dest_acc_en = true, .compile_args = compute_ct_args});

    // Each core owns rows [row_offset, row_offset+rows_for_core). Its 128-element scale blocks
    // form a flat stream read/written in tile_h-block batches.
    const uint32_t scale_blocks_per_core_row = scale_blocks_per_row;  // H / 128
    auto all_cores_vec = corerange_to_cores(all_cores, num_cores, true);
    uint32_t row_offset = 0;
    for (uint32_t i = 0; i < num_cores; ++i) {
        const auto& core = all_cores_vec[i];
        uint32_t rows_for_core =
            core_group_1.contains(core) ? rows_per_core_g1 : (core_group_2.contains(core) ? rows_per_core_g2 : 0);
        const uint32_t total_scale_blocks = rows_for_core * scale_blocks_per_core_row;
        const uint32_t num_blocks = tt::div_up(total_scale_blocks, tile_h);  // last block may be partial

        // Host-side invariant checks (no LLK investigation needed downstream if these hold).
        TT_FATAL(
            num_blocks == tt::div_up(rows_for_core * scale_blocks_per_core_row, tile_h),
            "per_token_cast_to_fp8: num_blocks invariant violated on a core");

        SetRuntimeArgs(
            program, reader_kernel_id, core, {src_buffer->address(), num_blocks, row_offset, rows_for_core, H});
        SetRuntimeArgs(
            program,
            writer_kernel_id,
            core,
            {dst_e4m3_buffer->address(), dst_scale_buffer->address(), num_blocks, row_offset, rows_for_core, H});
        SetRuntimeArgs(program, compute_kernel_id, core, {num_blocks});
        row_offset += rows_for_core;
    }

    return cached_program_t{
        std::move(program), {reader_kernel_id, writer_kernel_id, compute_kernel_id, std::move(all_cores_vec)}};
}

void PerTokenCastToFp8ProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const PerTokenCastToFp8Params& /*operation_attributes*/,
    const PerTokenCastToFp8Inputs& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& program = cached_program.program;
    auto& shared = cached_program.shared_variables;

    const auto& [output_e4m3, output_scale] = tensor_return_value;
    uint32_t src_addr = tensor_args.input_tensor.buffer()->address();
    uint32_t dst_e4m3_addr = output_e4m3.buffer()->address();
    uint32_t dst_scale_addr = output_scale.buffer()->address();

    for (const auto& core : shared.all_cores_vec) {
        auto& reader_args = tt::tt_metal::GetRuntimeArgs(program, shared.reader_kernel_id, core);
        reader_args[0] = src_addr;
        auto& writer_args = tt::tt_metal::GetRuntimeArgs(program, shared.writer_kernel_id, core);
        writer_args[0] = dst_e4m3_addr;
        writer_args[1] = dst_scale_addr;
    }
}

}  // namespace ttnn::experimental::prim::per_token_cast_to_fp8
