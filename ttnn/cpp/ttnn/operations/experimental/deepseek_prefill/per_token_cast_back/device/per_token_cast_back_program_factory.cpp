// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "per_token_cast_back_program_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>

#include "ttnn/operations/experimental/deepseek_prefill/common/fp8_quant_common.hpp"

// per_token_cast_back: LLK implementation (promoted from the experiments/e4m3-cast grouped spike).
// out = decode(e4m3) * scale, with one fp32 scale per token per 128-element group. Per (tile-row,
// 1024-col column-block): convert e4m3 -> fp32 (copy_tile), tilize, multiply each tile by its
// group's per-row scale broadcast from column 0 (mul_tiles_bcast_cols), and untilize to the output
// dtype (bf16 or fp32). The reader builds the per-group column-0 broadcast operands from the scale
// tensor. Requires H % 128 == 0; work is split across cores over tile-rows.

namespace ttnn::experimental::prim::per_token_cast_back {

namespace common = ttnn::operations::experimental::deepseek_prefill::fp8_quant_common;

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
    auto [M, H] = common::infer_M_H(shape);  // M = rows, H = width (last dim)

    // Tile / face dims come from the tensor's tile spec.
    const auto tile_shape = input_e4m3.tensor_spec().tile().get_tile_shape();
    const auto face_shape = input_e4m3.tensor_spec().tile().get_face_shape();
    const uint32_t tile_h = tile_shape[0];
    const uint32_t tile_w = tile_shape[1];
    const uint32_t face_h = face_shape[0];
    const uint32_t face_w = face_shape[1];

    // M and H are now arbitrary; the last tile-row / column-block may be partial (zero-padded by the
    // reader, written back only for real rows/columns by the writer). H stays a multiple of 128 so
    // groups are always full and the scale tensor's last dim is H/128.
    TT_FATAL(
        H % common::SCALE_GROUP_SIZE == 0,
        "per_token_cast_back: H={} must be a multiple of SCALE_GROUP_SIZE={}",
        H,
        common::SCALE_GROUP_SIZE);

    const uint32_t TILE_BYTES = tile_h * tile_w * sizeof(float);
    const uint32_t COL_BLOCK_TILES = common::COL_BLOCK_ELEMS / tile_w;                         // 32 for 32-wide tiles
    constexpr uint32_t GROUPS_PER_BLOCK =
        common::COL_BLOCK_ELEMS /
        common::SCALE_GROUP_SIZE;  // 8 groups of 128 elements per column block of 1024 elements

    const uint32_t tile_rows = tt::div_up(M, tile_h);                        // last tile-row may be partial
    const uint32_t num_col_blocks = tt::div_up(H, common::COL_BLOCK_ELEMS);  // last col-block may be partial
    const uint32_t e4m3_col_block_bytes = common::COL_BLOCK_ELEMS;  // 1 byte/elem
    const uint32_t out_elem_bytes = output.element_size();
    const uint32_t out_col_block_bytes = common::COL_BLOCK_ELEMS * out_elem_bytes;  // bf16: 2048, fp32: 4096
    const uint32_t scale_aligned_page_bytes = input_scale.buffer()->aligned_page_size();

    auto* src_e4m3_buffer = input_e4m3.buffer();
    auto* src_scale_buffer = input_scale.buffer();
    auto* dst_buffer = output.buffer();

    Program program{};

    auto* device = input_e4m3.device();
    auto compute_grid = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, rows_per_core_g1, rows_per_core_g2] =
        split_work_to_cores(compute_grid, tile_rows);

    const DataFormat fp8_df = DataFormat::Fp8_e4m3;
    const DataFormat fp32_df = DataFormat::Float32;
    const DataFormat output_df = datatype_to_dataformat_converter(operation_attributes.output_dtype);

    constexpr uint32_t cb_e4m3_idx = CBIndex::c_0;
    constexpr uint32_t cb_in_rm_idx = CBIndex::c_1;
    constexpr uint32_t cb_in_tile_idx = CBIndex::c_2;
    constexpr uint32_t cb_scale_bcast_idx = CBIndex::c_4;
    constexpr uint32_t cb_out_tile_idx = CBIndex::c_5;
    constexpr uint32_t cb_scale_scratch_idx = CBIndex::c_6;
    constexpr uint32_t cb_out_idx = CBIndex::c_16;

    auto make_fp32_tile_cb = [&](uint32_t cb_idx, uint32_t num_tiles) {
        CircularBufferConfig cfg =
            CircularBufferConfig(num_tiles * TILE_BYTES, {{cb_idx, fp32_df}}).set_page_size(cb_idx, TILE_BYTES);
        CreateCircularBuffer(program, all_cores, cfg);
    };

    // cb_e4m3: e4m3 input, one tile (1024 bytes) per page, double-buffered over a column-block.
    CircularBufferConfig cb_e4m3_cfg =
        CircularBufferConfig(2 * COL_BLOCK_TILES * e4m3_col_block_bytes, {{cb_e4m3_idx, fp8_df}})
            .set_page_size(cb_e4m3_idx, e4m3_col_block_bytes);
    CreateCircularBuffer(program, all_cores, cb_e4m3_cfg);

    make_fp32_tile_cb(cb_in_rm_idx, COL_BLOCK_TILES);             // e4m3 -> fp32 RM
    make_fp32_tile_cb(cb_in_tile_idx, COL_BLOCK_TILES);           // tilized fp32 input
    make_fp32_tile_cb(cb_scale_bcast_idx, 2 * GROUPS_PER_BLOCK);  // per-group col0 = scale
    make_fp32_tile_cb(cb_out_tile_idx, COL_BLOCK_TILES);          // divided tiles -> untilize

    // cb_out: row-major output (bf16/fp32), one column-block stick per page, double-buffered.
    CircularBufferConfig cb_out_cfg =
        CircularBufferConfig(2 * COL_BLOCK_TILES * out_col_block_bytes, {{cb_out_idx, output_df}})
            .set_page_size(cb_out_idx, out_col_block_bytes);
    CreateCircularBuffer(program, all_cores, cb_out_cfg);

    // cb_scale_scratch: reader-private staging for 32 tokens' full scale rows (page-aligned stride).
    const uint32_t scale_scratch_bytes = tile_h * scale_aligned_page_bytes;
    CircularBufferConfig cb_scale_scratch_cfg =
        CircularBufferConfig(scale_scratch_bytes, {{cb_scale_scratch_idx, fp32_df}})
            .set_page_size(cb_scale_scratch_idx, scale_scratch_bytes);
    CreateCircularBuffer(program, all_cores, cb_scale_scratch_cfg);

    // Reader (RISCV_1): e4m3 col-blocks + builds per-group column-0 scale broadcast operands.
    std::vector<uint32_t> reader_ct_args = {
        cb_e4m3_idx,
        cb_scale_bcast_idx,
        cb_scale_scratch_idx,
        e4m3_col_block_bytes,
        GROUPS_PER_BLOCK,
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
    std::vector<uint32_t> writer_ct_args = {cb_out_idx, out_col_block_bytes, tile_h};
    TensorAccessorArgs(dst_buffer).append_to(writer_ct_args);
    KernelHandle writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/per_token_cast_back/device/kernels/dataflow/"
        "writer_per_token_cast_back.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = writer_ct_args});

    // Compute (TRISC): e4m3 -> fp32 RM -> tilize -> per-group bcast multiply -> untilize to output.
    std::vector<uint32_t> compute_ct_args = {
        cb_e4m3_idx, cb_in_rm_idx, cb_in_tile_idx, cb_scale_bcast_idx, cb_out_tile_idx, cb_out_idx, tile_h, tile_w};
    // fp32_dest_acc_en=True required (e4m3 CB on core); HiFi4 (the ComputeConfig default) keeps the
    // broadcast multiply precise.
    KernelHandle compute_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/per_token_cast_back/device/kernels/compute/"
        "compute_per_token_cast_back.cpp",
        all_cores,
        ComputeConfig{.fp32_dest_acc_en = true, .compile_args = compute_ct_args});

    auto all_cores_vec = corerange_to_cores(all_cores, num_cores, true);
    uint32_t row_offset = 0;
    for (uint32_t i = 0; i < num_cores; ++i) {
        const auto& core = all_cores_vec[i];
        uint32_t rows_for_core =
            core_group_1.contains(core) ? rows_per_core_g1 : (core_group_2.contains(core) ? rows_per_core_g2 : 0);

        SetRuntimeArgs(
            program,
            reader_kernel_id,
            core,
            {src_e4m3_buffer->address(), src_scale_buffer->address(), rows_for_core, num_col_blocks, row_offset, M, H});
        SetRuntimeArgs(
            program, writer_kernel_id, core, {dst_buffer->address(), rows_for_core, num_col_blocks, row_offset, M, H});
        SetRuntimeArgs(program, compute_kernel_id, core, {rows_for_core, num_col_blocks});
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
