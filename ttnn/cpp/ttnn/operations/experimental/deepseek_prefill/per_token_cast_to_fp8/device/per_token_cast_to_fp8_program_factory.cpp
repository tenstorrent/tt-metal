// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "per_token_cast_to_fp8_program_factory.hpp"

#include <bit>
#include <map>
#include <string>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>

#include "ttnn/operations/experimental/deepseek_prefill/per_token_cast_to_fp8/per_token_cast_to_fp8.hpp"

// per_token_cast_to_fp8: LLK implementation. Per tile_h x 128 block, the compute kernel gets the input
// into tiles (tilize for ROW_MAJOR, copy for TILE), computes a per-128-element amax = max(|x|), forms
// scale = clamp(amax, 1e-4) / 448 and 1/scale, divides, and untilizes to output_e4m3. The writer extracts
// column 0 of the scale tiles into the [.., H/128] scale output. Both layouts share the same kernels and
// CB layout (built by build_kernels_and_cbs); the two factories differ only in work-split and runtime
// args. Requires H % 128 == 0.

namespace ttnn::experimental::prim::per_token_cast_to_fp8 {

namespace fp8 = ttnn::operations::experimental::deepseek_prefill::per_token_cast_to_fp8;

namespace {

std::pair<uint32_t, uint32_t> fold_M_H(const ttnn::Shape& shape) {
    uint64_t M = 1;
    for (size_t i = 0; i + 1 < shape.size(); ++i) {
        M *= static_cast<uint64_t>(shape[i]);
    }
    return {static_cast<uint32_t>(M), static_cast<uint32_t>(shape[shape.size() - 1])};
}

// split_work_to_cores returns a per-core unit count split across two core-range groups; this maps a core
// to its group's count. The "unit" is rows for the ROW_MAJOR factory and 128-blocks for the TILE factory.
uint32_t units_for_core_from_split(
    const CoreCoord& core,
    const CoreRangeSet& core_range_set_1,
    const CoreRangeSet& core_range_set_2,
    uint32_t units_per_core_g1,
    uint32_t units_per_core_g2) {
    if (core_range_set_1.contains(core)) {
        return units_per_core_g1;
    }
    if (core_range_set_2.contains(core)) {
        return units_per_core_g2;
    }
    return 0;
}

struct KernelIds {
    tt::tt_metal::KernelHandle reader = 0;
    tt::tt_metal::KernelHandle writer = 0;
    tt::tt_metal::KernelHandle compute = 0;
};

// Shared by both factories: creates every circular buffer and the three kernels on all_cores. The kernels
// and CB layout are layout-agnostic; the tile path is selected via `defines` (INPUT_TILE_LAYOUT) and only
// the scale-scratch footprint differs (one row for ROW_MAJOR, tile_h rows for TILE).
KernelIds build_kernels_and_cbs(
    tt::tt_metal::Program& program,
    const Tensor& input,
    const Tensor& output_e4m3,
    const Tensor& output_scale,
    const tt::tt_metal::CoreRangeSet& all_cores,
    uint32_t scale_scratch_bytes,
    bool round_scale_to_power_of_two,
    const std::map<std::string, std::string>& defines) {
    using namespace tt;
    using namespace tt::tt_metal;

    auto [M, H] = fold_M_H(input.logical_shape());  // M = rows, H = width (last dim)
    (void)M;                                        // only H is used here

    // Tile / face dims come from the tensor's tile spec.
    const auto tile_shape = input.tensor_spec().tile().get_tile_shape();
    const auto face_shape = input.tensor_spec().tile().get_face_shape();
    const uint32_t tile_h = tile_shape[0];
    const uint32_t tile_w = tile_shape[1];
    const uint32_t face_h = face_shape[0];
    const uint32_t face_w = face_shape[1];

    // Block geometry: a block is [tile_h x 128] = tiles_per_block tiles wide (BlockWt), one tile tall (BlockHt=1).
    constexpr uint32_t block_w = fp8::BLOCK_W;  // BlockW: 128 elements

    const uint32_t TILE_BYTES_FP32 = tile_h * tile_w * 4;
    const uint32_t block_wt = block_w / tile_w;  // BlockWt: tiles across the 128-wide block
    constexpr uint32_t block_ht = 1;             // BlockHt: one tile-height batch
    const uint32_t tiles_per_block = block_ht * block_wt;

    const uint32_t scale_blocks_per_row = H / fp8::BLOCK_W;  // H / 128
    const uint32_t in_elem_bytes = input.element_size();
    const uint32_t input_block_bytes = block_w * in_elem_bytes;      // reader CT[1]; ROW_MAJOR-only
    const uint32_t output_e4m3_block_bytes = block_w;                // one 128-element row, 1 byte/elem
    const uint32_t in_tile_bytes = tile_h * tile_w * in_elem_bytes;  // cb_in page = one input tile
    const uint32_t output_e4m3_page_bytes = tile_h * tile_w;         // cb_output_e4m3 page = one tile
    const uint32_t scale_aligned_page_bytes = output_scale.buffer()->aligned_page_size();

    auto* src_buffer = input.buffer();
    auto* dst_e4m3_buffer = output_e4m3.buffer();
    auto* dst_scale_buffer = output_scale.buffer();

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

    // cb_in: input, one tile per page; tiles_per_block pages = one 128-wide block, double-buffered.
    // ROW_MAJOR input: the reader fills the block ([tile_h x 128]) contiguously across these pages.
    // TILE input: the reader reads the tiles_per_block tiles individually into these pages.
    CircularBufferConfig cb_in_cfg = CircularBufferConfig(2 * tiles_per_block * in_tile_bytes, {{cb_in_idx, input_df}})
                                         .set_page_size(cb_in_idx, in_tile_bytes);
    CreateCircularBuffer(program, all_cores, cb_in_cfg);

    // cb_tile: input tiles in fp32. ROW_MAJOR: tilized input; TILE: copied input (already tiles).
    make_fp32_tile_cb(cb_tile_idx, tiles_per_block);
    make_fp32_tile_cb(cb_scaler_idx, 1);                      // reduce scaler (1.0), reader-filled
    make_fp32_tile_cb(cb_abs_idx, 2 * block_wt);              // abs tiles for one block row
    make_fp32_tile_cb(cb_scale_tiles_idx, 2 * block_ht);      // col0 = scale
    make_fp32_tile_cb(cb_inv_scale_tiles_idx, 2 * block_ht);  // col0 = 1/scale
    make_fp32_tile_cb(cb_out_tile_idx, tiles_per_block);      // divided tiles -> untilize

    // cb_output_e4m3: e4m3 output, always row-major, one tile per page, double-buffered.
    // Identical for both input layouts (ROW_MAJOR and TILE).
    CircularBufferConfig cb_output_e4m3_cfg =
        CircularBufferConfig(2 * tiles_per_block * output_e4m3_page_bytes, {{cb_output_e4m3_idx, fp8_df}})
            .set_page_size(cb_output_e4m3_idx, output_e4m3_page_bytes);
    CreateCircularBuffer(program, all_cores, cb_output_e4m3_cfg);

    // cb_scale_scratch: writer-private scale staging. Caller sizes it: one page (ROW_MAJOR, one token's
    // row) or tile_h pages (TILE, one row-tile's rows) — see each factory.
    CircularBufferConfig cb_scale_scratch_cfg =
        CircularBufferConfig(scale_scratch_bytes, {{cb_scale_scratch_idx, fp32_df}})
            .set_page_size(cb_scale_scratch_idx, scale_scratch_bytes);
    CreateCircularBuffer(program, all_cores, cb_scale_scratch_cfg);

    // Reader CT layout is shared; CT[1] input_block_bytes is read only on the ROW_MAJOR branch but is
    // passed by both so the TensorAccessorArgs offset (<7>) and later indices line up across layouts.
    std::vector<uint32_t> reader_ct_args = {
        cb_in_idx, input_block_bytes, cb_scaler_idx, tile_h, tile_w, face_h, face_w};
    TensorAccessorArgs(src_buffer).append_to(reader_ct_args);
    KernelIds ids;
    ids.reader = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/per_token_cast_to_fp8/device/kernels/dataflow/"
        "reader_per_token_cast_to_fp8.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_ct_args,
            .defines = defines});

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
    ids.writer = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/per_token_cast_to_fp8/device/kernels/dataflow/"
        "writer_per_token_cast_to_fp8.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_ct_args,
            .defines = defines});

    // Compute (TRISC): [tilize] -> block amax scale + 1/scale -> divide -> untilize to output_e4m3.
    const uint32_t clamp_min_bits = std::bit_cast<uint32_t>(fp8::SCALE_CLAMP_MIN);
    const uint32_t clamp_max_bits = std::bit_cast<uint32_t>(3.0e38f);
    const uint32_t inv_448_bits = std::bit_cast<uint32_t>(1.0f / fp8::E4M3_MAX_NORMAL);
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
        inv_448_bits,
        tile_w,
        static_cast<uint32_t>(round_scale_to_power_of_two)};
    // fp32_dest_acc_en=True is required whenever an 8-bit-float CB (output_e4m3) is on the core (DEST in
    // 32-bit family-agnostic mode); it also gives fp32 precision for the reduce/divide stages.
    ids.compute = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/per_token_cast_to_fp8/device/kernels/compute/"
        "compute_per_token_cast_to_fp8.cpp",
        all_cores,
        ComputeConfig{.fp32_dest_acc_en = true, .compile_args = compute_ct_args, .defines = defines});

    return ids;
}

// Shared create() body for both factories. `tile_layout` picks the input layout, which is the only thing
// that varies: it changes the work-split unit, the scale-scratch size, the INPUT_TILE_LAYOUT kernel define,
// and the per-core runtime args. Everything else (kernels, CBs, the split/build/loop skeleton) is identical.
ttnn::device_operation::CachedProgram<PerTokenCastToFp8SharedVariables> create_program(
    const PerTokenCastToFp8Params& attrs,
    const Tensor& input,
    const Tensor& output_e4m3,
    const Tensor& output_scale,
    bool tile_layout) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input_shape = input.logical_shape();
    const auto [M, H] = fold_M_H(input_shape);  // M = rows, H = width (last dim)
    const auto tile_shape = input.tensor_spec().tile().get_tile_shape();
    const uint32_t tile_h = tile_shape[0];
    const uint32_t scale_blocks_per_row = H / fp8::BLOCK_W;  // 128-wide column-blocks per row
    const uint32_t scale_page_bytes = output_scale.buffer()->aligned_page_size();

    // Work-split unit, scale-scratch size, and kernel define differ by layout. TILE also needs a bit of
    // extra geometry (computed here) to build its per-core args in the loop below.
    uint32_t total_units = 0;
    uint32_t scale_scratch_bytes = 0;
    std::map<std::string, std::string> defines;
    uint32_t rows_per_plane = 0;
    uint32_t row_tiles_per_plane = 0;
    if (tile_layout) {
        // Example — TILE input [2, 30, 1152], 32x32 tiles:
        //   rows_per_plane          = 30           (the 30, i.e. second-to-last dim)
        //   num_planes              = 2            (M=60 / 30)
        //   row_tiles_per_plane     = 1            (30 rows rounded up to 32 = 1 tile tall)
        //   total_units             = 2*1*9 = 18   (9 = 1152/128 blocks per row)
        // Each 30-row chunk is padded to 32 on its own, so rows 30,31 are padding the writer skips.
        rows_per_plane = static_cast<uint32_t>(input_shape[input_shape.size() - 2]);  // R = rows in one [R,H] plane
        const uint32_t num_planes = M / rows_per_plane;        // trailing [R,H] planes (1 for a 2D input)
        row_tiles_per_plane = div_up(rows_per_plane, tile_h);  // tile-rows per plane (R padded up to tile_h)
        total_units = num_planes * row_tiles_per_plane * scale_blocks_per_row;  // total 128-wide blocks
        scale_scratch_bytes = tile_h * scale_page_bytes;  // one plane's tile-row = tile_h scale rows
        defines = {{"INPUT_TILE_LAYOUT", "1"}};
    } else {
        // Split on rows (not tile-rows) so horizontal tensors (small M, large H) use the whole grid; the op is
        // DRAM/NoC-bound, so spreading rows across more cores spreads the data movement. Each core's
        // contiguous row range need not be tile-aligned (kernels address by absolute DRAM page).
        total_units = M;  // rows
        // scale_scratch: writer-private staging for ONE token's full scale row (page-aligned),
        // accumulated across blocks and flushed when the token's last scale block is emitted.
        scale_scratch_bytes = scale_page_bytes;
    }

    Program program{};
    auto compute_grid = input.device()->compute_with_storage_grid_size();
    auto [num_cores, all_cores, crs1, crs2, upc1, upc2] = split_work_to_cores(compute_grid, total_units);

    const KernelIds k = build_kernels_and_cbs(
        program,
        input,
        output_e4m3,
        output_scale,
        all_cores,
        scale_scratch_bytes,
        attrs.round_scale_to_power_of_two,
        defines);

    const uint32_t src_addr = input.buffer()->address();
    const uint32_t dst_e4m3_addr = output_e4m3.buffer()->address();
    const uint32_t dst_scale_addr = output_scale.buffer()->address();

    auto all_cores_vec = corerange_to_cores(all_cores, num_cores, true);
    uint32_t offset = 0;  // ROW_MAJOR: first row; TILE: first 128-block. Advances by this core's unit count.
    for (uint32_t i = 0; i < num_cores; ++i) {
        const auto& core = all_cores_vec[i];
        const uint32_t core_units = units_for_core_from_split(core, crs1, crs2, upc1, upc2);
        uint32_t num_blocks = 0;
        if (tile_layout) {
            num_blocks = core_units;  // the work unit is already a 128-block
            SetRuntimeArgs(program, k.reader, core, {src_addr, offset, num_blocks});
            SetRuntimeArgs(
                program,
                k.writer,
                core,
                {dst_e4m3_addr, dst_scale_addr, offset, num_blocks, rows_per_plane, row_tiles_per_plane});
        } else {
            num_blocks = div_up(core_units * scale_blocks_per_row, tile_h);  // core_units = rows; last may be partial
            SetRuntimeArgs(program, k.reader, core, {src_addr, num_blocks, offset, core_units, H});
            SetRuntimeArgs(program, k.writer, core, {dst_e4m3_addr, dst_scale_addr, num_blocks, offset, core_units, H});
        }
        SetRuntimeArgs(program, k.compute, core, {num_blocks});
        offset += core_units;
    }

    return {std::move(program), {k.reader, k.writer, k.compute, std::move(all_cores_vec)}};
}

// Shared override body: only the I/O buffer addresses change on a program-cache hit; everything else is
// shape-derived and captured at create time (and the shape is in the program hash).
void apply_io_overrides(
    ttnn::device_operation::CachedProgram<PerTokenCastToFp8SharedVariables>& cached_program,
    const PerTokenCastToFp8Inputs& tensor_args,
    std::tuple<Tensor, Tensor>& tensor_return_value) {
    const auto& [output_e4m3, output_scale] = tensor_return_value;
    const auto& shared = cached_program.shared_variables;
    const uint32_t src_addr = tensor_args.input_tensor.buffer()->address();
    const uint32_t dst_e4m3_addr = output_e4m3.buffer()->address();
    const uint32_t dst_scale_addr = output_scale.buffer()->address();
    for (const auto& core : shared.all_cores_vec) {
        auto& reader_args = tt::tt_metal::GetRuntimeArgs(cached_program.program, shared.reader_kernel_id, core);
        reader_args[0] = src_addr;
        auto& writer_args = tt::tt_metal::GetRuntimeArgs(cached_program.program, shared.writer_kernel_id, core);
        writer_args[0] = dst_e4m3_addr;
        writer_args[1] = dst_scale_addr;
    }
}

}  // namespace

PerTokenCastToFp8RowMajorProgramFactory::cached_program_t PerTokenCastToFp8RowMajorProgramFactory::create(
    const PerTokenCastToFp8Params& operation_attributes,
    const PerTokenCastToFp8Inputs& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto& [output_e4m3, output_scale] = tensor_return_value;
    return create_program(
        operation_attributes, tensor_args.input_tensor, output_e4m3, output_scale, /*tile_layout=*/false);
}

void PerTokenCastToFp8RowMajorProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const PerTokenCastToFp8Params& /*operation_attributes*/,
    const PerTokenCastToFp8Inputs& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    apply_io_overrides(cached_program, tensor_args, tensor_return_value);
}

PerTokenCastToFp8TileProgramFactory::cached_program_t PerTokenCastToFp8TileProgramFactory::create(
    const PerTokenCastToFp8Params& operation_attributes,
    const PerTokenCastToFp8Inputs& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto& [output_e4m3, output_scale] = tensor_return_value;
    return create_program(
        operation_attributes, tensor_args.input_tensor, output_e4m3, output_scale, /*tile_layout=*/true);
}

void PerTokenCastToFp8TileProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const PerTokenCastToFp8Params& /*operation_attributes*/,
    const PerTokenCastToFp8Inputs& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    apply_io_overrides(cached_program, tensor_args, tensor_return_value);
}

}  // namespace ttnn::experimental::prim::per_token_cast_to_fp8
