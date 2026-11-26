// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "per_token_cast_to_fp8_program_factory.hpp"

#include <bit>
#include <string>
#include <utility>

#include <tt-metalium/workload/legacy/host_api.hpp>
#include <tt-metalium/utils/work_split.hpp>
#include <tt-metalium/workload/legacy/tensor_accessor_args.hpp>
#include <tt-metalium/data_formats/tile_constants.hpp>
#include <tt-metalium/utils/math.hpp>

#include "ttnn/operations/experimental/quasar/deepseek_prefill/per_token_cast_to_fp8/per_token_cast_to_fp8.hpp"

// per_token_cast_to_fp8: LLK implementation. Per tile_h x 128 block, the compute kernel gets the input
// into tiles (tilize for ROW_MAJOR, copy for TILE), computes a per-128-element amax = max(|x|), forms
// scale = clamp(amax, 1e-4) / 448 and 1/scale, divides, and untilizes to output_e4m3. The writer extracts
// column 0 of the scale tiles into the [.., H/128] scale output. A single factory handles both layouts:
// they share the same kernels and CB layout (built by build_kernels_and_cbs) and differ only in work-split
// and runtime args. Requires H % 128 == 0.

namespace ttnn::experimental::prim::qsr::per_token_cast_to_fp8 {

namespace fp8 = ttnn::operations::experimental::quasar::deepseek_prefill::per_token_cast_to_fp8;

namespace {

std::pair<uint32_t, uint32_t> fold_M_H(const ttnn::Shape& shape) {
    uint64_t M = 1;
    for (size_t i = 0; i + 1 < shape.size(); ++i) {
        M *= static_cast<uint64_t>(shape[i]);
    }
    return {static_cast<uint32_t>(M), static_cast<uint32_t>(shape[shape.size() - 1])};
}

// split_work_to_cores returns a per-core unit count split across two core-range groups; this maps a core
// to its group's count. The "unit" is rows on the ROW_MAJOR path and 128-blocks on the TILE path.
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

// Kernel positions in ProgramDescriptor::kernels, fixed by the push_back order in build_kernels_and_cbs.
constexpr size_t kReaderKernelIdx = 0;
constexpr size_t kWriterKernelIdx = 1;
constexpr size_t kComputeKernelIdx = 2;

// Shared by both layout paths: describes every circular buffer and the three kernels on all_cores. The
// kernels and CB layout are layout-agnostic; the tile path is selected via `defines` (INPUT_TILE_LAYOUT)
// and only the scale-scratch footprint differs (one row for ROW_MAJOR, tile_h rows for TILE).
void build_kernels_and_cbs(
    tt::tt_metal::ProgramDescriptor& desc,
    const Tensor& input,
    const Tensor& output_e4m3,
    const Tensor& output_scale,
    const tt::tt_metal::CoreRangeSet& all_cores,
    uint32_t scale_scratch_bytes,
    bool round_scale_to_power_of_two,
    const tt::tt_metal::KernelDescriptor::Defines& defines) {
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
    const uint32_t scale_aligned_page_bytes = output_scale.mesh_buffer().aligned_page_size();

    const auto* src_buffer = &input.mesh_buffer();
    const auto* dst_e4m3_buffer = &output_e4m3.mesh_buffer();
    const auto* dst_scale_buffer = &output_scale.mesh_buffer();

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
    constexpr uint32_t cb_output_e4m3_idx = CBIndex::c_16;

    auto make_cb = [&](uint32_t cb_idx, DataFormat data_format, uint32_t total_size, uint32_t page_size) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = total_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(cb_idx),
                .data_format = data_format,
                .page_size = page_size,
            }}},
        });
    };
    auto make_fp32_tile_cb = [&](uint32_t cb_idx, uint32_t num_tiles) {
        make_cb(cb_idx, fp32_df, num_tiles * TILE_BYTES_FP32, TILE_BYTES_FP32);
    };

    // cb_in: input, one tile per page; tiles_per_block pages = one 128-wide block, double-buffered.
    // ROW_MAJOR input: the reader fills the block ([tile_h x 128]) contiguously across these pages.
    // TILE input: the reader reads the tiles_per_block tiles individually into these pages.
    make_cb(cb_in_idx, input_df, 2 * tiles_per_block * in_tile_bytes, in_tile_bytes);

    // cb_tile: input tiles in fp32. ROW_MAJOR: tilized input; TILE: copied input (already tiles).
    make_fp32_tile_cb(cb_tile_idx, tiles_per_block);
    make_fp32_tile_cb(cb_scaler_idx, 1);                      // reduce scaler (1.0), reader-filled
    make_fp32_tile_cb(cb_abs_idx, 2 * block_wt);              // abs tiles for one block row
    make_fp32_tile_cb(cb_scale_tiles_idx, 2 * block_ht);      // col0 = scale
    make_fp32_tile_cb(cb_inv_scale_tiles_idx, 2 * block_ht);  // col0 = 1/scale

    // cb_output_e4m3: e4m3 output, always row-major, one tile per page, double-buffered.
    // Identical for both input layouts (ROW_MAJOR and TILE).
    make_cb(cb_output_e4m3_idx, fp8_df, 2 * tiles_per_block * output_e4m3_page_bytes, output_e4m3_page_bytes);

    // cb_scale_scratch: writer-private scale staging. Caller sizes it: one page (ROW_MAJOR, one token's
    // row) or tile_h pages (TILE, one row-tile's rows) — see each layout path in create_descriptor.
    make_cb(cb_scale_scratch_idx, fp32_df, scale_scratch_bytes, scale_scratch_bytes);

    // Reader CT layout is shared; CT[1] input_block_bytes is read only on the ROW_MAJOR branch but is
    // passed by both so the TensorAccessorArgs offset (<7>) and later indices line up across layouts.
    std::vector<uint32_t> reader_ct_args = {
        cb_in_idx, input_block_bytes, cb_scaler_idx, tile_h, tile_w, face_h, face_w};
    TensorAccessorArgs(src_buffer).append_to(reader_ct_args);
    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/quasar/deepseek_prefill/per_token_cast_to_fp8/device/kernels/dataflow/"
        "reader_per_token_cast_to_fp8.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_ct_args);
    reader_desc.defines = defines;
    reader_desc.config = DataMovementConfigDescriptor{
        .processor = DataMovementProcessor::RISCV_1,
        .noc = NOC::RISCV_1_default,
    };

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
    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/quasar/deepseek_prefill/per_token_cast_to_fp8/device/kernels/dataflow/"
        "writer_per_token_cast_to_fp8.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_ct_args);
    writer_desc.defines = defines;
    writer_desc.config = DataMovementConfigDescriptor{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::RISCV_0_default,
    };

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
        cb_output_e4m3_idx,
        clamp_min_bits,
        clamp_max_bits,
        inv_448_bits,
        tile_w,
        static_cast<uint32_t>(round_scale_to_power_of_two)};
    // fp32_dest_acc_en=True is required whenever an 8-bit-float CB (output_e4m3) is on the core (DEST in
    // 32-bit family-agnostic mode); it also gives fp32 precision for the reduce/divide stages.
    KernelDescriptor compute_desc;
    compute_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/quasar/deepseek_prefill/per_token_cast_to_fp8/device/kernels/compute/"
        "compute_per_token_cast_to_fp8.cpp";
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = all_cores;
    compute_desc.compile_time_args = std::move(compute_ct_args);
    compute_desc.defines = defines;
    compute_desc.config = ComputeConfigDescriptor{.fp32_dest_acc_en = true};

    // Push order fixes kReaderKernelIdx / kWriterKernelIdx / kComputeKernelIdx.
    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));
}

}  // namespace

// Shared body for both layout paths. `tile_layout` picks the input layout, which is the only thing that varies:
// it changes the work-split unit, the scale-scratch size, the INPUT_TILE_LAYOUT kernel define, and the per-core
// runtime args. Everything else (kernels, CBs, the split/build/loop skeleton) is identical.
//
// Cache hits: everything here derives from the program hash (attributes + the input's TensorSpec, which also
// fixes both outputs' specs), so only the three I/O buffer addresses differ between two calls sharing a key.
// They are passed as Buffer* bindings and the framework re-points them on a hit; no override is needed.
tt::tt_metal::ProgramDescriptor PerTokenCastToFp8ProgramFactory::create_descriptor(
    const PerTokenCastToFp8Params& operation_attributes,
    const PerTokenCastToFp8Inputs& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input = tensor_args.input_tensor;
    const auto& [output_e4m3, output_scale] = tensor_return_value;
    const bool tile_layout = input.layout() == Layout::TILE;

    const auto& input_shape = input.logical_shape();
    const auto [M, H] = fold_M_H(input_shape);  // M = rows, H = width (last dim)
    const auto tile_shape = input.tensor_spec().tile().get_tile_shape();
    const uint32_t tile_h = tile_shape[0];
    const uint32_t scale_blocks_per_row = H / fp8::BLOCK_W;  // 128-wide column-blocks per row
    const uint32_t scale_page_bytes = output_scale.mesh_buffer().aligned_page_size();

    // Work-split unit, scale-scratch size, and kernel define differ by layout. TILE also needs a bit of
    // extra geometry (computed here) to build its per-core args in the loop below.
    uint32_t total_units = 0;
    uint32_t scale_scratch_bytes = 0;
    KernelDescriptor::Defines defines;
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

    ProgramDescriptor desc;
    auto compute_grid = input.device()->compute_with_storage_grid_size();
    auto [num_cores, all_cores, crs1, crs2, upc1, upc2] = split_work_to_cores(compute_grid, total_units);

    build_kernels_and_cbs(
        desc,
        input,
        output_e4m3,
        output_scale,
        all_cores,
        scale_scratch_bytes,
        operation_attributes.round_scale_to_power_of_two,
        defines);

    auto& reader_desc = desc.kernels[kReaderKernelIdx];
    auto& writer_desc = desc.kernels[kWriterKernelIdx];
    auto& compute_desc = desc.kernels[kComputeKernelIdx];

    // Buffer* (not ->address()) so each address slot is a binding the framework patches on a cache hit.
    Buffer* src_buffer = input.buffer();
    Buffer* dst_e4m3_buffer = output_e4m3.buffer();
    Buffer* dst_scale_buffer = output_scale.buffer();

    auto all_cores_vec = corerange_to_cores(all_cores, num_cores, true);
    uint32_t offset = 0;  // ROW_MAJOR: first row; TILE: first 128-block. Advances by this core's unit count.
    for (uint32_t i = 0; i < num_cores; ++i) {
        const auto& core = all_cores_vec[i];
        const uint32_t core_units = units_for_core_from_split(core, crs1, crs2, upc1, upc2);
        uint32_t num_blocks = 0;
        if (tile_layout) {
            num_blocks = core_units;  // the work unit is already a 128-block
            reader_desc.emplace_runtime_args(core, {src_buffer, offset, num_blocks});
            writer_desc.emplace_runtime_args(
                core, {dst_e4m3_buffer, dst_scale_buffer, offset, num_blocks, rows_per_plane, row_tiles_per_plane});
        } else {
            num_blocks = div_up(core_units * scale_blocks_per_row, tile_h);  // core_units = rows; last may be partial
            reader_desc.emplace_runtime_args(core, {src_buffer, num_blocks, offset, core_units, H});
            writer_desc.emplace_runtime_args(
                core, {dst_e4m3_buffer, dst_scale_buffer, num_blocks, offset, core_units, H});
        }
        compute_desc.emplace_runtime_args(core, {num_blocks});
        offset += core_units;
    }

    return desc;
}

}  // namespace ttnn::experimental::prim::qsr::per_token_cast_to_fp8
