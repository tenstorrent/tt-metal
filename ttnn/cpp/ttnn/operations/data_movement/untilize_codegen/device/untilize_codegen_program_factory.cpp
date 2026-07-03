// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/untilize_codegen/device/untilize_codegen_program_factory.hpp"

#include <algorithm>
#include <cstdint>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::prim {

using namespace tt::tt_metal;

namespace {

constexpr std::string_view kKernelDir = "ttnn/cpp/ttnn/operations/data_movement/untilize_codegen/device/kernels/";

// Port of builder._compute_block_ct_dim: largest divisor of Wt no bigger than max_bct.
uint32_t compute_block_ct_dim(uint32_t Wt, bool fp32) {
    const uint32_t max_bct = fp32 ? 4 : 8;
    for (uint32_t bct = max_bct; bct >= 1; --bct) {
        if (Wt % bct == 0) {
            return bct;
        }
    }
    return 1;
}

struct CbDepths {
    uint32_t cb_in_depth;
    uint32_t cb_out_depth;
    uint32_t read_batch;
};

// Port of builder._cb_depths: L1-aware tiered CB depth selection. Recomputed here
// (not an attribute) because it is a pure function of Wt/tile_size, which are already
// in the cache key via the input spec — so it is identical per cache key.
CbDepths cb_depths(uint32_t Wt, uint32_t tile_size, uint32_t block_ct_dim) {
    // Matches builder_utils.USABLE_L1 (physical L1 minus ~100KB firmware/stack).
    constexpr uint64_t USABLE_L1 = 1'400'000;
    const uint64_t double_both = static_cast<uint64_t>(2 * Wt + 2 * Wt) * tile_size;
    const uint64_t double_in = static_cast<uint64_t>(2 * Wt + Wt) * tile_size;
    const uint64_t single_both = static_cast<uint64_t>(Wt + Wt) * tile_size;

    if (double_both <= USABLE_L1) {
        return {2 * Wt, 2 * Wt, Wt};  // tier 1: full pipeline overlap
    }
    if (double_in <= USABLE_L1) {
        return {2 * Wt, Wt, Wt};  // tier 2: reader overlaps compute
    }
    if (single_both <= USABLE_L1) {
        return {Wt, Wt, block_ct_dim};  // tier 3: no overlap
    }
    return {std::max(Wt, block_ct_dim), Wt, block_ct_dim};  // tier 4: Wt too large for L1
}

}  // namespace

ProgramDescriptor UntilizeCodegenProgramFactory::create_descriptor(
    const UntilizeCodegenParams& /*operation_attributes*/,
    const UntilizeCodegenInputs& tensor_args,
    Tensor& tensor_return_value) {
    const auto& input = tensor_args.input;
    const auto& output = tensor_return_value;

    Buffer* src_buffer = input.buffer();
    Buffer* dst_buffer = output.buffer();

    const auto& shape = input.logical_shape();
    const uint32_t W = shape[3];
    const uint32_t H = shape[2];
    const uint32_t NC = shape[0] * shape[1];
    const uint32_t Wt = W / tt::constants::TILE_WIDTH;
    const uint32_t Ht = H / tt::constants::TILE_HEIGHT;
    const uint32_t total_tile_rows = NC * Ht;  // one tile-row = Wt tiles -> TILE_HEIGHT sticks

    const bool fp32 = (input.dtype() == DataType::FLOAT32);
    const tt::DataFormat df = datatype_to_dataformat_converter(input.dtype());
    const uint32_t tile_size = tt::tile_size(df);
    const uint32_t elem_size = input.element_size();

    IDevice* device = input.device();
    const CoreCoord grid = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, tpc1, tpc2] = split_work_to_cores(grid, total_tile_rows);

    const uint32_t block_ct_dim = compute_block_ct_dim(Wt, fp32);
    const CbDepths depths = cb_depths(Wt, tile_size, block_ct_dim);
    const uint32_t read_batch = depths.read_batch;

    constexpr uint32_t cb_in_id = 0;
    constexpr uint32_t cb_out_id = 16;
    const uint32_t max_bct = fp32 ? 4 : 8;

    ProgramDescriptor desc;

    // CB0: input tiles (reader -> compute). CB16: RM sticks (compute -> writer).
    desc.cbs.push_back(CBDescriptor{
        .total_size = depths.cb_in_depth * tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cb_in_id, .data_format = df, .page_size = tile_size}}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = depths.cb_out_depth * tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cb_out_id, .data_format = df, .page_size = tile_size}}},
    });

    // Reader: unified tile reader with the IDENTITY sequencer (seq_id=0). Positional
    // CT is TensorAccessorArgs only; addressing params are named.
    KernelDescriptor reader_desc;
    reader_desc.kernel_source = std::string(kKernelDir) + "reader_tile_interleaved_unified.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.named_compile_time_args = {{"seq_id", 0}, {"cb_id", cb_in_id}, {"batch", read_batch}};
    TensorAccessorArgs(*src_buffer).append_to(reader_desc.compile_time_args);
    reader_desc.config = ReaderConfigDescriptor{};

    // Writer: untilize interleaved writer. CT = [cb_out, unpadded_row_bytes] + TA + [Wt].
    const uint32_t unpadded_row_size_bytes = W * elem_size;
    const uint32_t padded_row_size_bytes = Wt * tt::constants::TILE_WIDTH * elem_size;  // == W since tile-aligned
    KernelDescriptor writer_desc;
    writer_desc.kernel_source = std::string(kKernelDir) + "writer_untilize_interleaved.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = {cb_out_id, unpadded_row_size_bytes};
    TensorAccessorArgs(*dst_buffer).append_to(writer_desc.compile_time_args);
    writer_desc.compile_time_args.push_back(Wt);
    writer_desc.config = WriterConfigDescriptor{};

    // Compute: pack_untilize. per_core_block_cnt (CT arg 0) is the per-core tile-row
    // count, which differs across the cliff split -> one descriptor per core group.
    auto make_compute = [&](const CoreRangeSet& cores, uint32_t tpc) {
        KernelDescriptor k;
        k.kernel_source = std::string(kKernelDir) + "compute_untilize.cpp";
        k.source_type = KernelDescriptor::SourceType::FILE_PATH;
        k.core_ranges = cores;
        k.compile_time_args = {tpc, Wt, cb_in_id, cb_out_id, max_bct};
        ComputeConfigDescriptor cc;
        cc.fp32_dest_acc_en = fp32;
        k.config = cc;
        return k;
    };
    KernelDescriptor compute_desc_1 = make_compute(core_group_1, tpc1);
    const bool has_cliff = core_group_2.num_cores() > 0;
    KernelDescriptor compute_desc_2;
    if (has_cliff) {
        compute_desc_2 = make_compute(core_group_2, tpc2);
    }

    // Per-core runtime args. Slot 0 of reader & writer MUST be the Buffer* (not a raw
    // address) so the framework binds it for the O(1) cache-hit patch path; a raw
    // address would silently defeat program caching.
    const auto cores = corerange_to_cores(all_cores, num_cores, /*row_wise=*/false);
    const uint32_t H_unpadded = 0;      // no height unpadding in this path
    const uint32_t padded_batch_h = 0;  // unused when H_unpadded == 0
    const uint32_t out_page_start = 0;
    uint32_t tile_row_offset = 0;
    for (const auto& core : cores) {
        const uint32_t n_tile_rows = core_group_1.contains(core) ? tpc1 : tpc2;
        const uint32_t start_tile = tile_row_offset * Wt;
        const uint32_t num_tiles = n_tile_rows * Wt;
        reader_desc.emplace_runtime_args(core, {src_buffer, num_tiles, start_tile});

        const uint32_t start_stick = tile_row_offset * tt::constants::TILE_HEIGHT;
        writer_desc.emplace_runtime_args(
            core,
            {dst_buffer, n_tile_rows, start_stick, padded_row_size_bytes, H_unpadded, padded_batch_h, out_page_start});

        tile_row_offset += n_tile_rows;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc_1));
    if (has_cliff) {
        desc.kernels.push_back(std::move(compute_desc_2));
    }
    return desc;
}

}  // namespace ttnn::prim
