// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "per_token_cast_back_masked_program_factory.hpp"

#include <algorithm>
#include <cstdint>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/operations/experimental/deepseek_prefill/per_token_cast_to_fp8/per_token_cast_to_fp8.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"

// Masked per_token_cast_back: decompress only this device's valid expert-region rows of a dispatch
// buffer, using scale row = metadata[row][1] (token_idx). Built as a per-mesh-coordinate
// ProgramDescriptor so each device's expert window (counter_offset) is derived from its linearized
// mesh coordinate. Reuses the plain path's block/broadcast machinery; the
// only kernel changes are: outer loop over this device's experts, scale gathered by token_idx, and a
// dynamic per-core num_blocks (depends on device-side counts) passed reader->compute via a small CB.

namespace ttnn::experimental::prim::per_token_cast_back {

namespace fp8 = ttnn::operations::experimental::deepseek_prefill::per_token_cast_to_fp8;

namespace {

using namespace tt;
using namespace tt::tt_metal;

std::pair<uint32_t, uint32_t> fold_M_H(const ttnn::Shape& shape) {
    uint64_t M = 1;
    for (size_t i = 0; i + 1 < shape.size(); ++i) {
        M *= static_cast<uint64_t>(shape[i]);
    }
    return {static_cast<uint32_t>(M), static_cast<uint32_t>(shape[shape.size() - 1])};
}

uint32_t aligned_page_bytes(const Tensor& t) { return static_cast<uint32_t>(t.buffer()->aligned_page_size()); }
uint32_t num_pages(const Tensor& t) { return static_cast<uint32_t>(t.buffer()->num_pages()); }

void push_cb(
    ProgramDescriptor& desc,
    const CoreRangeSet& cores,
    uint32_t cb_id,
    DataFormat df,
    uint32_t page_size,
    uint32_t num_pages_in_cb) {
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_pages_in_cb * page_size,
        .core_ranges = cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_id),
            .data_format = df,
            .page_size = page_size,
        }}},
    });
}

ProgramDescriptor build_program_for_coord(
    const PerTokenCastBackParams& attrs,
    const PerTokenCastBackInputs& inputs,
    Tensor& output,
    const ttnn::MeshCoordinate& coord) {
    ProgramDescriptor desc;

    const auto& input_e4m3 = inputs.input_e4m3;
    const auto& input_scale = inputs.input_scale;
    const auto& counts = *inputs.expert_token_counts;
    const auto& offsets = *inputs.expert_region_offsets;
    const auto& metadata = *inputs.metadata;

    auto* mesh_device = input_e4m3.device();
    const auto& mesh_view = mesh_device->get_view();
    const uint32_t mesh_rows = mesh_view.num_rows();
    const uint32_t mesh_cols = mesh_view.num_cols();
    const uint32_t linearized = ttnn::operations::ccl::common::get_linearized_index(coord, mesh_view);
    const uint32_t mesh_row = linearized / mesh_cols;
    const uint32_t mesh_col = linearized % mesh_cols;

    const auto& shape = input_e4m3.logical_shape();
    auto [T, H] = fold_M_H(shape);  // T = dispatch buffer token capacity, H = hidden dim

    // The scale tensor is the dispatch group's gathered per-token scale: its token dim spans all
    // dispatch_group_size source devices (ISL = dispatch_group_size * seq_len_per_chip). A dispatched
    // row's scale lives at the global row (source_position_in_group * seq_len_per_chip + token_idx),
    // where source_position_in_group = metadata[0] / mesh_cols (cluster_axis=0). seq_len_per_chip is
    // recovered from the gathered scale's token dim. On a 1x1 mesh metadata[0]=0, so this reduces to
    // scale[token_idx] (the non-dispatched single-source case).
    const auto& scale_shape = input_scale.logical_shape();
    const uint32_t scale_rows = static_cast<uint32_t>(scale_shape[scale_shape.size() - 2]);
    const uint32_t seq_len_per_chip = scale_rows / attrs.dispatch_group_size;

    const uint32_t num_routed_experts = static_cast<uint32_t>(counts.logical_shape()[-1]);
    const uint32_t num_dispatch_groups = num_routed_experts / (attrs.experts_per_chip * attrs.dispatch_group_size);
    TT_FATAL(
        num_dispatch_groups > 0 &&
            num_dispatch_groups * attrs.experts_per_chip * attrs.dispatch_group_size == num_routed_experts,
        "per_token_cast_back(masked): num_dispatch_groups derivation failed (routed_experts={}, experts_per_chip={}, "
        "dispatch_group_size={})",
        num_routed_experts,
        attrs.experts_per_chip,
        attrs.dispatch_group_size);
    const uint32_t dispatch_group_idx = mesh_col % num_dispatch_groups;
    const uint32_t experts_per_dispatch_group = attrs.experts_per_chip * mesh_rows;
    const uint32_t counter_offset = dispatch_group_idx * experts_per_dispatch_group + mesh_row * attrs.experts_per_chip;

    // Tile / face dims and block geometry — identical to the plain factory.
    const auto tile_shape = input_e4m3.tensor_spec().tile().get_tile_shape();
    const auto face_shape = input_e4m3.tensor_spec().tile().get_face_shape();
    const uint32_t tile_h = tile_shape[0];
    const uint32_t tile_w = tile_shape[1];
    const uint32_t face_h = face_shape[0];
    const uint32_t face_w = face_shape[1];

    constexpr uint32_t block_w = fp8::BLOCK_W;  // 128
    const uint32_t TILE_BYTES = tile_h * tile_w * sizeof(float);
    const uint32_t block_wt = block_w / tile_w;
    constexpr uint32_t block_ht = 1;
    const uint32_t tiles_per_block = block_ht * block_wt;

    const uint32_t input_e4m3_block_bytes = block_w;
    const uint32_t out_elem_bytes = output.element_size();
    const uint32_t out_block_bytes = block_w * out_elem_bytes;
    const uint32_t input_e4m3_tile_bytes = tile_h * tile_w;
    const uint32_t out_tile_bytes = tile_h * tile_w * out_elem_bytes;
    const uint32_t scale_aligned_page_bytes = aligned_page_bytes(input_scale);
    const uint32_t counts_aligned_page_bytes = aligned_page_bytes(counts);
    const uint32_t metadata_aligned_page_bytes = aligned_page_bytes(metadata);
    const uint32_t counts_pages = num_pages(counts);  // offsets has the same shape/pages

    // Split this device's experts across cores (one contiguous expert sub-range per core).
    auto compute_grid = mesh_device->compute_with_storage_grid_size();
    const uint32_t max_cores = compute_grid.x * compute_grid.y;
    const uint32_t num_cores = std::max<uint32_t>(1, std::min(attrs.experts_per_chip, max_cores));
    const uint32_t experts_per_core_range = tt::div_up(attrs.experts_per_chip, num_cores);
    auto all_cores = num_cores_to_corerangeset(num_cores, compute_grid, /*row_wise=*/true);
    auto all_cores_vec = corerange_to_cores(all_cores, num_cores, /*row_wise=*/true);

    const DataFormat fp8_df = DataFormat::Fp8_e4m3;
    const DataFormat fp32_df = DataFormat::Float32;
    const DataFormat output_df = datatype_to_dataformat_converter(attrs.output_dtype);
    const DataFormat counts_df = datatype_to_dataformat_converter(counts.dtype());
    const DataFormat offsets_df = datatype_to_dataformat_converter(offsets.dtype());
    const DataFormat metadata_df = datatype_to_dataformat_converter(metadata.dtype());

    // CB indices: c_0/c_1/c_2/c_4/c_5/c_6/c_16 mirror the plain factory; the rest are masked staging.
    constexpr uint32_t cb_input_e4m3 = CBIndex::c_0;
    constexpr uint32_t cb_in_rm = CBIndex::c_1;
    constexpr uint32_t cb_in_tile = CBIndex::c_2;
    constexpr uint32_t cb_counts = CBIndex::c_3;  // reader counts scratch
    constexpr uint32_t cb_scale_bcast = CBIndex::c_4;
    constexpr uint32_t cb_out_tile = CBIndex::c_5;
    constexpr uint32_t cb_scale_scratch = CBIndex::c_6;
    constexpr uint32_t cb_offsets = CBIndex::c_7;     // reader offsets scratch
    constexpr uint32_t cb_metadata = CBIndex::c_8;    // reader metadata scratch
    constexpr uint32_t cb_nblocks = CBIndex::c_9;     // reader -> compute dynamic block count
    constexpr uint32_t cb_counts_w = CBIndex::c_10;   // writer counts scratch
    constexpr uint32_t cb_offsets_w = CBIndex::c_11;  // writer offsets scratch
    constexpr uint32_t cb_out = CBIndex::c_16;

    // Decompress pipeline CBs (same as plain).
    push_cb(desc, all_cores, cb_input_e4m3, fp8_df, input_e4m3_tile_bytes, 2 * tiles_per_block);
    push_cb(desc, all_cores, cb_in_rm, fp32_df, TILE_BYTES, tiles_per_block);
    push_cb(desc, all_cores, cb_in_tile, fp32_df, TILE_BYTES, tiles_per_block);
    push_cb(desc, all_cores, cb_scale_bcast, fp32_df, TILE_BYTES, 2 * block_ht);
    push_cb(desc, all_cores, cb_out_tile, fp32_df, TILE_BYTES, tiles_per_block);
    push_cb(desc, all_cores, cb_out, output_df, out_tile_bytes, 2 * tiles_per_block);
    // Reader-private scratch for up to tile_h tokens' full scale rows (gathered by token_idx).
    push_cb(desc, all_cores, cb_scale_scratch, fp32_df, tile_h * scale_aligned_page_bytes, 1);

    // Masked staging CBs (tiny). Counts/offsets staged whole; metadata staged tile_h rows per block.
    push_cb(desc, all_cores, cb_counts, counts_df, counts_aligned_page_bytes, counts_pages);
    push_cb(desc, all_cores, cb_offsets, offsets_df, counts_aligned_page_bytes, counts_pages);
    // Reader-private metadata scratch: one page holding tile_h tokens' metadata rows (stride =
    // metadata_aligned_page_bytes), mirroring the scale scratch layout.
    push_cb(desc, all_cores, cb_metadata, metadata_df, tile_h * metadata_aligned_page_bytes, 1);
    push_cb(desc, all_cores, cb_nblocks, DataFormat::UInt32, sizeof(uint32_t) * 4, 1);
    push_cb(desc, all_cores, cb_counts_w, counts_df, counts_aligned_page_bytes, counts_pages);
    push_cb(desc, all_cores, cb_offsets_w, offsets_df, counts_aligned_page_bytes, counts_pages);

    const std::string kdir =
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/per_token_cast_back/device/kernels/";

    // ---- Reader ----
    std::vector<uint32_t> reader_ct = {
        cb_input_e4m3,
        cb_scale_bcast,
        cb_scale_scratch,
        cb_counts,
        cb_offsets,
        cb_metadata,
        cb_nblocks,
        input_e4m3_block_bytes,
        block_ht,
        scale_aligned_page_bytes,
        tile_h,
        tile_w,
        face_h,
        face_w,
        counts_aligned_page_bytes,
        metadata_aligned_page_bytes,
        counts_pages,
        counter_offset,
        attrs.experts_per_chip,
        T,
        num_routed_experts,
        mesh_cols,         // index 21: divides metadata[0] (source coord) -> source row in group
        seq_len_per_chip,  // index 22: per-source token count (gathered scale stride)
    };
    TensorAccessorArgs(input_e4m3.buffer()).append_to(reader_ct);
    TensorAccessorArgs(input_scale.buffer()).append_to(reader_ct);
    TensorAccessorArgs(counts.buffer()).append_to(reader_ct);
    TensorAccessorArgs(offsets.buffer()).append_to(reader_ct);
    TensorAccessorArgs(metadata.buffer()).append_to(reader_ct);

    KernelDescriptor reader_kd;
    reader_kd.kernel_source = kdir + "dataflow/reader_per_token_cast_back_masked.cpp";
    reader_kd.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_kd.core_ranges = all_cores;
    reader_kd.compile_time_args = std::move(reader_ct);
    reader_kd.config =
        DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default};
    const KernelHandle reader_id = static_cast<KernelHandle>(desc.kernels.size());
    desc.kernels.push_back(std::move(reader_kd));

    // ---- Compute ----
    std::vector<uint32_t> compute_ct = {
        cb_input_e4m3, cb_in_rm, cb_in_tile, cb_scale_bcast, cb_out_tile, cb_out, cb_nblocks, tile_h, tile_w};
    KernelDescriptor compute_kd;
    compute_kd.kernel_source = kdir + "compute/compute_per_token_cast_back_masked.cpp";
    compute_kd.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_kd.core_ranges = all_cores;
    compute_kd.compile_time_args = std::move(compute_ct);
    compute_kd.config = ComputeConfigDescriptor{.fp32_dest_acc_en = true};
    const KernelHandle compute_id = static_cast<KernelHandle>(desc.kernels.size());
    desc.kernels.push_back(std::move(compute_kd));

    // ---- Writer ----
    std::vector<uint32_t> writer_ct = {
        cb_out,
        cb_counts_w,
        cb_offsets_w,
        out_block_bytes,
        tile_h,
        tiles_per_block,
        counts_aligned_page_bytes,
        counts_pages,
        counter_offset,
        attrs.experts_per_chip,
        T,
    };
    TensorAccessorArgs(output.buffer()).append_to(writer_ct);
    TensorAccessorArgs(counts.buffer()).append_to(writer_ct);
    TensorAccessorArgs(offsets.buffer()).append_to(writer_ct);

    KernelDescriptor writer_kd;
    writer_kd.kernel_source = kdir + "dataflow/writer_per_token_cast_back_masked.cpp";
    writer_kd.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_kd.core_ranges = all_cores;
    writer_kd.compile_time_args = std::move(writer_ct);
    writer_kd.config =
        DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default};
    const KernelHandle writer_id = static_cast<KernelHandle>(desc.kernels.size());
    desc.kernels.push_back(std::move(writer_kd));

    // Per-core runtime args: each core owns expert sub-range [expert_start, expert_end).
    for (uint32_t i = 0; i < num_cores; ++i) {
        const auto& core = all_cores_vec[i];
        const uint32_t expert_start = i * experts_per_core_range;
        const uint32_t expert_end = std::min((i + 1) * experts_per_core_range, attrs.experts_per_chip);

        KernelDescriptor::RTArgList reader_rt;
        reader_rt.push_back(input_e4m3.buffer());
        reader_rt.push_back(input_scale.buffer());
        reader_rt.push_back(counts.buffer());
        reader_rt.push_back(offsets.buffer());
        reader_rt.push_back(metadata.buffer());
        reader_rt.push_back(expert_start);
        reader_rt.push_back(expert_end);
        reader_rt.push_back(H);
        desc.kernels[reader_id].emplace_runtime_args(core, reader_rt);

        // Compute reads num_blocks from cb_nblocks; no per-core RT args needed.
        desc.kernels[compute_id].emplace_runtime_args(core, KernelDescriptor::RTArgList{});

        KernelDescriptor::RTArgList writer_rt;
        writer_rt.push_back(output.buffer());
        writer_rt.push_back(counts.buffer());
        writer_rt.push_back(offsets.buffer());
        writer_rt.push_back(expert_start);
        writer_rt.push_back(expert_end);
        writer_rt.push_back(H);
        desc.kernels[writer_id].emplace_runtime_args(core, writer_rt);
    }

    return desc;
}

}  // namespace

WorkloadDescriptor MaskedPerTokenCastBackProgramFactory::create_workload_descriptor(
    const PerTokenCastBackParams& operation_attributes,
    const PerTokenCastBackInputs& tensor_args,
    Tensor& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    // Masked decompress is per-device single-shot (no cross-device communication), but it is
    // mesh-coord-dependent because counter_offset is baked into each program. Build one
    // ProgramDescriptor per coordinate; no GlobalSemaphores / Synchronize needed.
    WorkloadDescriptor workload_descriptor;
    for (const auto& coord : tensor_coords.coords()) {
        auto desc = build_program_for_coord(operation_attributes, tensor_args, tensor_return_value, coord);
        workload_descriptor.programs.push_back({ttnn::MeshCoordinateRange(coord), std::move(desc)});
    }
    return workload_descriptor;
}

}  // namespace ttnn::experimental::prim::per_token_cast_back
