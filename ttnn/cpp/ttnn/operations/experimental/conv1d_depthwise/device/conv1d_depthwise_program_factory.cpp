// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "conv1d_depthwise_device_operation.hpp"

#include <bit>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::experimental::conv1d_depthwise {

using namespace tt::tt_metal;

static constexpr const char* KERNELS_DIR = "ttnn/cpp/ttnn/operations/experimental/conv1d_depthwise/device/kernels/";

ProgramDescriptor Conv1dDepthwiseOperation::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    const Tensor& input = tensor_args.input;
    IDevice* device = output.device();
    auto grid = device->compute_with_storage_grid_size();

    const auto& in_shape = input.logical_shape();
    const uint32_t B = in_shape[0];
    const uint32_t T_pad = in_shape[1];
    const uint32_t C = in_shape[2];
    const uint32_t K = operation_attributes.taps.size();
    const uint32_t stride = operation_attributes.stride;
    const uint32_t T_out = (T_pad - K) / stride + 1;
    const uint32_t N_out = B * T_out;

    constexpr uint32_t TILE = tt::constants::TILE_WIDTH;  // 32
    const uint32_t C_pad = tt::round_up(C, TILE);
    const uint32_t block_w_tiles = C_pad / TILE;
    const uint32_t block_h_tiles = 1;  // BLOCK_T = 32 rows per block (v1)
    const uint32_t BLOCK_T = block_h_tiles * TILE;
    const uint32_t block_num_tiles = block_w_tiles * block_h_tiles;

    const uint32_t total_tile_rows = tt::div_up(N_out, BLOCK_T);
    auto [num_cores, all_cores, core_group_1, core_group_2, rows_per_core_1, rows_per_core_2] =
        split_work_to_cores(grid, total_tile_rows);

    const tt::DataFormat df = tt::DataFormat::Float32;
    const uint32_t tile_bytes = tile_size(df);

    ProgramDescriptor desc;

    constexpr uint32_t act_cb = tt::CBIndex::c_0;
    constexpr uint32_t scalar_cb = tt::CBIndex::c_1;
    constexpr uint32_t tilized_cb = tt::CBIndex::c_2;
    constexpr uint32_t scratch_cb = tt::CBIndex::c_3;
    constexpr uint32_t out_cb = tt::CBIndex::c_16;
    constexpr uint32_t out_rm_cb = tt::CBIndex::c_17;

    // B==1: the reader reads the per-block input-page union once into scratch_cb and gathers the
    // K tap windows from L1 (cuts the ~K× DRAM re-read; the op is read-bound). Any stride.
    const bool coalesce = (B == 1);
    const uint32_t scratch_num_tiles = tt::div_up((BLOCK_T - 1) * stride + K, TILE) * block_w_tiles;

    auto push_cb = [&](uint32_t cb_id, uint32_t num_tiles) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = num_tiles * tile_bytes,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(cb_id),
                .data_format = df,
                .page_size = tile_bytes,
            }}},
        });
    };
    // Only act_cb is double-buffered (reader/compute overlap); the tilized window and the
    // output accumulator are transient within one block, so 1x keeps large-C inside L1.
    push_cb(act_cb, 2 * block_num_tiles);
    push_cb(scalar_cb, K);  // K resident tap tiles, filled once
    push_cb(tilized_cb, block_num_tiles);
    push_cb(out_cb, block_num_tiles);
    push_cb(out_rm_cb, block_num_tiles);
    if (coalesce) {
        push_cb(scratch_cb, scratch_num_tiles);
    }

    // ---- Reader ----
    KernelDescriptor reader;
    reader.kernel_source = std::string(KERNELS_DIR) + "reader_conv1d_depthwise.cpp";
    reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader.core_ranges = all_cores;
    reader.config = ReaderConfigDescriptor{};
    reader.compile_time_args = {act_cb, scalar_cb, C, C_pad, stride, K, T_pad, T_out, block_h_tiles, scratch_cb, B};
    TensorAccessorArgs(*input.buffer()).append_to(reader.compile_time_args);
    reader.common_runtime_args.reserve(K);
    for (uint32_t j = 0; j < K; ++j) {
        reader.common_runtime_args.push_back(std::bit_cast<uint32_t>(operation_attributes.taps[j]));
    }

    // ---- Compute ----
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);

    KernelDescriptor compute;
    compute.kernel_source = std::string(KERNELS_DIR) + "compute_conv1d_depthwise.cpp";
    compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = all_cores;
    compute.compile_time_args = {act_cb, scalar_cb, tilized_cb, out_cb, out_rm_cb, block_w_tiles, block_h_tiles, K};
    compute.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = true,
        .dst_full_sync_en = dst_full_sync_en,
        .math_approx_mode = math_approx_mode,
    };

    // ---- Writer ----
    KernelDescriptor writer;
    writer.kernel_source = std::string(KERNELS_DIR) + "writer_conv1d_depthwise.cpp";
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = all_cores;
    writer.config = WriterConfigDescriptor{};
    writer.compile_time_args = {out_rm_cb, C, C_pad, block_h_tiles};
    TensorAccessorArgs(*output.buffer()).append_to(writer.compile_time_args);

    // ---- Per-core runtime args ----
    const uint32_t core_h = grid.y;
    uint32_t tile_row_offset = 0;
    for (uint32_t i = 0; i < num_cores; ++i) {
        CoreCoord core = {i / core_h, i % core_h};
        uint32_t tile_rows;
        if (core_group_1.contains(core)) {
            tile_rows = rows_per_core_1;
        } else if (core_group_2.contains(core)) {
            tile_rows = rows_per_core_2;
        } else {
            TT_FATAL(false, "conv1d_depthwise: core not in specified core ranges");
        }

        const uint32_t row_start = tile_row_offset * BLOCK_T;
        const uint32_t rows_avail = (row_start < N_out) ? (N_out - row_start) : 0;
        const uint32_t num_rows = std::min(tile_rows * BLOCK_T, rows_avail);

        reader.emplace_runtime_args(core, {input.buffer(), row_start, num_rows});
        compute.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs{tile_rows});
        writer.emplace_runtime_args(core, {output.buffer(), row_start, num_rows});

        tile_row_offset += tile_rows;
    }

    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(writer));
    desc.kernels.push_back(std::move(compute));

    return desc;
}

}  // namespace ttnn::operations::experimental::conv1d_depthwise
