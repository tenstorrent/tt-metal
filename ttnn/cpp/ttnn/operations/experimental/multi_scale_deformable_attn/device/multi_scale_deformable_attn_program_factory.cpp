// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <vector>

#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include "hostdevcommon/kernel_structs.h"

#include "multi_scale_deformable_attn_device_operation.hpp"

namespace ttnn::operations::experimental::multi_scale_deformable_attn {

using namespace tt::tt_metal;

namespace {

uint32_t aligned_page_size(uint32_t raw_bytes, BufferType buffer_type) {
    const uint32_t alignment = buffer_type == BufferType::DRAM ? tt::tt_metal::hal::get_dram_alignment()
                                                               : tt::tt_metal::hal::get_l1_alignment();
    return tt::round_up(raw_bytes, alignment);
}

}  // namespace

ProgramDescriptor MSDAOperation::create_descriptor(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args, Tensor& output) {
    ProgramDescriptor descriptor{};

    const auto& value = tensor_args.value;
    const auto& grid = tensor_args.grid;
    const auto& attn = tensor_args.attn;

    auto* device = value.device();
    const auto compute_grid = device->compute_with_storage_grid_size();

    // Shape extraction.
    const auto& vs = value.logical_shape();  // (N, h, w, D)
    const auto& as = attn.logical_shape();   // (N, Q, P)

    const uint32_t N = vs[0];
    const uint32_t h_in = vs[1];
    const uint32_t w_in = vs[2];
    const uint32_t D = vs[3];
    const uint32_t Q = as[1];
    const uint32_t P = as[2];
    const uint32_t reduction_size = 4 * P;

    constexpr uint32_t TILE_MAX_ROWS = 32;

    // Build the list of output tiles: each tile holds up to 32 contiguous
    // queries from a single batch index n. Q is not required to divide 32 —
    // the trailing tile in each n carries v_rows = Q % 32 if nonzero.
    struct TileAssignment {
        uint32_t n;
        uint32_t q_start;
        uint32_t v_rows;
    };
    std::vector<TileAssignment> tiles;
    tiles.reserve(N * ((Q + TILE_MAX_ROWS - 1) / TILE_MAX_ROWS));
    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t q_start = 0; q_start < Q; q_start += TILE_MAX_ROWS) {
            const uint32_t count = std::min(TILE_MAX_ROWS, Q - q_start);
            tiles.push_back({n, q_start, count});
        }
    }
    const uint32_t total_output_tiles = static_cast<uint32_t>(tiles.size());

    // Stick sizes (raw, before alignment).
    const uint32_t value_stick_raw = D * value.element_size();
    const uint32_t grid_stick_raw = 2u * grid.element_size();
    const uint32_t attn_stick_raw = P * attn.element_size();
    const uint32_t output_stick_raw = D * output.element_size();

    const uint32_t value_stick_aligned = aligned_page_size(value_stick_raw, value.buffer()->buffer_type());
    const uint32_t grid_stick_aligned = aligned_page_size(grid_stick_raw, grid.buffer()->buffer_type());
    const uint32_t attn_stick_aligned = aligned_page_size(attn_stick_raw, attn.buffer()->buffer_type());
    const uint32_t output_stick_aligned = aligned_page_size(output_stick_raw, output.buffer()->buffer_type());

    // Tile size (bf16): 32 * 32 * 2 = 2048 bytes.
    const auto data_format = datatype_to_dataformat_converter(DataType::BFLOAT16);
    const uint32_t tile_nbytes = tt::tile_size(data_format);

    // Work split across cores in tile units.
    auto [num_cores, all_cores, core_group_1, core_group_2, tiles_per_core_group_1, tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_grid, total_output_tiles);

    // Data formats (all bf16).
    const auto value_fmt = datatype_to_dataformat_converter(value.dtype());
    const auto grid_fmt = datatype_to_dataformat_converter(grid.dtype());
    const auto attn_fmt = datatype_to_dataformat_converter(attn.dtype());
    const auto output_fmt = datatype_to_dataformat_converter(output.dtype());

    // CB indices. CBFormatDescriptor::buffer_index is uint8_t — keep these
    // typed the same so push_cb's aggregate init doesn't trigger a narrowing
    // conversion (forbidden in brace-init).
    constexpr uint8_t value_scratch_cb = tt::CBIndex::c_0;   // raw stick scratch (reader-only)
    constexpr uint8_t grid_cb = tt::CBIndex::c_1;            // grid scratch (reader-only)
    constexpr uint8_t attn_cb = tt::CBIndex::c_2;            // attn scratch (reader-only)
    constexpr uint8_t input_tile_cb = tt::CBIndex::c_3;      // reader -> compute (tile)
    constexpr uint8_t scalar_tile_cb = tt::CBIndex::c_4;     // reader -> compute (tile)
    constexpr uint8_t output_tile_cb = tt::CBIndex::c_16;    // compute -> writer (tile)
    constexpr uint8_t output_scratch_cb = tt::CBIndex::c_5;  // writer-only stick scratch

    auto push_cb = [&](uint8_t idx, uint32_t pages, uint32_t page_size, tt::DataFormat fmt) {
        descriptor.cbs.push_back(CBDescriptor{
            .total_size = pages * page_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = idx,
                .data_format = fmt,
                .page_size = page_size,
            }}},
        });
    };
    // Reader-only scratches sized to hold one full output tile worth of
    // staging (up to 32 queries per tile). Reader reserves the whole CB
    // once at startup and treats each as a linear L1 arena.
    push_cb(value_scratch_cb, TILE_MAX_ROWS, value_stick_aligned, value_fmt);
    push_cb(grid_cb, TILE_MAX_ROWS * P, grid_stick_aligned, grid_fmt);
    push_cb(attn_cb, TILE_MAX_ROWS, attn_stick_aligned, attn_fmt);
    // Reader -> compute pipes: double-buffered tiles.
    push_cb(input_tile_cb, 2, tile_nbytes, data_format);
    push_cb(scalar_tile_cb, 2, tile_nbytes, data_format);
    // Compute -> writer pipe: double-buffered tiles.
    push_cb(output_tile_cb, 2, tile_nbytes, output_fmt);
    // Writer-only scratch: 1 page.
    push_cb(output_scratch_cb, 1, output_stick_aligned, output_fmt);

    // Reader kernel descriptor (CT args, NoC config, but runtime args filled later).
    KernelDescriptor::CompileTimeArgs reader_ct{
        value_scratch_cb,
        grid_cb,
        attn_cb,
        input_tile_cb,
        scalar_tile_cb,
        D,
        Q,
        P,
        h_in,
        w_in,
        value_stick_aligned,
        grid_stick_aligned,
        attn_stick_aligned,
        static_cast<uint32_t>(operation_attributes.align_corners),
    };
    TensorAccessorArgs(*value.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*grid.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*attn.buffer()).append_to(reader_ct);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/multi_scale_deformable_attn/device/kernels/dataflow/reader_msda.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_ct);
    reader_desc.config = ReaderConfigDescriptor{};

    // Compute kernel descriptor.
    KernelDescriptor compute_desc;
    compute_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/multi_scale_deformable_attn/device/kernels/compute/msda_compute.cpp";
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = all_cores;
    compute_desc.compile_time_args = {input_tile_cb, scalar_tile_cb, output_tile_cb, reduction_size};
    compute_desc.config = ComputeConfigDescriptor{};

    // Writer kernel descriptor.
    KernelDescriptor::CompileTimeArgs writer_ct{
        output_tile_cb,
        output_scratch_cb,
        output_stick_aligned,
    };
    TensorAccessorArgs(*output.buffer()).append_to(writer_ct);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/multi_scale_deformable_attn/device/kernels/dataflow/writer_msda.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_ct);
    writer_desc.config = WriterConfigDescriptor{};

    // Per-core runtime args. Buffer* entries auto-register as buffer bindings
    // so the framework patches addresses on cache hits (no override_runtime_arguments).
    const auto logical_cores = corerange_to_cores(all_cores, num_cores, /*row_wise=*/true);
    uint32_t tile_cursor = 0;
    for (const auto& core : logical_cores) {
        uint32_t tiles_here = 0;
        if (core_group_1.contains(core)) {
            tiles_here = tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            tiles_here = tiles_per_core_group_2;
        } else {
            continue;
        }

        KernelDescriptor::RTArgList reader_args;
        reader_args.reserve(4 + tiles_here * 3);
        reader_args.push_back(value.buffer());
        reader_args.push_back(grid.buffer());
        reader_args.push_back(attn.buffer());
        reader_args.push_back(tiles_here);

        KernelDescriptor::RTArgList writer_args;
        writer_args.reserve(2 + tiles_here * 2);
        writer_args.push_back(output.buffer());
        writer_args.push_back(tiles_here);

        for (uint32_t i = 0; i < tiles_here; ++i) {
            const auto& asn = tiles[tile_cursor + i];
            reader_args.push_back(asn.n);
            reader_args.push_back(asn.q_start);
            reader_args.push_back(asn.v_rows);
            writer_args.push_back(asn.n * Q + asn.q_start);
            writer_args.push_back(asn.v_rows);
        }

        reader_desc.emplace_runtime_args(core, reader_args);
        compute_desc.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs{tiles_here});
        writer_desc.emplace_runtime_args(core, writer_args);

        tile_cursor += tiles_here;
    }

    // Kernels pushed in a fixed order (reader=0, compute=1, writer=2).
    descriptor.kernels.push_back(std::move(reader_desc));
    descriptor.kernels.push_back(std::move(compute_desc));
    descriptor.kernels.push_back(std::move(writer_desc));

    return descriptor;
}

}  // namespace ttnn::operations::experimental::multi_scale_deformable_attn
