// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "engram_gate_prefill_program_factory.hpp"

#include <algorithm>
#include <bit>
#include <string>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::experimental::deepseek::engram_gate_prefill {

using namespace tt::tt_metal;

namespace {

constexpr std::string_view kKernelDir =
    "ttnn/cpp/ttnn/operations/experimental/deepseek/engram_gate_prefill/device/kernels";

constexpr uint32_t kTileHeight = tt::constants::TILE_HEIGHT;
constexpr uint32_t kTileWidth = tt::constants::TILE_WIDTH;

// Tiles of D a core holds at once. D = 5120 is 160 tiles, so this is 10 chunks per unit.
constexpr uint32_t kChunkTiles = 16;

// One work unit is a (32-token block, stream copy) and a core owns it whole, the full D included: the gate's three
// reductions then never leave the core, so no cross-core exchange (and no semaphore) is needed.
constexpr uint8_t kCbX = tt::CBIndex::c_0;      // this chunk of x, bf16
constexpr uint8_t kCbK = tt::CBIndex::c_1;      // this chunk of key, bf16
constexpr uint8_t kCbG = tt::CBIndex::c_2;      // this chunk of the gate vector, fp32
constexpr uint8_t kCbV = tt::CBIndex::c_3;      // this chunk of value, bf16
constexpr uint8_t kCbOut = tt::CBIndex::c_4;    // this chunk of the output, bf16
constexpr uint8_t kCbProd = tt::CBIndex::c_5;   // fp32 scratch: x^2, key^2 or x*gate*key
constexpr uint8_t kCbP0 = tt::CBIndex::c_6;     // per-chunk row sums of x^2
constexpr uint8_t kCbP1 = tt::CBIndex::c_7;     // per-chunk row sums of key^2
constexpr uint8_t kCbP2 = tt::CBIndex::c_8;     // per-chunk row sums of x*gate*key
constexpr uint8_t kCbStats = tt::CBIndex::c_9;  // the three full-D sums, one per tile
constexpr uint8_t kCbCol = tt::CBIndex::c_10;   // scratch column for a row sum before its transpose
constexpr uint8_t kCbGateRow = tt::CBIndex::c_11;
constexpr uint8_t kCbGateCol = tt::CBIndex::c_12;
constexpr uint8_t kCbScaler = tt::CBIndex::c_13;

}  // namespace

ProgramDescriptor EngramGatePrefillProgramFactory::create_descriptor(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    const auto& streams = tensor_args.streams;
    const auto& kv = tensor_args.kv;
    const auto& gate = tensor_args.gate_weight;
    auto* device = streams.device();

    const auto& shape = streams.logical_shape();
    const uint32_t hc = shape[1];
    const uint32_t tokens = shape[2];
    const uint32_t d = shape[3];
    const uint32_t blocks = tokens / kTileHeight;
    const uint32_t d_tiles = d / kTileWidth;
    const uint32_t num_chunks = d_tiles / kChunkTiles;
    const uint32_t num_units = blocks * hc;

    const CoreCoord grid = device->compute_with_storage_grid_size();
    const CoreRange all_cores{CoreCoord{0, 0}, CoreCoord{grid.x - 1, grid.y - 1}};
    const uint32_t num_cores = grid.x * grid.y;

    TT_FATAL(
        d_tiles % kChunkTiles == 0,
        "engram_gate_prefill: D is {} tiles, not a multiple of the {} tile chunk",
        d_tiles,
        kChunkTiles);

    const tt::DataFormat bf16_df = datatype_to_dataformat_converter(streams.dtype());
    const tt::DataFormat fp32_df = tt::DataFormat::Float32;
    const uint32_t bf16_page = tt::tile_size(bf16_df);
    const uint32_t fp32_page = tt::tile_size(fp32_df);

    ProgramDescriptor desc;

    auto add_cb = [&](uint8_t index, uint32_t pages, tt::DataFormat df, uint32_t page_bytes) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = pages * page_bytes,
            .core_ranges = CoreRangeSet(all_cores),
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = index, .data_format = df, .page_size = page_bytes}}},
        });
    };

    add_cb(kCbX, kChunkTiles, bf16_df, bf16_page);
    add_cb(kCbK, kChunkTiles, bf16_df, bf16_page);
    add_cb(kCbG, kChunkTiles, fp32_df, fp32_page);
    add_cb(kCbV, kChunkTiles, bf16_df, bf16_page);
    add_cb(kCbOut, kChunkTiles, bf16_df, bf16_page);
    add_cb(kCbProd, kChunkTiles, fp32_df, fp32_page);
    add_cb(kCbP0, num_chunks, fp32_df, fp32_page);
    add_cb(kCbP1, num_chunks, fp32_df, fp32_page);
    add_cb(kCbP2, num_chunks, fp32_df, fp32_page);
    add_cb(kCbStats, 3, fp32_df, fp32_page);
    add_cb(kCbCol, 1, fp32_df, fp32_page);
    add_cb(kCbGateRow, 1, fp32_df, fp32_page);
    add_cb(kCbGateCol, 1, fp32_df, fp32_page);
    add_cb(kCbScaler, 1, fp32_df, fp32_page);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), args.compute_kernel_config);

    std::vector<uint32_t> reader_ct = {kCbX, kCbK, kCbG, kCbV, kCbScaler, kChunkTiles, d_tiles, hc, blocks};
    TensorAccessorArgs(*streams.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*kv.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*gate.buffer()).append_to(reader_ct);

    std::vector<uint32_t> writer_ct = {kCbOut, kChunkTiles, d_tiles, hc, blocks};
    TensorAccessorArgs(*output.buffer()).append_to(writer_ct);

    std::vector<uint32_t> compute_ct = {
        kCbX,
        kCbK,
        kCbG,
        kCbV,
        kCbOut,
        kCbProd,
        kCbP0,
        kCbP1,
        kCbP2,
        kCbStats,
        kCbCol,
        kCbGateRow,
        kCbGateCol,
        kCbScaler,
        kChunkTiles,
        num_chunks,
        std::bit_cast<uint32_t>(1.0f / static_cast<float>(d)),
        std::bit_cast<uint32_t>(args.eps),
        std::bit_cast<uint32_t>(args.clamp_value),
    };

    KernelDescriptor reader;
    reader.kernel_source = std::string(kKernelDir) + "/dataflow/reader_engram_gate_prefill.cpp";
    reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader.core_ranges = all_cores;
    reader.compile_time_args = std::move(reader_ct);
    reader.config = ReaderConfigDescriptor{};

    KernelDescriptor writer;
    writer.kernel_source = std::string(kKernelDir) + "/dataflow/writer_engram_gate_prefill.cpp";
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = all_cores;
    writer.compile_time_args = std::move(writer_ct);
    writer.config = WriterConfigDescriptor{};

    KernelDescriptor compute;
    compute.kernel_source = std::string(kKernelDir) + "/compute/engram_gate_prefill.cpp";
    compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = all_cores;
    compute.compile_time_args = std::move(compute_ct);
    compute.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .dst_full_sync_en = dst_full_sync_en,
        .math_approx_mode = math_approx_mode,
    };

    // Units are a contiguous run per core: the kernels walk unit_start .. unit_start + units. Every core runs
    // regardless of whether it owns a unit, because the reader, compute and writer of a core hand off through CBs
    // and the scaler tile is pushed unconditionally.
    const std::vector<CoreCoord> cores = corerange_to_cores(all_cores, num_cores, /*row_wise=*/true);
    const uint32_t units_base = num_units / num_cores;
    const uint32_t units_rem = num_units % num_cores;
    for (uint32_t c = 0; c < num_cores; ++c) {
        const uint32_t units = units_base + (c < units_rem ? 1u : 0u);
        const uint32_t unit_start = c * units_base + std::min(c, units_rem);
        reader.emplace_runtime_args(cores[c], {streams.buffer(), kv.buffer(), gate.buffer(), unit_start, units});
        writer.emplace_runtime_args(cores[c], {output.buffer(), unit_start, units});
        compute.emplace_runtime_args(cores[c], {unit_start, units});
    }

    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(writer));
    desc.kernels.push_back(std::move(compute));
    return desc;
}

}  // namespace ttnn::operations::experimental::deepseek::engram_gate_prefill
