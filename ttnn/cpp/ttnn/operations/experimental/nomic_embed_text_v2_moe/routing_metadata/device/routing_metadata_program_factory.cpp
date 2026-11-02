// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "routing_metadata_device_operation.hpp"

#include <algorithm>
#include <string>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::experimental::nomic_embed_text_v2_moe {

namespace {

const std::string kKernelDirectory =
    "ttnn/cpp/ttnn/operations/experimental/nomic_embed_text_v2_moe/routing_metadata/device/kernels/dataflow/";

constexpr uint32_t kWordBytes = sizeof(uint32_t);
// Faces 0 and 2 of a 32 x 32 tile of a 4-byte dtype: the tile's columns 0-15, all 32 rows.
constexpr uint32_t kFacePairBytes = 2 * tt::constants::FACE_HW * kWordBytes;
constexpr uint32_t kFloat32TileBytes = tt::constants::TILE_HW * kWordBytes;
// Tile rows of combine_weights a core holds in L1 at once.
constexpr uint32_t kMaxWeightSlots = 4;

enum CircularBufferIndex : uint8_t {
    kIndexFaces = tt::CBIndex::c_0,
    kGatherScratch = tt::CBIndex::c_1,
    kCombineScratch = tt::CBIndex::c_2,
    kCountsTable = tt::CBIndex::c_3,
    kValueFaces = tt::CBIndex::c_4,
    kWeightTiles = tt::CBIndex::c_5,
};

enum SemaphoreId : uint32_t {
    kCountsReady,
    kStartsReady,
    kRowsReady,
};

// The circular buffers are plain L1 scratch: each kernel addresses its own through the write
// pointer and never pushes or pops. A program places each one at the same address on every core,
// which is what lets a core write into another core's scratch by its own address.
tt::tt_metal::CBDescriptor scratch_buffer(
    const CoreRangeSet& cores, uint8_t index, tt::DataFormat format, uint32_t page_bytes, uint32_t num_pages) {
    return tt::tt_metal::CBDescriptor{
        .total_size = page_bytes * num_pages,
        .core_ranges = cores,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = index,
            .data_format = format,
            .page_size = page_bytes,
        }}},
    };
}

// A rectangle of cores from logical core (0, 0), inside the block whose NoC coordinates are
// contiguous, so a multicast over it reaches these cores and nothing else. Of the rectangles that
// give every core at least one tile row, it takes the fewest cores among those with the fewest
// tile rows on the busiest core: the per-core passes scale with the rows, the count exchange with
// the cores.
CoreCoord choose_core_grid(const tt::tt_metal::distributed::MeshDevice& device, uint32_t tile_rows) {
    const auto grid = device.compute_with_storage_grid_size();
    const auto origin = device.worker_core_from_logical_core(CoreCoord(0, 0));
    uint32_t max_columns = 1;
    while (max_columns < grid.x &&
           device.worker_core_from_logical_core(CoreCoord(max_columns, 0)).x == origin.x + max_columns) {
        ++max_columns;
    }
    uint32_t max_rows = 1;
    while (max_rows < grid.y && device.worker_core_from_logical_core(CoreCoord(0, max_rows)).y == origin.y + max_rows) {
        ++max_rows;
    }

    CoreCoord best(1, 1);
    uint32_t best_rows_per_core = tile_rows;
    for (uint32_t columns = 1; columns <= std::min(max_columns, tile_rows); ++columns) {
        for (uint32_t rows = 1; rows <= std::min(max_rows, tile_rows / columns); ++rows) {
            const uint32_t num_cores = columns * rows;
            const uint32_t rows_per_core = tt::div_up(tile_rows, num_cores);
            if (rows_per_core < best_rows_per_core ||
                (rows_per_core == best_rows_per_core && num_cores < best.x * best.y)) {
                best = CoreCoord(columns, rows);
                best_rows_per_core = rows_per_core;
            }
        }
    }
    return best;
}

}  // namespace

tt::tt_metal::ProgramDescriptor RoutingMetadataDeviceOperation::create_descriptor(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args, tensor_return_value_t& outputs) {
    using tt::tt_metal::DataMovementConfigDescriptor;
    using tt::tt_metal::DataMovementProcessor;
    using tt::tt_metal::KernelDescriptor;
    using tt::tt_metal::NOC;
    using tt::tt_metal::TensorAccessorArgs;

    const auto layout = compute_layout(attributes, tensor_args);
    auto& [gather_index, combine_index, combine_weights, expert_counts] = outputs;
    auto* indices_buffer = tensor_args.indices.buffer();
    auto* values_buffer = tensor_args.values.buffer();
    const auto& device = *tensor_args.indices.device();

    // Core c of the rectangle (row-major) takes tile rows [first, first + count) of the window, the
    // first tile_rows % num_cores cores one more than the rest.
    const uint32_t tile_rows = layout.padded_tokens / tt::constants::TILE_HEIGHT;
    const CoreCoord core_grid = choose_core_grid(device, tile_rows);
    const uint32_t num_cores = core_grid.x * core_grid.y;
    const uint32_t base_rows = tile_rows / num_cores;
    const uint32_t extra_rows = tile_rows % num_cores;
    const uint32_t max_rows_per_core = base_rows + (extra_rows > 0 ? 1 : 0);
    const CoreCoord last_core(core_grid.x - 1, core_grid.y - 1);
    const CoreRangeSet cores{CoreRange(CoreCoord(0, 0), last_core)};

    // Core 0 gathers the counts, two to a word, into a table of num_cores rows and multicasts it back;
    // after it come the totals core 0 writes out and a scratch word. Rows are L1-aligned for the NoC.
    const uint32_t l1_alignment = tt::tt_metal::hal::get_l1_alignment();
    const uint32_t packed_row_bytes = tt::div_up(attributes.num_experts, 2) * kWordBytes;
    const uint32_t table_row_words = tt::round_up(packed_row_bytes, l1_alignment) / kWordBytes;
    const uint32_t totals_row_words = tt::round_up(attributes.num_experts * kWordBytes, l1_alignment) / kWordBytes;
    const uint32_t table_bytes = (num_cores * table_row_words + totals_row_words) * kWordBytes + l1_alignment;
    const uint32_t tokens_per_core = max_rows_per_core * tt::constants::TILE_HEIGHT;
    const uint32_t weight_slots = std::min(max_rows_per_core, kMaxWeightSlots);

    tt::tt_metal::ProgramDescriptor descriptor;
    descriptor.cbs.push_back(
        scratch_buffer(cores, kIndexFaces, tt::DataFormat::UInt32, kFacePairBytes, max_rows_per_core));
    descriptor.cbs.push_back(
        scratch_buffer(cores, kGatherScratch, tt::DataFormat::UInt32, layout.gathered_rows * kWordBytes, 1));
    descriptor.cbs.push_back(
        scratch_buffer(cores, kCombineScratch, tt::DataFormat::UInt32, tokens_per_core * kWordBytes, layout.top_k));
    descriptor.cbs.push_back(scratch_buffer(cores, kCountsTable, tt::DataFormat::UInt32, table_bytes, 1));
    descriptor.cbs.push_back(
        scratch_buffer(cores, kValueFaces, tt::DataFormat::Float32, kFacePairBytes, max_rows_per_core));
    descriptor.cbs.push_back(
        scratch_buffer(cores, kWeightTiles, tt::DataFormat::Float32, kFloat32TileBytes, weight_slots * layout.top_k));

    for (uint32_t id : {kCountsReady, kStartsReady, kRowsReady}) {
        descriptor.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
            .id = id,
            .core_type = tt::CoreType::WORKER,
            .core_ranges = cores,
            .initial_value = 0,
        });
    }

    // RISCV_0 on NOC0: indices -> gather_index, combine_index, expert_counts. The three outputs are
    // row-major pages whose size grows with num_tokens. Their page size goes in as a runtime argument
    // rather than through the buffer's accessor arguments, which would bake it into the kernel binary
    // and force a new compile for every 32 tokens.
    KernelDescriptor indices_kernel;
    indices_kernel.kernel_source = kKernelDirectory + "routing_metadata_indices.cpp";
    indices_kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    indices_kernel.core_ranges = cores;
    indices_kernel.named_compile_time_args = {
        {"index_faces_cb", kIndexFaces},
        {"gather_cb", kGatherScratch},
        {"combine_cb", kCombineScratch},
        {"table_cb", kCountsTable},
        {"counts_ready_semaphore", kCountsReady},
        {"starts_ready_semaphore", kStartsReady},
        {"rows_ready_semaphore", kRowsReady},
        {"num_experts", attributes.num_experts},
        {"top_k", layout.top_k},
        {"table_row_words", table_row_words},
        {"totals_row_words", totals_row_words},
    };
    TensorAccessorArgs(*indices_buffer).append_to(indices_kernel.compile_time_args);
    for (int output = 0; output < 3; ++output) {
        TensorAccessorArgs::create_dram_interleaved().append_to(indices_kernel.compile_time_args);
    }
    indices_kernel.config = DataMovementConfigDescriptor{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::RISCV_0_default,
    };

    // RISCV_1 on NOC1, otherwise idle: values -> combine_weights. Both are 4 KiB tile pages.
    KernelDescriptor weights_kernel;
    weights_kernel.kernel_source = kKernelDirectory + "routing_metadata_weights.cpp";
    weights_kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    weights_kernel.core_ranges = cores;
    weights_kernel.named_compile_time_args = {
        {"value_faces_cb", kValueFaces},
        {"weight_tiles_cb", kWeightTiles},
        {"top_k", layout.top_k},
    };
    TensorAccessorArgs(*values_buffer).append_to(weights_kernel.compile_time_args);
    TensorAccessorArgs(*combine_weights.buffer()).append_to(weights_kernel.compile_time_args);
    weights_kernel.config = DataMovementConfigDescriptor{
        .processor = DataMovementProcessor::RISCV_1,
        .noc = NOC::RISCV_1_default,
    };

    // NOC0 multicasts from the top-left corner to the bottom-right one.
    const auto first_core_noc = device.worker_core_from_logical_core(CoreCoord(0, 0));
    const auto last_core_noc = device.worker_core_from_logical_core(last_core);
    for (uint32_t core_index = 0; core_index < num_cores; ++core_index) {
        const CoreCoord core(core_index % core_grid.x, core_index / core_grid.x);
        const uint32_t first_tile_row = core_index * base_rows + std::min(core_index, extra_rows);
        const uint32_t num_tile_rows = base_rows + (core_index < extra_rows ? 1 : 0);

        indices_kernel.emplace_runtime_args(
            core,
            {indices_buffer,
             gather_index.buffer(),
             combine_index.buffer(),
             expert_counts.buffer(),
             attributes.token_start,
             attributes.num_tokens,
             layout.gathered_rows,
             static_cast<uint32_t>(gather_index.buffer()->aligned_page_size()),
             static_cast<uint32_t>(combine_index.buffer()->aligned_page_size()),
             static_cast<uint32_t>(expert_counts.buffer()->aligned_page_size()),
             core_index,
             num_cores,
             first_tile_row,
             num_tile_rows,
             tokens_per_core,
             static_cast<uint32_t>(first_core_noc.x),
             static_cast<uint32_t>(first_core_noc.y),
             static_cast<uint32_t>(last_core_noc.x),
             static_cast<uint32_t>(last_core_noc.y)});
        weights_kernel.emplace_runtime_args(
            core,
            {values_buffer,
             combine_weights.buffer(),
             attributes.token_start,
             attributes.num_tokens,
             tile_rows,
             first_tile_row,
             num_tile_rows,
             weight_slots,
             core_index});
    }

    descriptor.kernels.push_back(std::move(indices_kernel));
    descriptor.kernels.push_back(std::move(weights_kernel));
    return descriptor;
}

}  // namespace ttnn::experimental::nomic_embed_text_v2_moe
