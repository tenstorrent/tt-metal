// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_expert_rows.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <set>
#include <string_view>
#include <tuple>
#include <variant>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-umd/types/arch.hpp>

#include "kernels/expert_rows/expert_rows_common.h"
#include "kernels/moe_ring_common.h"
#include "ttnn/operation.hpp"
#include "moe_compute_device_operation.hpp"
#include "moe_compute_program_factory.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include "ttnn/operations/experimental/ccl/moe_compute/moe_core_placement.hpp"

namespace ttnn::experimental::prim {

namespace {

using expert_rows::GroupRun;
using ttnn::experimental::ccl::add_semaphore_descriptor;

constexpr uint32_t kWordBytes = sizeof(uint32_t);
constexpr uint32_t kBf16Bytes = sizeof(uint16_t);
constexpr uint32_t kTileBytesBf16 = tt::constants::TILE_HW * kBf16Bytes;
constexpr uint32_t kTileRowBytesBf16 = tt::constants::TILE_WIDTH * kBf16Bytes;
constexpr uint32_t kGroupTiles = expert_rows::group_tiles;
constexpr uint32_t kColumnsPerGroup = expert_rows::group_columns;  // each column gives one activated tile
constexpr uint32_t kMaxTopK = 32;  // rows_reader keeps one u16 per k of a token on its stack (its static_assert)
// the routing table holds rows and tokens in u16
constexpr uint32_t kMaxRoutedRows = std::numeric_limits<uint16_t>::max();
constexpr std::array<uint32_t, 3> kBlockSlots = {8, 6, 4};  // weight blocks per reader core, tried in order
constexpr uint32_t kBlockBytes = 16 * 1024;                 // weight bytes per read block
constexpr uint32_t kTwoReadersCoresPerBank = 8;  // worker cores per DRAM bank from which a bank gets 2 readers
// a reader keeps one weight block in flight when its bank feeds this many cores or more, else two
constexpr uint32_t kOneBlockInFlightBankCores = 3;
// cost model, cycles at the device clock: the ring's pass of one job per ring position; the combine feed's handshake
// per active expert, and its cost per row (SingleDevice; FullCcl sends over the fabric); writing one FP32 partial tile
// to L1 and reading it back into DEST, and one NoC command of the writer RISC (address and issue), both fitted to the
// fastest row tiles per job; one tile matmul per fidelity phase
constexpr double kRingHopCycles = 350.0;
constexpr double kFeedExpertCycles = 1500.0;
constexpr double kFeedRowCyclesLocal = 200.0;
constexpr double kFeedRowCyclesCcl = 400.0;
constexpr double kReloadTileCycles = 64.0;
constexpr double kWriterOpCycles = 60.0;
constexpr double kMatmulCyclesPerFidelityPhase = 16.0;
constexpr double kCostMargin = 0.10;  // cost model: the margin the expert rows path must predict over the ring
// peak DRAM bandwidth in GiB / s: OpPerformanceModel's figures (ttnn/core/operation.cpp)
constexpr double kDramPeakGiBsBlackhole = 512.0;
constexpr double kDramPeakGiBsWormhole = 258.0;
constexpr uint64_t kRingSmallCbBytes = 4096;  // the ring tilize cores' count / flag CBs
constexpr uint64_t kRingTokensPerChunk = 32;  // the ring tilize cores' input chunk
constexpr uint64_t kRingFitMargin = 32768;    // L1 the ring-fit estimate leaves for what it does not model
// The reader adds the deal times of a group's jobs (at most its ctl entries) in u32: one deal unit is the smallest
// power of two of cycles that keeps the sum below this.
constexpr double kDealLoadLimit = 4.0e9;
// Placement of a bank's further cores: the grid distance to its optimal reader core, plus a cost for another row, a
// smaller x, a smaller y, and a smaller x than an optimal core off grid column 0.
constexpr double kPlaceOtherRowCost = 0.1;
constexpr double kPlaceSmallerXCost = 0.3;
constexpr double kPlaceSmallerYCost = 0.05;
constexpr double kPlaceSmallerXOffColumnZeroCost = 6.0;
constexpr const char* kKernelDir = "ttnn/cpp/ttnn/operations/experimental/ccl/moe_compute/device/kernels/expert_rows/";
// Blackhole and Wormhole run the same kernels: their plans differ only in numbers (readers per bank, packets per read
// block) that plan_expert_rows derives from the device.
constexpr const char* kReaderKernel = "rows_reader.cpp";
constexpr const char* kWriterKernel = "rows_writer.cpp";
constexpr const char* kComputeKernel = "rows_compute.cpp";

// (x slots, intermediate slots) tried in order; without credits at least three intermediate slots (rows_writer.cpp)
constexpr std::array<std::pair<uint32_t, uint32_t>, 3> kBuffering = {{{2, 4}, {2, 3}, {1, 3}}};
// Several row tiles per job: two x chunk slots (a chunk moves while the previous one computes) and one intermediate
// slot, both handed back by credits (rows_writer.cpp). The x chunks keep the group cores within a few chunks of each
// other, so the intermediate slot is free again by the time it is needed; its L1 goes to larger x chunks (fewer partial
// reloads).
constexpr uint32_t kChunkXSlots = 2;
constexpr uint32_t kChunkIntermediateSlots = 1;
constexpr uint32_t kActivatedSlots = 2;  // cb_activated: the activated tiles of two jobs
constexpr uint32_t kRowsSlots = 2;       // cb_rows: two output groups of one row tile

// moe_compute's outputs (MoEComputeDeviceOperation::compute_output_specs)
enum MoEComputeOutput : uint32_t {
    kExpertTokenCounts = 0,
    kExpertActivation = 1,
    kETPages = 2,
    kTilizeOutput = 3,  // the double buffer that place.cpp or feed.cpp fills
    kMatmulOutput = 4,  // aliases kTilizeOutput
    kCombineOutput = 5,
};
constexpr uint32_t kOutputsWithCombine = 6;

// The expert rows program's kernels, in order
constexpr uint32_t kRowsReaderKernelIdx = 0;
constexpr uint32_t kRowsWriterKernelIdx = 1;
constexpr uint32_t kRowsComputeKernelIdx = 2;
// The placement program with the combine starts with selective_reduce_combine's kernels: its reader, then its writer
// (then, with FullCcl, its mux kernels).
constexpr uint32_t kPlaceCombineWriterKernelIdx = 1;

bool expert_rows_arch(tt::ARCH arch) { return arch == tt::ARCH::BLACKHOLE || arch == tt::ARCH::WORMHOLE_B0; }

uint32_t align_to_part(uint32_t bytes) { return tt::align(bytes, expert_rows::part_alignment); }

// Consecutive parts of a layout, each at a multiple of part_alignment.
struct LayoutCursor {
    uint32_t end = 0;
    uint32_t take(uint32_t bytes) {
        const uint32_t offset = end;
        end = align_to_part(end + bytes);
        return offset;
    }
};

// moe_compute's prepared weights (prepare_w0_w1_tensor_for_moe_compute / prepare_w2_tensor_for_moe_compute in
// moe_compute_utils.py, moe_ring_common.h) as the expert rows reader addresses them, in tiles. W0/W1, per (layer,
// expert): the ring positions' slices back to back (ShardLayout::w0_w1_first_tile), each its column groups (4 tiles per
// K row) at a stride of w0_w1_group_tiles, then, for an odd count in the compact layout, one half block-column (one
// column, 2 tiles per K row); that stream is cut into one piece of w0_w1_piece_tiles per DRAM bank, so a slice can go
// on in the next bank (with equal even counts and 14-tile transactions, piece r is position r's slice). W2, per (layer,
// expert): w2_slice_tiles in position r's own bank, its output groups (4 tiles per K row) at a stride of
// w2_group_tiles; the last of the w2_groups can be 2 output tiles wide. The K padding at the end of each group is not
// read.
struct PreparedLayout {
    uint32_t w0_w1_piece_tiles = 0, w0_w1_group_tiles = 0;
    uint32_t w2_slice_tiles = 0, w2_group_tiles = 0;
    GroupRun w2_groups{0, false};
};

PreparedLayout prepared_layout(uint32_t hidden_tiles, uint32_t intermediate_tiles, bool has_bias, uint32_t ring) {
    const uint32_t tiles_per_txn = moe_ring::tiles_per_txn_for_shape(hidden_tiles, intermediate_tiles, has_bias, ring);
    const uint32_t block = moe_ring::W0_W1_TXNS_PER_BLOCK * tiles_per_txn;
    const uint32_t bias_rows = has_bias ? 1 : 0;
    const uint32_t w0_w1_k_rows = hidden_tiles + bias_rows, w2_k_rows = intermediate_tiles + bias_rows;
    PreparedLayout layout;
    layout.w0_w1_piece_tiles =
        moe_ring::w0_w1_bank_blocks_per_expert(w0_w1_k_rows, intermediate_tiles, ring, ring, tiles_per_txn) * block;
    layout.w0_w1_group_tiles = tt::div_up(w0_w1_k_rows, moe_ring::block_tiles_h(tiles_per_txn)) * block;
    layout.w2_slice_tiles = moe_ring::w2_core_blocks_per_expert(hidden_tiles, w2_k_rows, ring, tiles_per_txn) * block;
    layout.w2_group_tiles = tt::div_up(w2_k_rows, moe_ring::block_tiles_h(tiles_per_txn)) * block;
    layout.w2_groups = {
        moe_ring::w2_num_a2a_iters(hidden_tiles, ring),
        moe_ring::w2_last_a2a_iter_half(hidden_tiles, ring, tiles_per_txn)};
    return layout;
}

// moe_compute's prepared layout of one ring position
struct ShardLayout {
    uint32_t columns = 0, first_column = 0;            // real intermediate columns
    uint32_t output_tiles = 0, first_output_tile = 0;  // W2 output tiles
    uint32_t w0_w1_first_tile = 0;                     // first tile of its W0/W1 slice in the (layer, expert) stream
    bool half_column = false;                          // its last column is a half block-column
    GroupRun column_groups() const { return {tt::div_up(columns, kColumnsPerGroup), half_column}; }
    uint32_t output_groups() const { return tt::div_up(output_tiles, kGroupTiles); }
};

std::vector<ShardLayout> shard_layouts(
    uint32_t hidden_tiles, uint32_t intermediate_tiles, bool has_bias, uint32_t ring) {
    const uint32_t tiles_per_txn = moe_ring::tiles_per_txn_for_shape(hidden_tiles, intermediate_tiles, has_bias, ring);
    const uint32_t w0_w1_k_rows = hidden_tiles + (has_bias ? 1 : 0);
    const uint32_t column_blocks = tt::div_up(w0_w1_k_rows, moe_ring::block_tiles_h(tiles_per_txn));
    const uint32_t half_column_blocks = tt::div_up(w0_w1_k_rows, moe_ring::half_block_tiles_h(tiles_per_txn));
    const bool compact = moe_ring::w0_w1_compact_for_shape(intermediate_tiles, ring);
    std::vector<ShardLayout> layouts(ring);
    uint32_t column = 0, output_tile = 0;
    for (uint32_t position = 0; position < ring; ++position) {
        layouts[position].columns = moe_ring::shard_tiles(intermediate_tiles, position, ring);
        layouts[position].first_column = column;
        column += layouts[position].columns;
        layouts[position].output_tiles = moe_ring::w2_shard_tiles(hidden_tiles, position, intermediate_tiles, ring);
        layouts[position].first_output_tile = output_tile;
        output_tile += layouts[position].output_tiles;
        layouts[position].w0_w1_first_tile =
            moe_ring::w0_w1_core_block_offset(intermediate_tiles, position, ring, column_blocks, half_column_blocks) *
            moe_ring::W0_W1_TXNS_PER_BLOCK * tiles_per_txn;
        layouts[position].half_column = compact && layouts[position].columns % 2 == 1;
    }
    return layouts;
}

// The routing table block (rows_reader.cpp writes it, place.cpp and feed.cpp read it), offsets from its start: a head
// (u32 routed rows), u16 per-expert counts and first rows, the routed rows' tokens, then the same entries in token
// order: expert, token, score and k (a token that names one expert at several k has one entry per k).
struct TableBlock {
    uint32_t counts = 0, offsets = 0, rows = 0;
    uint32_t entry_experts = 0, entry_tokens = 0, entry_scores = 0, entry_k = 0;
    uint32_t bytes = 0;
};

TableBlock table_block(uint32_t tokens, uint32_t top_k, uint32_t local_experts) {
    const uint32_t entries = tokens * top_k;
    LayoutCursor cursor;
    TableBlock block;
    cursor.take(expert_rows::part_alignment);  // the head
    block.counts = cursor.take(local_experts * kBf16Bytes);
    block.offsets = cursor.take(local_experts * kBf16Bytes);
    block.rows = cursor.take(entries * kBf16Bytes);
    block.entry_experts = cursor.take(entries * kBf16Bytes);
    block.entry_tokens = cursor.take(entries * kBf16Bytes);
    block.entry_scores = cursor.take(entries * kBf16Bytes);
    block.entry_k = cursor.take(entries * kBf16Bytes);
    block.bytes = cursor.end;
    return block;
}

// Byte offsets of the expert rows reader's routing scratch (cb_routing). [table, jobs) is the routing table block the
// root core writes to DRAM for the placement program.
struct RoutingLayout {
    uint32_t own_mapping = 0, source_mappings = 0, token_ids = 0, token_scores = 0, ctl = 0;
    uint32_t active_experts = 0, expert_rows = 0, expert_first_rows = 0;
    uint32_t table = 0, table_counts = 0, table_offsets = 0, row_tokens = 0;
    uint32_t entry_experts = 0, entry_tokens = 0, entry_scores = 0, entry_k = 0;
    uint32_t table_bytes = 0;
    uint32_t jobs = 0, entry_areas = 0, size = 0;
};

RoutingLayout routing_layout(
    const MoEExpertRowsShape& shape, uint32_t program_cores, uint32_t sources_per_core, uint32_t expert_groups) {
    const uint32_t tokens_per_core = tt::div_up(shape.tokens, program_cores);
    // entries per token at most: one per k (a repeated id is listed per k)
    const uint32_t area_bytes = expert_rows::entry_area_bytes(tokens_per_core * shape.top_k);
    const TableBlock block = table_block(shape.tokens, shape.top_k, shape.local_experts);
    LayoutCursor cursor;
    RoutingLayout layout;
    layout.own_mapping = cursor.take(shape.global_experts * kBf16Bytes);
    layout.source_mappings = cursor.take(sources_per_core * shape.global_experts * kBf16Bytes);
    layout.token_ids = cursor.take(tokens_per_core * shape.index_page_bytes);
    layout.token_scores = cursor.take(tokens_per_core * shape.index_page_bytes);
    layout.ctl = cursor.take(expert_rows::ctl_bytes(expert_groups));
    layout.active_experts = cursor.take(shape.local_experts * kBf16Bytes);
    layout.expert_rows = cursor.take(shape.local_experts * kBf16Bytes);
    layout.expert_first_rows = cursor.take(shape.local_experts * kBf16Bytes);
    layout.table = cursor.take(block.bytes);
    layout.table_counts = layout.table + block.counts;
    layout.table_offsets = layout.table + block.offsets;
    layout.row_tokens = layout.table + block.rows;
    layout.entry_experts = layout.table + block.entry_experts;
    layout.entry_tokens = layout.table + block.entry_tokens;
    layout.entry_scores = layout.table + block.entry_scores;
    layout.entry_k = layout.table + block.entry_k;
    layout.table_bytes = block.bytes;
    layout.jobs = cursor.take(shape.tokens * shape.top_k * sizeof(expert_rows::JobRecord));
    layout.entry_areas = cursor.take(program_cores * area_bytes);
    layout.size = cursor.end;
    return layout;
}

// The writer's scratch (cb_scratch): the x staging (tile_rows rows per row tile, each the core's piece of an x chunk of
// at most piece_max tiles), that piece as tiles, and the staging of one row tile of output rows (one output group).
uint64_t writer_scratch_bytes(uint32_t row_tiles, uint32_t piece_max) {
    return uint64_t{expert_rows::tile_rows} * row_tiles * piece_max * kTileRowBytesBf16 +
           uint64_t{piece_max} * row_tiles * kTileBytesBf16 + expert_rows::tile_rows * kGroupTiles * kTileRowBytesBf16;
}

// Weight blocks of cb_w when jobs hold units (several row tiles): whole chunk units (none wraps around the CB), at
// least two, at least block_slots.
uint32_t weight_cb_blocks(uint32_t chunk_blocks, uint32_t block_slots) {
    return chunk_blocks * std::max(2u, tt::div_up(block_slots, chunk_blocks));
}

// cb_ctl: u32 job count, then with several row tiles per job the row tiles of each of the group's jobs
uint32_t ctl_cb_bytes(uint32_t words) { return tt::align(kWordBytes * words, expert_rows::part_alignment); }

// Format of the partials between x or W2 chunks: what DEST holds.
tt::DataFormat partials_format(bool fp32_dest_acc_en) {
    return fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
}

// One buffering choice of a plan. Without credits the intermediate exchange orders slot reuse: one x chunk per job and
// at least three intermediate slots (rows_writer.cpp).
struct Buffering {
    uint32_t row_tiles = 1;
    uint32_t x_slots = 0, intermediate_slots = 0;
    uint32_t block_slots = 0;  // weight blocks of cb_w
    // hidden tiles per x chunk (all of them with one row tile per job), the most one core owns of a chunk, weight
    // blocks per chunk unit (the weight tiles of one chunk of one column group, or of one W2 chunk of one output
    // group), W2 rows per chunk
    uint32_t chunk_tiles = 0, chunk_piece_max = 0, chunk_blocks = 0, w2_chunk_rows = 0;
    uint32_t ctl_bytes = 0;
    bool slot_credits = false;  // slots handed back by credits (several chunks per job)
};

// Circular buffer bytes per core of one plan.
struct CbSizes {
    uint32_t x_chunks = 0, weights = 0, activated = 0, intermediate = 0, rows = 0, ctl = 0, routing = 0, scratch = 0;
    uint32_t ones = 0;      // with a bias row
    uint32_t partials = 0;  // with several x or W2 chunks per job
    uint32_t total() const {
        return x_chunks + weights + activated + intermediate + rows + ctl + routing + scratch + partials + ones;
    }
};

// The CBs of a buffering choice, when they fit cb_budget bytes per core.
std::optional<CbSizes> cb_sizes(
    const MoEExpertRowsShape& shape,
    const Buffering& buffering,
    uint32_t activated_tiles,
    uint32_t block_tiles,
    uint32_t routing_bytes,
    uint32_t partial_tile_bytes,
    uint32_t cb_budget) {
    const uint32_t hidden_tiles = shape.hidden_tiles(), intermediate_tiles = shape.intermediate_tiles();
    const bool spill = buffering.chunk_tiles < hidden_tiles || buffering.w2_chunk_rows < intermediate_tiles;
    const uint64_t x_chunks =
        uint64_t{buffering.x_slots} * buffering.chunk_tiles * buffering.row_tiles * kTileBytesBf16;
    const uint64_t weights = uint64_t{buffering.block_slots} * block_tiles * tt::tile_size(tt::DataFormat::Bfp4_b);
    const uint64_t activated = uint64_t{kActivatedSlots} * buffering.row_tiles * activated_tiles * kTileBytesBf16;
    const uint64_t intermediate =
        uint64_t{buffering.intermediate_slots} * buffering.row_tiles * intermediate_tiles * kTileBytesBf16;
    const uint64_t rows = kRowsSlots * kGroupTiles * kTileBytesBf16;
    const uint64_t scratch = writer_scratch_bytes(buffering.row_tiles, buffering.chunk_piece_max);
    // a 4-tile slot per (column group, row tile), also for a half group (rows_compute.cpp)
    const uint64_t partials = spill ? uint64_t{buffering.row_tiles} * std::max(activated_tiles, kColumnsPerGroup) *
                                          kGroupTiles / kColumnsPerGroup * partial_tile_bytes
                                    : 0;
    const uint64_t ones = shape.has_bias ? kTileBytesBf16 : 0;
    const uint64_t total = x_chunks + weights + activated + intermediate + rows + buffering.ctl_bytes + routing_bytes +
                           scratch + partials + ones;
    if (total > cb_budget) {
        return std::nullopt;
    }
    return CbSizes{
        .x_chunks = static_cast<uint32_t>(x_chunks),
        .weights = static_cast<uint32_t>(weights),
        .activated = static_cast<uint32_t>(activated),
        .intermediate = static_cast<uint32_t>(intermediate),
        .rows = static_cast<uint32_t>(rows),
        .ctl = buffering.ctl_bytes,
        .routing = routing_bytes,
        .scratch = static_cast<uint32_t>(scratch),
        .ones = static_cast<uint32_t>(ones),
        .partials = static_cast<uint32_t>(partials)};
}

// One core of the expert rows program: its ring position's W0/W1 column groups and W2 output groups (rows_reader.cpp),
// and the intermediate columns and output tiles they hold (rows_writer.cpp).
struct ExpertRowsCore {
    CoreCoord core;
    uint32_t ring_position = 0;  // shard index of the prepared weights = position in moe_compute's ring
    uint32_t bank = 0;           // DRAM bank holding that shard
    uint32_t group = 0;          // expert group (jobs dealt by rows_reader.cpp)
    uint32_t first_column_group = 0, column_groups = 0;
    bool half_column = false;  // the last column group is the shard's half block-column (one column, 2 tiles per K row)
    uint32_t first_column = 0, activated_columns = 0;  // real intermediate columns
    uint32_t first_output_group = 0, output_groups = 0;
    bool half_output = false;                          // the last output group is the shard's last, two tiles wide
    uint32_t first_output_tile = 0, output_tiles = 0;  // real output tiles
    GroupRun column_run() const { return {column_groups, half_column}; }
    GroupRun output_run() const { return {output_groups, half_output}; }
};

struct ExpertRowsPlan {
    uint32_t row_tiles = 1;
    uint32_t readers_per_bank = 0;
    uint32_t expert_groups = 0;
    uint32_t group_cores = 0;
    uint32_t x_slots = 0;
    uint32_t intermediate_slots = 0;
    uint32_t activated_tiles = 0;  // cb_activated pages per row tile: one per intermediate column of the busiest core
    uint32_t block_tiles = 0;      // weight tiles per read block
    uint32_t block_packets = 0;    // NoC packets per read block
    uint32_t block_slots = 0;
    uint32_t blocks_in_flight = 0;
    uint32_t chunk_tiles = 0, chunk_piece_max = 0, chunk_blocks = 0, w2_chunk_rows = 0;  // Buffering
    bool slot_credits = false;
    uint32_t ctl_bytes = 0;
    CbSizes cbs;
    RoutingLayout routing;
    std::vector<ExpertRowsCore> cores;  // group-major
};

// The plan, or why there is none.
using PlanOrRefusal = std::variant<ExpertRowsPlan, std::string>;

// The cores of one expert group: each ring position's column groups split over up to readers_per_bank cores (earlier
// cores take the remainder), and its output groups over the same cores.
struct CoreSplit {
    uint32_t readers_per_bank = 0, block_tiles = 0, block_packets = 0;
    uint32_t activated_tiles = 0;
    std::vector<ExpertRowsCore> members;
};

// Each bank's cores: its NOC-0-optimal DRAM reader core first, then the nearest free cores (round-robin over banks).
std::vector<std::vector<CoreCoord>> place_cores(
    const std::vector<CoreCoord>& optimal, const std::vector<uint32_t>& counts, CoreCoord grid) {
    std::set<std::pair<uint32_t, uint32_t>> taken;
    std::vector<std::vector<CoreCoord>> out(optimal.size());
    for (uint32_t bank = 0; bank < optimal.size(); ++bank) {
        if (counts[bank]) {
            out[bank].push_back(optimal[bank]);
            taken.insert({optimal[bank].x, optimal[bank].y});
        }
    }
    const uint32_t most = *std::max_element(counts.begin(), counts.end());
    for (uint32_t step = 1; step < most; ++step) {
        for (uint32_t bank = 0; bank < optimal.size(); ++bank) {
            if (out[bank].size() >= counts[bank]) {
                continue;
            }
            const int optimal_x = optimal[bank].x, optimal_y = optimal[bank].y;
            double best = std::numeric_limits<double>::max();
            CoreCoord pick{0, 0};
            for (uint32_t x = 0; x < grid.x; ++x) {
                for (uint32_t y = 0; y < grid.y; ++y) {
                    if (taken.contains({x, y})) {
                        continue;
                    }
                    const int delta_x = static_cast<int>(x) - optimal_x, delta_y = static_cast<int>(y) - optimal_y;
                    double cost = std::abs(delta_x) + std::abs(delta_y) + (delta_y != 0 ? kPlaceOtherRowCost : 0.0) +
                                  (delta_x < 0 ? kPlaceSmallerXCost : 0.0) + (delta_y < 0 ? kPlaceSmallerYCost : 0.0) +
                                  (optimal_x > 0 && delta_x < 0 ? kPlaceSmallerXOffColumnZeroCost : 0.0);
                    if (cost < best) {
                        best = cost;
                        pick = CoreCoord(x, y);
                    }
                }
            }
            TT_FATAL(
                best < std::numeric_limits<double>::max(),
                "moe_compute expert rows: the {}x{} worker grid is full; bank {} needs {} cores, has {}",
                grid.x,
                grid.y,
                bank,
                counts[bank],
                out[bank].size());
            out[bank].push_back(pick);
            taken.insert({pick.x, pick.y});
        }
    }
    return out;
}

// W2 output groups per member: contiguous runs in member order on top of its gate/up tiles, the busiest member least
// loaded, then the least sum of squared loads; earlier members take more on a tie (with equal runs, the
// least-loaded-member greedy deal). best[member_end][group_end]: (busiest, sum of squares) of members [0, member_end)
// taking groups [0, group_end), the last of them groups [run_begin, group_end); first[member_end][group_end]: that
// run_begin.
std::vector<uint32_t> split_output_groups(
    const std::vector<uint64_t>& gate_up_tiles, const std::vector<uint64_t>& output_prefix) {
    const uint32_t members = gate_up_tiles.size(), groups = output_prefix.size() - 1;
    using Cost = std::pair<uint64_t, uint64_t>;
    constexpr Cost kNone = {std::numeric_limits<uint64_t>::max(), 0};
    std::vector<std::vector<Cost>> best(members + 1, std::vector<Cost>(groups + 1, kNone));
    std::vector<std::vector<uint32_t>> first(members + 1, std::vector<uint32_t>(groups + 1, 0));
    best[0][0] = {0, 0};
    for (uint32_t member_end = 1; member_end <= members; ++member_end) {
        for (uint32_t group_end = 0; group_end <= groups; ++group_end) {
            for (uint32_t run_begin = 0; run_begin <= group_end; ++run_begin) {
                if (best[member_end - 1][run_begin] == kNone) {
                    continue;
                }
                const uint64_t load =
                    gate_up_tiles[member_end - 1] + output_prefix[group_end] - output_prefix[run_begin];
                const Cost cost = {
                    std::max(best[member_end - 1][run_begin].first, load),
                    best[member_end - 1][run_begin].second + load * load};
                if (cost <= best[member_end][group_end]) {
                    best[member_end][group_end] = cost;
                    first[member_end][group_end] = run_begin;
                }
            }
        }
    }
    std::vector<uint32_t> outputs(members);
    for (uint32_t member_end = members, group_end = groups; member_end > 0; --member_end) {
        outputs[member_end - 1] = group_end - first[member_end][group_end];
        group_end = first[member_end][group_end];
    }
    return outputs;
}

// The plan search of one call: the core splits, their placement and their buffering.
struct Planner {
    const MoEExpertRowsShape& shape;
    const std::vector<uint32_t>& shard_banks;  // DRAM bank of the prepared weights' shard r (= ring position r)
    const std::vector<CoreCoord>& optimal;     // NOC-0-optimal DRAM reader core of each bank
    CoreCoord grid;
    uint32_t cb_budget = 0;
    uint32_t partial_tile_bytes = 0;
    const std::vector<ShardLayout>& layouts;
    const PreparedLayout& prepared;

    uint32_t grid_cores() const { return grid.x * grid.y; }

    // sources one core's tokens can come from: its tokens over the tokens per source, plus one where they straddle two
    uint32_t sources_per_core(uint32_t program_cores) const {
        return tt::div_up(tt::div_up(shape.tokens, program_cores), shape.tokens / shape.num_sources) + 1;
    }

    CoreSplit split_cores(uint32_t readers_per_bank, uint32_t block_tiles, uint32_t block_packets) const {
        CoreSplit split{
            .readers_per_bank = readers_per_bank, .block_tiles = block_tiles, .block_packets = block_packets};
        const uint32_t w0_w1_k_rows = shape.hidden_tiles() + shape.bias_rows();
        const uint32_t w2_k_rows = shape.intermediate_tiles() + shape.bias_rows();
        for (uint32_t position = 0; position < shard_banks.size(); ++position) {
            const ShardLayout& shard = layouts[position];
            const GroupRun columns = shard.column_groups();
            const uint32_t readers = std::min(readers_per_bank, columns.count), output_groups = shard.output_groups();
            std::vector<uint32_t> member_groups(readers);
            std::vector<uint64_t> gate_up_tiles(readers), output_prefix(output_groups + 1, 0);
            for (uint32_t reader = 0, group = 0; reader < readers; ++reader) {
                member_groups[reader] = columns.count / readers + (reader < columns.count % readers ? 1 : 0);
                for (const uint32_t end = group + member_groups[reader]; group < end; ++group) {
                    gate_up_tiles[reader] += columns.width(group) * w0_w1_k_rows;
                }
            }
            for (uint32_t output_group = 0; output_group < output_groups; ++output_group) {
                output_prefix[output_group + 1] =
                    output_prefix[output_group] + prepared.w2_groups.width(output_group) * w2_k_rows;
            }
            const std::vector<uint32_t> member_outputs = split_output_groups(gate_up_tiles, output_prefix);
            uint32_t first_group = 0, first_output = 0;
            for (uint32_t reader = 0; reader < readers; ++reader) {
                ExpertRowsCore member;
                member.ring_position = position;
                member.bank = shard_banks[position];
                member.first_column_group = first_group;
                member.column_groups = member_groups[reader];
                member.half_column = shard.half_column && first_group + member_groups[reader] == columns.count;
                member.first_column = shard.first_column + kColumnsPerGroup * first_group;
                member.activated_columns =
                    std::min(shard.columns - kColumnsPerGroup * first_group, kColumnsPerGroup * member_groups[reader]);
                member.first_output_group = first_output;
                member.output_groups = member_outputs[reader];
                member.half_output = prepared.w2_groups.half_last && member_outputs[reader] > 0 &&
                                     first_output + member_outputs[reader] == prepared.w2_groups.count;
                member.first_output_tile = shard.first_output_tile + kGroupTiles * first_output;
                member.output_tiles = member_outputs[reader] ? std::min(
                                                                   shard.output_tiles - kGroupTiles * first_output,
                                                                   kGroupTiles * member_outputs[reader])
                                                             : 0;
                split.members.push_back(member);
                first_group += member_groups[reader];
                first_output += member_outputs[reader];
            }
        }
        for (const auto& member : split.members) {
            split.activated_tiles = std::max(split.activated_tiles, kColumnsPerGroup * member.column_groups);
        }
        return split;
    }

    // expert_groups copies of the split, each core near its bank's optimal reader core, group-major.
    void place_groups(ExpertRowsPlan& plan, const CoreSplit& split, uint32_t expert_groups) const {
        const uint32_t ring = shard_banks.size();
        std::vector<uint32_t> counts(ring, 0);
        for (const auto& member : split.members) {
            counts[member.ring_position] += expert_groups;
        }
        std::vector<uint32_t> bank_counts(optimal.size(), 0);
        for (uint32_t position = 0; position < ring; ++position) {
            bank_counts[shard_banks[position]] = counts[position];
        }
        const auto placed = place_cores(optimal, bank_counts, grid);
        std::vector<uint32_t> first_member(ring, 0);
        for (uint32_t member_index = split.members.size(); member_index-- > 0;) {
            first_member[split.members[member_index].ring_position] = member_index;
        }
        plan.readers_per_bank = split.readers_per_bank;
        plan.expert_groups = expert_groups;
        plan.group_cores = split.members.size();
        plan.activated_tiles = split.activated_tiles;
        plan.block_tiles = split.block_tiles;
        plan.block_packets = split.block_packets;
        plan.blocks_in_flight =
            *std::max_element(bank_counts.begin(), bank_counts.end()) >= kOneBlockInFlightBankCores ? 1 : 2;
        for (uint32_t group = 0; group < expert_groups; ++group) {
            for (uint32_t member_index = 0; member_index < split.members.size(); ++member_index) {
                ExpertRowsCore member = split.members[member_index];
                member.group = group;
                member.core =
                    placed[member.bank][(member_index - first_member[member.ring_position]) * expert_groups + group];
                plan.cores.push_back(member);
            }
        }
    }

    // The plan of `buffering` over expert_groups copies of the split, when its CBs fit cb_budget.
    std::optional<ExpertRowsPlan> make_plan(
        const CoreSplit& split,
        const Buffering& buffering,
        uint32_t expert_groups,
        const RoutingLayout& routing) const {
        const auto cbs = cb_sizes(
            shape, buffering, split.activated_tiles, split.block_tiles, routing.size, partial_tile_bytes, cb_budget);
        if (!cbs.has_value()) {
            return std::nullopt;
        }
        ExpertRowsPlan plan;
        plan.row_tiles = buffering.row_tiles;
        plan.x_slots = buffering.x_slots;
        plan.intermediate_slots = buffering.intermediate_slots;
        plan.block_slots = buffering.block_slots;
        plan.chunk_tiles = buffering.chunk_tiles;
        plan.chunk_piece_max = buffering.chunk_piece_max;
        plan.chunk_blocks = buffering.chunk_blocks;
        plan.w2_chunk_rows = buffering.w2_chunk_rows;
        plan.slot_credits = buffering.slot_credits;
        plan.ctl_bytes = buffering.ctl_bytes;
        plan.cbs = *cbs;
        plan.routing = routing;
        place_groups(plan, split, expert_groups);
        return plan;
    }

    // One row tile per job: x in one chunk (one slot per job), W2 in whole runs, nothing spilled; the intermediate
    // exchange orders slot reuse (rows_writer.cpp), so x / intermediate slots come before expert groups.
    std::optional<ExpertRowsPlan> plan_one_row_tile(const CoreSplit& split) const {
        const uint32_t group_cores = split.members.size();
        const uint32_t hidden_tiles = shape.hidden_tiles();
        const uint32_t piece_max = tt::div_up(hidden_tiles, group_cores);
        for (const uint32_t block_slots : kBlockSlots) {
            for (const auto& [x_slots, intermediate_slots] : kBuffering) {
                for (uint32_t expert_groups = grid_cores() / group_cores; expert_groups >= 1; --expert_groups) {
                    const uint32_t program_cores = expert_groups * group_cores;
                    const RoutingLayout routing =
                        routing_layout(shape, program_cores, sources_per_core(program_cores), expert_groups);
                    const Buffering buffering{
                        .row_tiles = 1,
                        .x_slots = x_slots,
                        .intermediate_slots = intermediate_slots,
                        .block_slots = block_slots,
                        .chunk_tiles = hidden_tiles,
                        .chunk_piece_max = piece_max,
                        .chunk_blocks =
                            expert_rows::run_blocks(kGroupTiles, hidden_tiles + shape.bias_rows(), split.block_tiles),
                        .w2_chunk_rows = shape.intermediate_tiles(),
                        .ctl_bytes = ctl_cb_bytes(1),
                        .slot_credits = false};
                    auto plan = make_plan(split, buffering, expert_groups, routing);
                    if (plan.has_value()) {
                        return plan;
                    }
                }
            }
        }
        return std::nullopt;
    }

    // Several row tiles per job: x moves in chunks of chunk_tiles hidden tiles (whole K rows of a column group per
    // weight block where possible; the largest chunk that fits, at most all of them), W2 in chunks that fill the same
    // number of weight blocks.
    std::optional<ExpertRowsPlan> plan_row_tiles(const CoreSplit& split, uint32_t row_tiles) const {
        const uint32_t group_cores = split.members.size();
        const uint32_t hidden_tiles = shape.hidden_tiles();
        const uint32_t chunk_unit = split.block_tiles / kGroupTiles;
        std::vector<uint32_t> chunks = {hidden_tiles};
        for (uint32_t chunk = tt::round_down(hidden_tiles - 1, chunk_unit); chunk >= chunk_unit; chunk -= chunk_unit) {
            chunks.push_back(chunk);
        }
        // jobs of at most 32 row_tiles rows of one expert: at most one partial job per expert plus the full ones
        const uint32_t jobs_max =
            shape.local_experts + shape.tokens * shape.top_k / (expert_rows::tile_rows * row_tiles);
        for (const uint32_t block_slots : kBlockSlots) {
            for (uint32_t expert_groups = grid_cores() / group_cores; expert_groups >= 1; --expert_groups) {
                const uint32_t program_cores = expert_groups * group_cores;
                const RoutingLayout routing =
                    routing_layout(shape, program_cores, sources_per_core(program_cores), expert_groups);
                const uint32_t ctl_bytes = ctl_cb_bytes(1 + tt::div_up(jobs_max, expert_groups));
                for (const uint32_t chunk : chunks) {
                    const uint32_t chunk_blocks =
                        expert_rows::run_blocks(kGroupTiles, chunk + shape.bias_rows(), split.block_tiles);
                    const Buffering buffering{
                        .row_tiles = row_tiles,
                        .x_slots = kChunkXSlots,
                        .intermediate_slots = kChunkIntermediateSlots,
                        .block_slots = weight_cb_blocks(chunk_blocks, block_slots),
                        .chunk_tiles = chunk,
                        .chunk_piece_max = tt::div_up(chunk, group_cores),
                        .chunk_blocks = chunk_blocks,
                        .w2_chunk_rows = chunk_blocks * split.block_tiles / kGroupTiles - shape.bias_rows(),
                        .ctl_bytes = ctl_bytes,
                        .slot_credits = true};
                    auto plan = make_plan(split, buffering, expert_groups, routing);
                    if (plan.has_value()) {
                        return plan;
                    }
                }
            }
        }
        return std::nullopt;
    }
};

// The plan for `shape` with jobs of `row_tiles` row tiles on this device within `cb_budget` bytes of circular buffers
// per core, or why there is none. shard_banks[r]: DRAM bank of the prepared weights' shard r (= moe_compute ring
// position r).
PlanOrRefusal plan_expert_rows(
    ttnn::MeshDevice* mesh_device,
    const MoEExpertRowsShape& shape,
    const std::vector<uint32_t>& shard_banks,
    uint32_t cb_budget,
    uint32_t row_tiles,
    tt::DataFormat partials) {
    const uint32_t hidden_tiles = shape.hidden_tiles();
    const uint32_t intermediate_tiles = shape.intermediate_tiles();
    const std::vector<CoreCoord> optimal =
        mesh_device->get_optimal_dram_bank_to_logical_worker_assignment(tt::tt_metal::NOC::NOC_0);
    const uint32_t ring = shard_banks.size();
    if (ring != optimal.size()) {
        return "the prepared weights have " + std::to_string(ring) + " shards for " + std::to_string(optimal.size()) +
               " DRAM banks";
    }
    if (row_tiles == 0) {
        return std::string("zero row tiles per job");
    }
    const CoreCoord grid = mesh_device->compute_with_storage_grid_size();
    const uint32_t grid_cores = grid.x * grid.y;
    // a read block is kBlockBytes of whole K rows of a column group in as many NoC packets as that takes; one packet
    // per block when the larger block does not fit
    const uint32_t weight_tile_bytes = tt::tile_size(tt::DataFormat::Bfp4_b);
    const uint32_t burst = tt::tt_metal::hal::get_noc_max_burst_size_bytes();
    const uint32_t packet_tiles = burst / weight_tile_bytes;
    std::vector<uint32_t> packet_counts = {std::max(1u, kBlockBytes / burst)};
    if (packet_counts[0] > 1) {
        packet_counts.push_back(1);
    }
    if (packet_tiles < kGroupTiles) {
        return fmt::format(
            "a NoC packet of {} B holds {} weight tiles of {} B, less than one K row of a column group ({} tiles)",
            burst,
            packet_tiles,
            weight_tile_bytes,
            kGroupTiles);
    }
    const auto layouts = shard_layouts(hidden_tiles, intermediate_tiles, shape.has_bias, ring);
    const PreparedLayout prepared = prepared_layout(hidden_tiles, intermediate_tiles, shape.has_bias, ring);
    uint32_t most_groups = 0;
    for (uint32_t position = 0; position < ring; ++position) {
        const ShardLayout& shard = layouts[position];
        if (shard.columns == 0) {
            return fmt::format(
                "DRAM bank {} (ring position {}) holds no intermediate column: {} intermediate tiles over {} banks",
                shard_banks[position],
                position,
                intermediate_tiles,
                ring);
        }
        if (shard.output_tiles == 0) {
            return fmt::format(
                "DRAM bank {} (ring position {}) holds no W2 output tile: {} hidden tiles over {} banks",
                shard_banks[position],
                position,
                hidden_tiles,
                ring);
        }
        most_groups = std::max(most_groups, shard.column_groups().count);
    }
    // reader cores per bank: 1 while the grid has fewer than kTwoReadersCoresPerBank worker cores per DRAM bank, else
    // 2; a larger split (up to the busiest bank's column groups) only when fewer block slots do not fit. Jobs of
    // several row tiles keep the first split: larger groups halve the groups that stream at once and double the
    // all-to-all.
    std::vector<uint32_t> readers_options;
    for (const uint32_t readers : {grid_cores / ring < kTwoReadersCoresPerBank ? 1u : 2u, 2u, most_groups}) {
        if (readers <= most_groups &&
            std::find(readers_options.begin(), readers_options.end(), readers) == readers_options.end()) {
            readers_options.push_back(readers);
        }
    }
    if (row_tiles > 1) {
        readers_options.resize(1);
    }
    const Planner planner{
        .shape = shape,
        .shard_banks = shard_banks,
        .optimal = optimal,
        .grid = grid,
        .cb_budget = cb_budget,
        .partial_tile_bytes = tt::tile_size(partials),
        .layouts = layouts,
        .prepared = prepared};
    for (const uint32_t block_packets : packet_counts) {
        for (const uint32_t readers : readers_options) {
            const uint32_t block_tiles =
                tt::round_down(std::min(block_packets * packet_tiles, kBlockBytes / weight_tile_bytes), kGroupTiles);
            const CoreSplit split = planner.split_cores(readers, block_tiles, block_packets);
            if (split.members.size() > grid_cores) {
                continue;
            }
            auto plan = row_tiles == 1 ? planner.plan_one_row_tile(split) : planner.plan_row_tiles(split, row_tiles);
            if (plan.has_value()) {
                return std::move(*plan);
            }
        }
    }
    return "no core split with " + std::to_string(row_tiles) +
           " row tiles per job fits the circular-buffer budget of " + std::to_string(cb_budget) + " B per core";
}

std::vector<uint32_t> shard_banks_of(const ttnn::Tensor& weights) {
    const auto& mapping = weights.buffer()->get_buffer_page_mapping();
    std::vector<uint32_t> banks;
    banks.reserve(mapping->all_cores.size());
    for (const auto& core : mapping->all_cores) {
        banks.push_back(static_cast<uint32_t>(core.x));
    }
    return banks;
}

// Virtual coordinate of every grid column / row (the kernels rebuild a core's NoC address from its linear index).
struct GridCoords {
    std::vector<uint32_t> xs, ys;
};

GridCoords grid_coords(ttnn::MeshDevice* mesh_device, CoreCoord grid) {
    GridCoords coords;
    for (uint32_t x = 0; x < grid.x; ++x) {
        coords.xs.push_back(mesh_device->worker_core_from_logical_core(CoreCoord(x, 0)).x);
    }
    for (uint32_t y = 0; y < grid.y; ++y) {
        coords.ys.push_back(mesh_device->worker_core_from_logical_core(CoreCoord(0, y)).y);
    }
    for (uint32_t x = 0; x < grid.x; ++x) {
        for (uint32_t y = 0; y < grid.y; ++y) {
            const CoreCoord virtual_core = mesh_device->worker_core_from_logical_core(CoreCoord(x, y));
            TT_FATAL(
                virtual_core.x == coords.xs[x] && virtual_core.y == coords.ys[y],
                "moe_compute expert rows: worker grid is not a product of rows and columns: logical ({}, {}) is "
                "virtual ({}, {}), its column and row give ({}, {})",
                x,
                y,
                virtual_core.x,
                virtual_core.y,
                coords.xs[x],
                coords.ys[y]);
        }
    }
    return coords;
}

std::vector<uint32_t> core_mask(const std::vector<CoreCoord>& cores, CoreCoord grid) {
    std::vector<uint32_t> words(tt::div_up(grid.x * grid.y, 32u), 0);
    for (const auto& core : cores) {
        const uint32_t linear_index = core.x * grid.y + core.y;
        words[linear_index / 32] |= 1u << (linear_index % 32);
    }
    return words;
}

uint32_t float_bits(float value) { return std::bit_cast<uint32_t>(value); }

// mapping rows of the devices whose tokens this device receives (moe_compute tilize_reader: the devices of the
// dispatch axis through this device, in axis order)
std::vector<uint32_t> source_rows(
    const ttnn::MeshDevice& mesh_device, const ttnn::MeshCoordinate& coord, uint32_t axis) {
    const auto& view = mesh_device.get_view();
    const uint32_t rows = view.num_rows(), cols = view.num_cols();
    std::vector<uint32_t> out;
    if (axis == 1) {
        for (uint32_t j = 0; j < cols; ++j) {
            out.push_back(coord[0] * cols + j);
        }
    } else {
        for (uint32_t i = 0; i < rows; ++i) {
            out.push_back(i * cols + coord[1]);
        }
    }
    return out;
}

uint32_t aligned_page(const ttnn::Tensor& tensor) { return tensor.buffer()->aligned_page_size(); }

// Peak DRAM bandwidth in bytes / s.
double dram_peak_bytes_per_s(tt::ARCH arch) {
    return (arch == tt::ARCH::BLACKHOLE ? kDramPeakGiBsBlackhole : kDramPeakGiBsWormhole) *
           static_cast<double>(1ull << 30);
}

// cost model: cycles of one tile matmul at `fidelity`
double matmul_tile_cycles(tt::tt_metal::MathFidelity fidelity) {
    return kMatmulCyclesPerFidelityPhase * tt::tt_metal::operation::OpPerformanceModel::fidelity_multiplier(fidelity);
}

// Expected jobs (at most job_rows rows of one expert) of one call: local_experts experts, each routed by each token
// with probability top_k / global_experts (uniform routing).
double expected_jobs(const MoEExpertRowsShape& shape, uint32_t job_rows) {
    const double route_probability = std::min(1.0, static_cast<double>(shape.top_k) / shape.global_experts);
    const uint32_t tokens = shape.tokens;
    if (route_probability >= 1.0) {
        return shape.local_experts * std::ceil(tokens / static_cast<double>(job_rows));
    }
    // sum over j >= 1 of P(rows > job_rows (j - 1)), rows ~ Binomial(tokens, route_probability)
    double pmf = std::pow(1.0 - route_probability, tokens), cdf = 0.0, jobs = 0.0;
    for (uint32_t row_count = 0; row_count <= tokens; ++row_count) {
        cdf += pmf;
        if (row_count % job_rows == 0) {
            jobs += std::max(0.0, 1.0 - cdf);  // P(rows > row_count)
        }
        pmf *= (static_cast<double>(tokens - row_count) / (row_count + 1)) *
               (route_probability / (1.0 - route_probability));
    }
    return shape.local_experts * jobs;
}

// The cost model's terms of one plan, per job: the stream time of its weights at the DRAM peak (the busiest bank's
// share of the group's weights x banks); per row tile, the busiest core's matmul tiles and the partial tiles it writes
// and reads back between chunks, for a job of several row tiles and for a one-row-tile job of the same program (x in
// chunks of single_chunk_tiles hidden tiles, W2 whole).
struct JobCost {
    double stream_s = 0, busiest = 0, reloads = 0, single_reloads = 0;
};

JobCost job_cost(const MoEExpertRowsShape& shape, const ExpertRowsPlan& plan, uint32_t banks, tt::ARCH arch) {
    const uint32_t hidden_tiles = shape.hidden_tiles(), intermediate_tiles = shape.intermediate_tiles();
    const uint32_t w0_w1_k_rows = hidden_tiles + shape.bias_rows(), w2_k_rows = intermediate_tiles + shape.bias_rows();
    const auto layouts = shard_layouts(hidden_tiles, intermediate_tiles, shape.has_bias, banks);
    const PreparedLayout prepared = prepared_layout(hidden_tiles, intermediate_tiles, shape.has_bias, banks);
    JobCost per_job;
    // tiles each bank (shard) streams for one job: group 0's W0/W1 runs split at the bank pieces, W2 in its own bank
    std::vector<double> bank_tiles(banks, 0.0);
    for (const auto& member : plan.cores) {
        // weight tiles per K row of the core's column groups and of its output groups
        const double gate_up_width = member.column_run().tiles();
        const double down_width = member.output_run().tiles();
        const double tiles = gate_up_width * w0_w1_k_rows + down_width * w2_k_rows;
        if (tiles > per_job.busiest) {
            per_job.busiest = tiles;
            if (plan.row_tiles > 1) {
                const uint32_t single_chunk =
                    expert_rows::single_chunk_tiles(plan.chunk_tiles, plan.row_tiles, hidden_tiles);
                per_job.reloads = gate_up_width * (tt::div_up(hidden_tiles, plan.chunk_tiles) - 1) +
                                  down_width * (tt::div_up(intermediate_tiles, plan.w2_chunk_rows) - 1);
                per_job.single_reloads = gate_up_width * (tt::div_up(hidden_tiles, single_chunk) - 1);
            }
        }
        if (member.group == 0) {
            const ShardLayout& shard = layouts[member.ring_position];
            for (uint32_t group = member.first_column_group; group < member.first_column_group + member.column_groups;
                 ++group) {
                uint32_t stream_tile = shard.w0_w1_first_tile + group * prepared.w0_w1_group_tiles;
                for (uint32_t left = shard.column_groups().width(group) * w0_w1_k_rows; left > 0;) {
                    const uint32_t piece = stream_tile / prepared.w0_w1_piece_tiles;
                    const uint32_t take = std::min(left, (piece + 1) * prepared.w0_w1_piece_tiles - stream_tile);
                    bank_tiles[piece] += take;
                    stream_tile += take;
                    left -= take;
                }
            }
            bank_tiles[member.ring_position] += down_width * w2_k_rows;
        }
    }
    const double busiest_bank = *std::max_element(bank_tiles.begin(), bank_tiles.end());
    per_job.stream_s = busiest_bank * banks * tt::tile_size(tt::DataFormat::Bfp4_b) / dram_peak_bytes_per_s(arch);
    return per_job;
}

// The cost model's prediction of one plan: its job terms and the seconds of the call.
struct PlanCost {
    double stream_s = 0, busiest = 0, reloads = 0, call_s = 0;
};

// The cost model of one call: tile_s seconds per tile matmul at the device clock.
struct CostModel {
    const MoEExpertRowsShape& shape;
    uint32_t banks = 0;
    tt::ARCH arch;
    double tile_s = 0, clock_hz = 0;

    PlanCost cost(const ExpertRowsPlan& plan) const {
        const JobCost per_job = job_cost(shape, plan, banks, arch);
        PlanCost predicted{.stream_s = per_job.stream_s, .busiest = per_job.busiest, .reloads = per_job.reloads};
        const double compute = predicted.busiest * tile_s + predicted.reloads * kReloadTileCycles / clock_hz;
        predicted.call_s = std::max(
            expected_jobs(shape, expert_rows::tile_rows * plan.row_tiles) * predicted.stream_s,
            std::ceil(expected_jobs(shape, expert_rows::tile_rows) / plan.expert_groups) * compute);
        return predicted;
    }
};

struct CostedPlan {
    ExpertRowsPlan plan;
    PlanCost cost;
};

// The placement program's view: the routing table block and place.cpp's scratch layout.
struct PlaceLayout {
    TableBlock table;
    uint32_t scratch_counts = 0, scratch_offsets = 0, scratch_e_t = 0, scratch_activation = 0, scratch_count_row = 0;
    uint32_t scratch_table = 0, scratch_bytes = 0;
    uint32_t count_words = 0, e_t_page_bytes = 0, activation_row_bytes = 0;
};

PlaceLayout place_layout(uint32_t tokens, uint32_t top_k, uint32_t local_experts) {
    const uint32_t l1_align = tt::tt_metal::hal::get_l1_alignment();
    PlaceLayout layout;
    layout.table = table_block(tokens, top_k, local_experts);
    layout.count_words = tt::align(local_experts * kWordBytes, l1_align) / kWordBytes;
    layout.e_t_page_bytes = (tokens + 1) * tt::align(kWordBytes, l1_align);
    layout.activation_row_bytes = tt::align((2 * local_experts + 1) * kWordBytes, l1_align);
    LayoutCursor cursor;
    layout.scratch_counts = cursor.take(local_experts * kWordBytes);
    layout.scratch_offsets = cursor.take(local_experts * kWordBytes);
    layout.scratch_e_t = cursor.take(layout.e_t_page_bytes);
    layout.scratch_activation = cursor.take(expert_rows::activation_buffers * layout.activation_row_bytes);
    layout.scratch_count_row = cursor.take(layout.count_words * kWordBytes);
    layout.scratch_table = cursor.take(layout.table.bytes);
    layout.scratch_bytes = cursor.end;
    return layout;
}

// feed.cpp's scratch: the table block's head, then from scratch_stage two staging buffers of batch_rows row slices (one
// ring position's W2 output columns of a row) at slice_stride.
struct FeedLayout {
    uint32_t slice_stride = 0, batch_rows = 0, scratch_stage = 0, bytes = 0;
};

FeedLayout feed_layout(const PlaceLayout& place, const std::vector<ShardLayout>& layouts, uint32_t tile_width_bytes) {
    constexpr uint32_t kStageBytes = 16 * 1024;  // per staging buffer
    FeedLayout layout;
    for (const auto& shard : layouts) {
        layout.slice_stride = std::max(layout.slice_stride, shard.output_tiles * tile_width_bytes);
    }
    layout.slice_stride = align_to_part(layout.slice_stride);
    layout.batch_rows = std::clamp(kStageBytes / layout.slice_stride, 1u, expert_rows::tile_rows);
    layout.scratch_stage = align_to_part(place.table.rows);
    layout.bytes = layout.scratch_stage + 2 * layout.batch_rows * layout.slice_stride;
    return layout;
}

// selective_reduce_combine's row offset per height shard in the double buffer (output kTilizeOutput), from its values
// per core.
uint32_t combine_row_stride(
    uint32_t values_per_core, uint32_t hidden_size, uint32_t width_shards, uint32_t height_shards) {
    return values_per_core / (hidden_size / width_shards / 2) / height_shards;
}

// TT_METAL_MOE_COMPUTE_KERNEL, a test and benchmark hook: =ring forces the ring; =expert_rows forces the expert rows
// path (over the cost model and the Wormhole gate; a call that path cannot run is refused with the reason).
enum class ForcedKernel { None, Ring, ExpertRows };

ForcedKernel forced_kernel() {
    const char* value = std::getenv("TT_METAL_MOE_COMPUTE_KERNEL");
    if (value == nullptr) {
        return ForcedKernel::None;
    }
    const std::string_view kernel(value);
    if (kernel == "ring") {
        return ForcedKernel::Ring;
    }
    return kernel == "expert_rows" ? ForcedKernel::ExpertRows : ForcedKernel::None;
}

// TT_METAL_MOE_COMPUTE_ROW_TILES=<row tiles>, a test and benchmark hook: jobs of that many row tiles, from 1 to
// max_row_tiles.
std::optional<uint32_t> forced_row_tiles(uint32_t max_row_tiles) {
    const char* value = std::getenv("TT_METAL_MOE_COMPUTE_ROW_TILES");
    if (value == nullptr) {
        return std::nullopt;
    }
    const std::string_view text(value);
    uint32_t row_tiles = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), row_tiles);
    TT_FATAL(
        error == std::errc() && end == text.data() + text.size() && row_tiles >= 1 && row_tiles <= max_row_tiles,
        "moe_compute: TT_METAL_MOE_COMPUTE_ROW_TILES={} outside 1..{} (the row tiles of the call's routed rows)",
        text,
        max_row_tiles);
    return row_tiles;
}

// The ring program for this call, logged with the reason; a call forced onto the expert rows path fails with it.
std::optional<MoEExpertRowsParams> ring_with_reason(ForcedKernel forced, const std::string& why) {
    TT_FATAL(
        forced != ForcedKernel::ExpertRows,
        "moe_compute: TT_METAL_MOE_COMPUTE_KERNEL=expert_rows, but the expert rows path cannot run: {}",
        why);
    log_debug(tt::LogOp, "moe_compute: ring kernel ({})", why);
    return std::nullopt;
}

// The first input layout rule of the expert rows path that the call breaks, with the observed value, or nullopt.
std::optional<std::string> input_layout_refusal(const MoEComputeInputs& tensor_args) {
    const ttnn::Tensor& input = tensor_args.tilize_input_tensor;
    const ttnn::Tensor& indices = tensor_args.tilize_expert_indices_tensor;
    const ttnn::Tensor& mapping = tensor_args.tilize_expert_mapping_tensor;
    if (input.logical_shape().rank() < 3) {
        return fmt::format("the input has rank {}, expected at least 3", input.logical_shape().rank());
    }
    if (indices.logical_shape().rank() < 2) {
        return fmt::format("the expert indices have rank {}, expected at least 2", indices.logical_shape().rank());
    }
    if (mapping.logical_shape().rank() != 2) {
        return fmt::format("the expert mapping has rank {}, expected 2", mapping.logical_shape().rank());
    }
    using NamedTensor = std::pair<std::string_view, const ttnn::Tensor*>;
    const std::array<NamedTensor, 4> routing_inputs = {{
        {"input", &input},
        {"expert indices", &indices},
        {"expert scores", &tensor_args.tilize_expert_scores_tensor},
        {"expert mapping", &mapping},
    }};
    for (const auto& [name, tensor] : routing_inputs) {
        if (tensor->layout() != Layout::ROW_MAJOR) {
            return fmt::format("the {} tensor has layout {}, expected ROW_MAJOR", name, tensor->layout());
        }
    }
    const std::array<NamedTensor, 2> weights = {{
        {"W0/W1", &tensor_args.matmul_w0_w1_tensor},
        {"W2", &tensor_args.matmul_w2_tensor},
    }};
    for (const auto& [name, tensor] : weights) {
        if (tensor->logical_shape().rank() != 6) {
            return fmt::format("the {} weights have rank {}, expected 6", name, tensor->logical_shape().rank());
        }
        if (tensor->dtype() != DataType::BFLOAT4_B) {
            return fmt::format("the {} weights are {}, expected BFLOAT4_B", name, tensor->dtype());
        }
        if (tensor->memory_config().memory_layout() != TensorMemoryLayout::HEIGHT_SHARDED) {
            return fmt::format(
                "the {} weights are {}, expected HEIGHT_SHARDED", name, tensor->memory_config().memory_layout());
        }
    }
    return std::nullopt;
}

// The shape of a moe_compute call. Rows kept per local expert (row_cap): its e_t page (one entry per token) and its
// double-buffer half (output kTilizeOutput: per core 2 halves of height-shard-split rows of
// hidden_size / num_data_parallel_cores values, read with the combine's row stride).
MoEExpertRowsShape expert_rows_shape(
    ttnn::MeshDevice& mesh_device,
    const MoEComputeParams& args,
    const MoEComputeInputs& tensor_args,
    const tt::tt_metal::TensorSpec& dense) {
    const auto& input = tensor_args.tilize_input_tensor;
    MoEExpertRowsShape shape;
    shape.hidden_size = input.logical_shape()[-1];
    shape.intermediate_size = args.intermediate_size;
    shape.local_experts = tensor_args.matmul_w0_w1_tensor.logical_shape()[2];
    shape.top_k = tensor_args.tilize_expert_indices_tensor.logical_shape()[-1];
    shape.tokens = input.logical_shape()[0] * input.logical_shape()[1];
    shape.global_experts = tensor_args.tilize_expert_mapping_tensor.logical_shape()[-1];
    const auto& view = mesh_device.get_view();
    shape.num_sources = args.cluster_axis().value_or(1) == 1 ? view.num_cols() : view.num_rows();
    shape.index_page_bytes = aligned_page(tensor_args.tilize_expert_indices_tensor);
    shape.has_bias = args.has_bias;
    const uint32_t width_shards = args.num_data_parallel_cores, height_shards = args.num_token_parallel_cores;
    const uint32_t values_per_core =
        dense.logical_shape().volume() / dense.memory_config().shard_spec()->grid.num_cores();
    const uint32_t combine_stride = combine_row_stride(values_per_core, shape.hidden_size, width_shards, height_shards);
    const uint32_t fits = values_per_core / 2 / (shape.hidden_size / width_shards);  // rows of one half on one core
    shape.row_cap = std::min(shape.tokens, height_shards * std::min(combine_stride, fits));
    return shape;
}

// The CB budget per core: circular buffers end below the lowest live L1 buffer (and below L1_SMALL); bucketed, it is a
// program-cache key.
uint32_t cb_budget_of(ttnn::MeshDevice& mesh_device) {
    const auto& alloc = mesh_device.allocator();
    const uint64_t base = alloc->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    const uint64_t top = std::min<uint64_t>(
        mesh_device.lowest_occupied_compute_l1_address().value_or(std::numeric_limits<uint64_t>::max()),
        ttnn::ccl::l1_small_floor_address(mesh_device));
    constexpr uint32_t kBudgetBucket = 16 * 1024;
    return top > base ? static_cast<uint32_t>((top - base) / kBudgetBucket * kBudgetBucket) : 0;
}

// moe_compute's L1 outputs per core (per-bank upper bound); the placement program runs after they are allocated, so its
// CBs end below them.
uint64_t l1_outputs_per_core(ttnn::MeshDevice& mesh_device, const std::vector<tt::tt_metal::TensorSpec>& specs) {
    uint64_t bytes = 0;
    const uint32_t l1_banks = mesh_device.allocator()->get_num_banks(tt::tt_metal::BufferType::L1);
    for (uint32_t i = 0; i < specs.size(); ++i) {
        if (i != kMatmulOutput && specs[i].memory_config().buffer_type() == tt::tt_metal::BufferType::L1) {
            bytes += specs[i].compute_consumed_memory_bytes_per_bank(tt::tt_metal::hal::get_l1_alignment(), l1_banks);
        }
    }
    return bytes;
}

// The ring program's token-sized CBs on its tilize cores: the e_t lists, activation rows, the second RISC's halves of
// both, the mapping and an input chunk. When they do not fit below moe_compute's outputs, the ring program refuses the
// call and the caller would have to split it, so the expert rows path runs it.
uint64_t ring_tilize_cb_bytes(
    ttnn::MeshDevice* mesh_device,
    const MoEComputeParams& args,
    const MoEExpertRowsShape& shape,
    const std::vector<tt::tt_metal::TensorSpec>& specs,
    const MoEComputeInputs& tensor_args) {
    const auto ring_cores = ttnn::operations::ccl::common::select_moe_compute_cores(
        mesh_device,
        args.num_token_parallel_cores,
        args.num_data_parallel_cores,
        shape.hidden_size,
        args.combine_params.has_value() ? args.combine_params->mux_core_range_set : CoreRangeSet{},
        args.bh_ring_size);
    const auto& map = tensor_args.tilize_expert_mapping_tensor;
    const uint64_t l1_align = tt::tt_metal::hal::get_l1_alignment();
    const uint64_t tokens = shape.tokens, local_experts = shape.local_experts;
    const uint64_t activation_row_bytes = tt::align((2 * local_experts + 1) * kWordBytes, l1_align);
    const uint64_t tilize_cores = std::max<size_t>(1, ring_cores.tilize_cores.size());
    const uint64_t x_page = tensor_args.tilize_input_tensor.buffer()->aligned_page_size();
    return local_experts * specs[kETPages].compute_page_size_bytes()                      // e_t lists
           + tokens * activation_row_bytes                                                // activation rows
           + tt::div_up(tokens, 2) * l1_align * local_experts                             // second RISC's e_t
           + tt::div_up(tokens, 2) * activation_row_bytes                                 // second RISC's rows
           + map.buffer()->num_pages() * map.buffer()->aligned_page_size()                // mapping
           + kRingTokensPerChunk * tt::align(tt::div_up(x_page, tilize_cores), l1_align)  // input
           + kRingSmallCbBytes;
}

// Weight-stationary jobs: the most row tiles per job that keep the weight stream the longest part of a job while the
// jobs in flight (the expert groups, fewer when the call has fewer expected jobs) share DRAM: the busiest core's
// matmuls and the writer's NoC commands for the job's rows (x reads per chunk, tile build, row writes) both within one
// job's stream. The compute bound is rounded when the call has at least two expected jobs per group (the deal in
// rows_reader.cpp then balances the groups' work) and rounded down otherwise (a longer job holds up its group while the
// others have nothing left to do). Rows per expert are not known on the host; uniform routing rarely gives an expert
// more than 32 rows of a call, hot experts and longer calls do, so several row tiles are kept on a tie (they cost about
// the same when they merge nothing). Returns the chosen plan, or why a forced row-tile count cannot run.
std::variant<CostedPlan, std::string> choose_row_tiles(
    ttnn::MeshDevice* mesh_device,
    const CostModel& model,
    const std::vector<uint32_t>& banks,
    uint32_t budget,
    tt::DataFormat partials,
    CostedPlan best) {
    const MoEExpertRowsShape& shape = model.shape;
    const std::optional<uint32_t> forced =
        forced_row_tiles(tt::div_up(shape.tokens * shape.top_k, expert_rows::tile_rows));
    const double jobs = expected_jobs(shape, expert_rows::tile_rows);
    const double compute_bound =
        std::min<double>(best.plan.expert_groups, jobs) * best.cost.stream_s / (best.cost.busiest * model.tile_s);
    const double compute_cap =
        std::round(jobs) >= 2.0 * best.plan.expert_groups ? std::round(compute_bound) : std::floor(compute_bound);
    uint32_t most = std::min<uint32_t>(compute_cap, tt::div_up(shape.tokens, expert_rows::tile_rows));
    uint32_t least = 2;
    if (forced.has_value()) {
        most = least = *forced;
    }
    // Wormhole's kernel config buffer does not hold the kernels of jobs of several row tiles.
    if (mesh_device->arch() == tt::ARCH::WORMHOLE_B0 && most > 1) {
        if (forced.has_value()) {
            return fmt::format("TT_METAL_MOE_COMPUTE_ROW_TILES={}: Wormhole runs one row tile per job", *forced);
        }
        most = 1;
    }
    for (uint32_t row_tiles = most; row_tiles >= least && row_tiles >= 2; --row_tiles) {
        auto candidate = plan_expert_rows(mesh_device, shape, banks, budget, row_tiles, partials);
        auto* plan = std::get_if<ExpertRowsPlan>(&candidate);
        if (plan == nullptr) {
            continue;
        }
        const PlanCost cost = model.cost(*plan);
        uint32_t most_output_groups = 0;
        for (const auto& member : plan->cores) {
            most_output_groups = std::max(most_output_groups, member.output_groups);
        }
        // the writer's NoC commands per row of a job: one x read per chunk, two face-row reads per tile of its x
        // piece, and per output group two face-row reads per tile and one row write
        const double x_reads = tt::div_up(shape.hidden_tiles(), plan->chunk_tiles);
        const double piece_reads = 2.0 * tt::div_up(shape.hidden_tiles(), plan->group_cores);
        const double output_commands = (2.0 * kGroupTiles + 1) * most_output_groups;
        const double row_ops = x_reads + piece_reads + output_commands;
        const double writer_bound =
            plan->expert_groups * cost.stream_s / (expert_rows::tile_rows * row_ops * kWriterOpCycles / model.clock_hz);
        if (!forced.has_value() && (row_tiles > std::lround(writer_bound) || cost.call_s > best.cost.call_s)) {
            continue;
        }
        best = {std::move(*plan), cost};
        break;
    }
    if (forced.has_value() && best.plan.row_tiles != most) {
        return fmt::format("TT_METAL_MOE_COMPUTE_ROW_TILES={}: no plan fits", *forced);
    }
    return best;
}

// Reader runtime args 5..6, which override_runtime_arguments patches: the layer's W0/W1 and W2 (the bank address of
// the layer's first expert), offsets into the weights that move with layer_id.
constexpr uint32_t kReaderLayerWeightArgs = 5;
// The bank address of `layer` of prepared weights whose (layer, expert) holds tiles_per_expert tiles per bank.
uint32_t layer_address(const ttnn::Tensor& weights, uint32_t layer, uint32_t local_experts, uint32_t tiles_per_expert) {
    const uint64_t address = uint64_t{weights.buffer()->address()} +
                             uint64_t{layer} * local_experts * tiles_per_expert * tt::tile_size(tt::DataFormat::Bfp4_b);
    TT_FATAL(
        address <= std::numeric_limits<uint32_t>::max(),
        "moe_compute expert rows: layer {} of the weights at {} B in a bank is past 32 bits",
        layer,
        address);
    return static_cast<uint32_t>(address);
}

std::array<uint32_t, 2> layer_weight_args(
    const MoEExpertRowsParams& args, const MoEExpertRowsInputs& tensor_args, const PreparedLayout& layout) {
    const uint32_t local_experts = args.shape.local_experts;
    return {
        layer_address(tensor_args.w0_w1_tensor, args.layer_id, local_experts, layout.w0_w1_piece_tiles),
        layer_address(tensor_args.w2_tensor, args.layer_id, local_experts, layout.w2_slice_tiles)};
}

// The tiles of one shard per (layer, expert) of prepared weights: their last three dims.
uint32_t prepared_tiles_per_expert(const ttnn::Shape& shape) {
    return shape[-3] * (shape[-2] / tt::constants::TILE_HEIGHT) * (shape[-1] / tt::constants::TILE_WIDTH);
}

void create_cb(
    tt::tt_metal::ProgramDescriptor& desc,
    const CoreRangeSet& cores,
    uint32_t index,
    uint32_t bytes,
    tt::DataFormat format,
    uint32_t page_bytes) {
    desc.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = bytes,
        .core_ranges = cores,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(index),
            .data_format = format,
            .page_size = page_bytes,
        }}},
    });
}

// `count` consecutive semaphores on `cores`, all 0; returns the first id.
uint32_t create_semaphores(tt::tt_metal::ProgramDescriptor& desc, const CoreRangeSet& cores, uint32_t count) {
    const uint32_t first = add_semaphore_descriptor(desc, cores, 0);
    for (uint32_t i = 1; i < count; ++i) {
        add_semaphore_descriptor(desc, cores, 0);
    }
    return first;
}

}  // namespace

//-----------------------------------------------------------------------------------------------------------------
// Selection between the ring program and the expert rows programs
//-----------------------------------------------------------------------------------------------------------------
std::optional<MoEExpertRowsParams> select_moe_compute_expert_rows(
    const MoEComputeParams& args, const MoEComputeInputs& tensor_args) {
    auto* mesh_device = tensor_args.tilize_input_tensor.device();
    const ForcedKernel forced = forced_kernel();
    if (forced == ForcedKernel::Ring) {
        return ring_with_reason(forced, "TT_METAL_MOE_COMPUTE_KERNEL=ring");
    }
    if (args.path == MoEComputePath::SingleCluster) {
        return ring_with_reason(
            forced, "an extent-1 cluster axis builds no combine for the expert rows feed; dm1 writes the output");
    }
    if (!expert_rows_arch(mesh_device->arch())) {
        return ring_with_reason(
            forced, fmt::format("the expert rows path runs on Blackhole and Wormhole, not {}", mesh_device->arch()));
    }
    if (mesh_device->arch() != tt::ARCH::BLACKHOLE && forced != ForcedKernel::ExpertRows) {
        return ring_with_reason(
            forced, "Wormhole takes the expert rows path only when TT_METAL_MOE_COMPUTE_KERNEL=expert_rows forces it");
    }
    if (const auto refusal = input_layout_refusal(tensor_args)) {
        return ring_with_reason(forced, *refusal);
    }
    if (args.num_shared_experts_per_device.value_or(0) > 0 && args.combine_params.has_value()) {
        return ring_with_reason(
            forced,
            fmt::format(
                "{} shared experts per device with the combine (split over the mesh)",
                *args.num_shared_experts_per_device));
    }
    const auto specs = MoEComputeDeviceOperation::compute_output_specs(args, tensor_args);
    const MoEExpertRowsShape shape = expert_rows_shape(*mesh_device, args, tensor_args, specs[kTilizeOutput]);
    if (shape.top_k < 1 || shape.top_k > kMaxTopK) {
        return ring_with_reason(forced, fmt::format("top_k {} outside 1..{}", shape.top_k, kMaxTopK));
    }
    const uint32_t scores_page_bytes = aligned_page(tensor_args.tilize_expert_scores_tensor);
    if (scores_page_bytes != shape.index_page_bytes) {
        return ring_with_reason(
            forced,
            fmt::format(
                "the expert scores pages are {} B, the expert indices pages {} B",
                scores_page_bytes,
                shape.index_page_bytes));
    }
    if (shape.tokens % shape.num_sources != 0) {
        return ring_with_reason(
            forced, fmt::format("{} tokens do not split evenly over {} sources", shape.tokens, shape.num_sources));
    }
    if (uint64_t{shape.tokens} * shape.top_k > kMaxRoutedRows) {
        return ring_with_reason(
            forced,
            fmt::format(
                "{} tokens * top_k {} = {} routed rows, more than the routing table's u16 rows hold ({})",
                shape.tokens,
                shape.top_k,
                uint64_t{shape.tokens} * shape.top_k,
                kMaxRoutedRows));
    }
    const uint32_t budget = cb_budget_of(*mesh_device);
    const auto banks = shard_banks_of(tensor_args.matmul_w0_w1_tensor);
    const tt::DataFormat partials = partials_format(args.fp32_dest_acc_en);
    auto first = plan_expert_rows(mesh_device, shape, banks, budget, 1, partials);
    if (const auto* refusal = std::get_if<std::string>(&first)) {
        return ring_with_reason(forced, *refusal);
    }
    const uint64_t outputs_per_core = l1_outputs_per_core(*mesh_device, specs);
    const PlaceLayout place = place_layout(shape.tokens, shape.top_k, shape.local_experts);
    uint64_t placement_cb_bytes = place.scratch_bytes;
    if (args.path != MoEComputePath::ComputeOnly) {
        placement_cb_bytes +=
            feed_layout(
                place,
                shard_layouts(shape.hidden_tiles(), shape.intermediate_tiles(), shape.has_bias, banks.size()),
                kTileRowBytesBf16)
                .bytes;
    }
    if (placement_cb_bytes + outputs_per_core > budget) {
        return ring_with_reason(
            forced,
            fmt::format(
                "placement program CBs ({} B) and moe_compute's L1 outputs ({} B per core) exceed the {} B CB budget",
                placement_cb_bytes,
                outputs_per_core,
                budget));
    }
    // Can the ring program run this call at all (ring_tilize_cb_bytes)?
    const bool ring_fits =
        ring_tilize_cb_bytes(mesh_device, args, shape, specs, tensor_args) + outputs_per_core + kRingFitMargin <=
        budget;
    // Cost model. Every path streams each touched expert's weights once per job: a job is at most 32 of its rows on the
    // ring and with one row tile per job, at most 32 row_tiles rows with weight-stationary jobs. A bank's weights come
    // at 1 / banks of the DRAM peak, so the busiest bank sets the stream time of a job (stream_s). The ring computes
    // each job on one core per bank and pays a ring pass per job; the expert rows path spreads the work over
    // readers_per_bank cores per bank and runs expert_groups groups in parallel. With the combine (SingleDevice,
    // FullCcl) the ring overlaps it with its compute, while the expert rows path feeds it one expert at a time after
    // its rows exist, so that path also pays the feed: per active expert and per row.
    //   t_ring = jobs(32) * (max(stream_s, ring_tiles * tile_s) + ring * hop)
    //   t_expert_rows = max(jobs(32 row_tiles) * stream_s,
    //                       ceil(jobs(32) / expert_groups) * (busiest * tile_s + reloads * reload))
    //                   [+ active_experts * feed_expert + rows * feed_row]
    // tile_s: seconds per tile matmul (matmul_tile_cycles at the device clock); hop, feed_expert, feed_row and reload
    // (kRingHopCycles, kFeedExpertCycles, kFeedRowCycles*, kReloadTileCycles): cycle counts at the clock; jobs(n):
    // expected jobs of at most n rows (expected_jobs: top_k / global experts of each token); row_tiles: row tiles per
    // job; reloads: the partial tiles a row tile of a job of several row tiles writes and reads back between chunks.
    // The expert rows path runs only with a clear predicted margin over the ring (kCostMargin); within it the ring is
    // kept.
    const uint32_t hidden_tiles = shape.hidden_tiles(), intermediate_tiles = shape.intermediate_tiles();
    const uint32_t ring = banks.size();
    const uint32_t w0_w1_k_rows = hidden_tiles + shape.bias_rows(), w2_k_rows = intermediate_tiles + shape.bias_rows();
    const double clock_hz = mesh_device->get_clock_rate_mhz() * 1e6;
    const CostModel model{
        .shape = shape,
        .banks = ring,
        .arch = mesh_device->arch(),
        .tile_s = matmul_tile_cycles(args.math_fidelity) / clock_hz,
        .clock_hz = clock_hz};
    ExpertRowsPlan& one_row_tile = std::get<ExpertRowsPlan>(first);
    const PlanCost one_row_tile_cost = model.cost(one_row_tile);
    auto chosen = choose_row_tiles(
        mesh_device, model, banks, budget, partials, CostedPlan{std::move(one_row_tile), one_row_tile_cost});
    if (const auto* why = std::get_if<std::string>(&chosen)) {
        return ring_with_reason(forced, *why);
    }
    const auto& [plan, cost] = std::get<CostedPlan>(chosen);
    const double jobs = expected_jobs(shape, expert_rows::tile_rows);
    uint32_t ring_columns = 0;  // gate/up columns of the ring's busiest core (moe_ring_common.h)
    for (uint32_t position = 0; position < ring; ++position) {
        ring_columns = std::max(ring_columns, moe_ring::w0_w1_stored_cols(intermediate_tiles, position, ring));
    }
    const PreparedLayout prepared = prepared_layout(hidden_tiles, intermediate_tiles, shape.has_bias, ring);
    // matmul tiles of the ring's busiest core per job: W0 and W1 of its columns, its W2 output groups
    const double ring_tiles =
        2.0 * w0_w1_k_rows * ring_columns + static_cast<double>(prepared.w2_groups.tiles()) * w2_k_rows;
    const double t_ring =
        jobs * (std::max(cost.stream_s, ring_tiles * model.tile_s) + ring * kRingHopCycles / clock_hz);
    double t_expert_rows = cost.call_s;
    if (args.path != MoEComputePath::ComputeOnly) {
        const double route_probability = std::min(1.0, static_cast<double>(shape.top_k) / shape.global_experts);
        const double active_experts = shape.local_experts * (1.0 - std::pow(1.0 - route_probability, shape.tokens));
        const double rows =
            static_cast<double>(shape.tokens) * shape.top_k * shape.local_experts / shape.global_experts;
        const double row_cycles = args.path == MoEComputePath::FullCcl ? kFeedRowCyclesCcl : kFeedRowCyclesLocal;
        t_expert_rows += (active_experts * kFeedExpertCycles + rows * row_cycles) / clock_hz;
    }
    if (ring_fits && t_expert_rows * (1.0 + kCostMargin) >= t_ring && forced != ForcedKernel::ExpertRows) {
        return ring_with_reason(
            forced,
            fmt::format(
                "cost model: ring {:.1f} us, expert rows {:.1f} us for {:.1f} expected jobs, no clear margin",
                t_ring * 1e6,
                t_expert_rows * 1e6,
                jobs));
    }
    log_debug(
        tt::LogOp,
        "moe_compute: expert rows kernel (cost model: expert rows {:.1f} us, ring {:.1f} us for {:.1f} expected jobs, "
        "ring program {} L1; {} row tiles per job, {} readers per bank, {} expert groups of {} cores, "
        "x / intermediate slots {} / {}, x chunk {} tiles, W2 chunk {} rows, {} B CBs)",
        t_expert_rows * 1e6,
        t_ring * 1e6,
        jobs,
        ring_fits ? "fits" : "does not fit",
        plan.row_tiles,
        plan.readers_per_bank,
        plan.expert_groups,
        plan.group_cores,
        plan.x_slots,
        plan.intermediate_slots,
        plan.chunk_tiles,
        plan.w2_chunk_rows,
        plan.cbs.total());
    return MoEExpertRowsParams{
        .shape = shape,
        .layer_id = args.layer_id,
        .cluster_axis = args.cluster_axis().value_or(1),
        .cb_budget = budget,
        .row_tiles = plan.row_tiles,
        .activation_type = args.activation_type,
        .activation_limit = args.activation_limit,
        .math_fidelity = args.math_fidelity,
        .fp32_dest_acc_en = args.fp32_dest_acc_en};
}

//-----------------------------------------------------------------------------------------------------------------
// The expert rows program
//-----------------------------------------------------------------------------------------------------------------
MoEExpertRowsDeviceOperation::program_factory_t MoEExpertRowsDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return MoEExpertRowsFactory{};
}

ttsl::hash::hash_t MoEExpertRowsDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return hash_without_layer_id(args, tensor_args);
}

void MoEExpertRowsDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void MoEExpertRowsDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_activation_limit(args.activation_type, args.activation_limit);
    const auto& shape = args.shape;
    TT_FATAL(
        shape.tokens % shape.num_sources == 0,
        "moe_compute expert rows: {} tokens over {} sources",
        shape.tokens,
        shape.num_sources);
    TT_FATAL(
        shape.top_k >= 1 && shape.top_k <= kMaxTopK,
        "moe_compute expert rows: top_k {} outside 1..{}",
        shape.top_k,
        kMaxTopK);
    TT_FATAL(
        uint64_t{shape.tokens} * shape.top_k <= kMaxRoutedRows,
        "moe_compute expert rows: tokens * top_k must fit 16 bits: {} tokens, top_k {}",
        shape.tokens,
        shape.top_k);
    const auto& w0_w1 = tensor_args.w0_w1_tensor;
    const auto& w2 = tensor_args.w2_tensor;
    TT_FATAL(
        w0_w1.dtype() == DataType::BFLOAT4_B && w2.dtype() == DataType::BFLOAT4_B,
        "moe_compute expert rows: weights must be bfloat4_b, got W0/W1 {} and W2 {}",
        w0_w1.dtype(),
        w2.dtype());
    const uint32_t indices_page = aligned_page(tensor_args.expert_indices_tensor);
    const uint32_t scores_page = aligned_page(tensor_args.expert_scores_tensor);
    TT_FATAL(
        indices_page == shape.index_page_bytes && scores_page == shape.index_page_bytes,
        "moe_compute expert rows: indices and scores must share one page size of {} B, got {} and {} B",
        shape.index_page_bytes,
        indices_page,
        scores_page);
    const auto w0_w1_banks = shard_banks_of(w0_w1);
    const auto w2_banks = shard_banks_of(w2);
    TT_FATAL(
        w0_w1_banks == w2_banks,
        "moe_compute expert rows: W0/W1 and W2 shards must sit on the same banks, got {} and {}",
        w0_w1_banks,
        w2_banks);
    TT_FATAL(
        args.layer_id < w0_w1.logical_shape()[1],
        "moe_compute expert rows: layer_id {} is out of range for {} layers of weights",
        args.layer_id,
        w0_w1.logical_shape()[1]);
}

MoEExpertRowsDeviceOperation::spec_return_value_t MoEExpertRowsDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& shape = args.shape;
    auto* mesh_device = tensor_args.input_tensor.device();
    const auto result = plan_expert_rows(
        mesh_device,
        shape,
        shard_banks_of(tensor_args.w0_w1_tensor),
        args.cb_budget,
        args.row_tiles,
        partials_format(args.fp32_dest_acc_en));
    const auto* plan = std::get_if<ExpertRowsPlan>(&result);
    TT_FATAL(plan != nullptr, "moe_compute expert rows: {}", std::get<std::string>(result));
    const uint32_t rows = std::min(shape.tokens * shape.top_k, shape.local_experts * shape.row_cap);
    const auto rows_spec = tt::tt_metal::TensorSpec(
        ttnn::Shape({rows, shape.hidden_size}),
        tt::tt_metal::TensorLayout(
            DataType::BFLOAT16, tt::tt_metal::PageConfig(Layout::ROW_MAJOR), ttnn::DRAM_MEMORY_CONFIG));
    const auto table_spec = tt::tt_metal::TensorSpec(
        ttnn::Shape({1, plan->routing.table_bytes / kBf16Bytes}),
        tt::tt_metal::TensorLayout(
            DataType::UINT16, tt::tt_metal::PageConfig(Layout::ROW_MAJOR), ttnn::DRAM_MEMORY_CONFIG));
    return {rows_spec, table_spec};
}

MoEExpertRowsDeviceOperation::tensor_return_value_t MoEExpertRowsDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto specs = compute_output_specs(args, tensor_args);
    auto* device = tensor_args.input_tensor.device();
    return {create_device_tensor(specs[0], device), create_device_tensor(specs[1], device)};
}

namespace {

// The expert rows program on the device at mesh_coordinate.
tt::tt_metal::ProgramDescriptor build_expert_rows_program_descriptor(
    const MoEExpertRowsParams& args,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const MoEExpertRowsInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value) {
    tt::tt_metal::ProgramDescriptor desc;
    auto* mesh_device = tensor_args.input_tensor.device();
    const auto& shape = args.shape;
    const uint32_t hidden_tiles = shape.hidden_tiles();
    const uint32_t intermediate_tiles = shape.intermediate_tiles();
    const auto shard_banks = shard_banks_of(tensor_args.w0_w1_tensor);
    const uint32_t ring = shard_banks.size();
    const auto result = plan_expert_rows(
        mesh_device, shape, shard_banks, args.cb_budget, args.row_tiles, partials_format(args.fp32_dest_acc_en));
    TT_FATAL(
        std::holds_alternative<ExpertRowsPlan>(result), "moe_compute expert rows: {}", std::get<std::string>(result));
    TT_FATAL(
        expert_rows_arch(mesh_device->arch()),
        "moe_compute expert rows: the expert rows program runs on Blackhole and Wormhole, not {}",
        mesh_device->arch());
    const ExpertRowsPlan& plan = std::get<ExpertRowsPlan>(result);
    const auto& routing = plan.routing;
    const CoreCoord grid = mesh_device->compute_with_storage_grid_size();
    const GridCoords virtual_coords = grid_coords(mesh_device, grid);
    const auto& view = mesh_device->get_view();
    const uint32_t device_id = mesh_coordinate[0] * view.num_cols() + mesh_coordinate[1];
    const auto sources = source_rows(*mesh_device, mesh_coordinate, args.cluster_axis);
    TT_FATAL(
        sources.size() == shape.num_sources,
        "moe_compute expert rows: {} sources, expected {}",
        sources.size(),
        shape.num_sources);

    const ttnn::Tensor& rows_tensor = tensor_return_value[0];
    const ttnn::Tensor& table_tensor = tensor_return_value[1];

    const uint32_t weight_tile_bytes = tt::tile_size(tt::DataFormat::Bfp4_b);
    const PreparedLayout layout = prepared_layout(hidden_tiles, intermediate_tiles, shape.has_bias, ring);
    const auto shards = shard_layouts(hidden_tiles, intermediate_tiles, shape.has_bias, ring);
    const uint32_t w0_w1_tiles = prepared_tiles_per_expert(tensor_args.w0_w1_tensor.logical_shape());
    const uint32_t w2_tiles = prepared_tiles_per_expert(tensor_args.w2_tensor.logical_shape());
    TT_FATAL(
        w0_w1_tiles == layout.w0_w1_piece_tiles && w2_tiles == layout.w2_slice_tiles,
        "moe_compute expert rows: weights do not have the prepared layout of hidden size {} / intermediate size {} on "
        "{} banks: W0/W1 {} / W2 {} tiles per (layer, expert) in a bank, expected {} / {}",
        shape.hidden_size,
        shape.intermediate_size,
        ring,
        w0_w1_tiles,
        w2_tiles,
        layout.w0_w1_piece_tiles,
        layout.w2_slice_tiles);

    std::vector<CoreCoord> all_cores;
    all_cores.reserve(plan.cores.size());
    for (const auto& member : plan.cores) {
        all_cores.push_back(member.core);
    }
    const CoreRangeSet all_set = CoreRangeSet(ttsl::Span<const CoreCoord>(all_cores));
    const std::vector<uint32_t> all_mask = core_mask(all_cores, grid);

    // circular buffers: x chunk slots, row tiles of activated and intermediate tiles per job; with several chunks per
    // job FP32 (or 16-bit DEST) partials
    enum ExpertRowsCb : uint32_t {
        cb_x,
        cb_w,
        cb_activated,
        cb_intermediate,
        cb_rows,
        cb_ctl,
        cb_routing,
        cb_scratch,
        cb_ones,
        cb_partials,
    };
    const CbSizes& cbs = plan.cbs;
    create_cb(desc, all_set, cb_x, cbs.x_chunks, tt::DataFormat::Float16_b, kTileBytesBf16);
    create_cb(desc, all_set, cb_w, cbs.weights, tt::DataFormat::Bfp4_b, weight_tile_bytes);
    create_cb(desc, all_set, cb_activated, cbs.activated, tt::DataFormat::Float16_b, kTileBytesBf16);
    create_cb(desc, all_set, cb_intermediate, cbs.intermediate, tt::DataFormat::Float16_b, kTileBytesBf16);
    create_cb(desc, all_set, cb_rows, cbs.rows, tt::DataFormat::Float16_b, kTileBytesBf16);
    create_cb(desc, all_set, cb_ctl, cbs.ctl, tt::DataFormat::UInt32, plan.ctl_bytes);
    create_cb(desc, all_set, cb_routing, cbs.routing, tt::DataFormat::UInt32, routing.size);
    create_cb(desc, all_set, cb_scratch, cbs.scratch, tt::DataFormat::UInt32, cbs.scratch);
    if (cbs.ones > 0) {
        create_cb(desc, all_set, cb_ones, cbs.ones, tt::DataFormat::Float16_b, kTileBytesBf16);
    }
    tt::tt_metal::ComputeConfigDescriptor::UnpackToDestModes unpack_to_dest(
        NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
    if (cbs.partials > 0) {
        // partials go back into DEST exactly: FP32 tiles unpacked straight to DEST, 16-bit tiles through SrcA
        const tt::DataFormat format = partials_format(args.fp32_dest_acc_en);
        create_cb(desc, all_set, cb_partials, cbs.partials, format, tt::tile_size(format));
        if (args.fp32_dest_acc_en) {
            unpack_to_dest[cb_partials] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
        }
    }

    // semaphores: routing (entries at the root, table ready), intermediate slots, x slots; with credits also those that
    // hand a slot back (every group core freed it)
    const uint32_t sem_entries = add_semaphore_descriptor(desc, all_set, 0);
    const uint32_t sem_table = add_semaphore_descriptor(desc, all_set, 0);
    const uint32_t sem_intermediate = create_semaphores(desc, all_set, plan.intermediate_slots);
    const uint32_t sem_x = create_semaphores(desc, all_set, plan.x_slots);
    const uint32_t sem_intermediate_free =
        plan.slot_credits ? create_semaphores(desc, all_set, plan.intermediate_slots) : 0;
    const uint32_t sem_x_free = plan.slot_credits ? create_semaphores(desc, all_set, plan.x_slots) : 0;

    // several row tiles per job: the root deals the jobs to the groups by the cost model's predicted job time
    // (rows_reader.cpp), in cycles scaled down by a power of two while a group's most jobs (its ctl list) could
    // overflow its 32-bit load
    const JobCost per_job = job_cost(shape, plan, ring, mesh_device->arch());
    const double tile_cycles = matmul_tile_cycles(args.math_fidelity);
    const double stream_cycles = plan.expert_groups * per_job.stream_s * mesh_device->get_clock_rate_mhz() * 1e6;
    const double row_tile_cycles = per_job.busiest * tile_cycles + per_job.reloads * kReloadTileCycles;
    const double single_cycles = per_job.busiest * tile_cycles + per_job.single_reloads * kReloadTileCycles;
    const double job_most = std::max({stream_cycles, plan.row_tiles * row_tile_cycles, single_cycles});
    const uint32_t ctl_words = plan.ctl_bytes / kWordBytes;
    double deal_unit = 1.0;
    while (job_most * ctl_words / deal_unit >= kDealLoadLimit) {
        deal_unit *= 2.0;
    }
    auto deal_units = [deal_unit](double cycles) { return static_cast<uint32_t>(std::ceil(cycles / deal_unit)); };
    tt::tt_metal::KernelDescriptor::NamedCompileTimeArgs reader_ct = {
        {"cb_w", cb_w},
        {"cb_routing", cb_routing},
        {"cb_ctl", cb_ctl},
        {"block_tiles", plan.block_tiles},
        {"weight_tile_bytes", weight_tile_bytes},
        {"block_slots", plan.block_slots},
        {"block_packets", plan.block_packets},
        {"blocks_in_flight", plan.blocks_in_flight},
        {"tokens", shape.tokens},
        {"local_experts", shape.local_experts},
        {"top_k", shape.top_k},
        {"global_experts", shape.global_experts},
        {"index_page_bytes", shape.index_page_bytes},
        {"mapping_page_bytes", aligned_page(tensor_args.expert_mapping_tensor)},
        {"tokens_per_source", shape.tokens / shape.num_sources},
        {"expert_groups", plan.expert_groups},
        {"job_rows", expert_rows::tile_rows * plan.row_tiles},
        {"row_tiles", plan.row_tiles},
        {"chunk_tiles", plan.chunk_tiles},
        {"chunk_blocks", plan.chunk_blocks},
        {"w2_chunk_rows", plan.w2_chunk_rows},
        {"hidden_tiles", hidden_tiles},
        {"intermediate_tiles", intermediate_tiles},
        {"has_bias", shape.has_bias ? 1u : 0u},
        {"ctl_words", ctl_words},
        {"deal_stream", deal_units(stream_cycles)},
        {"deal_row_tile", deal_units(row_tile_cycles)},
        {"deal_single", deal_units(single_cycles)},
        {"row_cap", shape.row_cap},
        {"grid_w", grid.x},
        {"grid_h", grid.y},
        {"mask_words", static_cast<uint32_t>(all_mask.size())},
        {"sem_entries", sem_entries},
        {"sem_table", sem_table},
        {"routing_own_mapping", routing.own_mapping},
        {"routing_source_mappings", routing.source_mappings},
        {"routing_token_ids", routing.token_ids},
        {"routing_token_scores", routing.token_scores},
        {"routing_ctl", routing.ctl},
        {"routing_active_experts", routing.active_experts},
        {"routing_expert_rows", routing.expert_rows},
        {"routing_expert_first_rows", routing.expert_first_rows},
        {"routing_table", routing.table},
        {"routing_table_counts", routing.table_counts},
        {"routing_table_offsets", routing.table_offsets},
        {"routing_row_tokens", routing.row_tokens},
        {"routing_entry_experts", routing.entry_experts},
        {"routing_entry_tokens", routing.entry_tokens},
        {"routing_entry_scores", routing.entry_scores},
        {"routing_entry_k", routing.entry_k},
        {"routing_table_bytes", routing.table_bytes},
        {"routing_jobs", routing.jobs},
        {"routing_entry_areas", routing.entry_areas},
        {"w0_w1_group_tiles", layout.w0_w1_group_tiles},
        {"w0_w1_piece_tiles", layout.w0_w1_piece_tiles},
        {"w2_group_tiles", layout.w2_group_tiles},
        {"w2_slice_tiles", layout.w2_slice_tiles},
    };
    std::vector<uint32_t> reader_ct_pos;
    tt::tt_metal::TensorAccessorArgs(*tensor_args.expert_indices_tensor.buffer()).append_to(reader_ct_pos);
    tt::tt_metal::TensorAccessorArgs(*tensor_args.expert_scores_tensor.buffer()).append_to(reader_ct_pos);
    tt::tt_metal::TensorAccessorArgs(*tensor_args.expert_mapping_tensor.buffer()).append_to(reader_ct_pos);
    tt::tt_metal::TensorAccessorArgs(*table_tensor.buffer()).append_to(reader_ct_pos);

    tt::tt_metal::KernelDescriptor::NamedCompileTimeArgs writer_ct = {
        {"cb_activated", cb_activated},
        {"cb_intermediate", cb_intermediate},
        {"cb_routing", cb_routing},
        {"cb_x", cb_x},
        {"cb_scratch", cb_scratch},
        {"cb_rows", cb_rows},
        {"group_cores", plan.group_cores},
        {"intermediate_tiles", intermediate_tiles},
        {"hidden_tiles", hidden_tiles},
        {"intermediate_slots", plan.intermediate_slots},
        {"x_slots", plan.x_slots},
        {"activated_tiles", plan.activated_tiles},
        {"routing_ctl", routing.ctl},
        {"routing_row_tokens", routing.row_tokens},
        {"routing_jobs", routing.jobs},
        {"grid_w", grid.x},
        {"grid_h", grid.y},
        {"mask_words", static_cast<uint32_t>(all_mask.size())},
        {"row_bytes", shape.hidden_size * kBf16Bytes},
        {"expert_groups", plan.expert_groups},
        {"sem_intermediate", sem_intermediate},
        {"sem_x", sem_x},
        {"sem_intermediate_free", sem_intermediate_free},
        {"sem_x_free", sem_x_free},
        {"row_tiles", plan.row_tiles},
        {"chunk_tiles", plan.chunk_tiles},
        {"chunk_piece_max", plan.chunk_piece_max},
        {"slot_credits", plan.slot_credits ? 1u : 0u},
    };
    std::vector<uint32_t> writer_ct_pos;
    tt::tt_metal::TensorAccessorArgs(*tensor_args.input_tensor.buffer()).append_to(writer_ct_pos);
    tt::tt_metal::TensorAccessorArgs(*rows_tensor.buffer()).append_to(writer_ct_pos);

    tt::tt_metal::KernelDescriptor::NamedCompileTimeArgs compute_ct = {
        {"block_tiles", plan.block_tiles},
        {"cb_w", cb_w},
        {"hidden_tiles", hidden_tiles},
        {"intermediate_tiles", intermediate_tiles},
        {"ring_cores", ring},
        {"has_bias", shape.has_bias ? 1u : 0u},
        {"cb_x", cb_x},
        {"cb_activated", cb_activated},
        {"cb_intermediate", cb_intermediate},
        {"cb_rows", cb_rows},
        {"cb_ctl", cb_ctl},
        {"cb_ones", cb_ones},
        {"activated_tiles", plan.activated_tiles},
        {"activation_function", static_cast<uint32_t>(args.activation_type)},
        {"activation_limit_bits", float_bits(args.activation_limit.value_or(0.0f))},
        {"row_tiles", plan.row_tiles},
        {"chunk_tiles", plan.chunk_tiles},
        {"chunk_blocks", plan.chunk_blocks},
        {"w2_chunk_rows", plan.w2_chunk_rows},
        {"cb_partials", cb_partials},
    };

    // readers on NOC 0, writers on NOC 1: one kernel each over the whole core set, one binary per program written to
    // the cores' rectangles
    desc.kernels.push_back(tt::tt_metal::KernelDescriptor{
        .kernel_source = std::string(kKernelDir) + kReaderKernel,
        .core_ranges = all_set,
        .compile_time_args = std::move(reader_ct_pos),
        .named_compile_time_args = std::move(reader_ct),
        .config =
            tt::tt_metal::DataMovementConfigDescriptor{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt::tt_metal::NOC::NOC_0,
            },
    });
    desc.kernels.push_back(tt::tt_metal::KernelDescriptor{
        .kernel_source = std::string(kKernelDir) + kWriterKernel,
        .core_ranges = all_set,
        .compile_time_args = std::move(writer_ct_pos),
        .named_compile_time_args = std::move(writer_ct),
        .config =
            tt::tt_metal::DataMovementConfigDescriptor{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
                .noc = tt::tt_metal::NOC::NOC_1,
            },
    });
    desc.kernels.push_back(tt::tt_metal::KernelDescriptor{
        .kernel_source = std::string(kKernelDir) + kComputeKernel,
        .core_ranges = all_set,
        .named_compile_time_args = std::move(compute_ct),
        .config =
            tt::tt_metal::ComputeConfigDescriptor{
                .math_fidelity = args.math_fidelity,
                .fp32_dest_acc_en = args.fp32_dest_acc_en,
                .dst_full_sync_en = false,
                .unpack_to_dest_mode = std::move(unpack_to_dest),
                .math_approx_mode = true,
            },
    });

    // runtime args: per core, plus common args shared by every core of a kernel
    std::vector<uint32_t> reader_common = {static_cast<uint32_t>(sources.size())};
    reader_common.insert(reader_common.end(), sources.begin(), sources.end());
    reader_common.insert(reader_common.end(), virtual_coords.xs.begin(), virtual_coords.xs.end());
    reader_common.insert(reader_common.end(), virtual_coords.ys.begin(), virtual_coords.ys.end());
    reader_common.insert(reader_common.end(), all_mask.begin(), all_mask.end());
    reader_common.insert(reader_common.end(), shard_banks.begin(), shard_banks.end());
    std::vector<uint32_t> writer_common = virtual_coords.xs;
    writer_common.insert(writer_common.end(), virtual_coords.ys.begin(), virtual_coords.ys.end());
    desc.kernels[kRowsReaderKernelIdx].common_runtime_args = std::move(reader_common);
    desc.kernels[kRowsWriterKernelIdx].common_runtime_args = std::move(writer_common);
    const CoreCoord root = mesh_device->worker_core_from_logical_core(plan.cores.front().core);
    const uint32_t program_cores = plan.cores.size();
    const auto layer_weights = layer_weight_args(args, tensor_args, layout);
    for (uint32_t core_index = 0; core_index < program_cores; ++core_index) {
        const ExpertRowsCore& member = plan.cores[core_index];
        std::vector<CoreCoord> group_cores;
        for (const auto& other : plan.cores) {
            if (other.group == member.group) {
                group_cores.push_back(other.core);
            }
        }
        tt::tt_metal::KernelDescriptor::RTArgList reader_args;
        reader_args.push_back(member.bank);
        reader_args.push_back(tensor_args.expert_indices_tensor.buffer());
        reader_args.push_back(tensor_args.expert_scores_tensor.buffer());
        reader_args.push_back(tensor_args.expert_mapping_tensor.buffer());
        reader_args.push_back(table_tensor.buffer());
        reader_args.append(
            {layer_weights[0],
             layer_weights[1],
             shards[member.ring_position].w0_w1_first_tile + member.first_column_group * layout.w0_w1_group_tiles,
             member.column_groups,
             member.first_output_group,
             member.output_groups,
             member.group,
             core_index,
             program_cores,
             static_cast<uint32_t>(root.x),
             static_cast<uint32_t>(root.y),
             device_id,
             member.half_column ? 1u : 0u,
             member.half_output ? 1u : 0u});
        desc.kernels[kRowsReaderKernelIdx].emplace_runtime_args(member.core, reader_args);

        tt::tt_metal::KernelDescriptor::RTArgList writer_args;
        writer_args.push_back(tensor_args.input_tensor.buffer());
        writer_args.push_back(rows_tensor.buffer());
        writer_args.append(
            {member.activated_columns,
             member.first_column,
             member.output_groups,
             member.first_output_tile,
             member.output_tiles,
             member.group,
             core_index % plan.group_cores});
        writer_args.append(core_mask(group_cores, grid));
        desc.kernels[kRowsWriterKernelIdx].emplace_runtime_args(member.core, writer_args);

        desc.kernels[kRowsComputeKernelIdx].emplace_runtime_args(
            member.core,
            {member.column_groups,
             member.output_groups,
             member.ring_position,
             member.half_column ? 1u : 0u,
             member.half_output ? 1u : 0u});
    }
    return desc;
}

}  // namespace

tt::tt_metal::WorkloadDescriptor MoEExpertRowsFactory::create_workload_descriptor(
    const MoEExpertRowsParams& args,
    const MoEExpertRowsInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    tt::tt_metal::WorkloadDescriptor workload;
    for (const auto& coord : tensor_coords.coords()) {
        workload.programs.push_back(
            {ttnn::MeshCoordinateRange(coord),
             build_expert_rows_program_descriptor(args, coord, tensor_args, tensor_return_value)});
    }
    return workload;
}

void MoEExpertRowsFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const MoEExpertRowsParams& args,
    const MoEExpertRowsInputs& tensor_args,
    std::vector<ttnn::Tensor>& /*tensor_return_value*/,
    const std::optional<ttnn::MeshCoordinate>& /*coord*/) {
    // Buffer addresses are bindings. The layer's weight addresses are offsets that layer_id moves.
    const auto& shape = args.shape;
    const uint32_t ring = shard_banks_of(tensor_args.w0_w1_tensor).size();
    const PreparedLayout layout =
        prepared_layout(shape.hidden_tiles(), shape.intermediate_tiles(), shape.has_bias, ring);
    const auto layer_weights = layer_weight_args(args, tensor_args, layout);
    auto& reader_args_by_core = tt::tt_metal::GetRuntimeArgs(program, kRowsReaderKernelIdx);
    for (auto& reader_args_column : reader_args_by_core) {
        for (auto& reader_args : reader_args_column) {
            if (reader_args.size() == 0) {
                continue;
            }
            for (uint32_t weight = 0; weight < layer_weights.size(); ++weight) {
                reader_args[kReaderLayerWeightArgs + weight] = layer_weights[weight];
            }
        }
    }
}

//-----------------------------------------------------------------------------------------------------------------
// The placement program: moe_compute's outputs from the rows and the routing table. ComputeOnly: place.cpp on the
// combine cores writes the metadata and the double buffer. SingleDevice / FullCcl: place.cpp writes the metadata from
// the feeder cores (one per ring position) and signals the combine; feed.cpp on the same cores streams each expert's
// rows into the double buffer with the ring program's handshake; the combine cores run selective_reduce_combine's own
// kernels (with FullCcl, its fabric path and mux cores).
//-----------------------------------------------------------------------------------------------------------------
namespace {

// The placement program on the device at mesh_coordinate.
tt::tt_metal::ProgramDescriptor build_place_program_descriptor(
    const MoEComputeParams& args,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const std::vector<ttnn::MeshCoordinate>& all_mesh_coordinates,
    const MoEComputeInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value,
    const std::optional<GlobalSemaphore>& init_barrier_semaphore,
    const std::optional<GlobalSemaphore>& final_barrier_semaphore) {
    const bool combine = args.path != MoEComputePath::ComputeOnly;
    const bool ccl = args.path == MoEComputePath::FullCcl;
    TT_FATAL(
        tensor_args.expert_rows_tensor.has_value() && tensor_args.expert_rows_table_tensor.has_value(),
        "moe_compute expert rows placement program: needs the expert rows program's outputs");
    TT_FATAL(
        !ccl || (init_barrier_semaphore.has_value() && final_barrier_semaphore.has_value()),
        "moe_compute expert rows placement program: FullCcl needs the combine's barrier semaphores");
    tt::tt_metal::ProgramDescriptor desc;
    auto* mesh_device = tensor_args.tilize_input_tensor.device();
    const ttnn::Tensor& rows = *tensor_args.expert_rows_tensor;
    const ttnn::Tensor& table = *tensor_args.expert_rows_table_tensor;
    const auto& input_shape = tensor_args.tilize_input_tensor.logical_shape();
    const uint32_t hidden_size = input_shape[-1];
    const uint32_t tokens = input_shape[0] * input_shape[1];
    const uint32_t top_k = tensor_args.tilize_expert_indices_tensor.logical_shape()[-1];
    const uint32_t local_experts = tensor_args.matmul_w0_w1_tensor.logical_shape()[2];
    const uint32_t height_shards = args.num_token_parallel_cores, width_shards = args.num_data_parallel_cores;
    const PlaceLayout place = place_layout(tokens, top_k, local_experts);
    TT_FATAL(
        table.buffer()->size() >= place.table.bytes,
        "moe_compute expert rows placement program: routing table is {} B, expected {}",
        table.buffer()->size(),
        place.table.bytes);

    const ttnn::Tensor& counts_out = tensor_return_value[kExpertTokenCounts];
    const ttnn::Tensor& activation_out = tensor_return_value[kExpertActivation];
    const ttnn::Tensor& e_t_out = tensor_return_value[kETPages];
    const ttnn::Tensor& dense_out = tensor_return_value[kTilizeOutput];
    const uint32_t dense_cores = dense_out.memory_config().shard_spec()->grid.num_cores();
    const uint32_t combine_stride =
        combine_row_stride(dense_out.logical_shape().volume() / dense_cores, hidden_size, width_shards, height_shards);
    const uint32_t segment_bytes = hidden_size / width_shards * kBf16Bytes;  // a row's part in one width shard
    const CoreCoord grid = mesh_device->compute_with_storage_grid_size();
    const GridCoords virtual_coords = grid_coords(mesh_device, grid);
    auto virt = [mesh_device](const CoreCoord& logical_core) {
        return mesh_device->worker_core_from_logical_core(logical_core);
    };

    std::vector<CoreCoord> cores;  // place.cpp's cores; with the combine also feed.cpp's, in ring order
    uint32_t done_sem = 0, meta_sem = 0;
    CoreCoord lead, sync;
    if (!combine) {
        cores = get_moe_combine_cores(
            mesh_device, height_shards, width_shards, hidden_size, CoreRangeSet{}, args.bh_ring_size);
    } else {
        TT_FATAL(
            args.combine_params.has_value() && args.combine_params->local_combine == !ccl &&
                tensor_return_value.size() == kOutputsWithCombine,
            "moe_compute expert rows placement program: SingleDevice / FullCcl need their combine parameters and {} "
            "outputs; got combine parameters {}, local_combine {}, {} outputs",
            kOutputsWithCombine,
            args.combine_params.has_value(),
            args.combine_params.has_value() && args.combine_params->local_combine,
            tensor_return_value.size());
        const auto selection = ttnn::operations::ccl::common::select_moe_compute_cores(
            mesh_device,
            height_shards,
            width_shards,
            hidden_size,
            args.combine_params->mux_core_range_set,
            args.bh_ring_size);
        cores = selection.matmul_cores;
        const uint32_t ring = cores.size();
        const auto& combine_params = *args.combine_params;
        TT_FATAL(
            combine_params.worker_cores.size() >= height_shards * width_shards,
            "moe_compute expert rows placement program: {} combine cores, need {}",
            combine_params.worker_cores.size(),
            height_shards * width_shards);
        const std::vector<CoreCoord> combine_cores(
            combine_params.worker_cores.begin(), combine_params.worker_cores.begin() + height_shards * width_shards);
        const CoreRangeSet feed_set = CoreRangeSet(ttsl::Span<const CoreCoord>(cores));
        const CoreRangeSet combine_set = CoreRangeSet(ttsl::Span<const CoreCoord>(combine_cores));
        // the combine reader multicasts its metadata semaphore over its cores' bounding box
        meta_sem = add_semaphore_descriptor(desc, CoreRangeSet(combine_set.bounding_box()).merge(feed_set), 0);
        const uint32_t sync_sem = add_semaphore_descriptor(desc, combine_set.merge(feed_set), 0);
        done_sem = add_semaphore_descriptor(desc, feed_set, 0);
        lead = virt(cores[0]);
        sync = virt(combine_cores[0]);

        const ttnn::experimental::prim::SelectiveReduceCombineTensors combine_tensors{
            .dense_input_tensor = tensor_return_value[kMatmulOutput],
            .dense_activations_tensor = activation_out,
            .dense_token_maps_tensor = e_t_out,
            .dense_token_counts_tensor = counts_out,
            .optional_output_tensor = tensor_args.optional_output_tensor};
        // Each combine column (width shard) is fed by the feeders whose W2 width slice overlaps it; with a ring size
        // that is not a multiple of the column count a slice straddles columns (as in the ring program's dm1)
        const uint32_t hidden_tiles = hidden_size / tt::constants::TILE_WIDTH;
        const uint32_t shard_width_tiles = hidden_tiles / width_shards;
        const auto layouts =
            shard_layouts(hidden_tiles, args.intermediate_size / tt::constants::TILE_WIDTH, args.has_bias, ring);
        std::vector<std::vector<CoreCoord>> feeders_by_column(width_shards);
        for (uint32_t position = 0; position < ring; ++position) {
            const uint32_t begin =
                moe_ring::w2_combine_col_begin(layouts[position].first_output_tile, shard_width_tiles);
            const uint32_t end = moe_ring::w2_combine_col_end(
                layouts[position].first_output_tile, layouts[position].output_tiles, shard_width_tiles);
            TT_FATAL(
                begin >= end || end <= width_shards,
                "moe_compute expert rows placement program: ring position {} feeds combine columns [{}, {}) of {}",
                position,
                begin,
                end,
                width_shards);
            for (uint32_t column = begin; column < end; ++column) {
                feeders_by_column[column].push_back(cores[position]);
            }
        }
        const uint32_t combine_writer_kernel_idx = append_selective_reduce_combine_to_descriptor(
            desc,
            combine_params,
            mesh_coordinate,
            all_mesh_coordinates,
            combine_tensors,
            tensor_return_value[kCombineOutput],
            init_barrier_semaphore,
            final_barrier_semaphore,
            meta_sem,
            sync_sem,
            feeders_by_column);
        TT_FATAL(
            combine_writer_kernel_idx == kPlaceCombineWriterKernelIdx,
            "moe_compute expert rows placement program: combine writer is kernel {}, expected {}",
            combine_writer_kernel_idx,
            kPlaceCombineWriterKernelIdx);

        // feed.cpp: each ring position's feeder streams the W2 output columns that position owns in the ring program
        const uint32_t tile_width_bytes = tt::constants::TILE_WIDTH * rows.element_size();
        const FeedLayout feed = feed_layout(place, layouts, tile_width_bytes);
        constexpr uint32_t kFeedCb = 1;
        create_cb(desc, feed_set, kFeedCb, feed.bytes, tt::DataFormat::UInt32, feed.bytes);
        tt::tt_metal::KernelDescriptor::NamedCompileTimeArgs feed_ct = {
            {"cb_scratch", kFeedCb},
            {"local_experts", local_experts},
            {"height_shards", height_shards},
            {"width_shards", width_shards},
            {"shard_width_tiles", shard_width_tiles},
            {"tile_width_bytes", tile_width_bytes},
            {"half_bytes", combine_stride * segment_bytes},
            {"row_bytes", hidden_size * kBf16Bytes},
            {"batch_rows", feed.batch_rows},
            {"slice_stride", feed.slice_stride},
            {"table_counts", place.table.counts},
            {"table_offsets", place.table.offsets},
            {"table_rows", place.table.rows},
            {"table_bytes", place.table.bytes},
            {"scratch_stage", feed.scratch_stage},
            {"combine_sync_semaphore_id", sync_sem},
        };
        std::vector<uint32_t> feed_pos;
        tt::tt_metal::TensorAccessorArgs(*rows.buffer()).append_to(feed_pos);
        tt::tt_metal::TensorAccessorArgs(*table.buffer()).append_to(feed_pos);
        const uint32_t feed_kernel = desc.kernels.size();
        desc.kernels.push_back(tt::tt_metal::KernelDescriptor{
            .kernel_source = std::string(kKernelDir) + "feed.cpp",
            .core_ranges = feed_set,
            .compile_time_args = std::move(feed_pos),
            .named_compile_time_args = std::move(feed_ct),
            .config =
                tt::tt_metal::DataMovementConfigDescriptor{
                    .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
                    .noc = tt::tt_metal::NOC::NOC_1,
                },
        });
        for (uint32_t position = 0; position < ring; ++position) {
            const uint32_t first_shard = layouts[position].first_output_tile / shard_width_tiles;
            const uint32_t last_shard =
                (layouts[position].first_output_tile + layouts[position].output_tiles - 1) / shard_width_tiles;
            TT_FATAL(
                last_shard - first_shard < expert_rows::max_slice_shards,
                "moe_compute expert rows placement program: a feeder slice spans more than {} width shards: ring "
                "position {} covers width shards {}..{}",
                expert_rows::max_slice_shards,
                position,
                first_shard,
                last_shard);
            tt::tt_metal::KernelDescriptor::RTArgList feed_args;
            feed_args.push_back(rows.buffer());
            feed_args.push_back(table.buffer());
            feed_args.push_back(dense_out.buffer());
            feed_args.append({layouts[position].first_output_tile, layouts[position].output_tiles});
            for (const auto& combine_core : combine_cores) {
                const CoreCoord virtual_core = virt(combine_core);
                feed_args.push_back(virtual_core.x);
                feed_args.push_back(virtual_core.y);
            }
            desc.kernels[feed_kernel].emplace_runtime_args(cores[position], feed_args);
        }
    }

    const CoreRangeSet place_set = CoreRangeSet(ttsl::Span<const CoreCoord>(cores));
    create_cb(desc, place_set, 0, place.scratch_bytes, tt::DataFormat::UInt32, place.scratch_bytes);

    tt::tt_metal::KernelDescriptor::NamedCompileTimeArgs place_ct = {
        {"cb_scratch", 0},
        {"tokens", tokens},
        {"top_k", top_k},
        {"local_experts", local_experts},
        {"height_shards", height_shards},
        {"segment_bytes", segment_bytes},
        {"half_bytes", combine_stride * segment_bytes},
        {"count_words", place.count_words},
        {"e_t_page_bytes", place.e_t_page_bytes},
        {"activation_row_bytes", place.activation_row_bytes},
        {"grid_w", grid.x},
        {"grid_h", grid.y},
        {"program_cores", static_cast<uint32_t>(cores.size())},
        {"row_bytes", hidden_size * kBf16Bytes},
        {"place_dense", combine ? 0u : 1u},
        {"done_semaphore_id", done_sem},
        {"metadata_semaphore_id", meta_sem},
        {"table_counts", place.table.counts},
        {"table_offsets", place.table.offsets},
        {"table_rows", place.table.rows},
        {"table_entry_experts", place.table.entry_experts},
        {"table_entry_tokens", place.table.entry_tokens},
        {"table_entry_scores", place.table.entry_scores},
        {"table_entry_k", place.table.entry_k},
        {"table_bytes", place.table.bytes},
        {"scratch_counts", place.scratch_counts},
        {"scratch_offsets", place.scratch_offsets},
        {"scratch_e_t", place.scratch_e_t},
        {"scratch_activation", place.scratch_activation},
        {"scratch_count_row", place.scratch_count_row},
        {"scratch_table", place.scratch_table},
    };
    std::vector<uint32_t> place_ct_pos;
    tt::tt_metal::TensorAccessorArgs(*rows.buffer()).append_to(place_ct_pos);
    tt::tt_metal::TensorAccessorArgs(*e_t_out.buffer()).append_to(place_ct_pos);
    tt::tt_metal::TensorAccessorArgs(*activation_out.buffer()).append_to(place_ct_pos);
    tt::tt_metal::TensorAccessorArgs(*table.buffer()).append_to(place_ct_pos);
    const uint32_t place_kernel = desc.kernels.size();
    desc.kernels.push_back(tt::tt_metal::KernelDescriptor{
        .kernel_source = std::string(kKernelDir) + "place.cpp",
        .core_ranges = place_set,
        .compile_time_args = std::move(place_ct_pos),
        .named_compile_time_args = std::move(place_ct),
        .config =
            tt::tt_metal::DataMovementConfigDescriptor{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt::tt_metal::NOC::NOC_0,
            },
    });
    for (uint32_t core_index = 0; core_index < cores.size(); ++core_index) {
        tt::tt_metal::KernelDescriptor::RTArgList place_args;
        place_args.push_back(rows.buffer());
        place_args.push_back(e_t_out.buffer());
        place_args.push_back(activation_out.buffer());
        place_args.push_back(counts_out.buffer());
        place_args.push_back(dense_out.buffer());
        place_args.push_back(table.buffer());
        place_args.append(
            {combine ? 0 : core_index / width_shards,
             combine ? 0 : core_index % width_shards,
             core_index,
             static_cast<uint32_t>(lead.x),
             static_cast<uint32_t>(lead.y),
             static_cast<uint32_t>(sync.x),
             static_cast<uint32_t>(sync.y)});
        place_args.append(virtual_coords.xs);
        place_args.append(virtual_coords.ys);
        desc.kernels[place_kernel].emplace_runtime_args(cores[core_index], place_args);
    }
    return desc;
}

}  // namespace

tt::tt_metal::WorkloadDescriptor MoEComputePlaceFactory::create_workload_descriptor(
    const MoEComputeParams& args,
    const MoEComputeInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    tt::tt_metal::WorkloadDescriptor workload;
    std::optional<GlobalSemaphore> init_barrier_semaphore;
    std::optional<GlobalSemaphore> final_barrier_semaphore;
    if (args.path == MoEComputePath::FullCcl) {
        std::tie(init_barrier_semaphore, final_barrier_semaphore) = add_selective_reduce_combine_barrier_semaphores(
            workload,
            tensor_args.tilize_input_tensor.device(),
            CoreRangeSet(ttsl::Span<const CoreCoord>(args.combine_params->worker_cores)),
            args.combine_params->optional_cross_device_semaphore);
    }
    const std::vector<ttnn::MeshCoordinate> all_coordinates = tensor_coords.coords();
    for (const auto& coord : all_coordinates) {
        workload.programs.push_back(
            {ttnn::MeshCoordinateRange(coord),
             build_place_program_descriptor(
                 args,
                 coord,
                 all_coordinates,
                 tensor_args,
                 tensor_return_value,
                 init_barrier_semaphore,
                 final_barrier_semaphore)});
    }
    return workload;
}

void MoEComputePlaceFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const MoEComputeParams& args,
    const MoEComputeInputs& /*tensor_args*/,
    std::vector<ttnn::Tensor>& /*tensor_return_value*/,
    const std::optional<ttnn::MeshCoordinate>& /*coord*/) {
    // Buffer addresses are bindings. Only the caller-owned cross-device semaphore address can move.
    if (args.path != MoEComputePath::FullCcl || !args.combine_params->optional_cross_device_semaphore.has_value()) {
        return;
    }
    override_selective_reduce_combine_cross_device_semaphore(
        program, kPlaceCombineWriterKernelIdx, *args.combine_params->optional_cross_device_semaphore);
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::vector<ttnn::Tensor> moe_expert_rows(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& expert_indices_tensor,
    const ttnn::Tensor& expert_scores_tensor,
    const ttnn::Tensor& expert_mapping_tensor,
    const ttnn::Tensor& w0_w1_tensor,
    const ttnn::Tensor& w2_tensor,
    const ttnn::experimental::prim::MoEExpertRowsParams& params) {
    using OperationType = ttnn::experimental::prim::MoEExpertRowsDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        params,
        OperationType::tensor_args_t{
            .input_tensor = input_tensor,
            .expert_indices_tensor = expert_indices_tensor,
            .expert_scores_tensor = expert_scores_tensor,
            .expert_mapping_tensor = expert_mapping_tensor,
            .w0_w1_tensor = w0_w1_tensor,
            .w2_tensor = w2_tensor});
}

}  // namespace ttnn::prim
