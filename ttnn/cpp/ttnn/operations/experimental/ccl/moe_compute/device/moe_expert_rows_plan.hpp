// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The host side of moe_compute's expert rows path, shared by its sources: constants, the layouts and the plan of the
// expert rows program (program 1), the placement program's (program 2) layouts.
//   moe_expert_rows_plan.cpp               program 1's plan search and the layouts
//   moe_expert_rows_select.cpp             the cost model; the choice between the ring and the expert rows path
//   moe_expert_rows.cpp                    the expert rows device operation and program 1's factory
//   moe_compute_place_program_factory.cpp  program 2's factory
#pragma once

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-umd/types/arch.hpp>

#include "kernels/expert_rows/expert_rows_common.h"
#include "moe_expert_rows.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim::expert_rows_host {

using expert_rows::GroupRun;

constexpr uint32_t kWordBytes = sizeof(uint32_t);
constexpr uint32_t kBf16Bytes = sizeof(uint16_t);
constexpr uint32_t kTileBytesBf16 = tt::constants::TILE_HW * kBf16Bytes;
constexpr uint32_t kGroupTiles = expert_rows::group_tiles;
constexpr uint32_t kColumnsPerGroup = expert_rows::group_columns;  // each column gives one activated tile
constexpr uint32_t kMaxTopK = 32;  // rows_reader keeps one u16 per k of a token on its stack (its static_assert)
// the routing table holds rows and tokens in u16
constexpr uint32_t kMaxRoutedRows = std::numeric_limits<uint16_t>::max();
constexpr const char* kKernelDir = "ttnn/cpp/ttnn/operations/experimental/ccl/moe_compute/device/kernels/expert_rows/";

// moe_compute's outputs (MoEComputeDeviceOperation::compute_output_specs)
enum MoEComputeOutput : uint32_t {
    kExpertTokenCounts = 0,
    kExpertActivation = 1,
    kETPages = 2,
    kTilizeOutput = 3,  // the double buffer that place.cpp fills
    kMatmulOutput = 4,  // aliases kTilizeOutput
    kCombineOutput = 5,
};

inline bool expert_rows_arch(tt::ARCH arch) { return arch == tt::ARCH::BLACKHOLE || arch == tt::ARCH::WORMHOLE_B0; }

// Format of the partials between x or W2 chunks: what DEST holds.
inline tt::DataFormat partials_format(bool fp32_dest_acc_en) {
    return fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
}

inline uint32_t aligned_page(const ttnn::Tensor& tensor) { return tensor.buffer()->aligned_page_size(); }

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

PreparedLayout prepared_layout(uint32_t hidden_tiles, uint32_t intermediate_tiles, bool has_bias, uint32_t num_shards);

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
    uint32_t hidden_tiles, uint32_t intermediate_tiles, bool has_bias, uint32_t num_shards);

// Byte offsets of the expert rows reader's routing scratch (cb_routing). [table, jobs) is the routing table block the
// root core writes to DRAM for the placement program.
struct RoutingLayout {
    uint32_t own_mapping = 0, source_mappings = 0, token_ids = 0, token_scores = 0, ctl = 0;
    uint32_t active_experts = 0, expert_rows = 0, expert_first_rows = 0;
    uint32_t table = 0, row_tokens = 0;  // the table block (expert_rows::table_layout) and its row tokens
    uint32_t jobs = 0, entry_areas = 0, size = 0;
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

// One core of the expert rows program: its ring position's W0/W1 column groups and W2 output groups (rows_reader.cpp),
// and the intermediate columns and output tiles they hold (rows_writer.cpp).
struct ExpertRowsCore {
    CoreCoord core;
    uint32_t ring_position = 0;  // shard index of the prepared weights = position in moe_compute's ring
    uint32_t bank = 0;           // DRAM bank holding that shard
    uint32_t expert_group = 0;   // jobs dealt by rows_reader.cpp
    uint32_t first_column_group = 0, column_groups = 0;
    bool half_column = false;  // the last column group is the shard's half block-column (one column, 2 tiles per K row)
    uint32_t first_column = 0, activated_columns = 0;  // real intermediate columns
    uint32_t first_output_group = 0, output_groups = 0;
    bool half_output = false;                          // the last output group is the shard's last, two tiles wide
    uint32_t first_output_tile = 0, output_tiles = 0;  // real output tiles
    GroupRun column_run() const { return {column_groups, half_column}; }
    GroupRun output_run() const { return {output_groups, half_output}; }
};

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
};

// A plan: its buffering, core split and placement.
struct ExpertRowsPlan : Buffering {
    uint32_t readers_per_bank = 0;
    uint32_t expert_groups = 0;
    uint32_t group_cores = 0;
    uint32_t activated_tiles = 0;  // cb_activated pages per row tile: one per intermediate column of the busiest core
    uint32_t block_tiles = 0;      // weight tiles per read block
    uint32_t block_packets = 0;    // NoC packets per read block
    uint32_t blocks_in_flight = 0;
    CbSizes cbs;
    RoutingLayout routing;
    std::vector<ExpertRowsCore> cores;  // group-major
};

// The plan, or why there is none.
using PlanOrRefusal = std::variant<ExpertRowsPlan, std::string>;

// The plan for `shape` with jobs of `row_tiles` row tiles on this device within `cb_budget` bytes of circular buffers
// per core, or why there is none. shard_banks[r]: DRAM bank of the prepared weights' shard r (= moe_compute ring
// position r).
PlanOrRefusal plan_expert_rows(
    ttnn::MeshDevice* mesh_device,
    const MoEExpertRowsShape& shape,
    const std::vector<uint32_t>& shard_banks,
    uint32_t cb_budget,
    uint32_t row_tiles,
    tt::DataFormat partials);

std::vector<uint32_t> shard_banks_of(const ttnn::Tensor& weights);

// Virtual coordinate of every grid column / row (the kernels rebuild a core's NoC address from its linear index).
struct GridCoords {
    std::vector<uint32_t> xs, ys;
};

GridCoords grid_coords(ttnn::MeshDevice* mesh_device, CoreCoord grid);

std::vector<uint32_t> core_mask(const std::vector<CoreCoord>& cores, CoreCoord grid);

// The placement program's view: the routing table block and place.cpp's scratch layout.
struct PlaceLayout {
    expert_rows::TableLayout table;
    uint32_t scratch_e_t = 0, scratch_activation = 0;
    uint32_t scratch_count_row = 0, scratch_table = 0, scratch_bytes = 0;
    uint32_t count_words = 0, e_t_entry_words = 0, e_t_page_bytes = 0, activation_row_bytes = 0;
};

PlaceLayout place_layout(uint32_t tokens, uint32_t top_k, uint32_t local_experts);

// Rows of one double-buffer half on one core (output kTilizeOutput), in selective_reduce_combine's fused-source layout,
// which the ring's dm1 and the combine also use.
uint32_t double_buffer_half_rows(const tt::tt_metal::TensorSpec& dense, uint32_t hidden_size, uint32_t width_shards);

// rows_combine.cpp's scratch: the table head; the sender's row tokens and row k (a DRAM read keeps its source offset
// modulo part_alignment, so each slice has part_alignment bytes of slack); its rows in send order; two staging buffers
// of batch_rows rows.
struct CombineLayout {
    uint32_t sender_rows_max = 0, row_stride = 0, batch_rows = 0;
    uint32_t scratch_head = 0, scratch_tokens = 0, scratch_k = 0, scratch_order = 0, scratch_stage = 0, bytes = 0;
};

CombineLayout combine_layout(uint32_t tokens, uint32_t top_k, uint32_t hidden_size, uint32_t senders);

// The first rule of the expert rows path that `shape` breaks, with its values, or nullopt.
std::optional<std::string> shape_refusal(const MoEExpertRowsShape& shape);

// The cost model's cycles of one job of a plan, by which rows_reader.cpp deals the jobs to the expert groups: the
// weight stream of the expert groups' jobs in flight, one row tile, and a job of one row tile.
struct JobCycles {
    double stream = 0, row_tile = 0, single = 0;
};

JobCycles job_cycles(
    const MoEExpertRowsShape& shape,
    const ExpertRowsPlan& plan,
    uint32_t num_shards,
    tt::ARCH arch,
    tt::tt_metal::MathFidelity fidelity);

void create_cb(
    tt::tt_metal::ProgramDescriptor& desc,
    const CoreRangeSet& cores,
    uint32_t index,
    uint32_t bytes,
    tt::DataFormat format,
    uint32_t page_bytes);

}  // namespace ttnn::experimental::prim::expert_rows_host
