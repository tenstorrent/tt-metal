// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "grouped_matmul_w2_device_operation.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/nomic_embed_text_v2_moe/grouped_matmul_w1/device/grouped_matmul_w1_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::experimental::nomic_embed_text_v2_moe {

namespace {

using tt::tt_metal::DataType;
using tt::tt_metal::Layout;
using tt::tt_metal::TensorMemoryLayout;

constexpr uint32_t kTileSize = tt::constants::TILE_HEIGHT;
constexpr uint32_t kGridX = 11;
constexpr uint32_t kGridY = 10;
constexpr uint32_t kShares = 8;
constexpr uint32_t kShareTiles = 3;
// The data-movement kernels keep a few words per expert on their stacks.
constexpr uint32_t kMaxExperts = 64;
// Column lanes once each lane has this many blocks: below it the forwarders, which pace the weight stream, cost more
// than the extra lanes save (2x512, 71 blocks: 78.7 us with 10 lanes against 83.6 with 13).
constexpr uint32_t kColumnLaneBlocks = 12 * 13;

}  // namespace

CoreCoord GroupedMatmulW2Map::lane_core(uint32_t lane, uint32_t index) const {
    return lane < kRowLanes ? CoreCoord(index, lane) : CoreCoord(lane - 2, index + 2);
}

CoreCoord GroupedMatmulW2Map::input_sender(uint32_t lane) const {
    if (lane < kDedicatedSenderLanes) {
        return CoreCoord(kShares, lane);
    }
    return lane < kRowLanes ? CoreCoord(kShares - 1, lane) : CoreCoord(lane - 2, kGridY - 1);
}

uint32_t GroupedMatmulW2Map::share_of(const CoreCoord& core) const {
    if (core.x < kShares) {
        return core.x;
    }
    return core.y == 2 ? kShares - 1 : core.y - 3;
}

CoreCoord GroupedMatmulW2Map::forwarder(uint32_t share) const {
    return share == kShares - 1 ? CoreCoord(kShares - 1, 1) : CoreCoord(share, share + 3);
}

uint32_t GroupedMatmulW2Map::forward_row(uint32_t share) const { return share == kShares - 1 ? 2 : share + 3; }

std::vector<CoreCoord> GroupedMatmulW2Map::compute_cores() const {
    std::vector<CoreCoord> cores;
    for (uint32_t lane = 0; lane < num_lanes; ++lane) {
        for (uint32_t index = 0; index < lane_cores; ++index) {
            cores.push_back(lane_core(lane, index));
        }
    }
    return cores;
}

std::vector<CoreCoord> GroupedMatmulW2Map::input_shard_cores(uint32_t input_pieces) const {
    std::vector<CoreCoord> cores;
    const uint32_t rounds = lane_cores / input_pieces;
    for (uint32_t turn = 0; turn < rounds * num_lanes; ++turn) {
        for (uint32_t piece = 0; piece < input_pieces; ++piece) {
            cores.push_back(lane_core(turn % num_lanes, input_pieces * (turn / num_lanes) + piece));
        }
    }
    return cores;
}

GroupedMatmulW2Map grouped_matmul_w2_map(uint32_t input_blocks, const CoreCoord& grid) {
    TT_FATAL(
        grid.x == kGridX && grid.y == kGridY,
        "grouped_matmul_w2: the core map is built for an {} x {} compute grid, got {} x {}",
        kGridX,
        kGridY,
        grid.x,
        grid.y);
    const bool column_lanes = input_blocks >= kColumnLaneBlocks;
    return GroupedMatmulW2Map{
        .num_lanes = GroupedMatmulW2Map::kRowLanes + (column_lanes ? kGridX - kShares : 0u),
        .lane_cores = kShares,
        .num_shares = kShares};
}

tt::tt_metal::MemoryConfig grouped_matmul_w2_input_memory_config(uint32_t rows, uint32_t k, const CoreCoord& grid) {
    TT_FATAL(
        rows % kTileSize == 0 && k % (kGroupedMatmulW2InputPieces * kTileSize) == 0,
        "grouped_matmul_w2: the input must be whole 32-row blocks of K in {} whole-tile pieces, got {} x {}",
        kGroupedMatmulW2InputPieces,
        rows,
        k);
    const auto map = grouped_matmul_w2_map(rows / kTileSize, grid);
    std::vector<CoreRange> ranges;
    for (const auto& core : map.input_shard_cores(kGroupedMatmulW2InputPieces)) {
        ranges.emplace_back(core);
    }
    // Single-core ranges keep their order: ROUND_ROBIN_1D deals the shards in it.
    return tt::tt_metal::MemoryConfig(
        tt::tt_metal::BufferType::L1,
        tt::tt_metal::NdShardSpec(
            ttnn::Shape({kTileSize, k / kGroupedMatmulW2InputPieces}),
            CoreRangeSet(std::move(ranges)),
            tt::tt_metal::ShardOrientation::ROW_MAJOR,
            tt::tt_metal::ShardDistributionStrategy::ROUND_ROBIN_1D));
}

uint32_t grouped_matmul_w2_circular_buffer_bytes(
    uint32_t k, uint32_t n, uint32_t first_slots, uint32_t second_slots, uint32_t num_experts, const CoreCoord& grid) {
    const auto chunks = grouped_weight_chunks(k / kTileSize, n / kTileSize, 1, false, 0, grid.x);
    const tt::tt_metal::Tile tile({kTileSize, kTileSize});
    const uint32_t bf8_tile = tile.get_tile_size(tt::DataFormat::Bfp8_b);
    const uint32_t bf16_tile = tile.get_tile_size(tt::DataFormat::Float16_b);
    const uint32_t half_tiles = k / kTileSize / 2 * chunks.share_n_tiles;
    const uint32_t input = kGroupedMatmulW2InputSlots * (k / kTileSize / kGroupedMatmulW2InputPieces) * bf8_tile;
    const uint32_t weights = (first_slots + second_slots) * half_tiles * bf8_tile;
    const uint32_t output = kGroupedMatmulW2OutputSlots * chunks.share_n_tiles * bf16_tile;
    // The schedule (a word an expert) and two counts pages (the counts tensor's page).
    const uint32_t expert_words = num_experts * static_cast<uint32_t>(sizeof(uint32_t));
    const uint32_t words = tt::round_up(expert_words, tt::tt_metal::hal::get_l1_alignment()) +
                           2 * tt::round_up(expert_words, tt::tt_metal::hal::get_dram_alignment());
    return input + weights + output + words;
}

void GroupedMatmulW2DeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    const auto& weight = tensor_args.weight;
    const auto& counts = tensor_args.expert_counts;
    const auto grid = input.device()->compute_with_storage_grid_size();

    TT_FATAL(
        input.dtype() == DataType::BFLOAT8_B && input.layout() == Layout::TILE,
        "grouped_matmul_w2: input_tensor must be BFLOAT8_B TILE, got {} {}",
        input.dtype(),
        input.layout());
    const auto& shape = input.logical_shape();
    TT_FATAL(
        shape.rank() == 4 && shape[0] == 1 && shape[1] == 1 && shape[2] % kTileSize == 0,
        "grouped_matmul_w2: input_tensor must be 1 x 1 x rows x K in whole 32-row blocks, got {}",
        shape);
    const uint32_t rows = shape[2];
    const uint32_t k = shape[3];
    TT_FATAL(
        input.memory_config() == grouped_matmul_w2_input_memory_config(rows, k, grid),
        "grouped_matmul_w2: input_tensor must be in grouped_matmul_w2_input_memory_config's layout for {} rows",
        rows);

    TT_FATAL(
        counts.device() == input.device() && counts.dtype() == DataType::UINT32 &&
            counts.layout() == Layout::ROW_MAJOR &&
            counts.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "grouped_matmul_w2: expert_counts must be UINT32, ROW_MAJOR and interleaved on the input's device, as "
        "routing_metadata writes it");
    const auto& counts_shape = counts.logical_shape();
    TT_FATAL(
        counts_shape.rank() == 2 && counts_shape[0] == 1 && counts_shape[1] >= 1 && counts_shape[1] <= kMaxExperts,
        "grouped_matmul_w2: expert_counts must be 1 x num_experts, num_experts 1 to {}, got {}",
        kMaxExperts,
        counts_shape);
    const uint32_t num_experts = counts_shape[1];

    const uint32_t n = attributes.weight_n;
    TT_FATAL(n > 0 && n % kTileSize == 0, "grouped_matmul_w2: weight_n must be whole tiles, got {}", n);
    const uint32_t k_tiles = k / kTileSize;
    const uint32_t n_tiles = n / kTileSize;
    const auto chunks =
        grouped_weight_chunks(k_tiles, n_tiles, num_experts, false, attributes.weight_chunk_tiles, grid.x);
    TT_FATAL(
        chunks.num_shares == kShares && chunks.share_n_tiles == kShareTiles && chunks.last_share_n_tiles == kShareTiles,
        "grouped_matmul_w2: the layout takes N as {} shares of {} tiles, got N = {}",
        kShares,
        kShareTiles,
        n);
    const uint32_t half_tiles = k_tiles / 2 * kShareTiles;
    TT_FATAL(
        k_tiles % (2 * kGroupedMatmulW2InputPieces) == 0 && attributes.weight_chunk_tiles > 0 &&
            half_tiles % attributes.weight_chunk_tiles == 0,
        "grouped_matmul_w2: weight_chunk_tiles {} must split each K half of a {}-tile slice evenly",
        attributes.weight_chunk_tiles,
        2 * half_tiles);
    TT_FATAL(
        weight.device() == input.device() && weight.dtype() == DataType::BFLOAT8_B && weight.layout() == Layout::TILE &&
            weight.memory_config().buffer_type() == tt::tt_metal::BufferType::DRAM &&
            weight.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED,
        "grouped_matmul_w2: weight must be BFLOAT8_B TILE, WIDTH_SHARDED in DRAM, as prepare_grouped_matmul_w2_weight "
        "writes it");
    const auto& weight_shape = weight.logical_shape();
    TT_FATAL(
        weight_shape.rank() == 4 && weight_shape[0] == 1 && weight_shape[1] == 1 && weight_shape[2] == kTileSize &&
            weight_shape[3] == kGroupedWeightBanks * chunks.shard_tiles * kTileSize,
        "grouped_matmul_w2: a weight for K = {}, N = {}, {} experts and chunks of {} tiles is 1 x 1 x 32 x {}, got {}; "
        "prepare it with the same arguments",
        k,
        n,
        num_experts,
        attributes.weight_chunk_tiles,
        kGroupedWeightBanks * chunks.shard_tiles * kTileSize,
        weight_shape);
    const auto& shard_spec = weight.memory_config().shard_spec();
    TT_FATAL(
        shard_spec.has_value() && shard_spec->shape[1] == chunks.shard_tiles * kTileSize &&
            shard_spec->grid.num_cores() == kGroupedWeightBanks,
        "grouped_matmul_w2: the weight's shards must be {} tiles wide, one a DRAM bank",
        chunks.shard_tiles);

    // A receiver grants piece p + 2 once p is in, ahead of forwarding p: p + 2 must not share p's place in its ring.
    const uint32_t ring_pieces = half_tiles / attributes.weight_chunk_tiles;
    TT_FATAL(
        attributes.first_slots * ring_pieces >= 3 && attributes.second_slots * ring_pieces >= 3,
        "grouped_matmul_w2: each weight ring must hold at least 3 pieces; {} and {} slots of {} pieces do not",
        attributes.first_slots,
        attributes.second_slots,
        ring_pieces);
    TT_FATAL(
        get_dest_reg_count(attributes.compute_kernel_config) >= kShareTiles,
        "grouped_matmul_w2: a share's {} output tiles must fit DST with this compute_kernel_config",
        kShareTiles);
    TT_FATAL(
        attributes.memory_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "grouped_matmul_w2: memory_config must be interleaved, got {}",
        attributes.memory_config.memory_layout());
}

GroupedMatmulW2DeviceOperation::spec_return_value_t GroupedMatmulW2DeviceOperation::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    using tt::tt_metal::PageConfig;
    using tt::tt_metal::TensorLayout;
    const auto& shape = tensor_args.input.logical_shape();
    return tt::tt_metal::TensorSpec(
        ttnn::Shape({1, 1, shape[2], attributes.weight_n}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::ROW_MAJOR), attributes.memory_config));
}

GroupedMatmulW2DeviceOperation::tensor_return_value_t GroupedMatmulW2DeviceOperation::create_output_tensors(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(attributes, tensor_args), tensor_args.input.device());
}

}  // namespace ttnn::experimental::nomic_embed_text_v2_moe

namespace ttnn::prim {

Tensor grouped_matmul_w2(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& expert_counts,
    const tt::tt_metal::MemoryConfig& memory_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    uint32_t weight_chunk_tiles,
    uint32_t weight_n,
    uint32_t first_slots,
    uint32_t second_slots) {
    using Operation = ttnn::experimental::nomic_embed_text_v2_moe::GroupedMatmulW2DeviceOperation;
    return ttnn::device_operation::launch<Operation>(
        Operation::operation_attributes_t{
            .memory_config = memory_config,
            .compute_kernel_config = compute_kernel_config,
            .weight_chunk_tiles = weight_chunk_tiles,
            .weight_n = weight_n,
            .first_slots = first_slots,
            .second_slots = second_slots},
        Operation::tensor_args_t{.input = input, .weight = weight, .expert_counts = expert_counts});
}

}  // namespace ttnn::prim
