// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "grouped_matmul_w1_device_operation.hpp"

#include <algorithm>
#include <cstdlib>
#include <string_view>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::experimental::nomic_embed_text_v2_moe {

namespace {

using tt::tt_metal::DataType;
using tt::tt_metal::Layout;
using tt::tt_metal::TensorMemoryLayout;

constexpr uint32_t kTileSize = tt::constants::TILE_HEIGHT;
// The data-movement kernels keep up to four words per expert on their stacks, 1 KB at 64 experts.
constexpr uint32_t kMaxExperts = 64;
// Output tiles a subblock may hold, before the DST limit.
constexpr uint32_t kMaxSubblockWidth = 4;
// Mapping (i): dedicated input senders a lane, east of the compute grid, once a lane has this many blocks. Below it
// the lane's east compute core sends: at 7 blocks a lane (2x512) it was faster, at 26 (8x512) 20% slower.
constexpr uint32_t kDedicatedInputSenders = 3;
constexpr uint32_t kDedicatedSenderLaneBlocks = 12;
constexpr uint32_t kStagingBlocks = 2;
// The folded layout: senders a row lane, east of the shares, and a column lane, below its cores. With none a row
// lane (its east core sending, 13 lanes on 104 cores) w2 at 8x512 took 157.9 against 120.5 us: the row lanes'
// senders could not keep up and every lane waits on the shared weight slots.
const uint32_t kFoldedRowLaneSenders = std::getenv("GM_RLS") ? std::atoi(std::getenv("GM_RLS")) : 2;
constexpr uint32_t kFoldedInputSenders = 2;
// Weight reads in flight a share sender keeps. w2 (read transposed) asks for its whole first slice at once: 8x512
// 117.9 -> 116.5 us; at 8, w1 and fc1 were 3 and 0.5 us slower.
constexpr uint32_t kWeightDepthTransposed = 8;
constexpr uint32_t kWeightDepth = 4;
// One expert: subblock-outer over the lane's whole set of blocks when it has at most this many, else in groups.
constexpr uint32_t kWholeLaneBlocks = 4;
constexpr uint32_t kGroupBlocks = 4;

// The framework rejects host tensors before this runs.
void check_tile_operand(const Tensor& tensor, std::string_view name) {
    TT_FATAL(
        tensor.dtype() == DataType::BFLOAT16 || tensor.dtype() == DataType::BFLOAT8_B,
        "grouped_matmul_w1: {} must be BFLOAT16 or BFLOAT8_B, got {}",
        name,
        tensor.dtype());
    TT_FATAL(
        tensor.layout() == Layout::TILE, "grouped_matmul_w1: {} must be TILE layout, got {}", name, tensor.layout());
    TT_FATAL(
        tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "grouped_matmul_w1: {} must be interleaved, got {}",
        name,
        tensor.memory_config().memory_layout());
    const auto tile = tensor.tensor_spec().tile();
    TT_FATAL(
        tile.get_height() == kTileSize && tile.get_width() == kTileSize && !tile.get_transpose_of_faces() &&
            !tile.get_transpose_within_face(),
        "grouped_matmul_w1: {} must use plain 32 x 32 tiles",
        name);
    const auto& shape = tensor.logical_shape();
    TT_FATAL(
        shape.rank() == 4 && shape[0] == 1 && shape[1] == 1,
        "grouped_matmul_w1: {} must be 1 x 1 x rows x columns, got {}",
        name,
        shape);
    TT_FATAL(
        shape[2] % kTileSize == 0 && shape[3] % kTileSize == 0,
        "grouped_matmul_w1: {} must be whole tiles, got {}",
        name,
        shape);
}

// The input: (1, 1, rows, K) in whole tiles, or (B, 1, S, K), whose tile rows, the padding of each
// sequence included, are the rows.
void check_input(const Tensor& tensor) {
    TT_FATAL(
        tensor.dtype() == DataType::BFLOAT16 || tensor.dtype() == DataType::BFLOAT8_B,
        "grouped_matmul_w1: input_tensor must be BFLOAT16 or BFLOAT8_B, got {}",
        tensor.dtype());
    // Any memory layout: the reader addresses its tiles by page id through a TensorAccessor.
    TT_FATAL(
        tensor.layout() == Layout::TILE,
        "grouped_matmul_w1: input_tensor must be TILE layout, got {}",
        tensor.layout());
    const auto tile = tensor.tensor_spec().tile();
    TT_FATAL(
        tile.get_height() == kTileSize && tile.get_width() == kTileSize && !tile.get_transpose_of_faces() &&
            !tile.get_transpose_within_face(),
        "grouped_matmul_w1: input_tensor must use plain 32 x 32 tiles");
    const auto& shape = tensor.logical_shape();
    TT_FATAL(
        shape.rank() == 4 && shape[1] == 1, "grouped_matmul_w1: input_tensor must be B x 1 x rows x K, got {}", shape);
    TT_FATAL(shape[3] % kTileSize == 0, "grouped_matmul_w1: input_tensor's K must be whole tiles, got {}", shape);
}

}  // namespace

GroupedMatmulW1Layout GroupedMatmulW1DeviceOperation::compute_layout(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    const auto& weight = tensor_args.weight;
    check_input(input);
    if (attributes.weight_chunk_tiles == 0) {
        check_tile_operand(weight, "weight");
    } else {
        TT_FATAL(
            weight.dtype() == DataType::BFLOAT16 || weight.dtype() == DataType::BFLOAT8_B,
            "grouped_matmul_w1: weight must be BFLOAT16 or BFLOAT8_B, got {}",
            weight.dtype());
    }
    TT_FATAL(weight.device() == input.device(), "grouped_matmul_w1: weight must be on the device of input_tensor");
    uint32_t num_experts = 1;
    if (tensor_args.expert_counts.has_value()) {
        const auto& counts = *tensor_args.expert_counts;
        TT_FATAL(
            counts.device() == input.device(),
            "grouped_matmul_w1: expert_counts must be on the device of input_tensor");
        TT_FATAL(
            counts.dtype() == DataType::UINT32 && counts.layout() == Layout::ROW_MAJOR &&
                counts.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "grouped_matmul_w1: expert_counts must be UINT32, ROW_MAJOR and interleaved, as routing_metadata writes "
            "it");
        const auto& counts_shape = counts.logical_shape();
        TT_FATAL(
            counts_shape.rank() == 2 && counts_shape[0] == 1,
            "grouped_matmul_w1: expert_counts must be 1 x num_experts, got {}",
            counts_shape);
        num_experts = counts_shape[1];
    }
    TT_FATAL(
        num_experts >= 1 && num_experts <= kMaxExperts,
        "grouped_matmul_w1: num_experts must be 1 to {}, got {}",
        kMaxExperts,
        num_experts);

    const auto& input_shape = input.logical_shape();
    const auto& weight_shape = weight.logical_shape();
    const uint32_t k = input_shape[3];
    const bool chunked = attributes.weight_chunk_tiles > 0;
    uint32_t n = 0;
    if (chunked) {
        // A chunked weight's shape holds no N: weight_n gives it, and the chunk geometry is checked below.
        n = attributes.weight_n;
        TT_FATAL(
            n > 0 && n % kTileSize == 0,
            "grouped_matmul_w1: a chunked weight needs weight_n, an expert's N in whole tiles, got {}",
            n);
        TT_FATAL(
            !tensor_args.bias.has_value(),
            "grouped_matmul_w1: a chunked weight carries its bias (weight_has_bias); bias must be None");
    } else {
        // weight is (1, 1, K, E * N), or (1, 1, N, E * K) read transposed: either way expert e owns the e-th E-th
        // of its columns.
        TT_FATAL(
            weight_shape[3] % num_experts == 0,
            "grouped_matmul_w1: the weight's {} columns do not split into {} experts",
            weight_shape[3],
            num_experts);
        const uint32_t expert_columns = weight_shape[3] / num_experts;
        const uint32_t weight_k = attributes.transpose_weight ? expert_columns : weight_shape[2];
        n = attributes.transpose_weight ? weight_shape[2] : expert_columns;
        TT_FATAL(
            weight_k == k,
            "grouped_matmul_w1: input_tensor has K = {} but the weight's experts take K = {}{}",
            k,
            weight_k,
            attributes.transpose_weight ? " (transpose_weight)" : "");
        TT_FATAL(
            !attributes.weight_has_bias,
            "grouped_matmul_w1: weight_has_bias needs a chunked weight (weight_chunk_tiles)");
    }
    TT_FATAL(n % kTileSize == 0, "grouped_matmul_w1: each expert's N must be whole tiles, got {}", n);
    const bool has_bias = tensor_args.bias.has_value() || attributes.weight_has_bias;

    TT_FATAL(
        attributes.dtype == DataType::BFLOAT16 || attributes.dtype == DataType::BFLOAT8_B,
        "grouped_matmul_w1: dtype must be BFLOAT16 or BFLOAT8_B, got {}",
        attributes.dtype);
    const bool row_major_output = attributes.output_layout == Layout::ROW_MAJOR;
    TT_FATAL(
        attributes.output_layout == Layout::TILE || row_major_output,
        "grouped_matmul_w1: output_layout must be TILE or ROW_MAJOR, got {}",
        attributes.output_layout);
    TT_FATAL(
        !row_major_output || attributes.dtype == DataType::BFLOAT16,
        "grouped_matmul_w1: a ROW_MAJOR output must be BFLOAT16, got {}",
        attributes.dtype);
    // An ND-sharded TILE output (grouped_matmul_w2's input layout) is written the same way: tiles by page id.
    TT_FATAL(
        attributes.memory_config.memory_layout() == TensorMemoryLayout::INTERLEAVED ||
            (!row_major_output && attributes.memory_config.nd_shard_spec().has_value()),
        "grouped_matmul_w1: memory_config must be interleaved, or ND-sharded for a TILE output, got {}",
        attributes.memory_config.memory_layout());
    if (has_bias) {
        // The bias joins the K loop as one more step, a tile of ones in the input's place.
        TT_FATAL(!attributes.transpose_weight, "grouped_matmul_w1: bias needs the weight untransposed");
        TT_FATAL(input.dtype() == DataType::BFLOAT16, "grouped_matmul_w1: bias needs a BFLOAT16 input");
    }
    if (tensor_args.bias.has_value()) {
        const auto& bias = *tensor_args.bias;
        TT_FATAL(bias.device() == input.device(), "grouped_matmul_w1: bias must be on the device of input_tensor");
        TT_FATAL(
            bias.dtype() == weight.dtype() && bias.layout() == Layout::TILE &&
                bias.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "grouped_matmul_w1: bias must be interleaved TILE of the weight's dtype {}, got {} {}",
            weight.dtype(),
            bias.dtype(),
            bias.layout());
        const auto& bias_shape = bias.logical_shape();
        TT_FATAL(
            bias_shape.rank() == 4 && bias_shape[0] == 1 && bias_shape[1] == 1 && bias_shape[2] == 1 &&
                bias_shape[3] == weight_shape[3],
            "grouped_matmul_w1: bias must be 1 x 1 x 1 x {}, each expert's N columns side by side, got {}",
            weight_shape[3],
            bias_shape);
    }
    if (attributes.activation.has_value()) {
        const auto& activation = *attributes.activation;
        const auto params = activation.get_params();
        TT_FATAL(
            activation.type() == ttnn::operations::unary::UnaryOpType::GELU_TANH && params.size() <= 1 &&
                (params.empty() || params[0] == 0.0f || params[0] == 1.0f || params[0] == 2.0f),
            "grouped_matmul_w1: activation must be GELU_TANH with no parameter, 0 (accurate), 1 (fast) or 2 "
            "(polynomial)");
    }

    // Lanes are the core rows and shares the core columns, each multiplying by its share of N. Per-expert lanes give
    // each expert a row of its own.
    const auto grid = input.device()->compute_with_storage_grid_size();
    TT_FATAL(
        !attributes.per_expert_lanes || num_experts <= grid.y,
        "grouped_matmul_w1: per_expert_lanes gives each expert a core row, so num_experts must be at most the grid's "
        "{} "
        "rows, got {}",
        grid.y,
        num_experts);
    const uint32_t k_tiles = k / kTileSize;
    const uint32_t n_tiles = n / kTileSize;
    const auto chunks =
        grouped_weight_chunks(k_tiles, n_tiles, num_experts, has_bias, attributes.weight_chunk_tiles, grid.x);
    const uint32_t dest_tiles = std::min(kMaxSubblockWidth, get_dest_reg_count(attributes.compute_kernel_config));
    TT_FATAL(
        chunks.subblock_w <= dest_tiles,
        "grouped_matmul_w1: subblocks of {} output tiles do not fit DST's {} tiles with this compute_kernel_config",
        chunks.subblock_w,
        dest_tiles);
    // Rows are untilized as whole output-row segments of a share.
    TT_FATAL(
        !row_major_output || chunks.last_share_n_tiles == chunks.share_n_tiles,
        "grouped_matmul_w1: a ROW_MAJOR output needs N split evenly over the core columns; {} tiles over {} columns "
        "does not",
        n_tiles,
        chunks.num_shares);
    // Rows are written by tile row, which only a flat input's row pages match one for one.
    TT_FATAL(
        !row_major_output || (input_shape[0] == 1 && input_shape[2] % kTileSize == 0),
        "grouped_matmul_w1: a ROW_MAJOR output needs input_tensor 1 x 1 x rows x K in whole tiles, got {}",
        input_shape);
    if (chunked) {
        TT_FATAL(
            weight.layout() == Layout::TILE && weight.memory_config().buffer_type() == tt::tt_metal::BufferType::DRAM &&
                weight.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED,
            "grouped_matmul_w1: a chunked weight must be TILE and WIDTH_SHARDED in DRAM, as "
            "prepare_grouped_matmul_w1_weight writes it");
        const uint32_t width = kGroupedWeightBanks * chunks.shard_tiles * kTileSize;
        TT_FATAL(
            weight_shape.rank() == 4 && weight_shape[0] == 1 && weight_shape[1] == 1 && weight_shape[2] == kTileSize &&
                weight_shape[3] == width,
            "grouped_matmul_w1: a chunked weight for K = {}, N = {}, {} experts{} and chunks of {} tiles is 1 x 1 x 32 "
            "x "
            "{} on this grid, got {}; prepare it with the same arguments",
            k,
            n,
            num_experts,
            has_bias ? " with a bias" : "",
            attributes.weight_chunk_tiles,
            width,
            weight_shape);
        const auto& shard_spec = weight.memory_config().shard_spec();
        TT_FATAL(
            shard_spec.has_value() && shard_spec->shape[1] == chunks.shard_tiles * kTileSize &&
                shard_spec->grid.num_cores() == kGroupedWeightBanks,
            "grouped_matmul_w1: a chunked weight's shards must be {} tiles wide, one a DRAM bank",
            chunks.shard_tiles);
    }

    const uint32_t rows = input_shape[0] * input.padded_shape()[2];
    const uint32_t input_blocks = rows / kTileSize;
    const uint32_t row_lanes = attributes.per_expert_lanes ? num_experts : static_cast<uint32_t>(grid.y);
    const uint32_t lane_blocks = tt::div_up(input_blocks, row_lanes);
    const bool dedicated = !attributes.per_expert_lanes && chunks.num_shares + kDedicatedInputSenders <= grid.x &&
                           lane_blocks >= kDedicatedSenderLaneBlocks;
    // The columns east of the row lanes' senders become lanes when each fits a share a row with its senders below
    // (w2 at 8x512: 117.5 against 120.5 us with three senders a row lane and no column lane, bit for bit the same).
    const bool folded = dedicated && num_experts > 1 && chunks.num_shares + kFoldedRowLaneSenders + 1 <= grid.x &&
                        chunks.num_shares + kFoldedInputSenders <= grid.y;
    const uint32_t row_lane_senders = folded ? kFoldedRowLaneSenders : 0u;
    const uint32_t num_lanes = folded ? row_lanes + grid.x - chunks.num_shares - row_lane_senders : row_lanes;
    // One expert, TILE output: subblock-outer, so each subblock of the slice serves all held blocks while the rest
    // streams in.
    const bool subblock_outer = num_experts == 1 && !row_major_output && !attributes.per_expert_lanes;
    const bool whole_lane = subblock_outer && lane_blocks <= kWholeLaneBlocks;
    // The tail blocks' subblocks spread over the lanes (expert_schedule.hpp lane_work). A whole lane holds its tail
    // block in its input ring; in groups the tail block joins the last group, which needs room for it, known here only
    // without counts (every block the expert's). QKV at 64 tile rows: lanes 0 to 3 took 7 blocks, the rest 6.
    const bool tail_split = whole_lane || (subblock_outer && !tensor_args.expert_counts.has_value() &&
                                           (input_blocks / row_lanes) % kGroupBlocks != 0);
    return GroupedMatmulW1Layout{
        .num_experts = num_experts,
        .rows = rows,
        .k_tiles = k_tiles,
        .n_tiles = n_tiles,
        .num_lanes = num_lanes,
        .num_columns = chunks.num_shares,
        .column_n_tiles = chunks.share_n_tiles,
        .last_column_n_tiles = chunks.last_share_n_tiles,
        .subblock_w = chunks.subblock_w,
        .has_bias = has_bias,
        .slot_tiles = (k_tiles + (has_bias ? 1u : 0u)) * chunks.share_n_tiles,
        .weight_slots = num_experts > 1 ? 2u : 1u,
        .chunks = chunks,
        .weight_depth = attributes.transpose_weight ? kWeightDepthTransposed : kWeightDepth,
        .input_senders = folded ? kFoldedInputSenders : (dedicated ? kDedicatedInputSenders : 0u),
        .staging_blocks = kStagingBlocks,
        .folded = folded,
        .row_lane_senders = row_lane_senders,
        .input_slots = whole_lane ? std::max(2u, lane_blocks) : (subblock_outer ? 2 * kGroupBlocks : 2u),
        .output_slots = subblock_outer && !whole_lane ? 2 * kGroupBlocks : 2u,
        .subblock_outer = subblock_outer,
        .group_blocks = subblock_outer && !whole_lane ? kGroupBlocks : 0u,
        .tail_split = tail_split,
    };
}

GroupedWeightChunks grouped_weight_chunks(
    uint32_t k_tiles, uint32_t n_tiles, uint32_t num_experts, bool has_bias, uint32_t chunk_tiles, uint32_t grid_x) {
    GroupedWeightChunks chunks;
    chunks.share_n_tiles = tt::div_up(n_tiles, grid_x);
    chunks.num_shares = tt::div_up(n_tiles, chunks.share_n_tiles);
    chunks.last_share_n_tiles = n_tiles - (chunks.num_shares - 1) * chunks.share_n_tiles;
    chunks.subblock_w = kMaxSubblockWidth;
    while (chunks.share_n_tiles % chunks.subblock_w != 0 || chunks.last_share_n_tiles % chunks.subblock_w != 0) {
        --chunks.subblock_w;
    }
    chunks.chunk_tiles = chunk_tiles;
    if (chunk_tiles > 0) {
        const uint32_t slice_tiles = (k_tiles + (has_bias ? 1u : 0u)) * chunks.share_n_tiles;
        chunks.max_chunks = tt::div_up(slice_tiles, chunk_tiles);
        chunks.bank_sharers = tt::div_up(chunks.num_shares, kGroupedWeightBanks);
        chunks.shard_tiles = num_experts * chunks.max_chunks * chunks.bank_sharers * chunk_tiles;
    }
    return chunks;
}

void GroupedMatmulW1DeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    compute_layout(attributes, tensor_args);
}

GroupedMatmulW1DeviceOperation::spec_return_value_t GroupedMatmulW1DeviceOperation::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    using tt::tt_metal::PageConfig;
    using tt::tt_metal::TensorLayout;

    const auto layout = compute_layout(attributes, tensor_args);
    const auto& input_shape = tensor_args.input.logical_shape();
    return tt::tt_metal::TensorSpec(
        ttnn::Shape({input_shape[0], 1, input_shape[2], layout.n_tiles * kTileSize}),
        TensorLayout(attributes.dtype, PageConfig(attributes.output_layout), attributes.memory_config));
}

GroupedMatmulW1DeviceOperation::tensor_return_value_t GroupedMatmulW1DeviceOperation::create_output_tensors(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(attributes, tensor_args), tensor_args.input.device());
}

}  // namespace ttnn::experimental::nomic_embed_text_v2_moe

namespace ttnn::prim {

Tensor grouped_matmul_w1(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& expert_counts,
    tt::tt_metal::DataType dtype,
    bool transpose_weight,
    const std::optional<ttnn::operations::unary::UnaryWithParam>& activation,
    tt::tt_metal::Layout output_layout,
    const tt::tt_metal::MemoryConfig& memory_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    bool per_expert_lanes,
    const std::optional<Tensor>& bias,
    uint32_t weight_chunk_tiles,
    uint32_t weight_n,
    bool weight_has_bias) {
    using Operation = ttnn::experimental::nomic_embed_text_v2_moe::GroupedMatmulW1DeviceOperation;
    return ttnn::device_operation::launch<Operation>(
        Operation::operation_attributes_t{
            .dtype = dtype,
            .transpose_weight = transpose_weight,
            .activation = activation,
            .output_layout = output_layout,
            .memory_config = memory_config,
            .compute_kernel_config = compute_kernel_config,
            .per_expert_lanes = per_expert_lanes,
            .weight_chunk_tiles = weight_chunk_tiles,
            .weight_n = weight_n,
            .weight_has_bias = weight_has_bias},
        Operation::tensor_args_t{.input = input, .weight = weight, .expert_counts = expert_counts, .bias = bias});
}

}  // namespace ttnn::prim
