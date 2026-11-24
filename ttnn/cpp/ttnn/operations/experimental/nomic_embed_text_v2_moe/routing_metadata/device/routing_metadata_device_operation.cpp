// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "routing_metadata_device_operation.hpp"

#include <string_view>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::experimental::nomic_embed_text_v2_moe {

namespace {

using tt::tt_metal::DataType;
using tt::tt_metal::Layout;

constexpr uint32_t kBlockRows = tt::constants::TILE_HEIGHT;
// The kernels read a tile's columns 0-15 from its faces 0 and 2 only.
constexpr uint32_t kMaxTopK = tt::constants::FACE_WIDTH;
constexpr uint32_t kMaxExperts = 64;
// The slots, top_k x num_tokens, of one routed expert pass; larger batches run as windows. The cores
// exchange their per-expert counts as 16-bit halves of a word, and a sum of the other cores' counts
// stays below 65536 only if the window's slots do: ids clamped to num_experts - 1 can put several of
// a token's slots on one expert, so bounding the tokens alone would not do. It also bounds each
// core's L1 scratch, to about 420 KB at top_k 16 and 64 experts; a clash with tensors already in L1
// raises when the program is created. That is 4096 tokens at top_k 16 and 32768 at top_k 2.
constexpr uint32_t kMaxSlots = 65536;

// The framework rejects host tensors before this runs.
void check_input(const Tensor& tensor, std::string_view name, DataType dtype) {
    TT_FATAL(tensor.dtype() == dtype, "routing_metadata: {} must be {}, got {}", name, dtype, tensor.dtype());
    TT_FATAL(
        tensor.layout() == Layout::TILE, "routing_metadata: {} must be TILE layout, got {}", name, tensor.layout());
    TT_FATAL(
        tensor.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
        "routing_metadata: {} must be interleaved, got {}",
        name,
        tensor.memory_config().memory_layout());
    const auto tile = tensor.tensor_spec().tile();
    const auto face_shape = tile.get_face_shape();
    TT_FATAL(
        tile.get_height() == tt::constants::TILE_HEIGHT && tile.get_width() == tt::constants::TILE_WIDTH &&
            face_shape[0] == tt::constants::FACE_HEIGHT && face_shape[1] == tt::constants::FACE_WIDTH,
        "routing_metadata: {} must use 32 x 32 tiles of 16 x 16 faces",
        name);
}

}  // namespace

RoutingMetadataLayout RoutingMetadataDeviceOperation::compute_layout(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto& [indices, values, scores] = tensor_args;
    const bool select = scores.has_value();
    TT_FATAL(
        select != indices.has_value() && indices.has_value() == values.has_value(),
        "routing_metadata: takes indices with values, or scores, not both");
    TT_FATAL(select == (attributes.top_k > 0), "routing_metadata: top_k goes with scores, and only with scores");
    const Tensor& input = select ? *scores : *indices;
    if (select) {
        check_input(input, "scores", DataType::FLOAT32);
    } else {
        check_input(input, "indices", DataType::UINT32);
        check_input(*values, "values", DataType::FLOAT32);
    }

    const auto& shape = input.logical_shape();
    TT_FATAL(
        shape.rank() == 4 && shape[0] == 1 && shape[1] == 1,
        "routing_metadata: {} must be 1 x 1 x tokens x {}, got {}",
        select ? "scores" : "indices",
        select ? "columns" : "top_k",
        shape);
    if (!select) {
        TT_FATAL(values->device() == input.device(), "routing_metadata: values must be on the device of indices");
        TT_FATAL(
            values->logical_shape() == shape,
            "routing_metadata: values must have the shape of indices {}, got {}",
            shape,
            values->logical_shape());
    }

    const uint32_t total_tokens = shape[2];
    const uint32_t top_k = select ? attributes.top_k : shape[3];
    TT_FATAL(top_k >= 1 && top_k <= kMaxTopK, "routing_metadata: top_k must be 1 to {}, got {}", kMaxTopK, top_k);
    TT_FATAL(
        attributes.num_experts >= 1 && attributes.num_experts <= kMaxExperts,
        "routing_metadata: num_experts must be 1 to {}, got {}",
        kMaxExperts,
        attributes.num_experts);
    if (select) {
        TT_FATAL(
            shape[3] >= attributes.num_experts,
            "routing_metadata: scores must have a column for each of the {} experts, got {}",
            attributes.num_experts,
            shape[3]);
        TT_FATAL(
            top_k <= attributes.num_experts,
            "routing_metadata: top_k {} is above the {} experts to select from",
            top_k,
            attributes.num_experts);
    }
    TT_FATAL(
        attributes.token_start % kBlockRows == 0,
        "routing_metadata: token_start must be a multiple of {}, got {}",
        kBlockRows,
        attributes.token_start);
    TT_FATAL(
        attributes.token_start < total_tokens,
        "routing_metadata: token_start must be below the token count {}, got {}",
        total_tokens,
        attributes.token_start);
    TT_FATAL(
        attributes.num_tokens >= 1 && attributes.num_tokens <= kMaxSlots / top_k,
        "routing_metadata: num_tokens must be 1 to {} at top_k {}, {} slots a window, got {}",
        kMaxSlots / top_k,
        top_k,
        kMaxSlots,
        attributes.num_tokens);
    TT_FATAL(
        attributes.num_tokens <= total_tokens - attributes.token_start,
        "routing_metadata: the window of {} tokens from token {} ends past the token count {}",
        attributes.num_tokens,
        attributes.token_start,
        total_tokens);

    const uint32_t num_blocks = tt::div_up(top_k * attributes.num_tokens, kBlockRows) + attributes.num_experts - 1;
    return RoutingMetadataLayout{
        .select = select,
        .score_groups = select ? tt::div_up(attributes.num_experts, tt::constants::FACE_WIDTH) : 0,
        .score_tiles_per_row = select ? tt::div_up(shape[3], tt::constants::TILE_WIDTH) : 0,
        .top_k = top_k,
        .num_blocks = num_blocks,
        .gathered_rows = num_blocks * kBlockRows,
        .padded_tokens = tt::round_up(attributes.num_tokens, kBlockRows),
    };
}

void RoutingMetadataDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    compute_layout(attributes, tensor_args);
}

RoutingMetadataDeviceOperation::spec_return_value_t RoutingMetadataDeviceOperation::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    using tt::tt_metal::PageConfig;
    using tt::tt_metal::TensorLayout;
    using tt::tt_metal::TensorSpec;

    const auto layout = compute_layout(attributes, tensor_args);
    const tt::tt_metal::MemoryConfig dram{
        tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};
    const auto row_major_uint32 = [&](const ttnn::Shape& shape) {
        return TensorSpec(shape, TensorLayout(DataType::UINT32, PageConfig(Layout::ROW_MAJOR), dram));
    };

    return {
        row_major_uint32(ttnn::Shape({1, layout.gathered_rows})),
        row_major_uint32(ttnn::Shape({layout.top_k, layout.padded_tokens})),
        // The layout attn_res_weighted_reduce_nc takes its weights in: column 0 of each tile is read.
        TensorSpec(
            ttnn::Shape({1, layout.top_k, layout.padded_tokens, 1}),
            TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE), dram)),
        row_major_uint32(ttnn::Shape({1, attributes.num_experts})),
    };
}

RoutingMetadataDeviceOperation::tensor_return_value_t RoutingMetadataDeviceOperation::create_output_tensors(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto [gather_index, combine_index, combine_weights, expert_counts] =
        compute_output_specs(attributes, tensor_args);
    auto* device = (tensor_args.scores.has_value() ? tensor_args.scores : tensor_args.indices)->device();
    return {
        create_device_tensor(gather_index, device),
        create_device_tensor(combine_index, device),
        create_device_tensor(combine_weights, device),
        create_device_tensor(expert_counts, device),
    };
}

}  // namespace ttnn::experimental::nomic_embed_text_v2_moe

namespace ttnn::prim {

std::tuple<Tensor, Tensor, Tensor, Tensor> routing_metadata(
    const std::optional<Tensor>& indices,
    const std::optional<Tensor>& values,
    const std::optional<Tensor>& scores,
    uint32_t num_experts,
    uint32_t top_k,
    uint32_t token_start,
    uint32_t num_tokens) {
    using Operation = ttnn::experimental::nomic_embed_text_v2_moe::RoutingMetadataDeviceOperation;
    return ttnn::device_operation::launch<Operation>(
        Operation::operation_attributes_t{
            .num_experts = num_experts, .top_k = top_k, .token_start = token_start, .num_tokens = num_tokens},
        Operation::tensor_args_t{.indices = indices, .values = values, .scores = scores});
}

}  // namespace ttnn::prim
