// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_and_split_heads_device_operation.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

using tt::tt_metal::TensorMemoryLayout;

void UntilizeAndSplitHeadsDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    const auto& in = args.input;
    TT_FATAL(in.storage_type() == StorageType::DEVICE, "untilize_and_split_heads: input must be on device");
    TT_FATAL(in.device()->arch() != tt::ARCH::QUASAR, "untilize_and_split_heads is not supported on Quasar");
    TT_FATAL(in.dtype() == DataType::BFLOAT16, "untilize_and_split_heads: input must be BFLOAT16, got {}", in.dtype());
    TT_FATAL(in.layout() == Layout::TILE, "untilize_and_split_heads: input must be TILE, got {}", in.layout());
    // The kernels read whole 2048 B tiles and untilize them as plain 32x32
    // tiles; nothing configures the unpacker for a transposed datum order.
    const auto tile = in.tensor_spec().tile();
    const auto tile_shape = tile.get_tile_shape();
    TT_FATAL(
        tile_shape[0] == tt::constants::TILE_HEIGHT && tile_shape[1] == tt::constants::TILE_WIDTH &&
            !tile.get_transpose_within_face() && !tile.get_transpose_of_faces(),
        "untilize_and_split_heads: input must use non-transposed 32x32 tiles, got {}x{}{}",
        tile_shape[0],
        tile_shape[1],
        tile.get_transpose_within_face() || tile.get_transpose_of_faces() ? " transposed" : "");
    TT_FATAL(
        in.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "untilize_and_split_heads: input must be INTERLEAVED");
    TT_FATAL(
        attrs.output_memory_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "untilize_and_split_heads: output memory_config must be INTERLEAVED");
    const auto& s = in.logical_shape();
    TT_FATAL(s.rank() == 3, "untilize_and_split_heads: input must be (B, S, H*D), got shape {}", s);
    TT_FATAL(s[0] > 0 && s[1] > 0 && s[2] > 0, "untilize_and_split_heads: B, S and H*D must be > 0, got shape {}", s);
    TT_FATAL(attrs.num_heads > 0, "untilize_and_split_heads: num_heads must be > 0");
    // Divide rather than multiply: num_heads * TILE_WIDTH can wrap a uint32 to 0.
    TT_FATAL(
        s[2] % attrs.num_heads == 0 && (s[2] / attrs.num_heads) % tt::constants::TILE_WIDTH == 0,
        "untilize_and_split_heads: H*D ({}) must split into {} heads of a whole number of tile widths; a head "
        "has to own its tiles",
        s[2],
        attrs.num_heads);
    // The reader turns (b, tile row, head) into tile ids from the logical
    // shape, so the tile grid must carry no padding beyond the last tile.
    const auto& ps = in.padded_shape();
    TT_FATAL(
        ps[-2] == tt::round_up(s[1], tt::constants::TILE_HEIGHT) && ps[-1] == s[2],
        "untilize_and_split_heads: input {} is padded to {}; only padding S up to a whole tile is supported, as "
        "ttnn.linear and ttnn.to_layout produce. Re-tilize it with ttnn.to_layout",
        s,
        ps);
}

void UntilizeAndSplitHeadsDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    validate_on_program_cache_miss(attrs, args);
}

tt::tt_metal::TensorSpec untilize_and_split_heads_output_spec(
    const ttnn::Shape& input_shape, uint32_t num_heads, const MemoryConfig& memory_config) {
    const uint32_t head_dim = input_shape[2] / num_heads;
    const Shape out_shape(
        {input_shape[0],
         num_heads,
         tt::div_up(input_shape[1], tt::constants::TILE_HEIGHT),
         tt::constants::TILE_HEIGHT * head_dim});
    return tt::tt_metal::TensorSpec(
        out_shape,
        tt::tt_metal::TensorLayout(DataType::BFLOAT16, tt::tt_metal::PageConfig(Layout::ROW_MAJOR), memory_config));
}

UntilizeAndSplitHeadsDeviceOperation::spec_return_value_t UntilizeAndSplitHeadsDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    // launch() builds the output before it validates, and the shape math below
    // divides by num_heads and indexes three dims.
    validate_on_program_cache_miss(attrs, args);
    return untilize_and_split_heads_output_spec(
        args.input.logical_shape(), attrs.num_heads, attrs.output_memory_config);
}

UntilizeAndSplitHeadsDeviceOperation::tensor_return_value_t UntilizeAndSplitHeadsDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    return create_device_tensor(compute_output_specs(attrs, args), args.input.device());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

ttnn::Tensor untilize_and_split_heads(
    const Tensor& input, uint32_t num_heads, const std::optional<MemoryConfig>& memory_config) {
    using OperationType = ttnn::experimental::prim::UntilizeAndSplitHeadsDeviceOperation;
    auto attrs = OperationType::operation_attributes_t{
        .num_heads = num_heads,
        .output_memory_config = memory_config.value_or(input.memory_config()),
    };
    auto args = OperationType::tensor_args_t{.input = input};
    return ttnn::device_operation::launch<OperationType>(attrs, args);
}

}  // namespace ttnn::prim
