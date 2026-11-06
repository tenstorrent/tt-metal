// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "head_major_value_device_operation.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/experimental/fused_msda/device/kernels/msda_value_layout.hpp"

namespace ttnn::operations::experimental::fused_msda {

using tt::tt_metal::TensorMemoryLayout;

void FusedMSDAHeadMajorValueOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    const auto& v = args.value;
    TT_FATAL(v.storage_type() == StorageType::DEVICE, "fused_msda_head_major_value: value must be on device");
    TT_FATAL(v.dtype() == DataType::BFLOAT16, "fused_msda_head_major_value: value must be BFLOAT16, got {}", v.dtype());
    TT_FATAL(v.layout() == Layout::TILE, "fused_msda_head_major_value: value must be TILE, got {}", v.layout());
    // The kernels read whole 2048 B tiles and untilize them as 32x32.
    const auto tile_shape = v.tensor_spec().tile().get_tile_shape();
    TT_FATAL(
        tile_shape[0] == tt::constants::TILE_HEIGHT && tile_shape[1] == tt::constants::TILE_WIDTH,
        "fused_msda_head_major_value: value must use 32x32 tiles, got {}x{}",
        tile_shape[0],
        tile_shape[1]);
    TT_FATAL(
        v.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "fused_msda_head_major_value: value must be INTERLEAVED");
    TT_FATAL(
        attrs.output_memory_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "fused_msda_head_major_value: output memory_config must be INTERLEAVED");
    const auto& vs = v.logical_shape();
    TT_FATAL(vs.rank() == 3, "fused_msda_head_major_value: value must be (B, S, H*D), got shape {}", vs);
    TT_FATAL(vs[0] > 0 && vs[1] > 0, "fused_msda_head_major_value: B and S must be > 0, got shape {}", vs);
    TT_FATAL(attrs.num_heads > 0, "fused_msda_head_major_value: num_heads must be > 0");
    TT_FATAL(
        vs[2] % (attrs.num_heads * tt::constants::TILE_WIDTH) == 0,
        "fused_msda_head_major_value: H*D ({}) must split into {} heads of a whole number of tile widths; a head "
        "has to own its tiles",
        vs[2],
        attrs.num_heads);
    // The reader turns (b, key block, head) into tile ids from the logical
    // shape, so the tile grid must carry no padding beyond the last tile.
    const auto& ps = v.padded_shape();
    TT_FATAL(
        ps[-2] == tt::round_up(vs[1], tt::constants::TILE_HEIGHT) && ps[-1] == vs[2],
        "fused_msda_head_major_value: value {} is padded to {}; only padding S up to a whole tile is supported, as "
        "ttnn.linear and ttnn.to_layout produce. Re-tilize it with ttnn.to_layout",
        vs,
        ps);
}

void FusedMSDAHeadMajorValueOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    validate_on_program_cache_miss(attrs, args);
}

tt::tt_metal::TensorSpec head_major_value_spec(
    const ttnn::Shape& packed_shape, uint32_t num_heads, const MemoryConfig& memory_config) {
    using namespace fused_msda_value_layout;
    const uint32_t head_dim = packed_shape[2] / num_heads;
    const Shape out_shape(
        {packed_shape[0], num_heads, head_major_pages_per_head(packed_shape[1]), HEAD_MAJOR_KEYS_PER_PAGE * head_dim});
    return tt::tt_metal::TensorSpec(
        out_shape,
        tt::tt_metal::TensorLayout(DataType::BFLOAT16, tt::tt_metal::PageConfig(Layout::ROW_MAJOR), memory_config));
}

FusedMSDAHeadMajorValueOperation::spec_return_value_t FusedMSDAHeadMajorValueOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    // launch() builds the output before it validates, and the shape math below
    // divides by num_heads and indexes three dims.
    validate_on_program_cache_miss(attrs, args);
    return head_major_value_spec(args.value.logical_shape(), attrs.num_heads, attrs.output_memory_config);
}

FusedMSDAHeadMajorValueOperation::tensor_return_value_t FusedMSDAHeadMajorValueOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    return create_device_tensor(compute_output_specs(attrs, args), args.value.device());
}

}  // namespace ttnn::operations::experimental::fused_msda

namespace ttnn::prim {

ttnn::Tensor fused_msda_head_major_value(
    const Tensor& value, uint32_t num_heads, const std::optional<MemoryConfig>& memory_config) {
    using OperationType = ttnn::operations::experimental::fused_msda::FusedMSDAHeadMajorValueOperation;
    auto attrs = OperationType::operation_attributes_t{
        .num_heads = num_heads,
        .output_memory_config = memory_config.value_or(value.memory_config()),
    };
    auto args = OperationType::tensor_args_t{.value = value};
    return ttnn::device_operation::launch<OperationType>(attrs, args);
}

}  // namespace ttnn::prim
