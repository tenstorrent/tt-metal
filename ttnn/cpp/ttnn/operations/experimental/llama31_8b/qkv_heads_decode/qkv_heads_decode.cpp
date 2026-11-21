// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/llama31_8b/qkv_heads_decode/qkv_heads_decode.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/llama31_8b/qkv_heads_decode/device/qkv_heads_decode_device_operation.hpp"

namespace ttnn::experimental::llama31_8b {

std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> qkv_heads_decode(
    const ttnn::Tensor& input_tensor,
    uint32_t num_heads,
    uint32_t num_kv_heads,
    const std::optional<MemoryConfig>& memory_config) {
    TT_FATAL(num_heads > 0 && num_kv_heads > 0, "qkv_heads_decode: num_heads and num_kv_heads must be positive");
    const uint32_t width = input_tensor.padded_shape()[3];
    TT_FATAL(width % (num_heads + 2 * num_kv_heads) == 0, "qkv_heads_decode: width {} not divisible by heads", width);
    const bool input_on_subcoregrids =
        input_tensor.is_sharded() &&
        (input_tensor.shard_spec().value().grid.ranges().size() > 1 ||
         input_tensor.shard_spec().value().grid.bounding_box().start_coord != CoreCoord{0, 0});
    CoreRangeSet output_core_grid;
    if (memory_config.has_value() && memory_config.value().shard_spec().has_value()) {
        output_core_grid = memory_config.value().shard_spec().value().grid;
    } else {
        const auto grid = input_tensor.device()->compute_with_storage_grid_size();
        output_core_grid = CoreRangeSet{CoreRange{CoreCoord{0, 0}, CoreCoord{grid.x - 1, grid.y - 1}}};
    }
    using namespace ttnn::prim::llama31_8b_qkv_heads_decode;
    auto out = ttnn::device_operation::launch<QkvHeadsDecodeDeviceOperation>(
        QkvHeadsDecodeDeviceOperation::operation_attributes_t{
            .num_q_heads = num_heads,
            .num_kv_heads = num_kv_heads,
            .head_dim = width / (num_heads + 2 * num_kv_heads),
            .overlap_qk_coregrid = true,
            .input_on_subcoregrids = input_on_subcoregrids,
            .slice_size = std::nullopt,
            .output_mem_config =
                MemoryConfig{
                    tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
                    tt::tt_metal::BufferType::L1,
                    tt::tt_metal::ShardSpec{output_core_grid, {}}}},
        QkvHeadsDecodeDeviceOperation::tensor_args_t{.input_tensor = input_tensor, .batch_offset = std::nullopt});
    return {out.at(0), out.at(1), out.at(2)};
}

}  // namespace ttnn::experimental::llama31_8b
