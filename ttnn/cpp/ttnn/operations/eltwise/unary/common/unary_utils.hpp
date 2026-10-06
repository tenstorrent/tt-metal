// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include <optional>

namespace ttnn::operations::unary {

/** True if native L1 sharding path can be used (input and output both L1, even sharding). */
bool is_native_L1_sharding(
    const tt::tt_metal::TensorSpec& input_spec, const tt::tt_metal::MemoryConfig& output_memory_config);

/** Shard spec for output when using native sharded path; nullopt if interleaved/fallback path. */
struct UnaryShardSpecs {
    tt::tt_metal::ShardSpec input_shard_spec;
    tt::tt_metal::ShardSpec output_shard_spec;
};

std::optional<UnaryShardSpecs> get_shard_specs(
    const tt::tt_metal::TensorSpec& input_spec, const tt::tt_metal::TensorSpec& output_spec);

const std::optional<tt::tt_metal::ShardSpec>& get_shard_spec(const tt::tt_metal::TensorSpec& tensor_spec);

bool is_uneven(const tt::tt_metal::TensorSpec& t);

/** DRAM height-sharded TILE in and out with the same shard spec. A shard's pages sit in one bank, so
 * the interleaved reader and writer visit every shard slot by slot (SHARD_ROTATE) instead of walking
 * page ids, which keeps consecutive pages, and each burst, on different banks. Whether it applies
 * depends only on the memory configs, which the program hash covers; the sizes below are runtime
 * args. Other sharded DRAM layouts keep the one-page interleaved order. */
struct DramHeightRotate {
    bool enabled = false;
    uint32_t shard_pages = 0;       // pages (slots) per full shard
    uint32_t num_shards = 0;        // shards holding data, the last may be partial
    uint32_t last_shard_pages = 0;  // pages in the last shard
};

DramHeightRotate get_dram_height_rotate(
    const tt::tt_metal::TensorSpec& input_spec, const tt::tt_metal::TensorSpec& output_spec);

tt::tt_metal::CoreRangeSet get_worker_grid(
    const Tensor& input_tensor,
    const std::optional<Tensor>& output_tensor,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids);

tt::tt_metal::ShardSpec adjust_to_shape(
    const tt::tt_metal::ShardSpec& shard_spec, const ttnn::Shape& from_shape, const ttnn::Shape& to_shape);

// Synthesize a populated-shard output ShardSpec for specless sharded eltwise-unary outputs.
// `output_element_size_bytes` reflects the output dtype (typecast can differ from input dtype).
tt::tt_metal::ShardSpec generate_output_shard_spec(
    const Tensor& input_tensor,
    const ttnn::Shape& padded_out_shape,
    tt::tt_metal::TensorMemoryLayout memory_layout,
    uint32_t output_element_size_bytes);

}  // namespace ttnn::operations::unary
