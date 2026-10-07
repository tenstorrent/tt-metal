// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include <optional>
#include <vector>

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

/** How a DRAM height-sharded (SHARD_ROTATE) unary op moves its pages. Measured on Blackhole p100a, bf16:
 * - StaticBurst: v1. Each core gets an even share of pages, read up to 8 per barrier.
 * - StaticOnePage: same split, one page per barrier. For compute-bound ops on small tensors, where a burst
 *   only delays the first tiles (mish_fast: 95.6 -> 88.8 us at 7 168 tiles).
 * - WorkQueue: cores pull fixed-size chunks from a scheduler that runs inside one core's writer, so
 *   cores that DRAM serves quickly take more of the work (silu: 36.0 -> 21.5 ms at 2 093 056 tiles).
 * The choice depends on the shape, so it is part of the program hash. */
enum class DramHeightFlow : uint8_t { None, StaticBurst, StaticOnePage, WorkQueue };

struct DramHeightPlan {
    DramHeightFlow flow = DramHeightFlow::None;
    uint32_t chunk_pages = 0;  // WorkQueue only
};

DramHeightPlan get_dram_height_plan(
    const std::vector<EltwiseUnaryWithParam>& op_chain,
    const tt::tt_metal::TensorSpec& input_spec,
    const tt::tt_metal::TensorSpec& output_spec,
    uint32_t num_cores);

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
