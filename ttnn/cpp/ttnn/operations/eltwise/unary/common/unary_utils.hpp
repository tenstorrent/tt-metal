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

/** DRAM height-sharded TILE in and out with one shard spec. Each shard sits in one DRAM bank, so the reader and
 * writer take slot 0 of every shard, then slot 1, ... (SHARD_ROTATE), which spreads consecutive pages over the
 * banks. The flow splits the pages over cores:
 * - StaticBurst: even split, several pages in flight.
 * - StaticOnePage: even split, one page in flight. Compute-heavy ops on small tensors.
 * - WorkQueue: cores take chunks from a scheduler on one core, so cores that finish early take more work.
 * The flow is hashed; the sizes are runtime args. */
enum class DramHeightFlow : uint8_t { None, StaticBurst, StaticOnePage, WorkQueue };

struct DramHeightPlan {
    DramHeightFlow flow = DramHeightFlow::None;
    uint32_t shard_pages = 0;       // pages per full shard
    uint32_t num_shards = 0;        // shards holding data
    uint32_t last_shard_pages = 0;  // pages in the last shard, which may be short
    uint32_t chunk_pages = 0;       // WorkQueue only
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
