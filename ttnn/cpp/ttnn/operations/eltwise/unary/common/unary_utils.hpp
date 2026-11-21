// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/buffer_distribution_spec.hpp>
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

// How a unary op on a DRAM sharded or interleaved TILE tensor reads its pages. The flow is part of the program hash.
enum class DramReadFlow : uint8_t {
    None,           // the default path: each core reads a contiguous range, one page per barrier
    StaticBurst,    // even split, rotated order, several pages per barrier
    StaticOnePage,  // even split, rotated order, one page per barrier
    WorkQueue,      // cores pull chunks of pages from a scheduler core
};

struct DramReadPlan {
    DramReadFlow flow = DramReadFlow::None;
    // Page order, see dram_read::RotatedPages.
    uint32_t shard_stride = 0;
    uint32_t num_shards = 0;
    uint32_t last_shard_pages = 0;
    uint32_t shard_width = 0;
    uint32_t row_pages = 0;
    uint32_t chunk_pages = 0;  // WorkQueue only
};

// Pages per read barrier and per write flush. They follow from the flow alone, so the program hash, which includes the
// flow, covers the defines and buffer depths built from them.
constexpr uint32_t read_burst(DramReadFlow flow) {
    return flow == DramReadFlow::StaticBurst || flow == DramReadFlow::WorkQueue ? 8 : 1;
}
constexpr uint32_t write_burst(DramReadFlow flow) {
    return flow == DramReadFlow::StaticBurst || flow == DramReadFlow::WorkQueue ? 2 : 1;
}

// Picks the read flow from the op chain's compute cost, the tensor layout and the pages per core; flow None for every
// other tensor. num_cores counts the op's worker grid, num_device_cores the device's whole compute grid.
DramReadPlan get_dram_read_plan(
    const std::vector<EltwiseUnaryWithParam>& op_chain,
    const tt::tt_metal::TensorSpec& input_spec,
    const tt::tt_metal::TensorSpec& output_spec,
    uint32_t num_cores,
    uint32_t num_device_cores,
    uint32_t num_dram_banks);

// The same, for an op on input over worker_grid.
DramReadPlan get_dram_read_plan(
    const std::vector<EltwiseUnaryWithParam>& op_chain,
    const Tensor& input,
    const tt::tt_metal::TensorSpec& output_spec,
    const tt::tt_metal::CoreRangeSet& worker_grid);

// The cores an op runs on: sub_core_grids, a shard grid, or the device's workers, fewer for small DRAM tensors.
tt::tt_metal::CoreRangeSet get_worker_grid(
    const std::vector<EltwiseUnaryWithParam>& op_chain,
    const Tensor& input_tensor,
    const std::optional<Tensor>& output_tensor,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids);

tt::tt_metal::ShardSpec adjust_to_shape(
    const tt::tt_metal::ShardSpec& shard_spec, const ttnn::Shape& from_shape, const ttnn::Shape& to_shape);

/** Throw if a sharded tensor's buffer is distributed differently from `spec_distribution`. A view keeps its
 * parent's buffer under a freshly computed spec and the same shard spec can resolve to different banks at
 * the view's shape. */
void require_buffer_distribution_matches_spec(
    const Tensor& tensor,
    const std::optional<tt::tt_metal::BufferDistributionSpec>& spec_distribution,
    const char* slot);

void require_buffer_distribution_matches_spec(const Tensor& tensor, const char* slot);

// Synthesize a populated-shard output ShardSpec for specless sharded eltwise-unary outputs.
// `output_element_size_bytes` reflects the output dtype (typecast can differ from input dtype).
tt::tt_metal::ShardSpec generate_output_shard_spec(
    const Tensor& input_tensor,
    const ttnn::Shape& padded_out_shape,
    tt::tt_metal::TensorMemoryLayout memory_layout,
    uint32_t output_element_size_bytes);

}  // namespace ttnn::operations::unary
