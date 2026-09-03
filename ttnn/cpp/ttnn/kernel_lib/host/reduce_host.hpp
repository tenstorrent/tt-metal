// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include <tt-metalium/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <umd/device/types/arch.hpp>

#include "ttnn/cpp/ttnn/kernel_lib/reduce_types.hpp"

namespace ttnn::kernel_lib::host {

using ReducePath = ttnn::kernel_lib::ReducePath;
using ReduceAuxiliaryTileType = ttnn::kernel_lib::ReduceAuxiliaryTileType;

enum class ReduceCbRole : std::uint8_t {
    Input,
    Output,
    Auxiliary,
    RowMajorStaging,
    TiledScratch,
    Accumulator,
    PaddingIdentity,
};

enum class ReduceCbAlias : std::uint8_t { None, InputTensor, OutputTensor };

struct ReduceHardwareConfig {
    tt::ARCH arch = tt::ARCH::Invalid;
    bool fp32_dest_acc_en = false;
    bool dst_full_sync_en = false;
    std::size_t available_l1_bytes = 0;
};

struct ReduceChunkPlan {
    // Tiles consumed along the reduction axis per synchronization chunk.
    std::uint32_t reduce_axis_tiles = 1;
    // Independent outputs retained in DEST for the chunk (greater than one for H reduction).
    std::uint32_t output_tiles = 1;
    // Number of chunks which fit concurrently in the input allocation.
    std::uint32_t buffers = 1;

    std::uint32_t input_tiles() const { return reduce_axis_tiles * output_tiles; }
};

struct ReduceCbRequirement {
    ReduceCbRole role;
    tt::DataFormat data_format;
    std::uint32_t page_size;
    std::uint32_t page_count;
    std::size_t total_size_bytes;
    ReduceCbAlias alias = ReduceCbAlias::None;

    bool owns_l1() const { return alias == ReduceCbAlias::None; }
};

// One concrete tile for the dataflow-side auxiliary recipe. The planner has
// already resolved why the tile is needed; the reader only needs these three
// physical properties to materialize it.
struct ReduceAuxiliaryTileSpec {
    float value = 0.0F;
    ReduceAuxiliaryTileType type = ReduceAuxiliaryTileType::Zero;
    std::uint32_t num_valid_elements = 0;
};

// Dense row-major geometry. This replaces the former factory-local RmPlan.
struct DenseRowMajorPlan {
    std::uint32_t H_logical = 0;
    std::uint32_t W_logical = 0;
    std::uint32_t Ht_rm = 0;
    std::uint32_t Wt = 0;
    std::uint32_t rm_rows_per_tile = 0;
    std::uint32_t wt_tiles_per_chunk = 1;
    std::uint32_t ht_tiles_per_chunk = 1;
    std::uint32_t chunk_row_bytes = 0;
    std::uint32_t rm_staging_page_size = 0;
    std::uint32_t padding_identity_bits = 0;
    std::uint32_t src_datum_size = 0;
    std::uint32_t dst_datum_size = 0;
    std::uint32_t staging_buffers = 1;
};

struct ReducePlan {
    ReducePath path = ReducePath::Tiled;
    tt::tt_metal::ReduceOpMath reduce_math = tt::tt_metal::ReduceOpMath::SUM;
    tt::tt_metal::ReduceOpDim reduce_dim = tt::tt_metal::ReduceOpDim::W;
    ReduceFp32Mode fp32_mode = ReduceFp32Mode::Fast;
    compute_kernel_lib::ReduceAlgorithm algorithm = compute_kernel_lib::ReduceAlgorithm::ReduceTile;
    compute_kernel_lib::ReduceInputPolicy input_policy = compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile;
    compute_kernel_lib::AccumulateReloadMode reload_mode = compute_kernel_lib::AccumulateReloadMode::CopySeedPairs;
    compute_kernel_lib::ReduceDataFormatReconfigMode reconfig_mode =
        compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT;
    compute_kernel_lib::ReduceWithinTile within_tile = compute_kernel_lib::ReduceWithinTile::Collapse;
    ReduceChunkPlan chunk;

    std::uint32_t Ht = 0;
    std::uint32_t Wt = 0;
    std::uint32_t batches = 0;
    // Zero means the ordinary contiguous Wt pitch.
    std::uint32_t input_row_stride_tiles = 0;
    std::uint32_t reduce_factor = 1;

    // post_scale is applied once, after reduction finalization and before any
    // caller callback. The auxiliary recipe is already lowered to physical tile
    // specifications in the order consumed by compute.
    float post_scale = 1.0F;
    compute_kernel_lib::ReducePartialMode partial_mode = compute_kernel_lib::ReducePartialMode::None;
    std::vector<ReduceAuxiliaryTileSpec> auxiliary_tiles;
    std::uint32_t partial_reduce_axis_elements = 0;

    std::optional<DenseRowMajorPlan> row_major;
    std::vector<ReduceCbRequirement> cb_requirements;
    std::size_t total_owned_l1_bytes = 0;

    const ReduceCbRequirement* find_cb(ReduceCbRole role) const;
};

// Per-input configuration for one call in a cross-CB reduction sequence. TensorSpec owns the shape, data
// type, tile, and memory-layout information; the optional zero-byte cap retains its single-call alias meaning.
struct ReduceCallConfig {
    tt::tt_metal::TensorSpec input_spec;
    tt::tt_metal::TensorSpec output_spec;
    tt::tt_metal::ReduceOpMath reduce_math;
    tt::tt_metal::ReduceOpDim reduce_dim;
    float scalar;
    ReduceFp32Mode fp32_mode;
    std::optional<std::size_t> max_input_cb_bytes = std::nullopt;
};

using ReduceCbConfig = std::pair<std::uint32_t, ReduceCallConfig>;

// These IDs bind planner roles into the caller's kernel CB namespace. They are explicit because only the
// caller knows which IDs are already occupied by the rest of a fused kernel.
struct ReduceSequenceCbIds {
    std::uint32_t auxiliary_cb_id;
    std::uint32_t accumulator_cb_id;
    std::uint32_t output_cb_id;
};

// CB binding for serializing an existing single-call ReducePlan.
struct ReduceCallCbIds {
    std::uint32_t input_cb_id;
    std::uint32_t auxiliary_cb_id;
    std::uint32_t output_cb_id;
};

// One complete kernel reduce() invocation. Accumulation behavior and index are
// explicit call properties; a kernel never derives them from this call's
// position in a list. `plan` is the complete existing single-CB plan for this
// input.
struct ReduceCallPlan {
    std::uint32_t input_cb_id;
    std::uint32_t auxiliary_cb_id;
    std::uint32_t output_cb_id;
    std::optional<std::uint32_t> accumulator_cb_id;
    ReduceAccumulationMode accumulation_mode = ReduceAccumulationMode::None;
    std::uint32_t accumulation_index = 0;
    ReducePlan plan;
};

struct ReduceSequencePlan {
    std::vector<ReduceCallPlan> calls;

    // Append the device wire-format suffix: call count followed by every
    // complete call in execution order. Existing caller-owned kernel arguments
    // remain untouched at the front of the vector.
    void append_to(std::vector<std::uint32_t>& compile_time_args) const;
    std::vector<std::uint32_t> get_compile_time_args() const;
};

// Host serializer for one independently decodable call. Its matching device
// view is ttnn::kernel_lib::ReduceCallArgs<CTA_OFFSET>.
class ReduceCallArgs {
public:
    explicit ReduceCallArgs(const ReduceCallPlan& call);
    ReduceCallArgs(const ReducePlan& plan, const ReduceCallCbIds& cb_ids);

    void append_to(std::vector<std::uint32_t>& compile_time_args) const;
    std::vector<std::uint32_t> get_compile_time_args() const;

private:
    std::vector<std::uint32_t> compile_time_args_;
};

// Plan a concrete reduction. A missing input-CB cap means "use the available
// reduction-owned L1 budget". A cap of zero is a sentinel for an input tensor
// already sharded in L1; in that case the input CB aliases the tensor and owns
// no scratch allocation.
ReducePlan make_reduce_plan(
    const tt::tt_metal::TensorSpec& input_spec,
    const tt::tt_metal::TensorSpec& output_spec,
    tt::tt_metal::ReduceOpMath reduce_math,
    tt::tt_metal::ReduceOpDim reduce_dim,
    float scalar,
    ReduceFp32Mode fp32_mode,
    const ReduceHardwareConfig& hardware,
    std::optional<std::size_t> max_input_cb_bytes = std::nullopt);

// Plan a kernel-ordered sequence of reductions whose results are accumulated together. The returned vector has
// exactly the same order and length as `reductions`; callers decide when to instantiate each reduce() call.
ReduceSequencePlan make_reduce_sequence_plan(
    const std::vector<ReduceCbConfig>& reductions,
    const ReduceSequenceCbIds& cb_ids,
    const ReduceHardwareConfig& hardware);

}  // namespace ttnn::kernel_lib::host
