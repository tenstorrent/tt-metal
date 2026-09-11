// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include <tt-metalium/tensor/spec/layout/layout.hpp>
#include <tt-metalium/tensor/tensor_types.hpp>
#include <tt-metalium/tile.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <umd/device/types/arch.hpp>

#include "ttnn/cpp/ttnn/kernel_lib/reduce_types.hpp"

/**
 * @file reduce_host.hpp
 * @brief Host-side planning and serialization for tiled reductions.
 *
 * One call to make_reduce_sequence_plan() produces one independent planning
 * unit: an ordered list of self-contained compute calls and one aggregate
 * auxiliary-tile recipe shared by those calls. Append the two parts to their
 * respective kernel compile-time-argument lists after any kernel-owned prefix:
 *
 * @code{.cpp}
 * auto unit = make_reduce_sequence_plan(reductions, cb_ids, hardware);
 * unit.append_to(compute_compile_time_args);
 * // compute suffix: [call_count][call_0]...[call_(call_count - 1)]
 *
 * unit.append_auxiliary_to(dataflow_compile_time_args);
 * // dataflow suffix: [one aggregate auxiliary descriptor]
 * @endcode
 *
 * Append multiple units to both lists in planner-invocation order. The device
 * kernels receive only these flat suffixes, not a ReduceSequencePlan object.
 * The compute kernel decides when to issue each call and may run arbitrary work
 * between them. In particular, multiple input descriptions may use the same CB
 * when the kernel refills or reuses it between calls.
 */

namespace ttnn::kernel_lib::host {

using ReducePath = ttnn::kernel_lib::ReducePath;
using ReduceAuxiliaryTileType = ttnn::kernel_lib::ReduceAuxiliaryTileType;

// Only kernels assigned to tail cores use these runtime arguments. Offsets are
// word offsets in the compute and auxiliary-producing kernels, respectively.
struct ReduceTailConfig {
    std::uint32_t compute_runtime_arg_offset = 0;
    std::uint32_t auxiliary_runtime_arg_offset = 0;
};

// Valid elements in one local block. The leading dimensions are flattened into
// batches, just as in ReduceBlockSpec. Empty cores should not issue a reduction.
struct ReduceValidShape {
    std::uint32_t height;
    std::uint32_t width;
    std::uint32_t batches = 1;
};

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
// already resolved why the tile is needed. A tail edge optionally takes its
// valid extent from a runtime shape rather than a constant.
struct ReduceAuxiliaryTileSpec {
    float value = 0.0F;
    ReduceAuxiliaryTileType type = ReduceAuxiliaryTileType::Zero;
    std::uint32_t num_valid_elements = 0;
    // When present, read this element extent from the auxiliary kernel's runtime
    // arguments. num_valid_elements is the tile extent; use the last tile's
    // remainder, or the whole extent when the runtime dimension is aligned.
    std::optional<std::uint32_t> runtime_extent_arg;
};

// The one shared auxiliary CB recipe for a complete planning unit. It carries
// the CB ID as well as the physical tiles to materialize. Calls refer to
// contiguous slices of `tiles`; equal call recipes share the same slice.
struct ReduceAuxiliaryPlan {
    std::uint32_t cb_id = 0;
    std::vector<ReduceAuxiliaryTileSpec> tiles;
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
    std::optional<ReduceTailConfig> tail;
    std::uint32_t logical_h = 0;
    std::uint32_t logical_w = 0;

    // post_scale is applied once, after reduction finalization and before any
    // caller callback. The auxiliary recipe is already lowered to physical tile
    // specifications in the order consumed by compute. SFPU reductions apply the
    // caller's scalar here because they do not consume scaler tiles. A non-unit
    // INT32 scale converts the reduced value to FLOAT32, multiplies, then converts
    // back to INT32 with truncation toward zero (matching reduce_post_mul_tile).
    float post_scale = 1.0F;
    compute_kernel_lib::ReducePartialMode partial_mode = compute_kernel_lib::ReducePartialMode::None;
    std::vector<ReduceAuxiliaryTileSpec> auxiliary_tiles;
    std::uint32_t partial_reduce_axis_elements = 0;

    std::optional<DenseRowMajorPlan> row_major;
    std::vector<ReduceCbRequirement> cb_requirements;
    std::size_t total_owned_l1_bytes = 0;

    const ReduceCbRequirement* find_cb(ReduceCbRole role) const;
    // Append these three words at the offsets in tail, on tail cores only.
    // Validates the shape against the planned local bounds before serialization.
    std::vector<std::uint32_t> get_runtime_shape_args(const ReduceValidShape& shape) const;
};

// The block consumed by one reduce invocation on one core. Shapes are in elements;
// padded extents describe the traversed block, independently of its allocation.
// The factory owns core assignment, tensor placement and reader/writer addressing.
struct ReduceBlockSpec {
    std::uint32_t logical_h = 0;
    std::uint32_t logical_w = 0;
    std::uint32_t padded_h = 0;
    std::uint32_t padded_w = 0;
    std::uint32_t batches = 1;
    tt::tt_metal::DataType input_dtype = tt::tt_metal::DataType::BFLOAT16;
    tt::tt_metal::DataType output_dtype = tt::tt_metal::DataType::BFLOAT16;
    tt::tt_metal::Layout input_layout = tt::tt_metal::Layout::TILE;
    tt::tt_metal::Layout output_layout = tt::tt_metal::Layout::TILE;
    tt::tt_metal::Tile input_tile;
    tt::tt_metal::Tile output_tile;
    // Tiled resident input only. Zero means contiguous at padded_w.
    std::uint32_t input_row_stride_tiles = 0;
    // Present: caller supplies an existing local allocation of this many tiles.
    // Input is already available to compute; output follows the ordinary pack protocol.
    // Absent: the planner sizes the corresponding FIFO/staging allocation.
    std::optional<std::uint32_t> resident_input_tiles;
    std::optional<std::uint32_t> resident_output_tiles;
    // Absent: entirely static shape. Present: logical extents are upper bounds;
    // this kernel reads [height, width, batches] at the configured runtime offset.
    // Resident inputs retain the planned row and batch pitches. FIFO producers
    // stream valid work in fixed-size packets of chunk.input_tiles() pages,
    // padding the final axis/output group so each packet fits the CB ring.
    std::optional<ReduceTailConfig> tail;

    // Convenience for a local tiled block with padding rounded to whole tiles.
    static ReduceBlockSpec tiled(
        std::uint32_t logical_h,
        std::uint32_t logical_w,
        tt::tt_metal::DataType input_dtype,
        tt::tt_metal::DataType output_dtype,
        std::uint32_t batches = 1,
        tt::tt_metal::Tile tile = {});
};

// Per-input configuration for one call in a cross-call reduction sequence.
// Multiple entries may name the same input CB when the kernel refills or reuses that CB between calls.
struct ReduceCallConfig {
    ReduceBlockSpec block;
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

// One complete kernel reduce() invocation. All planner-selected behavior,
// including accumulation and partial-tile handling, is an explicit call
// property; a kernel never derives it from this call's position in a list or
// from the auxiliary recipe. `plan` is the complete existing single-CB plan for
// this input.
struct ReduceCallPlan {
    std::uint32_t input_cb_id;
    std::uint32_t auxiliary_cb_id;
    std::uint32_t auxiliary_tile_offset = 0;
    std::uint32_t output_cb_id;
    std::optional<std::uint32_t> accumulator_cb_id;
    ReduceAccumulationMode accumulation_mode = ReduceAccumulationMode::None;
    std::uint32_t accumulation_index = 0;
    ReducePlan plan;
};

struct ReduceSequencePlan {
    std::vector<ReduceCallPlan> calls;
    ReduceAuxiliaryPlan auxiliary;

    // Append the compute-kernel suffix: call count followed by every call in
    // execution order. Existing caller-owned arguments remain at the front.
    // The count describes this unit only; it carries no call semantics.
    void append_to(std::vector<std::uint32_t>& compile_time_args) const;
    std::vector<std::uint32_t> get_compile_time_args() const;

    // Append the independent dataflow-kernel suffix: one shared auxiliary CB
    // description for this complete planning unit, regardless of call count.
    void append_auxiliary_to(std::vector<std::uint32_t>& compile_time_args) const;
    std::vector<std::uint32_t> get_auxiliary_compile_time_args() const;
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

// Host serializer for one sequence-level auxiliary CB description. Its
// matching device view is ttnn::kernel_lib::ReduceAuxiliaryArgs<CTA_OFFSET>.
class ReduceAuxiliaryArgs {
public:
    explicit ReduceAuxiliaryArgs(const ReduceAuxiliaryPlan& auxiliary);

    void append_to(std::vector<std::uint32_t>& compile_time_args) const;
    std::vector<std::uint32_t> get_compile_time_args() const;

private:
    std::vector<std::uint32_t> compile_time_args_;
};

// Plan one local reduction. A missing input-CB cap means "use the available
// reduction-owned L1 budget". A supplied cap must be positive. Existing local
// buffers are described by block.resident_input_tiles / resident_output_tiles.
// INT32 and accurate FLOAT32 use SFPU SUM/MAX/MIN along W or H on non-Quasar
// devices. Accurate FLOAT32 AVG must be lowered to SUM plus its normalization
// scalar; SFPU HW reductions must be split into W and H. Tiled SFPU calls require
// a tile-aligned reduction axis: callers with partial inputs must identity-pad
// that axis and describe the padded view. Dense row-major staging already pads
// its input to the reduction identity before tilizing.
ReducePlan make_reduce_plan(
    const ReduceBlockSpec& block,
    tt::tt_metal::ReduceOpMath reduce_math,
    tt::tt_metal::ReduceOpDim reduce_dim,
    float scalar,
    ReduceFp32Mode fp32_mode,
    const ReduceHardwareConfig& hardware,
    std::optional<std::size_t> max_input_cb_bytes = std::nullopt);

// Plan a kernel-ordered sequence of reductions whose results are accumulated
// together. The returned call vector has exactly the same order and length as
// `reductions`; callers decide when to issue each reduce() call. Input CB IDs
// need not be unique. An explicit algorithm retains the corresponding
// accumulation order when a fused operation requires it for numerical accuracy.
ReduceSequencePlan make_reduce_sequence_plan(
    const std::vector<ReduceCbConfig>& reductions,
    const ReduceSequenceCbIds& cb_ids,
    const ReduceHardwareConfig& hardware,
    std::optional<compute_kernel_lib::ReduceAlgorithm> algorithm = std::nullopt);

}  // namespace ttnn::kernel_lib::host
