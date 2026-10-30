// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <tt-metalium/base_types.hpp>
#include <tt-metalium/experimental/metal2_host_api/compute_hardware_config.hpp>
#include <tt-metalium/tensor/spec/layout/layout.hpp>
#include <tt-metalium/tensor/tensor_types.hpp>
#include <tt-metalium/tile.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <umd/device/types/arch.hpp>

#include "ttnn/kernel_lib/reduce_types.hpp"
#include "ttnn/kernel_lib/reduce_plan_args_common.hpp"

/**
 * @file reduce_host.hpp
 * @brief Host-side planning and serialization for tiled reductions.
 *
 * How to use the reduce helpers. Worked example: ttnn/examples/reduce_helpers/reduce_helpers.cpp.
 *
 * Ownership
 *   Factory   Cores, tensors, the order in which the reader delivers input tiles, CB IDs, and the capacities of
 *             the input, output and accumulator CBs. It also owns L1 fit checks.
 *   Planner   make_reduce_plan() or make_reduce_sequence_plan(). Chooses the algorithm, partial-tile masking,
 *             scaling and accumulation, and sizes the auxiliary CB.
 *   Reader    Delivers input tiles in exactly the order the factory described to the planner.
 *   Writer    Materializes the planned auxiliary recipe once, then writes outputs.
 *   Compute   Issues compute_kernel_lib::reduce<Call>() for each planned call and makes no reduction decisions.
 *
 * Factory steps
 *   1. Fix the block each reduce call consumes on one core and the order the reader delivers it in.
 *   2. Describe it with ReduceBlockSpec, usually ReduceBlockSpec::tiled():
 *      - logical_h/logical_w are the valid elements, even when they are not multiples of 32. The planner then
 *        excludes the padding in the last tile from the result and divides AVG by the valid count. SFPU
 *        reductions are the exception (see Constraints).
 *      - input_cb_tiles is the input CB capacity you will allocate. Streaming (WaitAndPopPerTile) needs at least
 *        two tiles for TileSum, BulkWaitBulkPop whole packets, and retained policies the whole block.
 *      - output_group (H only) is the number of output columns reduced together in DEST, at most
 *        max_reduce_output_group(). The reader must then deliver strips of that many columns, row-major within
 *        each strip.
 *      - tail describes one smaller shape that some cores reduce with the same compiled kernel.
 *   3. Derive the hardware view from the compute kernel's own config:
 *      make_reduce_hardware_config(arch, compute_hardware). Do not restate its fields by hand.
 *   4. Plan. One reduction: make_reduce_plan(). Several inputs accumulated into one result, e.g. a mean over two
 *      tensors without concatenating them: make_reduce_sequence_plan(), one ReduceCbConfig per call in the order
 *      compute issues them. Leave the scalar absent for a mean; pass one only for an explicit scale.
 *   5. Allocate CBs. The auxiliary CB comes from the plan (plan.auxiliary_cb, or sequence.auxiliary.cb) and every
 *      plan needs one, including SFPU plans. Input, output and accumulator CBs are yours; reduce() pushes one
 *      output tile at a time. A sequence's accumulator must hold one call's complete per-core output
 *      (ReduceSequenceCbIds::accumulator_cb_tiles).
 *   6. Serialize and append after each kernel's own compile-time arguments:
 *      - Compute: ReduceCallArgs(plan, {input, auxiliary, output}) for a single plan, or
 *        sequence.append_to() / get_compile_time_args() for a sequence.
 *      - Writer: ReduceAuxiliaryArgs(plan, auxiliary) for a single plan, or
 *        sequence.append_auxiliary_to() / get_auxiliary_compile_time_args() for a sequence.
 *      - Tails: plan.append_runtime_args() or sequence.append_runtime_args() on every core, with use_tail
 *        selecting the tail. The returned base is the device view's RTA_OFFSET.
 *
 * Kernel side
 *   Compute  compute_kernel_hw_startup() once, then reduce<Call>() with Call = ReduceCallArgs<CTA_OFFSET> or, for a
 *            sequence, ReduceCallAtT<CTA_OFFSET, I> (reduce_plan_args.hpp). Other work may run between calls.
 *   Writer   dataflow_kernel_lib::prepare_reduce_auxiliary_tiles<ReduceAuxiliaryArgs<CTA_OFFSET>>() before compute
 *            needs the tiles (reduce_helpers_dataflow.hpp).
 *   Metal 2  Serialize dense logical CB IDs and bind them to dfb::<name> with BoundReduceCallArgs and
 *            BoundReduceAuxiliaryArgs, as the example does.
 *
 * Rules
 *   - Do not choose the algorithm, scale, mask or accumulation in the kernel, and do not infer them from the call
 *     index, the call count or the auxiliary recipe. Each serialized call carries all of them.
 *   - If the reader's order differs from the description, the results are wrong without any error. For example,
 *     an H reduction of a 2x8-tile block with output_group = 4 expects r0c0-r0c3, r1c0-r1c3, r0c4-r0c7, r1c4-r1c7.
 *     A plain row-major reader delivers r0c0-r0c7 first, so compute adds r0c4-r0c7 into columns 0-3.
 *   - Override the algorithm only through make_reduce_sequence_plan()'s algorithm argument, and only when a fused
 *     operation needs a specific accumulation order for accuracy. Never hand-roll a reduction loop instead.
 *   - Several planning units in one kernel are appended to both suffixes in planner-invocation order; each
 *     device view's next_compile_time_args_offset() locates the next unit.
 *   - The Python binding ttnn.reduce_planner exposes the same API for ProgramDescriptor ops.
 *
 * Constraints (the planner rejects violations)
 *   - INT32, accurate FLOAT32 and BFLOAT16 MIN run on the SFPU: W or H only, a tile-aligned reduction axis
 *     (identity-pad it and describe the padded view), no Quasar. Accurate FLOAT32 needs fp32 DEST; lower its AVG
 *     to SUM with an explicit 1/N scalar.
 *   - Accumulated calls share math, dimension, fp32 mode and scalar, and produce the same number of outputs.
 *
 * Serialized layout
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
 * The device kernels receive only these flat suffixes, not a ReduceSequencePlan object. Multiple input
 * descriptions may use the same CB when the kernel refills or reuses it between calls.
 */

namespace ttnn::kernel_lib::host {

inline constexpr std::uint32_t no_cb_id = reduce_plan_args::no_cb_id;

using ReduceAuxiliaryTileType = ttnn::kernel_lib::ReduceAuxiliaryTileType;

// Valid elements in one local block. The leading dimensions are flattened into
// batches, just as in ReduceBlockSpec. Empty cores should not issue a reduction.
struct ReduceValidShape {
    std::uint32_t height;
    std::uint32_t width;
    std::uint32_t batches = 1;
};

// Plan both the ordinary block and this exact tail. At runtime [0, 0, 0] selects
// the ordinary block; [height, width, batches] selects the tail on any core using
// the same compiled kernel. Both auxiliary recipes are prepared upfront.
// Output lanes beyond the tail's valid non-reduced extent are unspecified.
struct ReduceTailConfig {
    ReduceValidShape shape;
};

struct ReduceHardwareConfig {
    tt::ARCH arch = tt::ARCH::Invalid;
    bool fp32_dest_acc_en = false;
    bool dst_full_sync_en = false;
    // Must match the enclosing compute kernel. Defaults to Metal's ComputeConfig fidelity.
    tt::tt_metal::MathFidelity math_fidelity = tt::tt_metal::MathFidelity::HiFi4;
};

ReduceHardwareConfig make_reduce_hardware_config(
    tt::ARCH arch, const tt::tt_metal::experimental::ComputeHardwareConfig& compute);

struct ReduceChunkPlan {
    // Independent outputs retained in DEST for the chunk: the block's output_group.
    std::uint32_t output_tiles = 1;
    // Number of chunks required by the default input allocation.
    std::uint32_t buffers = 1;
};

// The auxiliary CB the caller allocates for every planned reduction. It is the only reduction CB
// whose size and format the planner decides: the caller sizes input (input_cb_tiles), output and
// accumulator CBs itself.
struct ReduceAuxiliaryCb {
    tt::DataFormat data_format = tt::DataFormat::Invalid;
    std::uint32_t page_size = 0;
    std::uint32_t page_count = 0;
    std::size_t total_size_bytes = 0;
};

// One concrete tile for the dataflow-side auxiliary recipe. The planner has
// already resolved its value, pattern and exact valid extent, including tails.
struct ReduceAuxiliaryTileSpec {
    float value = 0.0F;
    ReduceAuxiliaryTileType type = ReduceAuxiliaryTileType::Zero;
    std::uint32_t num_valid_elements = 0;
};

// Resolve the tile's precision on the host while retaining FP32 recipe encoding.
// BF16 values are rounded to nearest-even and widened; FP32 values are unchanged.
float round_reduce_auxiliary_value(float value, tt::DataFormat data_format);

// The one shared auxiliary CB recipe for a complete planning unit. It carries
// the CB ID as well as the physical tiles to materialize. Calls refer to
// contiguous slices of `tiles`; equal call recipes share the same slice.
struct ReduceAuxiliaryPlan {
    std::uint32_t cb_id = no_cb_id;
    std::vector<ReduceAuxiliaryTileSpec> tiles;
    ReduceAuxiliaryCb cb;
};

struct ReducePlan {
    tt::tt_metal::ReduceOpMath reduce_math = tt::tt_metal::ReduceOpMath::SUM;
    tt::tt_metal::ReduceOpDim reduce_dim = tt::tt_metal::ReduceOpDim::W;
    ReduceFp32Mode fp32_mode = ReduceFp32Mode::Fast;
    compute_kernel_lib::ReduceAlgorithm algorithm = compute_kernel_lib::ReduceAlgorithm::Native;
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
    std::optional<ReduceTailConfig> tail;
    // The ordinary case is stored in this plan; the alternative is selected
    // at runtime. auxiliary_tiles contains the union of both recipes.
    std::shared_ptr<ReducePlan> tail_plan;
    std::uint32_t full_auxiliary_tile_count = 0;
    std::uint32_t tail_auxiliary_tile_offset = 0;
    // Assigned by the planner within the sequence's runtime section.
    std::uint32_t tail_runtime_arg_offset = reduce_plan_args::no_runtime_arg;
    std::uint32_t logical_h = 0;
    std::uint32_t logical_w = 0;

    // Multiplies each finished output before any caller callback on the TileSum and
    // SFPU paths: 1/N for an AVG, or the caller's explicit scalar. Native carries its
    // scale in the scaler tile and keeps 1. A non-unit INT32 scale converts the
    // reduced value to FLOAT32, multiplies, then converts back to INT32 with
    // truncation toward zero.
    float scale = 1.0F;
    compute_kernel_lib::ReducePartialMode partial_mode = compute_kernel_lib::ReducePartialMode::None;
    std::vector<ReduceAuxiliaryTileSpec> auxiliary_tiles;
    std::uint32_t partial_reduce_axis_elements = 0;

    // SFPU reductions read no auxiliary tile; they plan one placeholder so that every
    // reduction allocates and binds the same CB.
    ReduceAuxiliaryCb auxiliary_cb;
    std::uint32_t input_cb_tiles = 0;
    tt::DataFormat input_format = tt::DataFormat::Invalid;

    // Use the same compiled plan on full and tail cores. Initialize the runtime
    // marker on all cores: [0, 0, 0] selects full work, [height, width, batches] selects tail.
    // Both have three words, so arguments that follow sit at the same offset on every core.
    std::vector<std::uint32_t> get_runtime_shape_args(bool use_tail = true) const;
    // Append a standalone plan's record and return its base for ReduceCallArgs<CTA_OFFSET, RTA_OFFSET>.
    std::uint32_t append_runtime_args(std::vector<std::uint32_t>& runtime_args, bool use_tail = true) const;
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
    tt::tt_metal::Tile input_tile;
    tt::tt_metal::Tile output_tile;
    // Retained input policies only. Zero means contiguous at padded_w.
    std::uint32_t input_row_stride_tiles = 0;
    // Capacity of the input CB in tiles. Retained policies need the whole block,
    // BulkWaitBulkPop a whole number of bulk packets (Ht * output_group tiles for
    // H, Wt for W, Ht * Wt for HW), and streaming TileSum at least two tiles; a
    // one-tile streaming CB plans Native.
    std::uint32_t input_cb_tiles = 0;
    // H reduction only: output columns reduced together in DEST, at most
    // max_reduce_output_group(). Streaming and bulk inputs must arrive as strips of
    // this many columns, row-major within each strip; bulk input also requires it to
    // divide the block's (and tail's) column count. Streaming TileSum requires 1.
    // W and HW reductions require 1.
    std::uint32_t output_group = 1;
    // Absent: reduce the whole logical block. Present: also plan this known
    // smaller shape, including its normalization and auxiliary tiles. Runtime
    // arguments select full or tail work on each core of the same kernel grid.
    // In an accumulated sequence, the configured tails form one alternative
    // scenario and must be enabled together; calls without tails retain their
    // geometry but use that scenario's combined AVG divisor. Retained input
    // retains the full block's pitches. FIFO producers use the common planned
    // packet size and pad unused pages at the final axis/output group.
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
    // Explicit: use this scale unchanged; AVG computes scalar * sum(valid inputs).
    // Absent: AVG uses 1 / the valid reduction extent (the combined extent for
    // accumulated calls); other operations use 1. Geometry still determines masks.
    std::optional<float> scalar = std::nullopt;
    ReduceFp32Mode fp32_mode;
    compute_kernel_lib::ReduceInputPolicy input_policy = compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile;
};

using ReduceCbConfig = std::pair<std::uint32_t, ReduceCallConfig>;

// These IDs bind planner roles into the caller's kernel CB namespace. They are explicit because only the
// caller knows which IDs are already occupied by the rest of a fused kernel.
struct ReduceSequenceCbIds {
    std::uint32_t auxiliary_cb_id;
    std::uint32_t accumulator_cb_id;
    std::uint32_t output_cb_id;
    // Capacity of the accumulator CB in tiles; read only when the sequence accumulates. Every call
    // packs all of its partials before the next call reloads any, so it must hold one call's
    // complete per-core output.
    std::uint32_t accumulator_cb_tiles = 0;
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
    // Contiguous, deduplicated tail records; call descriptors carry their relative offsets.
    std::vector<std::uint32_t> tail_runtime_args;

    // Static sequences return no arguments; tail work returns all records and full work returns
    // as many zeros, so arguments that follow sit at the same offset on every core.
    std::vector<std::uint32_t> get_runtime_shape_args(bool use_tail = true) const;
    // Append after caller-owned arguments and return the section base for RTA_OFFSET.
    std::uint32_t append_runtime_args(std::vector<std::uint32_t>& runtime_args, bool use_tail = true) const;

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
    ReduceAuxiliaryArgs(const ReducePlan& plan, std::uint32_t auxiliary_cb_id);

    void append_to(std::vector<std::uint32_t>& compile_time_args) const;
    std::vector<std::uint32_t> get_compile_time_args() const;

private:
    std::vector<std::uint32_t> compile_time_args_;
};

// Largest ReduceBlockSpec::output_group the hardware's DEST holds for this reduction.
std::uint32_t max_reduce_output_group(
    const ReduceHardwareConfig& hardware,
    tt::tt_metal::DataType input_dtype,
    tt::tt_metal::ReduceOpMath reduce_math,
    ReduceFp32Mode fp32_mode);

// Plan one local tiled reduction using the requested input policy. The planner
// reports the auxiliary CB it needs; the factory owns allocation and L1 fit checks.
// Tilization and row-major staging belong to the caller.
// INT32 and accurate FLOAT32 use SFPU SUM/MAX/MIN, and BFLOAT16 MIN uses SFPU MIN,
// along W or H on non-Quasar devices. Accurate FLOAT32 AVG must be lowered to SUM plus its normalization
// scalar; SFPU HW reductions must be split into W and H. Tiled SFPU calls require
// a tile-aligned reduction axis: callers with partial inputs must identity-pad
// that axis and describe the padded view.
// scalar follows ReduceCallConfig: an explicit scale overrides AVG normalization.
ReducePlan make_reduce_plan(
    const ReduceBlockSpec& block,
    tt::tt_metal::ReduceOpMath reduce_math,
    tt::tt_metal::ReduceOpDim reduce_dim,
    std::optional<float> scalar,
    ReduceFp32Mode fp32_mode,
    const ReduceHardwareConfig& hardware,
    compute_kernel_lib::ReduceInputPolicy input_policy = compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile);

// Plan a kernel-ordered sequence of reductions whose results are accumulated
// together. The returned call vector has exactly the same order and length as
// `reductions`; callers decide when to issue each reduce() call. Input CB IDs
// need not be unique. An explicit algorithm retains the corresponding
// accumulation order when a fused operation requires it for numerical accuracy.
// All calls must either omit scalar or supply the same explicit scalar.
ReduceSequencePlan make_reduce_sequence_plan(
    const std::vector<ReduceCbConfig>& reductions,
    const ReduceSequenceCbIds& cb_ids,
    const ReduceHardwareConfig& hardware,
    std::optional<compute_kernel_lib::ReduceAlgorithm> algorithm = std::nullopt);

}  // namespace ttnn::kernel_lib::host
