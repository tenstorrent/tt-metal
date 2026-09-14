// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <utility>

#include "ttnn/operations/data_movement/squeeze/squeeze.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"

#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"

namespace tt::tt_metal {
struct ProgramDescriptor;
struct TileDescriptor;
}  // namespace tt::tt_metal

namespace ttnn::operations::data_movement {

ttnn::Shape squeeze_shape_to_ND(const ttnn::Shape& output_shape, uint32_t);
ttnn::Shape squeeze_shape_to_4D(const ttnn::Shape& output_shape);
ttnn::Shape squeeze_shape_to_3D(const ttnn::Shape& output_shape);
ttnn::Tensor squeeze_from_ND_to_4D(
    const ttnn::Tensor& tensor, const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
ttnn::Shape unsqueeze_shape_to_3D(const ttnn::Shape& shape);
ttnn::Shape unsqueeze_shape_to_4D(const ttnn::Shape& shape);

ttnn::Shape unsqueeze_shape_to_nd(const ttnn::Shape& shape, uint32_t n);

ttnn::Shape squeeze_or_unsqueeze_shape_to_ND(const ttnn::Shape& shape, uint32_t n);

// Estimate NOC transfer cycles for a batch of transactions.
// Returns {bw_cycles, latency_cycles} — BW is the steady-state transfer time,
// latency is the per-transaction pipeline startup cost. Callers can model
// pipelining by separating these: max(bw_terms...) + sum(latency_terms).
std::pair<uint32_t, uint32_t> get_cycles_for_transaction_size(
    uint32_t transaction_size, bool is_dram, bool is_local, uint32_t num_transactions, tt::ARCH arch, bool is_read);
int common_tm_bw_model(
    const Tensor& input_tensor,
    const Tensor& output_tensor,
    bool output_only = false,
    int compute_cycles = 0,
    bool per_faceline = false,
    bool split_op = false,
    bool bcast_local = false,
    bool concat_op = false);

// Extra staging CBs (e.g. tilize block factory's c_1) must pass staging_bytes_per_tile / fixed_staging_bytes.
uint32_t get_estimated_size_of_cbs(
    const Tensor& input_tensor_a,
    uint32_t input_single_tile_size,
    uint32_t output_single_tile_size,
    uint32_t num_tiles_per_row,
    uint32_t staging_bytes_per_tile = 0,
    uint32_t fixed_staging_bytes = 0);

uint32_t get_max_l1_space(const Tensor& input_tensor_a);

// One set of buffers, sized for a single block width.
//
// The `_multi_core_block[_interleaved]` tilize and untilize factories split work into blocks and
// hand cores one of two block widths (the full width, and a narrower cliff-row width). A block's
// width fixes the size of the buffers that carry it, and that size is a *correctness* property
// rather than a performance one, because one endpoint of the block buffer walks a whole block as
// raw contiguous L1 rather than page by page:
//
//   * tilize: the reader fills a whole block through one raw linear write starting at
//     `get_write_ptr()`, and `cb_push_back` requires the producer to write contiguously -- it only
//     wraps when the write pointer lands exactly on `fifo_limit` ("producer always writes into
//     contiguous memory, it cannot wrap").
//   * untilize: the writer takes `get_read_ptr()` once per block and walks the block's rows
//     forward from it, so the consumer likewise reads a contiguous run and cannot wrap mid-block.
//
// Either way, a buffer whose size is not an exact multiple of the block pushed into it overruns
// into its neighbour instead of wrapping.
//
// So each block width gets its own set of buffers, with its own indices, sized for that width,
// rather than one set of indices re-used at different sizes on disjoint cores (issue #51305). The
// sets live on disjoint core ranges, so L1 usage is unchanged: a core allocates only the set
// belonging to its own block width.
//
// This is shared rather than private to each factory because every factory using the model must
// apply the *same* index and sizing rules -- a private copy can drift and reintroduce the
// corruption the split prevents.
struct BlockBufferSet {
    // Per-row DRAM-alignment staging buffer (a reader-private scratchpad). Only the tilize
    // direction needs it: there the *reader* touches DRAM and must fix up alignment before the
    // block lands in L1. Untilize reads whole tile pages in and writes sticks out, so it has no
    // staging buffer and leaves this unset -- `push_buffer_set` then emits only input and output.
    std::optional<uint8_t> staging_index;
    uint8_t input_index = 0;   // row-major block the reader fills and compute tilizes (untilize: the
                               // tilized block the reader fills and compute untilizes)
    uint8_t output_index = 0;  // block compute produces and the writer drains
    uint32_t block_tiles = 0;  // block width in tiles -- the page count of input/output
    tt::tt_metal::CoreRangeSet core_ranges;

    bool empty() const { return core_ranges.empty(); }
};

// The block work split plus the two buffer sets derived from it.
//
// Shared so every `_multi_core_block[_interleaved]` factory -- tilize, tilize_with_val_padding,
// untilize and untilize_with_unpadding -- applies one set of index and sizing rules. A private copy
// per factory drifts, and a CB-index or sizing change landing in only some of them silently
// reintroduces the overrun BlockBufferSet exists to prevent. What legitimately differs between the
// callers is passed in, not forked: see BlockDirection and BlockCoreOrder.
//
// Tile sizes are passed in rather than derived from a Tile: the factories compute them differently
// (`Tile::get_tile_size`, which folds in runtime L1 alignment, versus the constexpr
// `tt::tile_size`) and unifying that here would quietly change CB sizes.
//
// Call this only on a cache miss. It is **not** reproducible on a later cache hit: the block-size
// limit folds in `get_max_l1_space`, which reads live L1 occupancy
// (`lowest_occupied_compute_l1_address`), and the program cache does not key on that. Two calls
// with identical attributes and tensor specs can therefore split differently, so anything the
// cache-hit hook needs must be recorded in the program at miss time rather than recomputed.
struct BlockPlan {
    ttnn::BlockSplitWH split;
    BlockBufferSet full;
    BlockBufferSet cliffrow;

    // Reader/writer kernels are emitted as one (reader, writer) pair per non-empty set, in
    // full-then-cliffrow order, ahead of the compute kernels.
    uint32_t num_dm_pairs() const { return (full.empty() ? 0u : 1u) + (cliffrow.empty() ? 0u : 1u); }
};

// Which way the block factory runs. This selects the two things that genuinely differ between the
// directions, both of which are correctness-relevant:
//
//   * Staging. Only the tilize direction's reader touches DRAM row by row and has to fix up
//     alignment before the block lands in L1, so only it gets a staging buffer -- and only it has
//     to reserve room for one when computing the block-size limit. Untilize reads whole tile pages
//     in and writes sticks out, so its sets leave `staging_index` unset.
//   * Which shape drives the split. Tilize splits over the *output* (tiled) shape, untilize over
//     the *input* (tiled) shape. For plain untilize the two are the same, but for
//     untilize_with_unpadding the output is smaller, and splitting over it would under-count the
//     tiles that actually have to be read.
enum class BlockDirection : uint8_t {
    Tilize,
    Untilize,
};

// The order the work split hands cores out in. A factory's runtime-arg loop must walk its cores in
// the SAME order the split assigned them, or a core's args land on another core's buffer set. The
// two orders are not interchangeable, so each factory passes the one its loop already uses:
//
//   ColumnMajor - `corerange_to_cores(grid)`, i.e. the CoreRangeSet work-split overload. Pair with
//                 a loop over `corerange_to_cores(available_grid)`.
//   RowMajor    - the CoreCoord work-split overload, which walks (x, y) across the full grid. Pair
//                 with a loop over `grid_to_cores(ncores, x, y, /*row_wise=*/true)`.
//
// RowMajor cannot be combined with `sub_core_grids`, since the CoreCoord overload has no way to
// honour a restricted grid.
enum class BlockCoreOrder : uint8_t {
    ColumnMajor,
    RowMajor,
};

BlockPlan make_block_plan(
    BlockDirection direction,
    BlockCoreOrder core_order,
    const Tensor& input_tensor,
    const Tensor& output_tensor,
    uint32_t input_single_tile_size,
    uint32_t output_single_tile_size,
    uint32_t tile_height,
    uint32_t tile_width,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids);

// The buffer set whose block width this core was assigned. Asserts the core belongs to exactly one
// set: an unmatched core silently defaulting to `full_set` would bind the wrong-width buffers, and
// a downstream size check only catches that when the widths happen to differ numerically.
const BlockBufferSet& buffer_set_for_core(const BlockPlan& plan, const tt::tt_metal::CoreCoord& core);

// Append the CBDescriptors for one buffer set: input, output, and -- only if the set declares a
// `staging_index` -- the staging buffer ahead of them. `dram_alignment` and `tile_height` size the
// staging buffer and are unused by a set without one.
//
// `tile` is the only other factory-specific piece -- a factory that supports non-default tile
// shapes passes its TileDescriptor so the input/output buffers carry it; one that is fixed to the
// standard tile leaves it unset.
void push_buffer_set(
    tt::tt_metal::ProgramDescriptor& desc,
    const BlockBufferSet& set,
    uint32_t input_single_tile_size,
    uint32_t output_single_tile_size,
    tt::DataFormat input_cb_data_format,
    tt::DataFormat output_cb_data_format,
    uint32_t dram_alignment,
    uint32_t tile_height,
    const std::optional<tt::tt_metal::TileDescriptor>& tile = std::nullopt);

// reserved_l1_bytes_per_core: per-core L1 that is not yet allocated at the time of this
// call but provably will be before the program's circular buffers are placed -- most
// notably the op's own output buffer. Interleaved L1 buffers are allocated top-down while
// static CBs grow bottom-up from the allocator base, so ignoring the pending output
// buffer overestimates the room available to the CBs and can route to a factory whose
// CBs then collide with it.
bool is_enough_space(
    const Tensor& input_tensor_a,
    uint32_t input_single_tile_size,
    uint32_t output_single_tile_size,
    uint32_t num_tiles_per_row,
    uint32_t staging_bytes_per_tile = 0,
    uint32_t fixed_staging_bytes = 0,
    uint32_t reserved_l1_bytes_per_core = 0);

// Per-core L1 footprint that `output_memory_config` will require for a tensor of
// `output_padded_shape`/`output_dtype`, or 0 if it will not live in L1.
//
// If the TensorSpec cannot be constructed (unsupported dtype/layout combination) and
// `require_constructible` is false, this falls back to reserving nothing -- the pre-existing
// behavior for callers that treat a failed reservation as advisory. Callers that use the return
// value to decide eligibility for a path with no other correctness backstop (e.g. concat's
// unaligned-width routing) should pass `require_constructible = true` so an unconstructible spec
// throws instead of silently making the eligibility check more permissive than it should be.
uint32_t get_pending_l1_output_reservation(
    const Tensor& input_tensor_a,
    const ttnn::Shape& output_padded_shape,
    const MemoryConfig& output_memory_config,
    DataType output_dtype,
    Layout output_layout,
    bool require_constructible = false);

ttnn::Tensor pad_to_tile_vol(
    const ttnn::Tensor& tensor, float value, bool use_multicore, const std::optional<MemoryConfig>& memory_config);

uint32_t wrap_index(int index, int size);

uint16_t float_to_uint16(float f);

uint32_t pack_two_uint16_into_uint32(std::pair<uint16_t, uint16_t> two_uint16s);

template <typename OpOutputType, typename... OpInputTypes>
struct MassagedOperationParams {
    using OwnedArgsType = std::tuple<std::decay_t<OpInputTypes>...>;
    using PredicateFunc = std::function<bool(OpInputTypes...)>;
    using PreTransformFunc = std::function<OwnedArgsType(OpInputTypes...)>;
    using PostTransformFunc = std::function<OpOutputType(const OpOutputType&)>;
    using OpType = std::function<OpOutputType(OpInputTypes...)>;

    PredicateFunc predicate;           // Function to determine if formatting should be applied
    PreTransformFunc pre_transform;    // Function to pre-process input arguments
    PostTransformFunc post_transform;  // Function to post-process the operation output
    OpType operation;                  // The main operation to be performed
};

template <typename OpOutputType, typename... OpInputTypes>
class MassagedOperation {
public:
    using OwnedArgsType = std::tuple<std::decay_t<OpInputTypes>...>;
    using PredicateFunc = std::function<bool(OpInputTypes...)>;
    using PreTransformFunc = std::function<OwnedArgsType(OpInputTypes...)>;
    // post transform takes the output and optionally the args; it may use
    // the args in order to know if it needs to post process the output.
    using PostTransformFunc = std::function<OpOutputType(const OpOutputType&)>;
    using OpType = std::function<OpOutputType(OpInputTypes...)>;

    MassagedOperation(const MassagedOperationParams<OpOutputType, OpInputTypes...>& params) :
        predicate_(params.predicate),
        pre_transform_(params.pre_transform),
        post_transform_(params.post_transform),
        operation_(params.operation) {}

    bool should_format(OpInputTypes... args) const { return predicate_(args...); }

    OwnedArgsType pre_format(OpInputTypes... args) const { return pre_transform_(args...); }

    OpOutputType post_format(const OpOutputType& output) const { return post_transform_(output); }

    OpOutputType operator()(OpInputTypes... args) const {
        if (should_format(args...)) {
            auto formatted_input = pre_format(args...);
            auto op_output = std::apply(operation_, formatted_input);
            return post_format(op_output);
        }
        return operation_(args...);
    }

    MassagedOperation sequence(const MassagedOperation& other) {
        std::shared_ptr<bool> t1_required = std::make_shared<bool>(false);
        std::shared_ptr<bool> t2_required = std::make_shared<bool>(false);
        std::shared_ptr<bool> t1_then_t2_required = std::make_shared<bool>(false);

        auto merged_predicate =
            [p1 = this->predicate_, p2 = other.predicate_, t1_required, t2_required](OpInputTypes... args) -> bool {
            if (p1(args...)) {
                *t1_required = true;
            }
            if (p2(args...)) {
                *t2_required = true;
            }
            return *t1_required or * t2_required;
        };

        auto merged_pre_transform = [t1 = this->pre_transform_,
                                     t2 = other.pre_transform_,
                                     p1 = this->predicate_,
                                     p2 = other.predicate_,
                                     t1_required,
                                     t2_required,
                                     t1_then_t2_required](OpInputTypes... args) -> OwnedArgsType {
            if (*t1_required) {
                auto transformed_args = t1(args...);
                if (std::apply(p2, transformed_args)) {
                    *t1_then_t2_required = true;
                    return std::apply(t2, transformed_args);
                }
                return transformed_args;
            }
            if (*t2_required) {
                return t2(args...);
            }
            return std::make_tuple(args...);
        };

        auto merged_post_transform =
            [t1 = this->post_transform_, t2 = other.post_transform_, t1_then_t2_required, t1_required, t2_required](
                OpOutputType output) -> OpOutputType {
            if (*t1_then_t2_required) {
                // we go backwards for post-transform
                auto t2_output = t2(output);
                auto t1_output = t1(t2_output);
                return t1_output;
            }
            if (*t1_required) {
                return t1(output);
            }
            if (*t2_required) {
                return t2(output);
            }
            return output;
        };

        return MassagedOperation(MassagedOperationParams<OpOutputType, OpInputTypes...>{
            .predicate = merged_predicate,
            .pre_transform = merged_pre_transform,
            .post_transform = merged_post_transform,
            .operation = this->operation_});
    }

    // getters for all private members
    PredicateFunc get_predicate() const { return predicate_; }
    PreTransformFunc get_pre_transform() const { return pre_transform_; }
    PostTransformFunc get_post_transform() const { return post_transform_; }
    OpType get_operation() const { return operation_; }

    // setters for all private members
    void set_predicate(PredicateFunc predicate) { predicate_ = std::move(predicate); }
    void set_pre_transform(PreTransformFunc pre_transform) { pre_transform_ = pre_transform; }
    void set_post_transform(PostTransformFunc post_transform) { post_transform_ = post_transform; }
    void set_operation(OpType operation) { operation_ = operation; }

private:
    PredicateFunc predicate_;
    PreTransformFunc pre_transform_;
    PostTransformFunc post_transform_;
    OpType operation_;
};

ttnn::Shape compute_padded_shape(
    ttnn::Shape logical_shape,
    uint32_t tile_height = tt::constants::TILE_HEIGHT,
    uint32_t tile_width = tt::constants::TILE_WIDTH);

/**
 * Pads a shape to align with tile dimensions
 * @param unpadded_shape Original shape to be padded
 * @return Padded shape aligned to tile dimensions
 */
ttnn::Shape pad_to_tile_shape(const ttnn::Shape& unpadded_shape);

enum class ShardStrategy { BLOCK, HEIGHT, WIDTH };

// Helper function for creating a sharded memory configuration for a tensor
// based on its logical shape, a shard strategy and orientation, and a core
// grid. Optionally, you may pass a preferred shard shape to use. If not
// provided, the shard shape will be inferred from the tensor shape and the
// shard strategy.
ttnn::MemoryConfig create_sharded_memory_config(
    const ttnn::Shape& logical_shape,
    const tt::tt_metal::CoreRangeSet& core_grid,
    const ShardStrategy& strategy,
    const tt::tt_metal::ShardOrientation& orientation,
    std::optional<std::array<uint32_t, 2>> shard_shape = std::nullopt,
    const tt::tt_metal::Layout& layout = tt::tt_metal::Layout::ROW_MAJOR);

std::pair<uint32_t, std::array<uint32_t, 2>> tensor_coord_to_height_sharded_coord(
    const std::span<const uint32_t>& tensor_shape,
    const std::span<const uint32_t>& shard_shape,
    const std::span<const uint32_t>& tensor_coord);

uint32_t get_num_pages(const ttnn::Tensor& tensor);

// B/W-sh → shard_W*E (feeds split branch); other sharded → buffer->aligned_page_size() (16-aligned L1 stride).
uint32_t per_shard_page_size_bytes(const ttnn::Tensor& tensor, uint32_t row_bytes);

}  // namespace ttnn::operations::data_movement
