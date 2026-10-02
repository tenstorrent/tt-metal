// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include <tt_stl/small_vector.hpp>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::data_movement::repeat_codegen {

// Whether a buffer of `logical_shape`/`layout` placed at `memory_config` has the same page grid its
// interleaved twin would have. The codegen kernels address every buffer by page id through a
// TensorAccessor, so such a placement is read or written in place with no reshard hop. Always true
// for an interleaved config.
bool shard_spec_is_page_identical(
    const MemoryConfig& memory_config, const ttnn::Shape& logical_shape, tt::tt_metal::Layout layout);

// A TILE input whose H or W is sub-tile, repeated along H or W. Copying tile pages along such an axis
// would interleave its tile padding into the logical result, so the whole repeat runs row-major
// between one untilize and one retilize.
bool needs_row_major_round_trip(const Tensor& input, const ttsl::SmallVector<uint32_t>& repeat_dims);

// Whether a caller-supplied output carries the page the codegen path would have built itself: the
// input's dtype, layout and tile. Every CB slot and transfer is cut from the output page while the
// kernels address the input's geometry, so any other destination is silently mis-strided. Tile
// shape and transpose flags are compared one by one; Tile's equality ignores the transpose flags.
bool output_matches_input_page(const Tensor& input, const Tensor& output);

// How a whole repeat call decomposes into single-dim prim::repeat_codegen legs. The router executes
// it and the whole-call gate budgets L1 against it, so the two cannot disagree on which buffers a
// leg's CB shares L1 with.
struct CodegenLegPlan {
    // A sharded input that cannot be read in place is unsharded to interleaved DRAM before any leg.
    bool unshard_input = false;
    bool round_trip = false;
    // Placement of every leg output but the last.
    MemoryConfig intermediate_mc;
    // Placement of the last leg's output: the requested one when its pages line up with it, else
    // interleaved in the requested buffer type, followed by one placement hop.
    MemoryConfig final_mc;
    bool final_in_place = false;
    // Whether the last leg can write a preallocated output directly. A fold leaves the last leg's
    // shape short of the requested one, which the leg's output validation rejects.
    bool final_into_prealloc = false;
    // Per-axis leg repeat counts. A repeat of a size-1 axis followed by one of the next axis is folded
    // into the latter as the product: both orders visit the output pages identically, and the fold
    // saves a whole-tensor leg. The folded axis keeps count 1 and the router views the result back to
    // the requested shape.
    ttsl::SmallVector<uint32_t> leg_repeats;
    // Axes with a leg, in execution order.
    std::vector<uint32_t> rep_dims;
    // How many legs run row-major: all of them for a ROW_MAJOR input and on the round trip, none for a
    // TILE input copying tile pages. The round trip runs its outer-axis legs row-major too, between the
    // one untilize and the one retilize, since copying padded tile planes moves more bytes.
    size_t row_major_legs = 0;
};

CodegenLegPlan plan_codegen_legs(
    const Tensor& input, const ttsl::SmallVector<uint32_t>& repeat_dims, const MemoryConfig& output_mem_config);

// Correctness gate for a single-dim codegen repeat step, as seen by
// prim::repeat_codegen: `input` is already reshaped into the 4D-padded space
// its kernels assume, so rep_dim is in [0, 3]. `output_mem_config` is where the
// step's output lands.
bool supported_by_codegen(
    const Tensor& input, uint32_t rep_dim, uint32_t num_repeats, const MemoryConfig& output_mem_config);

// Correctness gate for a whole (possibly multi-dim) ttnn::repeat call, on the
// original tensor/repeat vector before per-dim decomposition and 4D padding.
// Consulted by the free function's routing and by repeat_force_codegen, which both act on it alone:
// it rejects a zero repetition and a sharded output with no shard_spec itself.
bool supported_by_codegen(
    const Tensor& input, const ttsl::SmallVector<uint32_t>& repeat_dims, const MemoryConfig& output_mem_config);

// Routing-only, like is_demoted: whether each row-major leg's CB still gets at least one slot from
// the L1 free now, net of the buffers the call has yet to allocate. supported_by_codegen budgets the
// static window so that it agrees with itself across a cache miss; this keeps a call dispatched under
// L1 pressure off a codegen leg whose CB would not place, and on native, which stages a narrower stick.
// `output_preallocated` says the call's output already exists; a last leg that writes it in place then
// adds nothing to what the free window has already paid for.
bool row_major_cbs_fit_free_l1(
    const Tensor& input,
    const ttsl::SmallVector<uint32_t>& repeat_dims,
    const MemoryConfig& output_mem_config,
    bool output_preallocated);

// Perf-demotion gate: correct but not worth the codegen path. Routing-only --
// consulted by ttnn::repeat only, never by validate and never by
// repeat_force_codegen. `output_mem_config` is the placement the call resolves to.
bool is_demoted(
    const Tensor& input, const ttsl::SmallVector<uint32_t>& repeat_dims, const MemoryConfig& output_mem_config);

}  // namespace ttnn::operations::data_movement::repeat_codegen
