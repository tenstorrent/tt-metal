// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <optional>

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct ScatterCodegenParams;
struct ScatterCodegenInputs;

// The runtime page-map descriptor is a fixed-width [rank, input_dims x8, index_dims x8] triple
// (scatter_common.hpp's map_scatter_input_page), so it can only address a page ordinal folded from
// at most this many leading dimensions.
constexpr uint32_t kScatterMaxPageRank = 8;
constexpr uint32_t kScatterPageMapSize = 1 + 2 * kScatterMaxPageRank;
using ScatterPageMap = std::array<uint32_t, kScatterPageMapSize>;

// Number of leading page dimensions a tensor of this (already-normalized) logical shape folds to:
// TILE drops the last two dims (W, H) and replaces H with its tile-row count; ROW_MAJOR drops only
// the last dim (W). Both leave `rank - 1` page dimensions for rank >= 2, and a single unit dimension
// for rank <= 1.
uint32_t scatter_page_rank(const ttnn::Shape& shape, bool tiled);

// Builds the fixed-width logical-prefix page map shared by every reader: [rank, input_dims
// (zero-padded to kScatterMaxPageRank with 1s), index_dims (likewise)]. `tiled` selects between the
// TILE folding (H collapsed to tile-rows) and the ROW_MAJOR folding (no collapse). Clamps a page
// rank past kScatterMaxPageRank rather than failing: it runs ahead of the prim's own checks
// (through build_scatter_codegen_params), and the device operation's cache-miss checks are where
// the equal-rank and max-rank TT_FATALs live.
ScatterPageMap compute_scatter_page_map(const ttnn::Shape& input_shape, const ttnn::Shape& index_shape, bool tiled);

// Tile-page geometry, computed from the (already dim==-1, already-normalized) input/index/src
// tensors' PADDED shapes. TILE-only.
struct ScatterTileGeometry {
    uint32_t Ht = 0;
    uint32_t Wt_output = 0;
    uint32_t Wt_index = 0;
    uint32_t output_logical_w = 0;
    uint32_t idx_valid_h_last = 0;
    uint32_t idx_valid_w_last = 0;
    uint32_t Ht_per_batch_index = 0;
};
ScatterTileGeometry compute_scatter_tile_geometry(
    const Tensor& input_tensor, const Tensor& index_tensor, const Tensor& src_tensor);

// Stick geometry for the ROW_MAJOR factories, computed from LOGICAL shapes (ROW_MAJOR tensors carry
// no tile padding on any dimension).
struct ScatterRmGeometry {
    uint32_t num_sticks = 0;
    uint32_t input_stick_elems = 0;
    uint32_t index_stick_elems = 0;
};
ScatterRmGeometry compute_scatter_rm_geometry(const Tensor& input_tensor, const Tensor& index_tensor);

// The output buffer's ALIGNED page size, usable before the output buffer itself exists.
// The output spec always carries the input's own dtype and tile, so the unaligned
// page size is the input's; only the alignment can differ, since Buffer::alignment() is
// BufferType-dependent (tt_metal/impl/buffers/buffer.cpp: DRAM and L1 draw from distinct allocator
// alignments) and the caller's output_mem_config (or a preallocated output) need not share the
// input's buffer type. A preallocated output already has a real buffer to ask directly.
uint64_t scatter_output_aligned_page_size(
    const Tensor& input_tensor,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<Tensor>& output_tensor);

// The ALIGNED per-page byte stride a ROW_MAJOR stick of `stick_elems` elements occupies in `tensor`'s
// buffer type -- the single formula both the feasibility gate (evaluated pre-transpose, before the
// stick's own buffer exists) and the RM program factories (post-transpose, reading an already-
// allocated buffer) must agree on, since Buffer::aligned_page_size() rounds page_size() up to a
// BufferType-only device alignment (tt_metal/impl/buffers/buffer.cpp) with no dependence on which
// dimension the stick was drawn from.
uint64_t scatter_rm_stick_page_bytes(const Tensor& tensor, uint32_t stick_elems);

// The STATIC per-core CB budget: the allocator-managed L1 bank, ignoring what is live in it, so a
// buffer allocated between two evaluations cannot change the answer. What supported_by_codegen()
// plans against, because it is evaluated at the router and again in the prim's validate with this
// call's own create_output_tensors() in between; the forced entry and direct prim callers see
// nothing else.
uint64_t scatter_static_l1(const Tensor& input_tensor);

// The per-core CB ceiling the next program will actually find: the static budget less whatever is
// live in L1 right now. Read by the routing gates that deduct this call's own pending allocations
// (is_demoted()'s detour check, codegen_plan_fits_live_l1), by compute_program_hash on every call so
// a moved frontier keys a fresh plan, and by the factories that size CBs from it.
uint64_t scatter_usable_l1(const Tensor& input_tensor);

// Whether the TILE interleaved plan's four-CB footprint -- Wt_output-deep output and input, one-tile
// index and src -- fits the given L1 budget.
bool scatter_interleaved_fits_l1(
    uint64_t l1_budget,
    uint32_t Wt_output,
    uint64_t output_page_bytes,
    uint64_t input_page_bytes,
    uint64_t index_page_bytes,
    uint64_t src_page_bytes);

// RM: the bytes a ROW_MAJOR factory keeps resident per core whatever the chunking -- the input and
// output sticks, plus a same-sized FP32 accumulator (2x a BF16 page) for the bf16-reduce factory.
// The one formula the routing gate, the program hash and both RM factories size from.
uint64_t scatter_rm_fixed_bytes(uint64_t input_page_bytes, uint64_t output_page_bytes, bool bf16_reduce);

// RM: whether the smallest viable plan -- scatter_rm_fixed_bytes() plus the smallest
// NOC-alignment-safe index/src chunk (32 elements) -- fits `l1_budget`, the static bank for the
// scope gate or the live frontier less pending allocations for the routing gates. The sticks stay
// fully resident by construction (no shallower RM plan exists to scale down to), so a call this
// rejects has no feasible RM dispatch at all and the caller must send it to native.
bool scatter_rm_min_plan_fits_l1(
    uint64_t l1_budget, uint64_t input_page_bytes, uint32_t index_elem_size, uint32_t src_elem_size, bool bf16_reduce);

// RM: index/src CB depth in elements, scaled down from the 8192-element default to whatever the live
// L1 frontier leaves once the fully-resident input/output pages (and, for bf16_reduce, the FP32
// accumulator page) are seated. Rounded to a multiple of 32 once scaling is actually needed, so every
// chunk boundary offset stays NOC-aligned on Wormhole (32 B) and Blackhole (64 B) for the 2- and
// 4-byte dtypes in scope, then floored to that same 32-element minimum so a frontier tighter than
// what the caller planned against (the forced entry and direct prim callers only check the static
// bank) can never round the result down to zero (which would stall the reader kernels' chunk loop
// instead of failing).
uint32_t scatter_rm_chunk_elems(
    uint64_t usable_l1,
    uint64_t fixed_bytes,
    uint32_t index_stick_elems,
    uint32_t index_elem_size,
    uint32_t src_elem_size);

// TILE, full input/src row resident in L1. "Interleaved" is the name the gather port gave this
// row-resident plan; every factory here reads interleaved (non-sharded) buffers.
struct ScatterCodegenProgramFactoryInterleaved {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const ScatterCodegenParams& attributes, const ScatterCodegenInputs& tensor_args, Tensor& output_tensor);
};

// TILE, split by output column. Chosen by select_program_factory() when the interleaved plan's row
// does not fit L1, or when the tile-row count underfills the grid while more than one output column
// exists (Ht < column_cores && Wt_output > 1), so every core gets work.
struct ScatterCodegenProgramFactoryStreaming {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const ScatterCodegenParams& attributes, const ScatterCodegenInputs& tensor_args, Tensor& output_tensor);
};

// ROW_MAJOR fast path.
struct ScatterCodegenProgramFactoryRowMajor {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const ScatterCodegenParams& attributes, const ScatterCodegenInputs& tensor_args, Tensor& output_tensor);
};

// ROW_MAJOR, bfloat16 reduction deferred to FP32 arithmetic.
struct ScatterCodegenProgramFactoryBf16ReduceRowMajor {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const ScatterCodegenParams& attributes, const ScatterCodegenInputs& tensor_args, Tensor& output_tensor);
};

}  // namespace ttnn::prim
