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
// TILE folding (H collapsed to tile-rows) and the ROW_MAJOR folding (no collapse). TT_FATALs if
// either tensor's page rank exceeds kScatterMaxPageRank; supported_by_codegen() must reject that
// case first so this never fires under normal routing.
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
    uint32_t Ht_per_batch_input = 0;
    uint32_t Ht_per_batch_src = 0;
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

// Reduction-arithmetic dtype tag consumed by scatter_common.hpp's scatter_reduce_value (1=float32,
// 2=int32, 3=uint32, 4=uint16, 5=bfloat16); 0 (unused) when reduction_mode == 0.
uint32_t scatter_value_kind(tt::tt_metal::DataType dtype);

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

// The device's STATIC per-core CB budget (the allocator-managed L1 window, ignoring what is live in
// it). The only budget a routing gate may plan against: supported_by_codegen() is evaluated at the
// router and again at the codegen prim's validate, with this call's own create_output_tensors() in
// between, so an L1 output tensor must not change the answer between those two evaluations.
uint64_t scatter_static_l1(const Tensor& input_tensor);

// The device's real per-core CB ceiling, accounting for live L1 occupancy (e.g. this call's own
// output when placed in L1). Confined to the program factory and the program-cache hash, which both
// run once per cache miss with nothing else moving underneath them.
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

// RM: whether the smallest viable plan -- the fully-resident input/output sticks (doubled again for
// the FP32 accumulator when bf16_reduce is requested) plus the smallest NOC-alignment-safe index/src
// chunk (32 elements) -- fits the device's STATIC per-core L1 window. The sticks stay fully resident
// by construction (no shallower RM plan exists to scale down to), so a call this rejects has no
// feasible RM dispatch at all and the routing gate must send it to native.
bool scatter_rm_min_plan_fits_l1(
    uint64_t static_l1, uint64_t input_page_bytes, uint32_t index_elem_size, uint32_t src_elem_size, bool bf16_reduce);

// RM: index/src CB depth in elements, scaled down from the 8192-element default to whatever the live
// L1 frontier leaves once the fully-resident input/output pages (and, for bf16_reduce, the FP32
// accumulator page) are seated. Rounded to a multiple of 32 once scaling is actually needed, so every
// chunk boundary offset stays a multiple of the 32-byte NOC alignment for every supported dtype, then
// floored to that same 32-element minimum so a live frontier tighter than the static ceiling
// supported_by_codegen() validated at routing time can never round the result down to zero (which
// would stall the reader kernels' chunk loop instead of failing).
uint32_t scatter_rm_chunk_elems(
    uint64_t usable_l1,
    uint64_t fixed_bytes,
    uint32_t index_stick_elems,
    uint32_t index_elem_size,
    uint32_t src_elem_size);

// Whether a post-transpose, post-4D-fold TILE scatter is cheaper to serve by converting input,
// index and src to ROW_MAJOR, running the per-stick RM factory, and converting the result back,
// instead of dispatching the tile-row-parallel interleaved/streaming factories directly. Those
// factories split work by output tile-ROW (one core per 32-row band), so a shape with few tile-rows
// leaves most of the device idle; the RM factory splits by individual logical row (one core per
// row) instead. Below a small tile-row count this trade wins regardless of row width; above it, the
// tile-row-parallel kernels already keep the device busy and the extra conversion round trip is not
// worth paying. Reduction is never requested on this branch -- supported_by_codegen() only admits
// add/multiply reduction on ROW_MAJOR input, so a TILE call here always carries reduction_mode == 0.
bool scatter_tile_prefers_rm_strategy(const Tensor& input_tensor, const Tensor& index_tensor, const Tensor& src_tensor);

// Assembles the full cache-key/attributes struct for one scatter_codegen() call, branching on
// input_tensor.layout() to fill either the TILE or the ROW_MAJOR geometry fields (the other side's
// fields are zero) and computing the shared page map, value_kind and (for ROW_MAJOR) the frontier-
// derived chunk depth. `reduction_mode` and `sub_core_grids` are threaded straight through.
ScatterCodegenParams build_scatter_codegen_params(
    const Tensor& input_tensor,
    const Tensor& index_tensor,
    const Tensor& src_tensor,
    uint32_t reduction_mode,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<CoreRangeSet>& sub_core_grids);

// TILE, full input/src row resident in L1.
struct ScatterCodegenProgramFactoryInterleaved {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const ScatterCodegenParams& attributes, const ScatterCodegenInputs& tensor_args, Tensor& output_tensor);
};

// TILE, chunked streaming fallback for rows too wide for the interleaved plan's L1 budget.
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
