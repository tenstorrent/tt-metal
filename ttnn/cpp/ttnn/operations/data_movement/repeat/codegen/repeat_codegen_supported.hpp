// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include <tt_stl/small_vector.hpp>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::data_movement::repeat_codegen {

// Kernels address buffers by page id, so a placement with its interleaved twin's page grid needs no reshard hop.
bool shard_spec_is_page_identical(
    const MemoryConfig& memory_config, const ttnn::Shape& logical_shape, tt::tt_metal::Layout layout);

// Copying tile pages along a sub-tile H/W axis would interleave tile padding into the result.
bool needs_row_major_round_trip(const Tensor& input, const ttsl::SmallVector<uint32_t>& repeat_dims);

// CBs are cut from the output page but kernels use the input's geometry, so any other page is silently mis-strided.
bool output_matches_input_page(const Tensor& input, const Tensor& output);

// Shared by the router and the whole-call L1 gate so both budget the same legs.
struct CodegenLegPlan {
    bool unshard_input = false;
    bool round_trip = false;
    MemoryConfig intermediate_mc;
    MemoryConfig final_mc;
    bool final_in_place = false;
    // False after a fold: the last leg's shape is short of the requested one.
    bool final_into_prealloc = false;
    // A size-1 axis repeat is folded into the next axis's count (same page order, one leg fewer).
    ttsl::SmallVector<uint32_t> leg_repeats;
    std::vector<uint32_t> rep_dims;
    size_t row_major_legs = 0;
};

CodegenLegPlan plan_codegen_legs(
    const Tensor& input, const ttsl::SmallVector<uint32_t>& repeat_dims, const MemoryConfig& output_mem_config);

// Correctness gate for a single-dim codegen repeat step, as seen by
// prim::repeat_codegen: `input` is already reshaped into the 4D-padded space
// its kernels assume, so rep_dim is in [0, 3].
bool supported_by_codegen(
    const Tensor& input, uint32_t rep_dim, uint32_t num_repeats, const MemoryConfig& output_mem_config);

// Correctness gate for a whole (possibly multi-dim) ttnn::repeat call, on the
// original tensor/repeat vector before per-dim decomposition and 4D padding.
// Consulted by the free function's routing and by repeat_force_codegen.
bool supported_by_codegen(
    const Tensor& input, const ttsl::SmallVector<uint32_t>& repeat_dims, const MemoryConfig& output_mem_config);

// Routing-only: checks the live free L1, unlike supported_by_codegen's static budget, which must be cache-stable.
bool row_major_cbs_fit_free_l1(
    const Tensor& input,
    const ttsl::SmallVector<uint32_t>& repeat_dims,
    const MemoryConfig& output_mem_config,
    bool output_preallocated);

// Perf-demotion gate: correct but not worth the codegen path. Routing-only --
// consulted by ttnn::repeat only, never by validate and never by
// repeat_force_codegen. Same call shape as the whole-call correctness gate above.
bool is_demoted(
    const Tensor& input, const ttsl::SmallVector<uint32_t>& repeat_dims, const MemoryConfig& output_mem_config);

}  // namespace ttnn::operations::data_movement::repeat_codegen
