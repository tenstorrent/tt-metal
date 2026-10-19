// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <tt_stl/small_vector.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement::permute_codegen {

// Correctness gate for a ttnn::permute call. Consulted by the free function's
// routing, by the codegen prim's validation step, and by permute_force_codegen.
// Covers device, L1 and page-geometry feasibility, never performance.
bool supported_by_codegen(
    const Tensor& input, const ttsl::SmallVector<uint32_t>& dims, const tt::tt_metal::MemoryConfig& output_mem_config);

// Perf-demotion gate: correct but not worth the codegen path. Routing-only --
// consulted by ttnn::permute only, never by the prim's validation step and never
// by permute_force_codegen.
bool is_demoted(const Tensor& input, const ttsl::SmallVector<uint32_t>& dims);

// True when `dims` names each of the `rank` input axes exactly once.
bool is_valid_permutation(const ttsl::SmallVector<uint32_t>& dims, uint32_t rank);

// Spec of the permuted tensor. Tile and layout come from the input, so a caller
// building an output buffer and a caller checking one agree on the page geometry.
tt::tt_metal::TensorSpec permute_output_spec(
    const Tensor& input, const ttsl::SmallVector<uint32_t>& dims, const tt::tt_metal::MemoryConfig& output_mem_config);

// True when a caller-provided output tensor has the shape, dtype, layout, tile and
// memory config the op would have produced itself.
bool output_spec_matches(const tt::tt_metal::TensorSpec& expected, const tt::tt_metal::TensorSpec& actual);

}  // namespace ttnn::operations::data_movement::permute_codegen
