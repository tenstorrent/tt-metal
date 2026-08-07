// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <string_view>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/tile.hpp>

#include "tilize_codegen_device_operation_types.hpp"

namespace ttnn::prim {

// Execution controls: the free function's parameters that decide WHERE work lands or with what tile
// geometry, rather than what the result contains. None of the codegen builders honours these — every
// one places work over the full compute_with_storage_grid_size() with the standard 32x32 tile, and
// TilizeCodegenParams carries neither a core set nor a tile shape — so accepting such a call on
// codegen would land work on cores the caller reserved, or silently produce the wrong tile geometry.
//
// Returns nullptr when every control on the call is one codegen can honour, otherwise the name of
// the offending control. Kept out of supported_by_codegen() (the codegen prim has no such
// attributes, so its validate has nothing to check) and out of is_demoted() (this is correctness,
// not perf), but declared here so the `auto` gate and the forced-codegen TT_FATAL share one answer.
//
// use_multicore / use_low_perf are deliberately NOT here: both are cache-key attributes the codegen
// factory really does honour (tilize_codegen_dispatch's RowSingleCore route), so they are a perf
// question for is_demoted(), not a feasibility one.
const char* unsupported_execution_control(
    const tt::tt_metal::Tile& tile, const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids);

// Correctness gate: can the codegen prim produce a bit-exact result for these inputs?
// Transcribed from common/sweeps/codegen_tilize.py's invalidate_vector (which delegates
// same-dtype/interleaved cases to upstream sweeps.tilize) plus ops/tilize/tilize.py's own
// guards. Takes tensor_args too (not just the cache-key attrs) because the tile-alignment
// and layout checks need the tensor's raw logical shape/layout, which NC/Ht/Wt (already
// tile-rounded) cannot recover.
bool supported_by_codegen(const TilizeCodegenParams& operation_attributes, const TilizeCodegenInputs& tensor_args);

// Perf gate (auto-routing only): in-scope cases not worth the codegen path under `auto`. Never
// consulted by the codegen prim's validate, so a demoted case still runs under
// implementation=codegen and keeps being measured. Carries two general conditions — a
// caller-forced single-worker route, and Wt == 1, where every builder degenerates to a single tile
// per compute block and the batched writer is off — plus an enumerated table for the remaining
// configurations no mechanism was found for.
bool is_demoted(const TilizeCodegenParams& operation_attributes, const TilizeCodegenInputs& tensor_args);

enum class ImplementationSelector { Auto, Native, Codegen };

ImplementationSelector parse_implementation(std::string_view implementation);

}  // namespace ttnn::prim
