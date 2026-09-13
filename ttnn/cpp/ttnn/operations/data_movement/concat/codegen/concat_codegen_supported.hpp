// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include <tt-metalium/device.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement::concat_codegen {

// Per-core L1 a circular-buffer plan may claim. Deliberately a STATIC device property: it must
// not consult live occupancy (see supported_by_codegen below). How much of it is actually free
// at program-build time is the factory's business, via get_max_l1_space().
uint32_t usable_l1_bytes(const tt::tt_metal::IDevice* device);

// Correctness/feasibility gate for the four row-major builders. Consulted by the free
// function's routing, by concat_force_codegen, and by prim::concat_codegen's validate, so
// all three agree on the supported scope. `dim` is already normalized (0..rank-1).
//
// MUST stay a pure function of the tensors, the requested config, and static device geometry.
// The three call sites evaluate it at different moments of one dispatch and are consistent only
// because the answer cannot change between them -- and create_output_tensors() runs before
// validate_on_program_cache_miss(), so the op's own output allocation moves live L1 in between.
// A live-L1 term here means routing admits, the output lands, and validate then TT_FATALs on a
// case native would have served. Live accounting belongs in fits_live_l1() (routing-only) and
// in the factory, which samples once per cache miss.
bool supported_by_codegen(
    const std::vector<Tensor>& input_tensors, uint32_t dim, const tt::tt_metal::MemoryConfig& output_mem_config);

// Routing-only: does the plan fit L1 as it stands right now, once the output this call will
// allocate is accounted for? Same status as is_demoted() -- consulted by the auto branch,
// never re-asserted by validate, so it is free to depend on mutable device state.
bool fits_live_l1(
    const std::vector<Tensor>& input_tensors, uint32_t dim, const tt::tt_metal::MemoryConfig& output_mem_config);

// Perf-demotion gate: correct but not worth the codegen path. Routing-only --
// consulted by the auto branch only, never by validate.
bool is_demoted(const std::vector<Tensor>& input_tensors, uint32_t dim);

// Native execution controls (`groups`, `sub_core_grids`) that no ConcatCodegen
// builder honours. Kept out of supported_by_codegen()/is_demoted(): the codegen
// prim carries neither attribute, so this is a free-function-only routing
// concern, shared between the auto branch and the forced-codegen TT_FATAL.
bool supported_execution_controls(unsigned int groups, const std::optional<ttnn::CoreRangeSet>& sub_core_grids);

}  // namespace ttnn::operations::data_movement::concat_codegen
