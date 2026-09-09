// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt-metalium/core_coord.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/memory_config/memory_config.hpp"

namespace tt::tt_metal {
class IDevice;
}  // namespace tt::tt_metal

namespace ttnn::operations::data_movement::untilize_codegen {

// Mirrors codegen builder_utils.USABLE_L1: the CB budget every codegen builder plans against.
// Queried from the device's static L1 budget (total L1 minus the allocator's reserved base)
// rather than a hardcoded constant, so the gate and the factory can never disagree with what
// the allocator will actually hand out. Shared between the two so they stay in lockstep.
//
// Deliberately a STATIC device property: it must not consult live L1 occupancy (see
// supported_by_codegen below). How much of this budget is actually free at program-build time
// is the program factory's business, via get_max_l1_space() -- it is the only place that can
// observe it once, consistently.
uint32_t usable_l1_bytes(const tt::tt_metal::IDevice* device);

// Correctness-only: true iff the codegen build_untilize_tile path can produce a bit-exact
// result for this (input, output_mem_config) case. Consulted by the free function's forced
// "codegen" branch and by prim::untilize_codegen's validate -- never gated on performance.
//
// MUST stay a pure function of static tensor/memory-config properties (layout, dtype, shape,
// sharding, and static device geometry). It is evaluated independently at three call sites --
// ttnn::untilize's routing gate, detail::untilize_force_codegen's TT_FATAL, and
// UntilizeCodegenDeviceOperation::validate_on_program_cache_miss -- at different moments in the
// same dispatch, and those sites are only consistent with each other because the answer cannot
// change between them. Making it depend on mutable device state (e.g. live L1 occupancy, which
// the op's own create_output_tensors() moves by allocating the output) breaks that invariant:
// routing sees true, dispatches to codegen, and validate then TT_FATALs on the same tensor.
// Live-L1 accounting belongs in the program factory, which decides once per cache miss.
bool supported_by_codegen(const Tensor& input, const tt::tt_metal::MemoryConfig& output_mem_config);

// Every codegen builder places work over the full compute-with-storage grid and has no
// single-core variant, so it can honour neither of the native op's core-placement controls.
// False means the case must go to native, or be rejected outright by untilize_force_codegen.
// Separate from supported_by_codegen() because these are free-function attributes:
// the codegen prim carries no such fields, so its validate has nothing to check.
bool supported_execution_controls(bool use_multicore, const std::optional<CoreRangeSet>& sub_core_grids);

// Perf-only: true for the enumerated set of in-scope cases where codegen is correct but does
// not beat native on device. Consulted ONLY by ttnn::untilize's routing, alongside
// supported_by_codegen(); never by validate, and never by untilize_force_codegen.
bool is_demoted(const Tensor& input, const tt::tt_metal::MemoryConfig& output_mem_config);

}  // namespace ttnn::operations::data_movement::untilize_codegen
