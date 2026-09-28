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
// supported_by_codegen below). How much of this budget is actually free is the business of
// codegen_cb_plan_fits_live_l1 (routing) and the codegen op's hash/program factory (CB tier),
// via get_max_l1_space().
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
// Live-L1 accounting is codegen_cb_plan_fits_live_l1 below (routing only) and the codegen op's
// own hash/program factory (CB tier), never this predicate.
bool supported_by_codegen(const Tensor& input, const tt::tt_metal::MemoryConfig& output_mem_config);

// Live-L1 routing gate, deliberately NOT pure: true iff at least one codegen CB plan fits the L1
// that is free on the device right now, once the output tensor this call is about to allocate is
// reserved out of it (get_pending_l1_output_reservation, the same accounting untilize_native's
// enough_space_height applies). Consulted ONLY by ttnn::untilize's routing, alongside
// supported_by_codegen() and is_demoted(): false sends the whole call to the native untilize prims
// through their ordinary entry points. Never consulted by validate (it would TT_FATAL a case that
// routing already sent elsewhere) and never by untilize_force_codegen (which must not fall back).
//
// This is the same chooser the codegen op runs in compute_program_hash and create_descriptor
// (choose_codegen_cb_plan), evaluated one allocation earlier with that allocation reserved, so the
// tier those two later see is the one predicted here; the codegen op itself never builds a native
// program and treats "no plan fits" as an error. A host-resident or unallocated input has no L1 to
// plan against and returns true, so the codegen op's validate reports the established error.
bool codegen_cb_plan_fits_live_l1(const Tensor& input, const tt::tt_metal::MemoryConfig& output_mem_config);

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
