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
// Queried from the device's *live* L1 occupancy rather than a hardcoded constant, so the gate
// and the factory can never disagree with what the allocator will actually hand out. Shared
// between the two so they stay in lockstep.
//
// Statically allocated CBs grow upward from the allocator's base L1 address; L1 tensors are
// allocated downward from the top of L1. The budget is therefore the gap between the two, i.e.
// lowest_occupied_compute_l1_address() - base -- the exact quantity
// ProgramImpl::validate_circular_buffer_region() checks the CB region end against. Budgeting
// against total L1 instead would ignore buffers already resident in L1 (model weights, trace
// buffers) and plan a CB region that provably clashes with them.
uint32_t usable_l1_bytes(const tt::tt_metal::IDevice* device);

// Correctness-only: true iff the codegen path can produce a bit-exact result for this
// (input, output_mem_config) case. Consulted by ttnn::untilize's routing, by
// untilize_force_codegen, and by prim::untilize_codegen's validate -- never gated on performance.
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
