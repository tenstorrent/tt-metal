// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

// Nested per the sibling ports' convention: these names (`supported_by_codegen`, `is_demoted`) are
// generic enough that the bare data_movement namespace cannot hold one op's copy without colliding
// with the next.
namespace ttnn::operations::data_movement::repeat_interleave_codegen {

// Correctness gate for a whole ttnn::repeat_interleave call, on the caller's arguments before dim
// normalization. Consulted by the free function's routing, by repeat_interleave_force_codegen, and
// by prim::repeat_interleave_codegen's validate, so all three agree on the supported scope.
bool supported_by_codegen(
    const Tensor& input, uint32_t repeats, int32_t dim, const std::optional<MemoryConfig>& output_mem_config);

// A ROW_MAJOR codegen CB slot holds one whole stick, so its byte size scales with the input's
// last dim -- unlike the TILE path, whose slot is always one tile. Shared by supported_by_codegen()
// (which rejects a stick too wide for the smallest viable CB) and the program factory (which
// shrinks its read/write batch to whatever per-core L1 admits); the two must agree or the gate
// would claim a config the factory cannot instantiate.
struct RmCbBudget {
    uint32_t slot_stride;  // bytes per CB slot
    uint32_t max_slots;    // slots that fit in per-core L1
};
RmCbBudget rm_cb_budget(const Tensor& input, const std::optional<MemoryConfig>& output_mem_config);

// The RM writer waits on the previous batch plus the current one, so one in-flight page per side
// is the floor below which the kernels deadlock.
inline constexpr uint32_t kRmCbMinSlots = 2;

// Perf-demotion gate: true means a configuration is correct on the codegen path but not worth
// taking it. Routing-only -- consulted by ttnn::repeat_interleave only, never by validate and never
// by repeat_interleave_force_codegen. Demotes nothing today; no measured in-scope configuration
// loses to native on device.
bool is_demoted(
    const Tensor& input, uint32_t repeats, int32_t dim, const std::optional<MemoryConfig>& output_mem_config);

}  // namespace ttnn::operations::data_movement::repeat_interleave_codegen
