// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <string>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

// Nested per the sibling ports' convention: these names (`supported_by_codegen`, `is_demoted`,
// `ImplementationSelector`) are generic enough that the bare data_movement namespace cannot hold one
// port's copy without colliding with the next.
namespace ttnn::operations::data_movement::repeat_interleave_codegen {

// Correctness-only: transcribed from codegen_repeat_interleave.py's invalidate_vector plus the
// op's own guards (RepeatInterleaveCodegen.repeat_interleave). Must agree with every case in
// repeat_interleave.yaml (`scope: in` -> true, `scope: out` -> false). Consulted by both the
// free function's `auto`/`codegen` branches and prim::repeat_interleave_codegen's validate.
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

// Perf-only: enumerated cases that supported_by_codegen() accepts but that measured worse than
// the native prim on device. Consulted ONLY by the free function's `auto` branch -- never by
// validate -- so a forced implementation="codegen" call still runs these.
bool is_demoted(
    const Tensor& input, uint32_t repeats, int32_t dim, const std::optional<MemoryConfig>& output_mem_config);

enum class ImplementationSelector { Auto, Native, Codegen };

ImplementationSelector parse_implementation(const std::string& implementation);

}  // namespace ttnn::operations::data_movement::repeat_interleave_codegen
