// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <string>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement {

// Correctness-only: transcribed from codegen_repeat_interleave.py's invalidate_vector plus the
// op's own guards (RepeatInterleaveCodegen.repeat_interleave). Must agree with every case in
// repeat_interleave.yaml (`scope: in` -> true, `scope: out` -> false). Consulted by both the
// free function's `auto`/`codegen` branches and prim::repeat_interleave_codegen's validate.
bool supported_by_codegen(
    const Tensor& input, uint32_t repeats, int32_t dim, const std::optional<MemoryConfig>& output_mem_config);

// Perf-only: enumerated cases that supported_by_codegen() accepts but that measured worse than
// the native prim on device. Consulted ONLY by the free function's `auto` branch -- never by
// validate -- so a forced implementation="codegen" call still runs these.
bool is_demoted(
    const Tensor& input, uint32_t repeats, int32_t dim, const std::optional<MemoryConfig>& output_mem_config);

enum class ImplementationSelector { Auto, Native, Codegen };

ImplementationSelector parse_implementation(const std::string& implementation);

}  // namespace ttnn::operations::data_movement
