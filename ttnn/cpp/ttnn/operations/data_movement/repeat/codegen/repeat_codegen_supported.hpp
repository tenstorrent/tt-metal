// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

// Single source of truth for repeat_codegen eligibility. Mirrors the scope recorded in
// tt-dm-codegen's agentic_port/manifests/repeat.yaml: unsharded 4D TILE bfloat16, repeat_dim in
// {0, 1, 2} (W / dim 3 is a real-kernel-limit, not a scoping choice), H and W tile-aligned.
//
// Must be called with the same (normalized) tensor and dim from both the host free-function
// routing branch and RepeatCodegenDeviceOperation::validate_on_program_cache_miss — see
// review-checklist R4 / R7 in tt-dm-codegen agentic_port/knowledge/review-checklist.md.
bool supported_by_codegen(const ttnn::Tensor& tensor, int32_t repeat_dim);

}  // namespace ttnn::prim

namespace ttnn::operations::data_movement::repeat {

// Selects which prim ttnn::repeat() dispatches to. `Auto` defers to supported_by_codegen();
// `Native` / `Codegen` force a side (and TT_FATAL early on an ineligible forced-codegen input).
enum class ImplementationSelector { Auto, Native, Codegen };

// Parses the `implementation` pybind kwarg ("auto" / "native" / "codegen", case-sensitive).
// nullopt (parameter omitted) is treated as "auto". TT_FATALs on any other string.
ImplementationSelector parse_implementation(const std::optional<std::string>& value);

}  // namespace ttnn::operations::data_movement::repeat
