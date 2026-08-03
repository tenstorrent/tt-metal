// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <string>

#include <tt-metalium/core_coord.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/memory_config/memory_config.hpp"

namespace ttnn::operations::data_movement::untilize_codegen {

// Mirrors codegen builder_utils.USABLE_L1: the CB budget every codegen builder plans against.
// Shared with the program factory so the gate and the factory cannot disagree about what fits.
constexpr uint64_t kUsableL1 = 1'400'000;

enum class ImplementationSelector { Auto, Native, Codegen };

// Parses the "implementation" kwarg ("auto" | "native" | "codegen"); defaults to Auto for
// an empty string. TT_FATALs on any other value.
ImplementationSelector parse_implementation(const std::string& implementation);

// Correctness-only: true iff the codegen build_untilize_tile path can produce a bit-exact
// result for this (input, output_mem_config) case. Consulted by the free function's forced
// "codegen" branch and by prim::untilize_codegen's validate -- never gated on performance.
bool supported_by_codegen(const Tensor& input, const tt::tt_metal::MemoryConfig& output_mem_config);

// Every codegen builder places work over the full compute-with-storage grid and has no
// single-core variant, so it can honour neither of the native op's core-placement controls.
// False means the case must go to native (under "auto") or be rejected outright (under a forced
// "codegen"). Separate from supported_by_codegen() because these are free-function attributes:
// the codegen prim carries no such fields, so its validate has nothing to check.
bool supported_execution_controls(bool use_multicore, const std::optional<CoreRangeSet>& sub_core_grids);

// Perf-only: true for the enumerated set of in-scope cases where codegen is correct but does
// not beat native on device. Consulted ONLY by the free function's "auto" branch, alongside
// supported_by_codegen(); never by validate, and never under forced implementation="codegen".
bool is_demoted(const Tensor& input, const tt::tt_metal::MemoryConfig& output_mem_config);

}  // namespace ttnn::operations::data_movement::untilize_codegen
