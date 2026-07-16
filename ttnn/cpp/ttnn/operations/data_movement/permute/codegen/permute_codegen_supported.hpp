// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string_view>

#include "ttnn/tensor/tensor.hpp"
#include <tt_stl/span.hpp>

namespace ttnn::operations::data_movement::permute_codegen {

// Correctness gate: can the codegen prim produce a bit-exact result for these inputs?
// Placeholder — phase 4a transcribes the real predicate from codegen_permute.py's
// invalidate_vector and the _fused_wh_ok gate (see manifest cases with scope: out).
bool supported_by_codegen(const Tensor& input_tensor, ttsl::Span<const uint32_t> dims);

// Perf gate, auto-routing only: enumerated in-scope cases known not to win on device.
// Placeholder — phase 4a (and Iterate) populate this from the perf-demote list.
bool is_demoted(const Tensor& input_tensor, ttsl::Span<const uint32_t> dims);

enum class ImplementationSelector { Auto, Native, Codegen };

ImplementationSelector parse_implementation(std::string_view implementation);

}  // namespace ttnn::operations::data_movement::permute_codegen
