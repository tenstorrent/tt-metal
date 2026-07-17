// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string_view>

#include "ttnn/tensor/tensor.hpp"
#include <tt_stl/span.hpp>

namespace ttnn::operations::data_movement::permute_codegen {

// Correctness gate: can the codegen prim produce a bit-exact result for these inputs?
// Transcribed from an internal reference implementation's invalidate_vector and _fused_wh_ok
// gate. Consulted by both the free function and validate_on_program_cache_miss.
bool supported_by_codegen(const Tensor& input_tensor, ttsl::Span<const uint32_t> dims);

// Perf gate, auto-routing only: enumerated in-scope cases known not to win on device. Never
// consulted by validate — a demoted case must still run under forced implementation="codegen".
bool is_demoted(const Tensor& input_tensor, ttsl::Span<const uint32_t> dims);

enum class ImplementationSelector { Auto, Native, Codegen };

ImplementationSelector parse_implementation(std::string_view implementation);

}  // namespace ttnn::operations::data_movement::permute_codegen
