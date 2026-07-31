// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

#include "ttnn/tensor/tensor.hpp"
#include <tt_stl/span.hpp>

namespace ttnn::operations::data_movement {

// Dispatch predicate deciding whether a call lands on PermuteCodegenDeviceOperation vs. the
// native prim. Must replicate PermuteCodegen.permute's ops/permute/spec.py gating (row-major only,
// and rejecting the fused-WH delegation to TransposeCodegen — see permute.yaml's scope=out cases).
// Correctness-only: never folds in perf demotions (see is_demoted below).
bool supported_by_codegen(const Tensor& input_tensor, const ttsl::SmallVector<uint32_t>& dims);

// Perf-only routing gate, consulted by the "auto" selector alone (never by validate or a forced
// "codegen" call). See permute.yaml's perf-demoted ledger: no general mechanism was found for these
// cases, so each is an exact (shape, dims) regression-example branch — dtype-independent, since
// bf16/fp32/int32 were all measured slow for the same shape+dims.
bool is_demoted(const Tensor& input_tensor, const ttsl::SmallVector<uint32_t>& dims);

enum class ImplementationSelector { Auto, Native, Codegen };

// Parses the "implementation" kwarg ("auto" | "native" | "codegen"); TT_THROWs otherwise.
ImplementationSelector parse_implementation(const std::string& value);

}  // namespace ttnn::operations::data_movement
