// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string_view>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::data_movement::gather {

// Selects which prim ttnn::gather() dispatches to; shared by the free function's routing
// and the codegen prim's validation step (see call_parity.routing in the gather manifest).
enum class ImplementationSelector { kAuto, kNative, kCodegen };

ImplementationSelector parse_implementation(std::string_view implementation);

// Correctness gate: true iff the codegen prim can produce a result matching native's contract
// for this input/dim/index combination. Placeholder — phase 4a fills in the real predicate.
bool supported_by_codegen(const Tensor& input_tensor, int8_t dim, const Tensor& input_index_tensor);

// Perf gate consulted only when ImplementationSelector::kAuto and supported_by_codegen() is true:
// true means fall back to native despite codegen support. Placeholder — phase 4a fills this in.
bool is_demoted(const Tensor& input_tensor, int8_t dim, const Tensor& input_index_tensor);

}  // namespace ttnn::operations::data_movement::gather
