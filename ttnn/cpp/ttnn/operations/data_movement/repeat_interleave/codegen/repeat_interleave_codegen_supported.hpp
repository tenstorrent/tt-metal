// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string_view>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::data_movement::repeat_interleave {

enum class ImplementationSelector { kAuto, kNative, kCodegen };

ImplementationSelector parse_implementation(std::string_view implementation);

// Correctness gate: true only for shapes/layouts/dtypes the codegen kernels actually cover.
bool supported_by_codegen(const Tensor& input_tensor, uint32_t repeats, int32_t dim);

// Perf gate (auto routing only): true when codegen is supported but known slower than native.
bool is_demoted(const Tensor& input_tensor, uint32_t repeats, int32_t dim);

}  // namespace ttnn::operations::data_movement::repeat_interleave
