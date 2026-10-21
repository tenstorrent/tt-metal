// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include "ttnn/operations/data_movement/concat/codegen/concat_codegen_program_factory.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement::concat_codegen {

// Compile-time ceiling of the N-way readers.
inline constexpr uint32_t kConcatMaxNwayInputs = 8;

// The correctness gate shared by routing and the codegen prim's validation step.
bool supported_by_codegen(const std::vector<Tensor>& input_tensors, const ttnn::prim::ConcatCodegenParams& attrs);

}  // namespace ttnn::operations::data_movement::concat_codegen
