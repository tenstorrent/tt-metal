// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::data_movement::repeat_codegen {

// Correctness gate for the codegen prim. Placeholder: phase 4a fills in the real predicate.
bool supported_by_codegen(const Tensor& input, uint32_t rep_dim, uint32_t num_repeats);

}  // namespace ttnn::operations::data_movement::repeat_codegen
