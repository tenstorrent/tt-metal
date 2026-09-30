// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::data_movement::scatter {

// Correctness gate: true iff the codegen prim can produce a result matching native's contract for
// this input/index/src combination. Consulted by ttnn::scatter()'s routing, by
// scatter_force_codegen(), and by the codegen prim's own validation step. Placeholder until phase 4a.
bool supported_by_codegen(const Tensor& input_tensor, const Tensor& index_tensor, const Tensor& src_tensor);

// Perf gate consulted only by ttnn::scatter()'s routing, and only when supported_by_codegen() is
// true: true means fall back to native despite codegen support. Placeholder until phase 4a.
bool is_demoted(const Tensor& input_tensor, const Tensor& index_tensor, const Tensor& src_tensor);

}  // namespace ttnn::operations::data_movement::scatter
