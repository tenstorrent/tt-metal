// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <tt_stl/small_vector.hpp>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::data_movement::permute_codegen {

// Correctness gate for a ttnn::permute call. Consulted by the free function's
// routing, by the codegen prim's validation step, and by permute_force_codegen.
bool supported_by_codegen(const Tensor& input, const ttsl::SmallVector<uint32_t>& dims);

// Perf-demotion gate: correct but not worth the codegen path. Routing-only --
// consulted by ttnn::permute only, never by the prim's validation step and never
// by permute_force_codegen.
bool is_demoted(const Tensor& input, const ttsl::SmallVector<uint32_t>& dims);

}  // namespace ttnn::operations::data_movement::permute_codegen
