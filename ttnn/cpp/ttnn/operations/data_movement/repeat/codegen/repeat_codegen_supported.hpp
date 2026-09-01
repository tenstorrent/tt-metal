// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <tt_stl/small_vector.hpp>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::data_movement::repeat_codegen {

// Correctness gate for a single-dim codegen repeat step, as seen by
// prim::repeat_codegen: `input` is already reshaped into the 4D-padded space
// its kernels assume, so rep_dim is in [0, 3].
bool supported_by_codegen(const Tensor& input, uint32_t rep_dim, uint32_t num_repeats);

// Correctness gate for a whole (possibly multi-dim) ttnn::repeat call, on the
// original tensor/repeat vector before per-dim decomposition and 4D padding.
// Consulted by the free function's routing and by repeat_force_codegen.
bool supported_by_codegen(const Tensor& input, const ttsl::SmallVector<uint32_t>& repeat_dims);

// Perf-demotion gate: correct but not worth the codegen path. Routing-only --
// consulted by ttnn::repeat only, never by validate and never by
// repeat_force_codegen. Same call shape as the whole-call correctness gate above.
bool is_demoted(const Tensor& input, const ttsl::SmallVector<uint32_t>& repeat_dims);

}  // namespace ttnn::operations::data_movement::repeat_codegen
