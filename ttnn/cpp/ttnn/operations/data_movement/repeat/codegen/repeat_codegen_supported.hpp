// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>

#include <tt_stl/small_vector.hpp>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::data_movement::repeat_codegen {

enum class ImplementationSelector { Auto, Native, Codegen };

// Parses the `implementation` kwarg. TT_FATALs on an unrecognized value.
ImplementationSelector parse_implementation(const std::string& implementation);

// Correctness gate for a single-dim codegen repeat step, as seen by
// prim::repeat_codegen: `input` is already reshaped into the 4D-padded space
// the copied kernels assume, so rep_dim is in [0, 3].
bool supported_by_codegen(const Tensor& input, uint32_t rep_dim, uint32_t num_repeats);

// Correctness gate for a whole (possibly multi-dim) ttnn::repeat call, on the
// original tensor/repeat vector before per-dim decomposition and 4D padding.
// Used by the free function's auto/forced-codegen routing.
bool supported_by_codegen(const Tensor& input, const ttsl::SmallVector<uint32_t>& repeat_dims);

// Perf-demotion gate: correct but not worth the codegen path. Routing-only —
// consulted by the auto branch only, never by validate. Same call shape as
// the whole-call correctness gate above.
bool is_demoted(const Tensor& input, const ttsl::SmallVector<uint32_t>& repeat_dims);

}  // namespace ttnn::operations::data_movement::repeat_codegen
