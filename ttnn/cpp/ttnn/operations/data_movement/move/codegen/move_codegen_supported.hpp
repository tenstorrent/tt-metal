// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::data_movement::move_codegen {

// Selects which prim backs `ttnn::move`. Co-located with `supported_by_codegen()` (below) and
// `parse_implementation()` because eligibility and selection are one concern with a single source
// of truth, shared by the free function's `auto` routing and `prim::move_codegen`'s validate.
enum class ImplementationSelector { Auto, Native, Codegen };

// Parses the Python-facing `implementation` kwarg ("auto"|"native"|"codegen"). TT_THROWs on any
// other value.
ImplementationSelector parse_implementation(const std::string& implementation);

// Single source of truth for whether the codegen prim can handle a given call, shared by the
// `ttnn::move` free function's `auto` routing and `prim::move_codegen`'s validate. Transcribed from
// tt-dm-codegen's ops/move/move.py dispatch (layout guard) and common/sweeps/codegen_move.py's
// invalidate_vector (TILE tile-alignment, RM block-float rejection), plus manifests/move.yaml's
// scope:out case (sharded in/out stays on native's MULTI_CORE_SHARDED strategy).
bool supported_by_codegen(const Tensor& input_tensor, const tt::tt_metal::MemoryConfig& output_mem_config);

}  // namespace ttnn::operations::data_movement::move_codegen
