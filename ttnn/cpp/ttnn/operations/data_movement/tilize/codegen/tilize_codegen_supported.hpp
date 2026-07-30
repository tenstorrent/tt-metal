// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string_view>
#include "tilize_codegen_device_operation_types.hpp"

namespace ttnn::prim {

// Correctness gate: can the codegen prim produce a bit-exact result for these inputs?
// Transcribed from common/sweeps/codegen_tilize.py's invalidate_vector (which delegates
// same-dtype/interleaved cases to upstream sweeps.tilize) plus ops/tilize/tilize.py's own
// guards. Takes tensor_args too (not just the cache-key attrs) because the tile-alignment
// and layout checks need the tensor's raw logical shape/layout, which NC/Ht/Wt (already
// tile-rounded) cannot recover.
bool supported_by_codegen(const TilizeCodegenParams& operation_attributes, const TilizeCodegenInputs& tensor_args);

// Perf gate (auto-routing only): in-scope cases not worth the codegen path under `auto`.
bool is_demoted(const TilizeCodegenParams& operation_attributes, const TilizeCodegenInputs& tensor_args);

enum class ImplementationSelector { Auto, Native, Codegen };

ImplementationSelector parse_implementation(std::string_view implementation);

}  // namespace ttnn::prim
