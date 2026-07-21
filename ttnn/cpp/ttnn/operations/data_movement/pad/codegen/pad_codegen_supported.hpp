// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string_view>

#include "pad_codegen_device_operation_types.hpp"

namespace ttnn::prim {

// Correctness gate: can the codegen path produce a bit-exact result for these inputs?
// Transcribed from codegen_pad.py's invalidate_vector + ops/pad/pad.py's guards.
bool supported_by_codegen(const PadCodegenParams& operation_attributes, const PadCodegenInputs& tensor_args);

// Perf gate, auto-only: an in-scope case that codegen can do but shouldn't be routed to.
bool is_demoted(const PadCodegenParams& operation_attributes, const PadCodegenInputs& tensor_args);

enum class ImplementationSelector { Auto, Native, Codegen };

ImplementationSelector parse_implementation(std::string_view implementation);

}  // namespace ttnn::prim
