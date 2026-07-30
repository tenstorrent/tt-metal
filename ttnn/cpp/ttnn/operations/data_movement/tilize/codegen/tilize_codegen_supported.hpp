// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string_view>
#include "tilize_codegen_device_operation_types.hpp"

namespace ttnn::prim {

// Correctness gate: can the codegen prim produce a bit-exact result for these inputs?
// Placeholder — phase 4a fills this in from tt-dm-codegen's invalidate_vector / op guards.
bool supported_by_codegen(const TilizeCodegenParams& operation_attributes);

// Perf gate (auto-routing only): in-scope cases not worth the codegen path under `auto`.
// Placeholder — phase 4a fills this in from demotion analysis.
bool is_demoted(const TilizeCodegenParams& operation_attributes);

enum class ImplementationSelector { Auto, Native, Codegen };

ImplementationSelector parse_implementation(std::string_view implementation);

}  // namespace ttnn::prim
