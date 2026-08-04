// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gather_codegen_supported.hpp"

#include <tt-metalium/assert.hpp>

namespace ttnn::operations::data_movement::gather {

ImplementationSelector parse_implementation(std::string_view implementation) {
    if (implementation == "native") {
        return ImplementationSelector::kNative;
    }
    if (implementation == "codegen") {
        return ImplementationSelector::kCodegen;
    }
    TT_FATAL(implementation == "auto", "Unknown gather implementation selector: {}", implementation);
    return ImplementationSelector::kAuto;
}

bool supported_by_codegen(const Tensor&, int8_t, const Tensor&) {
    // Placeholder: phase 4a fills in the real correctness predicate (TILE layout, bfloat16,
    // dim already normalized to last -- see the gather manifest's coverage/port_scope/cases).
    return false;
}

bool is_demoted(const Tensor&, int8_t, const Tensor&) { return false; }

}  // namespace ttnn::operations::data_movement::gather
