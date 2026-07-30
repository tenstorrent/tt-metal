// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_codegen_supported.hpp"

#include <tt_stl/assert.hpp>

namespace ttnn::prim {

bool supported_by_codegen(const TilizeCodegenParams& /*operation_attributes*/) {
    // Placeholder — phase 4a derives the real predicate from tt-dm-codegen.
    return false;
}

bool is_demoted(const TilizeCodegenParams& /*operation_attributes*/) { return false; }

ImplementationSelector parse_implementation(std::string_view implementation) {
    if (implementation == "auto") {
        return ImplementationSelector::Auto;
    }
    if (implementation == "native") {
        return ImplementationSelector::Native;
    }
    if (implementation == "codegen") {
        return ImplementationSelector::Codegen;
    }
    TT_THROW("tilize: unknown implementation selector '{}'", implementation);
}

}  // namespace ttnn::prim
