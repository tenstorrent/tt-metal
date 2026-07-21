// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "pad_codegen_supported.hpp"

namespace ttnn::prim {

bool supported_by_codegen(const PadCodegenParams& /*operation_attributes*/, const PadCodegenInputs& /*tensor_args*/) {
    return false;
}

bool is_demoted(const PadCodegenParams& /*operation_attributes*/, const PadCodegenInputs& /*tensor_args*/) {
    return false;
}

ImplementationSelector parse_implementation(std::string_view implementation) {
    if (implementation == "native") {
        return ImplementationSelector::Native;
    }
    if (implementation == "codegen") {
        return ImplementationSelector::Codegen;
    }
    return ImplementationSelector::Auto;
}

}  // namespace ttnn::prim
