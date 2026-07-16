// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "permute_codegen_supported.hpp"

#include <tt_stl/assert.hpp>

namespace ttnn::operations::data_movement::permute_codegen {

bool supported_by_codegen(const Tensor& /*input_tensor*/, ttsl::Span<const uint32_t> /*dims*/) { return false; }

bool is_demoted(const Tensor& /*input_tensor*/, ttsl::Span<const uint32_t> /*dims*/) { return false; }

ImplementationSelector parse_implementation(std::string_view implementation) {
    if (implementation == "native") {
        return ImplementationSelector::Native;
    }
    if (implementation == "codegen") {
        return ImplementationSelector::Codegen;
    }
    TT_FATAL(implementation == "auto", "Unknown permute implementation selector: {}", implementation);
    return ImplementationSelector::Auto;
}

}  // namespace ttnn::operations::data_movement::permute_codegen
