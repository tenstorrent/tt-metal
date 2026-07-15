// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/repeat_interleave/codegen/repeat_interleave_codegen_supported.hpp"

namespace ttnn::operations::data_movement::repeat_interleave {

ImplementationSelector parse_implementation(std::string_view implementation) {
    if (implementation == "native") {
        return ImplementationSelector::kNative;
    }
    if (implementation == "codegen") {
        return ImplementationSelector::kCodegen;
    }
    return ImplementationSelector::kAuto;
}

bool supported_by_codegen(const Tensor& /*input_tensor*/, uint32_t /*repeats*/, int32_t /*dim*/) { return false; }

bool is_demoted(const Tensor& /*input_tensor*/, uint32_t /*repeats*/, int32_t /*dim*/) { return false; }

}  // namespace ttnn::operations::data_movement::repeat_interleave
