// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/concat/codegen/concat_codegen_supported.hpp"

namespace ttnn::operations::data_movement::concat_codegen {

bool supported_by_codegen(
    const std::vector<Tensor>& /*input_tensors*/, const ttnn::prim::ConcatCodegenParams& /*attrs*/) {
    return false;
}

}  // namespace ttnn::operations::data_movement::concat_codegen
