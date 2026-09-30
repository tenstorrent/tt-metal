// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "scatter_codegen_supported.hpp"

namespace ttnn::operations::data_movement::scatter {

bool supported_by_codegen(
    const Tensor& /*input_tensor*/, const Tensor& /*index_tensor*/, const Tensor& /*src_tensor*/) {
    return false;
}

bool is_demoted(const Tensor& /*input_tensor*/, const Tensor& /*index_tensor*/, const Tensor& /*src_tensor*/) {
    return false;
}

}  // namespace ttnn::operations::data_movement::scatter
