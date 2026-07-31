// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "permute_codegen_supported.hpp"

namespace ttnn::operations::data_movement {

bool supported_by_codegen(const Tensor& /*input_tensor*/, const ttsl::SmallVector<uint32_t>& /*dims*/) { return false; }

}  // namespace ttnn::operations::data_movement
