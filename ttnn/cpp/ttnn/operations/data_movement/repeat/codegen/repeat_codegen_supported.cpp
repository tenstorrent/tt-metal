// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/repeat/codegen/repeat_codegen_supported.hpp"

namespace ttnn::operations::data_movement::repeat_codegen {

bool supported_by_codegen(const Tensor& /*input*/, uint32_t /*rep_dim*/, uint32_t /*num_repeats*/) { return false; }

}  // namespace ttnn::operations::data_movement::repeat_codegen
